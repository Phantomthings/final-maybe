// dispo_pdc
package main

import (
	"crypto/sha256"
	"database/sql"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"

	_ "github.com/go-sql-driver/mysql"
	influx "github.com/influxdata/influxdb1-client/v2"
)

var (
	influxHost       = env("INFLUX_HOST", "tsdbe.nidec-asi-online.com")
	influxPort       = env("INFLUX_PORT", "443")
	influxUser       = env("INFLUX_USER", "Elto")
	influxPw         = env("INFLUX_PW", "NidecItadmElto")
	influxDB         = env("INFLUX_DB", "Elto")
	influxMeas       = env("INFLUX_MEAS", "elto1sec_borne")
	influxTagProject = env("INFLUX_TAG_PROJECT", "project")

	mysqlHost = env("MYSQL_HOST", "141.94.31.144")
	mysqlPort = env("MYSQL_PORT", "3306")
	mysqlUser = env("MYSQL_USER", "AdminNidec")
	mysqlPw   = env("MYSQL_PASSWORD", "u6Ehe987XBSXxa4")
	mysqlDB   = env("MYSQL_DB", "indicator")

	minIndispoMinutes = envInt("MIN_INDISPO_MINUTES", 5)

	defaultProjects = func() []string {
		raw := env("PROJECTS_LIST", "8822_001,8822_002,8822_003,8822_004,8822_005,8822_006")
		var out []string
		for _, p := range strings.Split(raw, ",") {
			p = strings.TrimSpace(p)
			if p != "" {
				out = append(out, p)
			}
		}
		return out
	}()
)

var paris = mustLoad("Europe/Paris")

type pdcConfig struct {
	ID     int
	Prefix string
	Label  string
}

type sample struct {
	ready *bool
	pc1   *int
	ic1   *int
}

type bloc struct {
	site          string
	pdcID         string
	dateDebut     time.Time
	dateFin       time.Time
	etat          int
	cause         *string
	rawPointCount int
	batchID       string
	hashSig       string
}

var pdcConfigs = []pdcConfig{
	{ID: 1, Prefix: "SEQ12", Label: "PDC1"},
	{ID: 2, Prefix: "SEQ22", Label: "PDC2"},
	{ID: 3, Prefix: "SEQ13", Label: "PDC3"},
	{ID: 4, Prefix: "SEQ23", Label: "PDC4"},
	{ID: 5, Prefix: "SEQ14", Label: "PDC5"},
	{ID: 6, Prefix: "SEQ24", Label: "PDC6"},
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}

func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		var x int
		if _, err := fmt.Sscanf(v, "%d", &x); err == nil {
			return x
		}
	}
	return def
}

func mustLoad(name string) *time.Location {
	loc, err := time.LoadLocation(name)
	if err != nil {
		log.Fatalf("load tz %s: %v", name, err)
	}
	return loc
}

func sanitizeSite(site string) string {
	s := strings.TrimSpace(site)
	return strings.ReplaceAll(s, "-", "_")
}

func influxClient() influx.Client {
	addr := fmt.Sprintf("https://%s:%s", influxHost, influxPort)
	c, err := influx.NewHTTPClient(influx.HTTPConfig{
		Addr:     addr,
		Username: influxUser,
		Password: influxPw,
	})
	if err != nil {
		log.Fatalf("Influx client: %v", err)
	}
	return c
}

func mysqlDBConn() *sql.DB {
	dsn := fmt.Sprintf("%s:%s@tcp(%s:%s)/%s?parseTime=true&charset=utf8mb4&loc=Europe%%2FParis",
		mysqlUser, mysqlPw, mysqlHost, mysqlPort, mysqlDB)
	db, err := sql.Open("mysql", dsn)
	if err != nil {
		log.Fatalf("MySQL open: %v", err)
	}
	if err := db.Ping(); err != nil {
		log.Fatalf("MySQL ping: %v", err)
	}
	return db
}

func fieldReady(cfg pdcConfig) string { return cfg.Prefix + ".B_Ready" }
func fieldPC1(cfg pdcConfig) string   { return cfg.Prefix + ".OLI.A.PC1" }
func fieldIC1(cfg pdcConfig) string   { return cfg.Prefix + ".OLI.A.IC1" }

func fetchMinuteGridUTC(ic influx.Client, project string, cfg pdcConfig, startUTC, endUTC time.Time) (map[time.Time]sample, error) {
	startISO := startUTC.UTC().Format(time.RFC3339)
	endISO := endUTC.UTC().Format(time.RFC3339)

	q := fmt.Sprintf(`
SELECT
  last("%s") AS "%s",
  last("%s") AS "%s",
  last("%s") AS "%s"
FROM "%s"
WHERE time >= '%s' AND time < '%s' AND "%s"='%s'
GROUP BY time(1m) fill(none)
`,
		fieldReady(cfg), fieldReady(cfg),
		fieldPC1(cfg), fieldPC1(cfg),
		fieldIC1(cfg), fieldIC1(cfg),
		influxMeas, startISO, endISO, influxTagProject, project)

	resp, err := ic.Query(influx.NewQuery(q, influxDB, ""))
	if err != nil {
		return nil, fmt.Errorf("influx query: %w", err)
	}
	if resp.Error() != nil {
		return nil, fmt.Errorf("influx resp: %w", resp.Error())
	}

	out := map[time.Time]sample{}

	if len(resp.Results) > 0 && len(resp.Results[0].Series) > 0 {
		s := resp.Results[0].Series[0]
		cols := s.Columns
		if len(cols) >= 4 {
			iTime := 0
			iReady, iPC, iIC := 1, 2, 3

			toBoolPtr := func(x interface{}) *bool {
				if x == nil {
					return nil
				}
				switch v := x.(type) {
				case bool:
					b := v
					return &b
				case float64:
					b := v != 0
					return &b
				case json.Number:
					if iv, err := strconv.Atoi(v.String()); err == nil {
						b := iv != 0
						return &b
					}
					return nil
				default:
					return nil
				}
			}

			toIntPtr := func(x interface{}) *int {
				if x == nil {
					return nil
				}
				switch v := x.(type) {
				case float64:
					i := int(v)
					return &i
				case json.Number:
					if iv, err := strconv.Atoi(v.String()); err == nil {
						return &iv
					}
					return nil
				default:
					return nil
				}
			}

			for _, row := range s.Values {
				ts, _ := row[iTime].(string)
				t, err := time.Parse(time.RFC3339, ts)
				if err != nil {
					continue
				}
				out[t.UTC()] = sample{
					ready: toBoolPtr(row[iReady]),
					pc1:   toIntPtr(row[iPC]),
					ic1:   toIntPtr(row[iIC]),
				}
			}
		}
	}

	for t := startUTC.UTC(); t.Before(endUTC.UTC()); t = t.Add(time.Minute) {
		if _, ok := out[t]; !ok {
			out[t] = sample{}
		}
	}
	return out, nil
}

func classify(s sample) (int, *string) {
	if s.ic1 == nil {
		if s.ready == nil && s.pc1 == nil {
			msg := "Donnée manquante"
			return -1, &msg
		}
		msg := "Donnée partielle"
		return -1, &msg
	}
	if *s.ic1 == 1024 {
		return 1, nil
	}
	if *s.ic1 == 0 {
		if s.pc1 != nil && *s.pc1 == 0 {
			return 1, nil
		}
	}
	cause := buildCause(s)
	return 0, &cause
}

func buildCause(s sample) string {
	pc := "NA"
	ic := "NA"
	if s.pc1 != nil {
		pc = strconv.Itoa(*s.pc1)
	}
	if s.ic1 != nil {
		ic = strconv.Itoa(*s.ic1)
	}
	return fmt.Sprintf("PC1=%s,IC1=%s", pc, ic)
}

func buildBlocks(site string, cfg pdcConfig, series map[time.Time]sample, batchID string) []bloc {
	if len(series) == 0 {
		return nil
	}
	type row struct {
		t     time.Time
		state int
		cause *string
		val   sample
	}
	rows := make([]row, 0, len(series))
	for t, s := range series {
		st, cause := classify(s)
		rows = append(rows, row{t: t, state: st, cause: cause, val: s})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].t.Before(rows[j].t) })
	if len(rows) == 0 {
		return nil
	}

	curState := rows[0].state
	curCause := rows[0].cause
	start := rows[0].t
	rawCount := 0
	if rows[0].val.ready != nil || rows[0].val.ic1 != nil {
		rawCount = 1
	}

	blocks := []bloc{}
	for i := 1; i < len(rows); i++ {
		r := rows[i]
		causeKey := func(c *string) string {
			if c == nil {
				return "OK"
			}
			return *c
		}
		if r.state != curState || causeKey(r.cause) != causeKey(curCause) {
			end := rows[i-1].t.Add(time.Minute)
			if curState == 0 && int(end.Sub(start).Minutes()) < minIndispoMinutes {
				curState = r.state
				curCause = r.cause
				start = r.t
				rawCount = 0
				if r.val.ready != nil || r.val.ic1 != nil {
					rawCount = 1
				}
				continue
			}
			h := hashSig(site, cfg.Label, start, end, curState, curCause)
			blocks = append(blocks, bloc{
				site:          site,
				pdcID:         cfg.Label,
				dateDebut:     start,
				dateFin:       end,
				etat:          curState,
				cause:         curCause,
				rawPointCount: rawCount,
				batchID:       batchID,
				hashSig:       h,
			})
			curState = r.state
			curCause = r.cause
			start = r.t
			rawCount = 0
		}
		if r.val.ready != nil || r.val.ic1 != nil {
			rawCount++
		}
	}
	end := rows[len(rows)-1].t.Add(time.Minute)
	if curState != 0 || int(end.Sub(start).Minutes()) >= minIndispoMinutes {
		h := hashSig(site, cfg.Label, start, end, curState, curCause)
		blocks = append(blocks, bloc{
			site:          site,
			pdcID:         cfg.Label,
			dateDebut:     start,
			dateFin:       end,
			etat:          curState,
			cause:         curCause,
			rawPointCount: rawCount,
			batchID:       batchID,
			hashSig:       h,
		})
	}
	return blocks
}

func hashSig(site, equip string, start, end time.Time, state int, cause *string) string {
	causeStr := "OK"
	if cause != nil {
		causeStr = *cause
	}
	src := fmt.Sprintf("%s|%s|%s|%s|%d|%s",
		site, equip, start.UTC().Format(time.RFC3339), end.UTC().Format(time.RFC3339), state, causeStr)
	sum := sha256.Sum256([]byte(src))
	return hex.EncodeToString(sum[:])
}

func tableName(site string, cfg pdcConfig) string {
	return fmt.Sprintf("indicator.dispo_pdc_n%d_%s", cfg.ID, sanitizeSite(site))
}

func upsertStmt(tbl string) string {
	return fmt.Sprintf(`
INSERT INTO %s
(site, pdc_id, type_label, date_debut, date_fin, etat, cause, raw_point_count, processed_at, batch_id, hash_signature)
VALUES (?, ?, 'PDC', ?, ?, ?, ?, ?, UTC_TIMESTAMP(), ?, ?)
ON DUPLICATE KEY UPDATE
  etat=VALUES(etat),
  cause=VALUES(cause),
  raw_point_count=VALUES(raw_point_count),
  processed_at=UTC_TIMESTAMP(),
  batch_id=VALUES(batch_id),
  hash_signature=VALUES(hash_signature)
`, tbl)
}

func saveBlocks(db *sql.DB, site string, cfg pdcConfig, blocks []bloc) (int, error) {
	if len(blocks) == 0 {
		return 0, nil
	}
	stmt, err := db.Prepare(upsertStmt(tableName(site, cfg)))
	if err != nil {
		return 0, err
	}
	defer stmt.Close()

	tx, err := db.Begin()
	if err != nil {
		return 0, err
	}
	n := 0
	for _, b := range blocks {
		cause := sql.NullString{}
		if b.cause != nil {
			cause = sql.NullString{String: *b.cause, Valid: true}
		}
		startStore := b.dateDebut.In(paris)
		endStore := b.dateFin.In(paris)
		args := []any{b.site, b.pdcID, startStore, endStore, b.etat, cause, b.rawPointCount, b.batchID, b.hashSig}
		if _, execErr := tx.Stmt(stmt).Exec(args...); execErr != nil {
			_ = tx.Rollback()
			return n, execErr
		}
		n++
	}
	if err := tx.Commit(); err != nil {
		return n, err
	}
	return n, nil
}

func processWindow(site string, cfg pdcConfig, startUTC, endUTC time.Time, ic influx.Client, db *sql.DB) (int, error) {
	log.Printf("%s %s : %s → %s (UTC)", cfg.Label, site, startUTC.Format(time.RFC3339), endUTC.Format(time.RFC3339))
	series, err := fetchMinuteGridUTC(ic, site, cfg, startUTC, endUTC)
	if err != nil {
		return 0, err
	}
	batchID := fmt.Sprintf("%s-%s", time.Now().UTC().Format("20060102T150405Z"), cfg.Label)
	blocks := buildBlocks(site, cfg, series, batchID)
	n, err := saveBlocks(db, site, cfg, blocks)
	if err != nil {
		return 0, err
	}
	log.Printf("%s %s: %d bloc(s) upsertés", site, cfg.Label, n)
	return n, nil
}

func firstDateInflux(site string, cfg pdcConfig, ic influx.Client) (*time.Time, error) {
	q := fmt.Sprintf(`SELECT first("%s") FROM "%s" WHERE "%s"='%s'`, fieldReady(cfg), influxMeas, influxTagProject, site)
	resp, err := ic.Query(influx.NewQuery(q, influxDB, ""))
	if err != nil {
		return nil, err
	}
	if resp.Error() != nil {
		return nil, resp.Error()
	}
	if len(resp.Results) == 0 || len(resp.Results[0].Series) == 0 || len(resp.Results[0].Series[0].Values) == 0 {
		return nil, nil
	}
	ts, _ := resp.Results[0].Series[0].Values[0][0].(string)
	tUTC, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		return nil, err
	}
	dp := time.Date(tUTC.In(paris).Year(), tUTC.In(paris).Month(), tUTC.In(paris).Day(), 0, 0, 0, 0, paris)
	return &dp, nil
}

func lastDateDB(site string, cfg pdcConfig, db *sql.DB) (*time.Time, error) {
	var d sql.NullTime
	q := fmt.Sprintf(`SELECT MAX(date_debut) FROM %s`, tableName(site, cfg))
	if err := db.QueryRow(q).Scan(&d); err != nil {
		return nil, err
	}
	if d.Valid {
		t := d.Time
		return &t, nil
	}
	return nil, nil
}

func processDayParis(site string, cfg pdcConfig, day time.Time, ic influx.Client, db *sql.DB) (int, error) {
	startLocal := time.Date(day.Year(), day.Month(), day.Day(), 0, 0, 0, 0, paris)
	endLocal := startLocal.Add(24 * time.Hour)
	return processWindow(site, cfg, startLocal.UTC(), endLocal.UTC(), ic, db)
}

func autoUpdateAll(projects []string) error {
	ic := influxClient()
	defer ic.Close()
	db := mysqlDBConn()
	defer db.Close()

	nowP := time.Now().In(paris)
	todayP := time.Date(nowP.Year(), nowP.Month(), nowP.Day(), 0, 0, 0, 0, paris)
	yesterdayP := todayP.Add(-24 * time.Hour)

	total := 0
	for _, site := range projects {
		for _, cfg := range pdcConfigs {
			ld, err := lastDateDB(site, cfg, db)
			if err != nil {
				log.Printf("lastDateDB %s %s: %v", site, cfg.Label, err)
				continue
			}
			var startDay time.Time
			if ld != nil {
				startDay = time.Date(ld.In(paris).Year(), ld.In(paris).Month(), ld.In(paris).Day(), 0, 0, 0, 0, paris).Add(24 * time.Hour)
			} else {
				fd, err := firstDateInflux(site, cfg, ic)
				if err != nil {
					log.Printf("firstDateInflux %s %s: %v", site, cfg.Label, err)
					continue
				}
				if fd == nil {
					startDay = todayP
				} else {
					startDay = *fd
				}
			}

			for d := startDay; !d.After(yesterdayP); d = d.Add(24 * time.Hour) {
				n, err := processDayParis(site, cfg, d, ic, db)
				if err != nil {
					log.Printf("processDay %s %s %s: %v", site, cfg.Label, d.Format("2006-01-02"), err)
					continue
				}
				total += n
				time.Sleep(250 * time.Millisecond)
			}
		}
		log.Printf("OK %s", site)
	}
	log.Printf("Terminé: %d blocs PDC", total)
	return nil
}

type multiString []string

func (m *multiString) String() string     { return strings.Join(*m, ",") }
func (m *multiString) Set(v string) error { *m = append(*m, v); return nil }

func main() {
	var projFlags multiString
	flag.Var(&projFlags, "project", "Projet (répétable). Ex: -project 8822_001 -project 8822_004")
	start := flag.String("start-date", "", "YYYY-MM-DD (Paris) — mode manuel (inclus)")
	end := flag.String("end-date", "", "YYYY-MM-DD (Paris) — mode manuel (exclu)")
	flag.Parse()

	projects := defaultProjects
	if len(projFlags) > 0 {
		projects = projFlags
	}

	ic := influxClient()
	defer ic.Close()
	db := mysqlDBConn()
	defer db.Close()

	if *start != "" && *end != "" {
		startP, err := time.ParseInLocation("2006-01-02", *start, paris)
		if err != nil {
			log.Fatalf("start-date invalide: %v", err)
		}
		endP, err := time.ParseInLocation("2006-01-02", *end, paris)
		if err != nil {
			log.Fatalf("end-date invalide: %v", err)
		}
		total := 0
		for _, site := range projects {
			for _, cfg := range pdcConfigs {
				n, err := processWindow(site, cfg, startP.UTC(), endP.UTC(), ic, db)
				if err != nil {
					log.Printf("Erreur %s %s: %v", site, cfg.Label, err)
					continue
				}
				total += n
				time.Sleep(250 * time.Millisecond)
			}
		}
		log.Printf("MANUEL terminé: %d blocs PDC", total)
		return
	}

	if err := autoUpdateAll(projects); err != nil {
		log.Fatalf("autoUpdateAll: %v", err)
	}
}
