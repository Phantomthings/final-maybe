// dispo_ac
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

// Config
var (
	// Influx
	influxHost       = env("INFLUX_HOST", "tsdbe.nidec-asi-online.com")
	influxPort       = env("INFLUX_PORT", "443")
	influxUser       = env("INFLUX_USER", "Elto")
	influxPw         = env("INFLUX_PW", "NidecItadmElto")
	influxDB         = env("INFLUX_DB", "Elto")
	influxMeas       = env("INFLUX_MEAS", "elto1sec_box")
	influxTagProject = env("INFLUX_TAG_PROJECT", "project")

	// MySQL
	mysqlHost = env("MYSQL_HOST", "141.94.31.144")
	mysqlPort = env("MYSQL_PORT", "3306")
	mysqlUser = env("MYSQL_USER", "AdminNidec")
	mysqlPw   = env("MYSQL_PASSWORD", "u6Ehe987XBSXxa4")
	mysqlDB   = env("MYSQL_DB", "indicator")

	fieldSEQ01PC1 = "SEQ01.OLI.A.PC1"
	fieldSEQ01IC1 = "SEQ01.OLI.A.IC1"

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

// TZ Paris
var paris = mustLoad("Europe/Paris")

// Types
type pair struct {
	pc1 *int
	ic1 *int
}
type bloc struct {
	site          string
	equip         string
	dateDebut     time.Time
	dateFin       time.Time
	estDispo      int
	cause         *string
	rawPointCount int
	batchID       string
	hashSig       string
}

// Utils
func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
func envInt(k string, def int) int {
	if v := os.Getenv(k); v != "" {
		var x int
		_, err := fmt.Sscanf(v, "%d", &x)
		if err == nil {
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
func midnightParis(t time.Time) time.Time {
	lt := t.In(paris)
	return time.Date(lt.Year(), lt.Month(), lt.Day(), 0, 0, 0, 0, paris)
}
func sanitizeSite(site string) string {
	s := strings.TrimSpace(site)
	s = strings.ReplaceAll(s, "-", "_")
	return s
}

// Clients
func influxClient() influx.Client {
	u := fmt.Sprintf("https://%s:%s", influxHost, influxPort)
	c, err := influx.NewHTTPClient(influx.HTTPConfig{
		Addr:     u,
		Username: influxUser,
		Password: influxPw,
	})
	if err != nil {
		log.Fatalf("Influx client: %v", err)
	}
	return c
}
func mysqlDBConn() *sql.DB {
	// MODIFICATION: Ajout de &loc=Europe%2FParis pour forcer le fuseau horaire
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

// Influx
func fetchMinuteGridUTC_AC(cli influx.Client, project string, startUTC, endUTC time.Time) (map[time.Time]pair, error) {
	startISO := startUTC.UTC().Format(time.RFC3339)
	endISO := endUTC.UTC().Format(time.RFC3339)

	q := fmt.Sprintf(`
SELECT
  last("%s") AS "%s",
  last("%s") AS "%s"
FROM "%s"
WHERE time >= '%s' AND time < '%s' AND "%s"='%s'
GROUP BY time(1m) fill(none)
`,
		fieldSEQ01PC1, fieldSEQ01PC1,
		fieldSEQ01IC1, fieldSEQ01IC1,
		influxMeas, startISO, endISO, influxTagProject, project)

	resp, err := cli.Query(influx.NewQuery(q, influxDB, ""))
	if err != nil {
		return nil, fmt.Errorf("influx query: %w", err)
	}
	if resp.Error() != nil {
		return nil, fmt.Errorf("influx resp: %w", resp.Error())
	}

	ac := map[time.Time]pair{}

	if len(resp.Results) > 0 && len(resp.Results[0].Series) > 0 {
		s := resp.Results[0].Series[0]
		cols := s.Columns
		if len(cols) >= 3 {
			iTime := 0
			iPC, iIC := 1, 2

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

			for _, v := range s.Values {
				tsStr, _ := v[iTime].(string)
				t, err := time.Parse(time.RFC3339, tsStr)
				if err != nil {
					continue
				}
				ac[t.UTC()] = pair{pc1: toIntPtr(v[iPC]), ic1: toIntPtr(v[iIC])}
			}
		}
	}

	for t := startUTC.UTC(); t.Before(endUTC.UTC()); t = t.Add(time.Minute) {
		if _, ok := ac[t]; !ok {
			ac[t] = pair{pc1: nil, ic1: nil}
		}
	}
	return ac, nil
}

// Classification
func classifyAC(pc1, ic1 *int) (int, *string) {
	if pc1 == nil || ic1 == nil {
		msg := "Donnée manquante"
		return -1, &msg
	}
	if *pc1 == 0 && *ic1 == 0 {
		return 1, nil
	}
	msg := fmt.Sprintf("PC1=%d,IC1=%d", *pc1, *ic1)
	return 0, &msg
}

func buildBlocksAC(site string, series map[time.Time]pair, batchID string) []bloc {
	type row struct {
		t     time.Time
		state int
		cause *string
		pc1   *int
		ic1   *int
	}
	rows := make([]row, 0, len(series))
	for t, v := range series {
		st, cause := classifyAC(v.pc1, v.ic1)
		rows = append(rows, row{t: t, state: st, cause: cause, pc1: v.pc1, ic1: v.ic1})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].t.Before(rows[j].t) })
	if len(rows) == 0 {
		return nil
	}

	curState := rows[0].state
	curCause := rows[0].cause
	start := rows[0].t
	rawCount := 0
	if rows[0].pc1 != nil && rows[0].ic1 != nil {
		rawCount = 1
	}

	blocks := []bloc{}
	for i := 1; i < len(rows); i++ {
		r := rows[i]
		c1 := causeKey(curCause)
		c2 := causeKey(r.cause)
		if r.state != curState || c1 != c2 {
			end := rows[i-1].t.Add(time.Minute)
			if curState == 0 && int(end.Sub(start).Minutes()) < minIndispoMinutes {
				curState = r.state
				curCause = r.cause
				start = r.t
				rawCount = 0
				if r.pc1 != nil && r.ic1 != nil {
					rawCount = 1
				}
				continue
			}
			h := hashSig(site, "AC", start, end, curState, curCause)
			blocks = append(blocks, bloc{
				site:          site,
				equip:         "AC",
				dateDebut:     start,
				dateFin:       end,
				estDispo:      curState,
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
		if r.pc1 != nil && r.ic1 != nil {
			rawCount++
		}
	}
	end := rows[len(rows)-1].t.Add(time.Minute)
	if curState != 0 || int(end.Sub(start).Minutes()) >= minIndispoMinutes {
		h := hashSig(site, "AC", start, end, curState, curCause)
		blocks = append(blocks, bloc{
			site:          site,
			equip:         "AC",
			dateDebut:     start,
			dateFin:       end,
			estDispo:      curState,
			cause:         curCause,
			rawPointCount: rawCount,
			batchID:       batchID,
			hashSig:       h,
		})
	}
	return blocks
}
func causeKey(c *string) string {
	if c == nil {
		return "OK"
	}
	return *c
}
func hashSig(site, equip string, start, end time.Time, state int, cause *string) string {
	src := fmt.Sprintf("%s|%s|%s|%s|%d|%s",
		site, equip, start.UTC().Format(time.RFC3339), end.UTC().Format(time.RFC3339), state, causeKey(cause))
	sum := sha256.Sum256([]byte(src))
	return hex.EncodeToString(sum[:])
}

// SQL
func tableAC(site string) string {
	return fmt.Sprintf("indicator.dispo_blocs_ac_%s", sanitizeSite(site))
}
func upsertStmtAC(t string) string {
	return fmt.Sprintf(`
INSERT INTO %s
(site, equipement_id, type_equipement, date_debut, date_fin, est_disponible, cause,
 raw_point_count, processed_at, batch_id, hash_signature)
VALUES (?, 'AC', 'AC', ?, ?, ?, ?, ?, UTC_TIMESTAMP(), ?, ?)
ON DUPLICATE KEY UPDATE
  est_disponible=VALUES(est_disponible),
  cause=VALUES(cause),
  raw_point_count=VALUES(raw_point_count),
  processed_at=UTC_TIMESTAMP(),
  batch_id=VALUES(batch_id),
  hash_signature=VALUES(hash_signature)`, t)
}

func saveBlocksPerSiteAC(db *sql.DB, site string, blocks []bloc) (int, error) {
	if len(blocks) == 0 {
		return 0, nil
	}
	stmtAC, err := db.Prepare(upsertStmtAC(tableAC(site)))
	if err != nil {
		return 0, err
	}
	defer stmtAC.Close()

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
		// GARDER .In(paris) - maintenant que loc=Europe/Paris est dans la DSN,
		// le driver va correctement interpréter le fuseau horaire
		startStore := b.dateDebut.In(paris)
		endStore := b.dateFin.In(paris)
		args := []any{b.site, startStore, endStore, b.estDispo, cause, b.rawPointCount, b.batchID, b.hashSig}
		if _, execErr := tx.Stmt(stmtAC).Exec(args...); execErr != nil {
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
func processWindowAC(project string, startUTC, endUTC time.Time, ic influx.Client, db *sql.DB) (int, error) {
	log.Printf("AC %s : %s → %s (UTC)", project, startUTC.Format(time.RFC3339), endUTC.Format(time.RFC3339))

	acm, err := fetchMinuteGridUTC_AC(ic, project, startUTC, endUTC)
	if err != nil {
		return 0, err
	}
	batchID := time.Now().UTC().Format("20060102T150405Z")

	blocks := buildBlocksAC(project, acm, batchID)
	n, err := saveBlocksPerSiteAC(db, project, blocks)
	if err != nil {
		return 0, err
	}
	log.Printf("%s: %d bloc(s) AC upsertés", project, n)
	return n, nil
}

func firstDateInfluxAC(site string, ic influx.Client) (*time.Time, error) {
	q := fmt.Sprintf(`SELECT first("%s") FROM "%s" WHERE "%s"='%s'`,
		fieldSEQ01PC1, influxMeas, influxTagProject, site)
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
	dParis := midnightParis(tUTC)
	return &dParis, nil
}
func lastDateDBAC(site string, db *sql.DB) (*time.Time, error) {
	var d sql.NullTime
	q := fmt.Sprintf(`SELECT MAX(date_debut) FROM %s`, tableAC(site))
	if err := db.QueryRow(q).Scan(&d); err != nil {
		return nil, err
	}
	if d.Valid {
		t := d.Time
		return &t, nil
	}
	return nil, nil
}
func processDayParisAC(site string, dayParis time.Time, ic influx.Client, db *sql.DB) (int, error) {
	startLocal := time.Date(dayParis.Year(), dayParis.Month(), dayParis.Day(), 0, 0, 0, 0, paris)
	endLocal := startLocal.Add(24 * time.Hour)
	return processWindowAC(site, startLocal.UTC(), endLocal.UTC(), ic, db)
}
func autoUpdateAllAC(projects []string) error {
	ic := influxClient()
	defer ic.Close()
	db := mysqlDBConn()
	defer db.Close()

	nowP := time.Now().In(paris)
	todayP := time.Date(nowP.Year(), nowP.Month(), nowP.Day(), 0, 0, 0, 0, paris)
	yesterdayP := todayP.Add(-24 * time.Hour)

	totalAC := 0
	for i, site := range projects {
		ld, err := lastDateDBAC(site, db)
		var startDayP time.Time
		if err != nil {
			log.Printf("lastDateDBAC %s: %v", site, err)
			continue
		}
		if ld != nil {
			startDayP = midnightParis(*ld).Add(24 * time.Hour)
		} else {
			fd, err := firstDateInfluxAC(site, ic)
			if err != nil {
				log.Printf("firstDateInfluxAC %s: %v", site, err)
				continue
			}
			if fd == nil {
				startDayP = todayP
			} else {
				startDayP = *fd
			}
		}

		for d := startDayP; !d.After(yesterdayP); d = d.Add(24 * time.Hour) {
			n, err := processDayParisAC(site, d, ic, db)
			if err != nil {
				log.Printf("processDayAC %s %s: %v", site, d.In(paris).Format("2006-01-02"), err)
				continue
			}
			totalAC += n
			time.Sleep(250 * time.Millisecond)
		}

		log.Printf("OK %s (%d/%d)", site, i+1, len(projects))
	}
	log.Printf("Terminé: %d blocs AC", totalAC)
	return nil
}

// CLI
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
			n, err := processWindowAC(site, startP.UTC(), endP.UTC(), ic, db)
			if err != nil {
				log.Printf("Erreur %s: %v", site, err)
				continue
			}
			total += n
			time.Sleep(250 * time.Millisecond)
		}
		log.Printf("MANUEL terminé: %d blocs AC", total)
		return
	}

	// Mode autoupdate
	if err := autoUpdateAllAC(projects); err != nil {
		log.Fatalf("autoUpdateAllAC: %v", err)
	}
}
