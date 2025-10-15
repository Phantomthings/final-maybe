package main

import (
	"database/sql"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	_ "github.com/go-sql-driver/mysql"
)

var (
	mysqlHost = env("MYSQL_HOST", "141.94.31.144")
	mysqlPort = env("MYSQL_PORT", "3306")
	mysqlUser = env("MYSQL_USER", "AdminNidec")
	mysqlPw   = env("MYSQL_PASSWORD", "u6Ehe987XBSXxa4")
	mysqlDB   = env("MYSQL_DB", "indicator")

	paris = mustLoad("Europe/Paris")

	stepDuration       = 10 * time.Minute
	checkpointsPerStep = 10
	checkpointInterval = time.Minute
)

type checkpoint struct {
	t time.Time
}

type stepData struct {
	start       time.Time
	end         time.Time
	checkpoints []checkpoint
	isT3        bool
	hasData     bool
	value       float64
}

type blocRecord struct {
	equipID   string
	dateDebut time.Time
	dateFin   time.Time
	etat      int
}

type exclusionRecord struct {
	dateDebut time.Time
	dateFin   time.Time
}

func env(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
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

// Génère tous les pas de 10 min pour le mois
func generateSteps(year int, month time.Month) []stepData {
	start := time.Date(year, month, 1, 0, 0, 0, 0, paris)
	var end time.Time
	if month == 12 {
		end = time.Date(year+1, 1, 1, 0, 0, 0, 0, paris)
	} else {
		end = time.Date(year, month+1, 1, 0, 0, 0, 0, paris)
	}

	var steps []stepData
	for t := start; t.Before(end); t = t.Add(stepDuration) {
		stepEnd := t.Add(stepDuration)
		var cps []checkpoint
		for i := 0; i < checkpointsPerStep; i++ {
			cpTime := t.Add(time.Duration(i)*checkpointInterval + 30*time.Second)
			cps = append(cps, checkpoint{t: cpTime})
		}
		steps = append(steps, stepData{
			start:       t,
			end:         stepEnd,
			checkpoints: cps,
			isT3:        false,
			hasData:     false,
			value:       0.0,
		})
	}
	return steps
}

// Charge les blocs AC/BATT (avec est_disponible)
func loadBlocsACBatt(db *sql.DB, site, table string, start, end time.Time) ([]blocRecord, error) {
	query := fmt.Sprintf(`
		SELECT COALESCE(equipement_id, ''), date_debut, date_fin, 
		       COALESCE(est_disponible, -1) as etat
		FROM %s
		WHERE site = ? 
		  AND date_fin > ? 
		  AND date_debut < ?
		ORDER BY date_debut
	`, table)

	rows, err := db.Query(query, site, start.Format("2006-01-02 15:04:05"), end.Format("2006-01-02 15:04:05"))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var blocs []blocRecord
	for rows.Next() {
		var b blocRecord
		if err := rows.Scan(&b.equipID, &b.dateDebut, &b.dateFin, &b.etat); err != nil {
			return nil, err
		}
		blocs = append(blocs, b)
	}
	return blocs, rows.Err()
}

// Charge les blocs PDC (avec etat et pdc_id au lieu de equipement_id)
func loadBlocsPDC(db *sql.DB, site, table string, start, end time.Time) ([]blocRecord, error) {
	query := fmt.Sprintf(`
		SELECT COALESCE(pdc_id, ''), date_debut, date_fin, 
		       COALESCE(etat, -1) as etat
		FROM %s
		WHERE site = ? 
		  AND date_fin > ? 
		  AND date_debut < ?
		ORDER BY date_debut
	`, table)

	rows, err := db.Query(query, site, start.Format("2006-01-02 15:04:05"), end.Format("2006-01-02 15:04:05"))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var blocs []blocRecord
	for rows.Next() {
		var b blocRecord
		if err := rows.Scan(&b.equipID, &b.dateDebut, &b.dateFin, &b.etat); err != nil {
			return nil, err
		}
		blocs = append(blocs, b)
	}
	return blocs, rows.Err()
}

// Charge les exclusions (T3) pour le site
func loadExclusions(db *sql.DB, site string, start, end time.Time) ([]exclusionRecord, error) {
	query := `
		SELECT date_debut, date_fin
		FROM dispo_annotations
		WHERE site = ? 
		  AND type_annotation = 'exclusion'
		  AND actif = 1
		  AND date_fin > ? 
		  AND date_debut < ?
		ORDER BY date_debut
	`

	rows, err := db.Query(query, site, start.Format("2006-01-02 15:04:05"), end.Format("2006-01-02 15:04:05"))
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var excl []exclusionRecord
	for rows.Next() {
		var e exclusionRecord
		if err := rows.Scan(&e.dateDebut, &e.dateFin); err != nil {
			return nil, err
		}
		excl = append(excl, e)
	}
	return excl, rows.Err()
}

// Vérifie si un pas est couvert par une exclusion (même partiellement)
func isStepExcluded(step stepData, exclusions []exclusionRecord) bool {
	for _, excl := range exclusions {
		if excl.dateDebut.Before(step.end) && excl.dateFin.After(step.start) {
			return true
		}
	}
	return false
}

// Trouve l'état d'un équipement à un instant donné
func getStateAt(t time.Time, blocs []blocRecord) *int {
	for _, b := range blocs {
		if !t.Before(b.dateDebut) && t.Before(b.dateFin) {
			return &b.etat
		}
	}
	return nil
}

// Calcule la disponibilité pour un mois
func calculateMonthly(db *sql.DB, site string, year int, month time.Month) error {
	log.Printf("Calcul pour %s - %d-%02d", site, year, month)

	periodStart := time.Date(year, month, 1, 0, 0, 0, 0, paris)
	var periodEnd time.Time
	if month == 12 {
		periodEnd = time.Date(year+1, 1, 1, 0, 0, 0, 0, paris)
	} else {
		periodEnd = time.Date(year, month+1, 1, 0, 0, 0, 0, paris)
	}

	// Génération des pas
	steps := generateSteps(year, month)

	// Chargement des exclusions
	exclusions, err := loadExclusions(db, site, periodStart, periodEnd)
	if err != nil {
		return fmt.Errorf("load exclusions: %w", err)
	}

	// Marquer les pas T3
	for i := range steps {
		steps[i].isT3 = isStepExcluded(steps[i], exclusions)
	}

	// Chargement des données
	acBlocs, err := loadBlocsACBatt(db, site, fmt.Sprintf("dispo_blocs_ac_%s", strings.ReplaceAll(site, "-", "_")), periodStart, periodEnd)
	if err != nil {
		log.Printf("Warning: AC blocs: %v", err)
	}

	dc1Blocs, err := loadBlocsACBatt(db, site, fmt.Sprintf("dispo_blocs_batt_%s", strings.ReplaceAll(site, "-", "_")), periodStart, periodEnd)
	if err != nil {
		log.Printf("Warning: DC1 blocs: %v", err)
	}

	dc2Blocs, err := loadBlocsACBatt(db, site, fmt.Sprintf("dispo_blocs_batt2_%s", strings.ReplaceAll(site, "-", "_")), periodStart, periodEnd)
	if err != nil {
		log.Printf("Warning: DC2 blocs: %v", err)
	}

	var pdcBlocs [6][]blocRecord
	for i := 1; i <= 6; i++ {
		blocs, err := loadBlocsPDC(db, site, fmt.Sprintf("dispo_pdc_n%d_%s", i, strings.ReplaceAll(site, "-", "_")), periodStart, periodEnd)
		if err != nil {
			log.Printf("Warning: PDC%d blocs: %v", i, err)
		}
		pdcBlocs[i-1] = blocs
	}

	// Traitement des pas
	t2Count := 0
	t3Count := 0
	tSum := 0.0

	for i := range steps {
		step := &steps[i]

		if step.isT3 {
			t3Count++
			t2Count++
			continue
		}

		// Vérifier si le pas a des données (data-driven T2)
		hasAnyData := false
		for _, cp := range step.checkpoints {
			if getStateAt(cp.t, acBlocs) != nil {
				hasAnyData = true
				break
			}
			if getStateAt(cp.t, dc1Blocs) != nil || getStateAt(cp.t, dc2Blocs) != nil {
				hasAnyData = true
				break
			}
			for pdcIdx := 0; pdcIdx < 6; pdcIdx++ {
				if getStateAt(cp.t, pdcBlocs[pdcIdx]) != nil {
					hasAnyData = true
					break
				}
			}
			if hasAnyData {
				break
			}
		}

		if !hasAnyData {
			continue // Pas non retenu dans T2
		}

		t2Count++
		step.hasData = true

		// Évaluation du pas
		stepBlocked := false
		var validContributions []float64

		for _, cp := range step.checkpoints {
			// Gate 1: AC
			acState := getStateAt(cp.t, acBlocs)
			if acState != nil && *acState == 0 {
				stepBlocked = true
				break
			}

			// Gate 2: Batteries
			dc1State := getStateAt(cp.t, dc1Blocs)
			dc2State := getStateAt(cp.t, dc2Blocs)
			if dc1State != nil && dc2State != nil && *dc1State == 0 && *dc2State == 0 {
				stepBlocked = true
				break
			}

			// Gate 3: PDC (règle ≥3 down + contribution)
			withData := 0
			upCount := 0
			for pdcIdx := 0; pdcIdx < 6; pdcIdx++ {
				pdcState := getStateAt(cp.t, pdcBlocs[pdcIdx])
				if pdcState != nil && *pdcState != -1 {
					withData++
					if *pdcState == 1 {
						upCount++
					}
				}
			}

			if withData > 0 {
				downCount := withData - upCount
				if downCount >= 3 {
					stepBlocked = true
					break
				}
				validContributions = append(validContributions, float64(upCount)/float64(withData))
			}
		}

		if stepBlocked {
			step.value = 0.0
		} else if len(validContributions) > 0 {
			sum := 0.0
			for _, c := range validContributions {
				sum += c
			}
			step.value = sum / float64(len(validContributions))
		} else {
			step.value = 0.0
		}

		tSum += step.value
	}

	// Calcul de la disponibilité
	availabilityPct := 0.0
	if t2Count > 0 {
		availabilityPct = ((tSum + float64(t3Count)) / float64(t2Count)) * 100
	}

	// Sauvegarde
	upsertQuery := `
		INSERT INTO dispo_contract_monthly 
		(site, period_start, t2, t3, t_sum, availability_pct, computed_at)
		VALUES (?, ?, ?, ?, ?, ?, UTC_TIMESTAMP())
		ON DUPLICATE KEY UPDATE
		  t2 = VALUES(t2),
		  t3 = VALUES(t3),
		  t_sum = VALUES(t_sum),
		  availability_pct = VALUES(availability_pct),
		  computed_at = UTC_TIMESTAMP()
	`

	_, err = db.Exec(upsertQuery, site, periodStart.Format("2006-01-02"), t2Count, t3Count, tSum, availabilityPct)
	if err != nil {
		return fmt.Errorf("upsert result: %w", err)
	}

	log.Printf("%s %d-%02d: T2=%d, T3=%d, T_sum=%.2f, Dispo=%.2f%%",
		site, year, month, t2Count, t3Count, tSum, availabilityPct)

	return nil
}

type multiString []string

func (m *multiString) String() string     { return strings.Join(*m, ",") }
func (m *multiString) Set(v string) error { *m = append(*m, v); return nil }

func main() {
	var sites multiString
	flag.Var(&sites, "site", "Site à traiter (répétable). Si absent, traite tous les sites par défaut")
	yearFlag := flag.Int("year", 0, "Année (0 = auto: tous les mois depuis le début)")
	monthFlag := flag.Int("month", 0, "Mois (1-12, 0 = tous les mois de l'année)")
	allFlag := flag.Bool("all", false, "Calculer tous les sites et tous les mois historiques")
	flag.Parse()

	defaultSites := []string{
		"8822_001", "8822_002", "8822_003",
		"8822_004", "8822_005", "8822_006",
	}

	if len(sites) == 0 {
		sites = defaultSites
	}

	db := mysqlDBConn()
	defer db.Close()

	// Mode auto : détecter la plage de dates à calculer
	if *allFlag || *yearFlag == 0 {
		log.Println("Mode automatique : calcul de tous les mois disponibles")
		for _, site := range sites {
			if err := calculateAllMonths(db, site); err != nil {
				log.Printf("Erreur %s: %v", site, err)
			}
		}
		log.Println("Terminé")
		return
	}

	// Mode manuel : année/mois spécifiques
	if *monthFlag == 0 {
		// Tous les mois de l'année
		for m := 1; m <= 12; m++ {
			for _, site := range sites {
				if err := calculateMonthly(db, site, *yearFlag, time.Month(m)); err != nil {
					log.Printf("Erreur %s %d-%02d: %v", site, *yearFlag, m, err)
				}
			}
		}
	} else {
		// Un mois spécifique
		for _, site := range sites {
			if err := calculateMonthly(db, site, *yearFlag, time.Month(*monthFlag)); err != nil {
				log.Printf("Erreur %s: %v", site, err)
			}
		}
	}

	log.Println("Terminé")
}

// Calcule tous les mois disponibles pour un site
func calculateAllMonths(db *sql.DB, site string) error {
	// Détecter le premier mois disponible (depuis PDC1)
	var firstDate sql.NullTime
	query := fmt.Sprintf(`
		SELECT MIN(date_debut) 
		FROM dispo_pdc_n1_%s
		WHERE site = ?
	`, strings.ReplaceAll(site, "-", "_"))

	if err := db.QueryRow(query, site).Scan(&firstDate); err != nil {
		return fmt.Errorf("detect first date: %w", err)
	}

	if !firstDate.Valid {
		log.Printf("%s: aucune donnée disponible", site)
		return nil
	}

	start := firstDate.Time.In(paris)
	startYear := start.Year()
	startMonth := start.Month()

	now := time.Now().In(paris)
	endYear := now.Year()
	endMonth := now.Month()

	log.Printf("%s: calcul depuis %d-%02d jusqu'à %d-%02d", site, startYear, startMonth, endYear, endMonth)

	for year := startYear; year <= endYear; year++ {
		monthStart := 1
		if year == startYear {
			monthStart = int(startMonth)
		}
		monthEnd := 12
		if year == endYear {
			monthEnd = int(endMonth) - 1 // Mois en cours exclu
		}

		for month := monthStart; month <= monthEnd; month++ {
			if err := calculateMonthly(db, site, year, time.Month(month)); err != nil {
				log.Printf("Erreur %s %d-%02d: %v", site, year, month, err)
			}
			time.Sleep(100 * time.Millisecond)
		}
	}

	return nil
}
