from __future__ import annotations
import os
import pandas as pd
from datetime import datetime
from sqlalchemy import create_engine, text

MYSQL_HOST = os.getenv("MYSQL_HOST", "141.94.31.144")
MYSQL_PORT = int(os.getenv("MYSQL_PORT", 3306))
MYSQL_USER = os.getenv("MYSQL_USER", "AdminNidec")
MYSQL_PW   = os.getenv("MYSQL_PASSWORD", "u6Ehe987XBSXxa4")
MYSQL_DB   = os.getenv("MYSQL_DB", "indicator")

def mysql_engine():
    return create_engine(
        f"mysql+mysqlconnector://{MYSQL_USER}:{MYSQL_PW}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
    )

INSERT_PCT_MOIS = text("""
INSERT INTO indicator.dispo_pct_mois
(site, equipement_id, mois, pct_brut, pct_excl, total_minutes, processed_at)
VALUES (:site, :equip, :mois, :pct_brut, :pct_excl, :total_minutes, UTC_TIMESTAMP())
ON DUPLICATE KEY UPDATE
  pct_brut = VALUES(pct_brut),
  pct_excl = VALUES(pct_excl),
  total_minutes = VALUES(total_minutes),
  processed_at = UTC_TIMESTAMP()
""")

def calculate_availability(df: pd.DataFrame, include_exclusions: bool = False) -> dict:
    if df.empty:
        return {"total_minutes": 0, "available_minutes": 0, "pct_available": 0.0}
    total = int(df["duration_minutes"].sum())
    if include_exclusions:
        available = int(
            df.loc[(df["est_disponible"] == 1) | (df["is_excluded"] == 1), "duration_minutes"].sum()
        )
    else:
        available = int(df.loc[df["est_disponible"] == 1, "duration_minutes"].sum())
    pct_available = (available / total * 100) if total > 0 else 0.0
    return {"total_minutes": total, "available_minutes": available, "pct_available": pct_available}

def update_monthly():
    eng = mysql_engine()
    query = """
        SELECT site, equipement_id, date_debut, date_fin, est_disponible,
               TIMESTAMPDIFF(MINUTE, date_debut, date_fin) as duration_minutes,
               0 as is_excluded  -- pas encore de gestion exclusions ici
        FROM indicator.dispo_blocs_ac
        UNION ALL
        SELECT site, equipement_id, date_debut, date_fin, est_disponible,
               TIMESTAMPDIFF(MINUTE, date_debut, date_fin) as duration_minutes,
               0 as is_excluded
        FROM indicator.dispo_blocs_batt
        UNION ALL
        SELECT site, equipement_id, date_debut, date_fin, est_disponible,
               TIMESTAMPDIFF(MINUTE, date_debut, date_fin) as duration_minutes,
               0 as is_excluded
        FROM indicator.dispo_blocs_batt2
    """
    df = pd.read_sql(query, eng)

    if df.empty:
        print("⚠️ Pas de données disponibles")
        return

    df["date_debut"] = pd.to_datetime(df["date_debut"], utc=True)
    df["month"] = df["date_debut"].dt.to_period("M").dt.to_timestamp()

    with eng.begin() as conn:
        for (site, equip), group_site in df.groupby(["site", "equipement_id"]):
            for month, group in group_site.groupby("month"):
                stats_raw = calculate_availability(group, include_exclusions=False)
                stats_excl = calculate_availability(group, include_exclusions=True)

                conn.execute(INSERT_PCT_MOIS, {
                    "site": site,
                    "equip": equip,
                    "mois": month.to_pydatetime().date(),
                    "pct_brut": stats_raw["pct_available"],
                    "pct_excl": stats_excl["pct_available"],
                    "total_minutes": stats_raw["total_minutes"],
                })
    print("✅ Table dispo_pct_mois mise à jour avec succès !")

if __name__ == "__main__":
    update_monthly()
