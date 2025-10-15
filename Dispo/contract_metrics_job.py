from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from contract_calculator import (
    AvailabilityTimeline,
    ContractCalculator,
    IntervalCollection,
    build_timeline,
    localize_to_paris,
)

logger = logging.getLogger("contract_job")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

CONTRACT_TABLE = "dispo_contract_monthly"


@dataclass
class DBConfig:
    user: str
    password: str
    host: str
    port: int
    database: str


def get_db_config() -> DBConfig:
    return DBConfig(
        user=os.getenv("MYSQL_USER", "AdminNidec"),
        password=os.getenv("MYSQL_PASSWORD", "u6Ehe987XBSXxa4"),
        host=os.getenv("MYSQL_HOST", "141.94.31.144"),
        port=int(os.getenv("MYSQL_PORT", 3306)),
        database=os.getenv("MYSQL_DB", "indicator"),
    )


def build_engine() -> Engine:
    cfg = get_db_config()
    uri = (
        f"mysql+pymysql://{cfg.user}:{cfg.password}@{cfg.host}:{cfg.port}/{cfg.database}?charset=utf8mb4"
    )
    engine = create_engine(uri, pool_recycle=3600, pool_pre_ping=True)
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    return engine


def execute_query(engine: Engine, query: str, params: Optional[Dict] = None) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql_query(text(query), conn, params=params or {})


def execute_write(engine: Engine, query: str, params: Optional[Dict] = None) -> None:
    with engine.begin() as conn:
        conn.execute(text(query), params or {})


def _normalize_blocks_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    for col in ["date_debut", "date_fin", "processed_at"]:
        if col in out.columns:
            series = pd.to_datetime(out[col], errors="coerce")
            try:
                if series.dt.tz is None:
                    series = series.dt.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
                else:
                    series = series.dt.tz_convert("Europe/Paris")
            except Exception:
                pass
            out[col] = series
    for col in ["est_disponible", "raw_point_count", "duration_minutes", "is_excluded"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
        else:
            if col == "is_excluded":
                out[col] = 0
    return out.sort_values("date_debut").reset_index(drop=True)


def _list_tables(engine: Engine, pattern: str) -> pd.DataFrame:
    cfg = get_db_config()
    query = """
        SELECT TABLE_NAME AS table_name
        FROM information_schema.tables
        WHERE TABLE_SCHEMA = :db
          AND TABLE_NAME REGEXP :pattern
        ORDER BY TABLE_NAME
    """
    params = {"db": cfg.database, "pattern": pattern}
    try:
        return execute_query(engine, query, params)
    except SQLAlchemyError as exc:
        logger.warning("Impossible de lister les tables (%s)", exc)
        return pd.DataFrame(columns=["table_name"])


def _list_ac_tables(engine: Engine) -> pd.DataFrame:
    df = _list_tables(engine, r"^dispo_blocs_ac_[0-9]+(_[0-9]+)?$")
    if df.empty:
        return pd.DataFrame(columns=["site_code", "table_name"])
    df.columns = [c.lower() for c in df.columns]
    df["site_code"] = df["table_name"].str.replace("dispo_blocs_ac_", "", regex=False)
    return df[["site_code", "table_name"]]


def _list_batt_tables(engine: Engine) -> pd.DataFrame:
    df = _list_tables(engine, r"^dispo_blocs_batt[0-9]*_[0-9]+(_[0-9]+)?$")
    if df.empty:
        return pd.DataFrame(columns=["site_code", "table_name"])
    df.columns = [c.lower() for c in df.columns]
    df["site_code"] = df["table_name"].str.extract(r"dispo_blocs_(?:batt|batt2)_(.*)")[0]
    return df[["site_code", "table_name"]]


def _list_pdc_tables(engine: Engine) -> pd.DataFrame:
    df = _list_tables(engine, r"^dispo_pdc_n[0-9]+_[0-9]+(_[0-9]+)?$")
    if df.empty:
        return pd.DataFrame(columns=["site_code", "pdc_id", "table_name"])
    df.columns = [c.lower() for c in df.columns]
    if "table_name" not in df.columns:
        return pd.DataFrame(columns=["site_code", "pdc_id", "table_name"])

    def _parse(tbl: str) -> pd.Series:
        t = str(tbl)
        prefix = "dispo_pdc_"
        if not t.startswith(prefix):
            return pd.Series([None, None, t])
        payload = t[len(prefix):]
        parts = payload.split("_", 1)
        if len(parts) != 2:
            return pd.Series([None, None, t])
        num = parts[0].lstrip("nN")
        pdc_id = f"PDC{num}" if num else None
        return pd.Series([parts[1], pdc_id, t])

    out = df["table_name"].apply(_parse)
    out.columns = ["site_code", "pdc_id", "table_name"]
    return out.dropna(subset=["site_code", "pdc_id"]).reset_index(drop=True)


def _collect_all_sites(engine: Engine) -> List[str]:
    sites: set[str] = set()
    for df in (_list_ac_tables(engine), _list_batt_tables(engine), _list_pdc_tables(engine)):
        if df.empty:
            continue
        column = "site_code" if "site_code" in df.columns else None
        if column is None:
            continue
        sites.update(df[column].dropna().astype(str).tolist())
    return sorted(sites)


def _query_union_bounds(engine: Engine, union_sql: str, site: str) -> Tuple[Optional[datetime], Optional[datetime]]:
    if not union_sql:
        return None, None
    query = f"""
        SELECT MIN(date_debut) AS min_start, MAX(date_fin) AS max_end
        FROM ({union_sql}) AS unioned
        WHERE site = :site
    """
    try:
        df = execute_query(engine, query, {"site": site})
    except SQLAlchemyError:
        return None, None
    if df.empty:
        return None, None

    def _convert(value: object) -> Optional[datetime]:
        if pd.isna(value):
            return None
        ts = pd.to_datetime(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is not None:
            ts = ts.tz_convert("Europe/Paris").tz_localize(None)
        return ts.to_pydatetime()

    return _convert(df.loc[0, "min_start"]), _convert(df.loc[0, "max_end"])


def _infer_site_bounds(engine: Engine, site: str) -> Optional[Tuple[datetime, datetime]]:
    bounds: List[Tuple[Optional[datetime], Optional[datetime]]] = []
    for union_sql in (
        _ac_union_sql_for_site(engine, site),
        _batt_union_sql_for_site(engine, site),
        _pdc_union_sql_for_site(engine, site),
    ):
        bounds.append(_query_union_bounds(engine, union_sql, site))

    starts = [start for start, _ in bounds if start is not None]
    ends = [end for _, end in bounds if end is not None]
    if not starts or not ends:
        return None
    return min(starts), max(ends)


def _ac_union_sql_for_site(engine: Engine, site: str) -> str:
    tables = _list_ac_tables(engine)
    subset = tables[tables["site_code"] == site]
    if subset.empty:
        return "SELECT * FROM (SELECT NULL AS site, NULL AS equipement_id, NULL AS type_equipement, NULL AS date_debut, NULL AS date_fin, NULL AS est_disponible, NULL AS cause, NULL AS raw_point_count, NULL AS processed_at, NULL AS batch_id, NULL AS hash_signature) x WHERE 1=0"
    parts = [
        f"SELECT site, equipement_id, type_equipement, date_debut, date_fin, est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature FROM `{tbl}`"
        for tbl in subset["table_name"].tolist()
    ]
    return " UNION ALL ".join(parts)


def _batt_union_sql_for_site(engine: Engine, site: str) -> str:
    tables = _list_batt_tables(engine)
    subset = tables[tables["site_code"] == site]
    if subset.empty:
        return "SELECT * FROM (SELECT NULL AS site, NULL AS equipement_id, NULL AS type_equipement, NULL AS date_debut, NULL AS date_fin, NULL AS est_disponible, NULL AS cause, NULL AS raw_point_count, NULL AS processed_at, NULL AS batch_id, NULL AS hash_signature) x WHERE 1=0"
    parts = [
        f"SELECT site, equipement_id, type_equipement, date_debut, date_fin, est_disponible, cause, raw_point_count, processed_at, batch_id, hash_signature FROM `{tbl}`"
        for tbl in subset["table_name"].tolist()
    ]
    return " UNION ALL ".join(parts)


def _pdc_union_sql_for_site(engine: Engine, site: str) -> str:
    tables = _list_pdc_tables(engine)
    subset = tables[tables["site_code"] == site]
    if subset.empty:
        return "SELECT * FROM (SELECT NULL AS site, NULL AS equipement_id, NULL AS type_equipement, NULL AS date_debut, NULL AS date_fin, NULL AS est_disponible, NULL AS cause, NULL AS raw_point_count, NULL AS processed_at, NULL AS batch_id, NULL AS hash_signature) x WHERE 1=0"
    parts = []
    for _, row in subset.iterrows():
        tbl = row["table_name"]
        parts.append(
            f"""
            SELECT
              site,
              pdc_id AS equipement_id,
              type_label AS type_equipement,
              date_debut,
              date_fin,
              etat AS est_disponible,
              cause,
              raw_point_count,
              processed_at,
              batch_id,
              hash_signature
            FROM `{tbl}`
            """
        )
    return " UNION ALL ".join(parts)


def _load_filtered_blocks_equipment(
    engine: Engine,
    site: str,
    equip: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    params = {"site": site, "equip": equip, "start": start_dt, "end": end_dt}
    try:
        q_view = """
            SELECT *
            FROM dispo_blocs_with_exclusion_flag
            WHERE site = :site
              AND equipement_id = :equip
              AND date_debut < :end
              AND date_fin > :start
            ORDER BY date_debut
        """
        df = execute_query(engine, q_view, params)
        if not df.empty:
            return _normalize_blocks_df(df)
    except SQLAlchemyError:
        pass

    union_ac = _ac_union_sql_for_site(engine, site)
    union_bt = _batt_union_sql_for_site(engine, site)
    q = f"""
        WITH ac AS (
            {union_ac}
        ),
        batt AS (
            {union_bt}
        ),
        base AS (
            SELECT * FROM ac
            UNION ALL
            SELECT * FROM batt
        )
        SELECT
          b.site,
          b.equipement_id,
          b.type_equipement,
          b.date_debut,
          b.date_fin,
          b.est_disponible,
          b.cause,
          b.raw_point_count,
          b.processed_at,
          b.batch_id,
          b.hash_signature,
          TIMESTAMPDIFF(MINUTE, b.date_debut, b.date_fin) AS duration_minutes,
          CAST(EXISTS (
            SELECT 1 FROM dispo_annotations a
            WHERE a.actif = 1
              AND a.type_annotation = 'exclusion'
              AND a.site = b.site
              AND a.equipement_id = b.equipement_id
              AND NOT (a.date_fin <= b.date_debut OR a.date_debut >= b.date_fin)
          ) AS UNSIGNED) AS is_excluded
        FROM base b
        WHERE b.site = :site
          AND b.equipement_id = :equip
          AND b.date_debut < :end
          AND b.date_fin > :start
        ORDER BY b.date_debut
    """
    df = execute_query(engine, q, params)
    return _normalize_blocks_df(df)


def _load_filtered_blocks_pdc(
    engine: Engine,
    site: str,
    equip: str,
    start_dt: datetime,
    end_dt: datetime,
) -> pd.DataFrame:
    params = {"site": site, "equip": equip, "start": start_dt, "end": end_dt}
    union_sql = _pdc_union_sql_for_site(engine, site)
    q = f"""
        WITH pdc AS (
            {union_sql}
        )
        SELECT
          p.site,
          p.equipement_id,
          p.type_equipement,
          p.date_debut,
          p.date_fin,
          p.est_disponible,
          p.cause,
          p.raw_point_count,
          p.processed_at,
          p.batch_id,
          p.hash_signature,
          TIMESTAMPDIFF(MINUTE, p.date_debut, p.date_fin) AS duration_minutes,
          CAST(EXISTS (
            SELECT 1 FROM dispo_annotations a
            WHERE a.actif = 1
              AND a.type_annotation = 'exclusion'
              AND a.site = p.site
              AND a.equipement_id = p.equipement_id
              AND NOT (a.date_fin <= p.date_debut OR a.date_debut >= p.date_fin)
          ) AS UNSIGNED) AS is_excluded
        FROM pdc p
        WHERE p.site = :site
          AND p.equipement_id = :equip
          AND p.date_debut < :end
          AND p.date_fin > :start
        ORDER BY p.date_debut
    """
    df = execute_query(engine, q, params)
    return _normalize_blocks_df(df)


def _load_site_pdc_ids(engine: Engine, site: str) -> List[str]:
    tables = _list_pdc_tables(engine)
    subset = tables[tables["site_code"] == site]
    if subset.empty:
        return []
    pdc_ids = subset["pdc_id"].dropna().unique().tolist()
    return sorted(pdc_ids)[:6]


def _load_exclusion_intervals(
    engine: Engine,
    site: str,
    start_dt: datetime,
    end_dt: datetime,
) -> IntervalCollection:
    query = """
        SELECT date_debut, date_fin
        FROM dispo_annotations
        WHERE actif = 1
          AND type_annotation = 'exclusion'
          AND site = :site
          AND date_debut < :end
          AND date_fin > :start
    """
    params = {"site": site, "start": start_dt, "end": end_dt}
    try:
        df = execute_query(engine, query, params)
    except SQLAlchemyError:
        return IntervalCollection([])

    intervals: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for _, row in df.iterrows():
        start_ts = pd.to_datetime(row.get("date_debut"))
        end_ts = pd.to_datetime(row.get("date_fin"))
        if pd.isna(start_ts) or pd.isna(end_ts):
            continue
        if start_ts.tzinfo is None:
            start_ts = start_ts.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
        else:
            start_ts = start_ts.tz_convert("Europe/Paris")
        if end_ts.tzinfo is None:
            end_ts = end_ts.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
        else:
            end_ts = end_ts.tz_convert("Europe/Paris")
        if pd.isna(start_ts) or pd.isna(end_ts) or start_ts >= end_ts:
            continue
        intervals.append((start_ts, end_ts))

    intervals.sort(key=lambda tpl: tpl[0])
    return IntervalCollection(intervals)


def _build_equipment_timelines(
    engine: Engine,
    site: str,
    start_dt: datetime,
    end_dt: datetime,
) -> Dict[str, AvailabilityTimeline]:
    start = localize_to_paris(start_dt)
    end = localize_to_paris(end_dt)
    timelines: Dict[str, AvailabilityTimeline] = {}
    for equip in ["AC", "DC1", "DC2"]:
        try:
            df = _load_filtered_blocks_equipment(engine, site, equip, start_dt, end_dt)
        except Exception as exc:
            logger.warning("Erreur chargement blocs %s (%s)", equip, exc)
            df = pd.DataFrame()
        timelines[equip] = build_timeline(df, start, end)
    return timelines


def _build_pdc_timelines(
    engine: Engine,
    site: str,
    start_dt: datetime,
    end_dt: datetime,
) -> List[AvailabilityTimeline]:
    start = localize_to_paris(start_dt)
    end = localize_to_paris(end_dt)
    timelines: List[AvailabilityTimeline] = []
    for pdc_id in _load_site_pdc_ids(engine, site):
        try:
            df = _load_filtered_blocks_pdc(engine, site, pdc_id, start_dt, end_dt)
        except Exception as exc:
            logger.warning("Erreur chargement blocs PDC %s (%s)", pdc_id, exc)
            df = pd.DataFrame()
        timelines.append(build_timeline(df, start, end))
    return timelines


def ensure_contract_table(engine: Engine) -> None:
    query = f"""
        CREATE TABLE IF NOT EXISTS {CONTRACT_TABLE} (
            site VARCHAR(64) NOT NULL,
            period_start DATE NOT NULL,
            t2 INT NOT NULL,
            t3 INT NOT NULL,
            t_sum DOUBLE NOT NULL,
            availability_pct DOUBLE NOT NULL,
            notes TEXT NULL,
            computed_at DATETIME NOT NULL,
            PRIMARY KEY (site, period_start)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
    """
    execute_write(engine, query)


def store_monthly_results(
    engine: Engine,
    site: str,
    monthly_df: pd.DataFrame,
    warnings: Sequence[str],
) -> None:
    ensure_contract_table(engine)
    notes = "\n".join(sorted(set(warnings))) if warnings else None
    now = datetime.utcnow()
    for _, row in monthly_df.iterrows():
        month_str = str(row["Mois"])
        period_start = datetime.strptime(month_str + "-01", "%Y-%m-%d")
        params = {
            "site": site,
            "period_start": period_start,
            "t2": int(row["T2"]),
            "t3": int(row["T3"]),
            "t_sum": float(row["T(11..16)"]),
            "availability_pct": float(row["Disponibilité (%)"]),
            "notes": notes,
            "computed_at": now,
        }
        query = f"""
            INSERT INTO {CONTRACT_TABLE}
                (site, period_start, t2, t3, t_sum, availability_pct, notes, computed_at)
            VALUES
                (:site, :period_start, :t2, :t3, :t_sum, :availability_pct, :notes, :computed_at)
            ON DUPLICATE KEY UPDATE
                t2 = VALUES(t2),
                t3 = VALUES(t3),
                t_sum = VALUES(t_sum),
                availability_pct = VALUES(availability_pct),
                notes = VALUES(notes),
                computed_at = VALUES(computed_at)
        """
        execute_write(engine, query, params)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calcule et stocke la disponibilité contractuelle mensuelle."
            " Sans argument, traite tous les sites depuis le début des données."
        )
    )
    parser.add_argument("site", nargs="?", help="Code site à traiter (optionnel)")
    parser.add_argument(
        "start",
        nargs="?",
        help="Date de début (YYYY-MM ou YYYY-MM-DD). Par défaut: première donnée disponible du site.",
    )
    parser.add_argument(
        "end",
        nargs="?",
        help="Date de fin (YYYY-MM ou YYYY-MM-DD). Par défaut: dernière donnée disponible du site.",
    )
    return parser.parse_args()


def _parse_date(value: str) -> datetime:
    for fmt in ("%Y-%m", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value, fmt)
            if fmt == "%Y-%m":
                dt = dt.replace(day=1)
            return dt
        except ValueError:
            continue
    raise ValueError(f"Format de date invalide: {value}")


def _month_range(start: datetime, end: datetime) -> Tuple[datetime, datetime]:
    start_first = start.replace(day=1)
    if end.day != 1:
        end_first = end.replace(day=1)
    else:
        end_first = end
    return start_first, (end_first + pd.offsets.MonthBegin(1)).to_pydatetime()


def main() -> None:
    args = parse_args()
    engine = build_engine()
    if args.site:
        sites = [args.site]
    else:
        sites = _collect_all_sites(engine)
    if not sites:
        logger.warning("Aucun site trouvé pour le calcul contractuel.")
        return
    site = args.site
    start_input = _parse_date(args.start) if args.start else None
    end_input = _parse_date(args.end) if args.end else None
    if (start_input is not None) and (end_input is not None) and end_input < start_input:
        raise ValueError("La date de fin doit être postérieure à la date de début")

    start_dt = None
    end_dt = None

    calculator = ContractCalculator(
        lambda s, start, end: _build_equipment_timelines(engine, s, start, end),
        lambda s, start, end: _build_pdc_timelines(engine, s, start, end),
        lambda s, start, end: _load_exclusion_intervals(engine, s, start, end),
        t2_mode="data_driven", 
    )

    for site in sites:
        bounds = _infer_site_bounds(engine, site)
        if bounds is None and not start_input:
            logger.warning("Aucune donnée source trouvée pour le site %s", site)
            continue

        if start_input:
            s_in = start_input
        elif bounds is not None:
            s_in = bounds[0]
        else:
            logger.warning("Date de début introuvable pour le site %s", site)
            continue

        if end_input:
            e_in = end_input
        elif bounds is not None:
            e_in = bounds[1]
        else:
            logger.warning("Date de fin introuvable pour le site %s", site)
            continue

        if e_in < s_in:
            logger.warning("Plage temporelle invalide pour %s (%s > %s)", site, s_in, e_in)
            continue

        start_dt, end_dt = _month_range(s_in, e_in)
        logger.info("Calcul disponibilité contractuelle pour %s (%s → %s)", site, start_dt, end_dt)

        monthly_df, warnings = calculator.calculate_monthly(site, start_dt, end_dt)
        if monthly_df.empty:
            logger.warning("Aucun résultat calculé pour le site %s", site)
            continue

        if warnings:
            for warning in warnings:
                logger.warning("⚠️ %s", warning)

        store_monthly_results(engine, site, monthly_df, warnings)
        logger.info("Calcul contractuel terminé pour %s", site)


if __name__ == "__main__":
    main()
