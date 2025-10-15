"""PDF export utilities for availability statistics."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from html import escape
from io import BytesIO
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from Binaire import get_equip_config, translate_ic_pc  # type: ignore


EXCLUSION_MODE_NONE = "none"
EXCLUSION_MODE_AVAILABLE = "available"
EXCLUSION_MODE_UNAVAILABLE = "unavailable"


def _format_exclusion_label(is_excluded: Any, mode: Any) -> str:
    try:
        flag = int(is_excluded)
    except Exception:
        flag = 0
    if flag != 1:
        return "Non"
    mode_str = str(mode) if mode is not None else EXCLUSION_MODE_AVAILABLE
    if mode_str == EXCLUSION_MODE_UNAVAILABLE:
        return "Comme indisponible"
    return "Comme disponible"


STYLES = getSampleStyleSheet()
THEME_BLUE = colors.HexColor("#1f77b4")
THEME_TEXT = colors.HexColor("#0b1f33")
CARD_BACKGROUND = colors.HexColor("#f0f2f6")
TABLE_ROW_ALT = [colors.whitesmoke, colors.HexColor("#f9fafb")]

TITLE_STYLE = ParagraphStyle(
    "ReportTitle",
    parent=STYLES["Heading1"],
    fontSize=18,
    leading=22,
    textColor=THEME_TEXT,
    spaceAfter=0,
    spaceBefore=0,
    alignment=0,
)

META_STYLE = ParagraphStyle(
    "Meta",
    parent=STYLES["Normal"],
    fontSize=10,
    textColor=colors.HexColor("#1f2933"),
    spaceAfter=6,
)

SITE_TITLE_STYLE = ParagraphStyle(
    "SiteTitle",
    parent=STYLES["Heading2"],
    fontSize=14,
    leading=18,
    textColor=THEME_TEXT,
    spaceBefore=12,
    spaceAfter=6,
)

SECTION_STYLE = ParagraphStyle(
    "SectionTitle",
    parent=STYLES["Heading3"],
    fontSize=11,
    leading=14,
    textColor=THEME_BLUE,
    spaceBefore=10,
    spaceAfter=4,
)

CARD_LABEL_STYLE = ParagraphStyle(
    "CardLabel",
    parent=STYLES["Normal"],
    fontSize=8,
    textColor=THEME_BLUE,
    leading=10,
)

CARD_VALUE_STYLE = ParagraphStyle(
    "CardValue",
    parent=STYLES["Normal"],
    fontSize=16,
    leading=18,
    textColor=THEME_TEXT,
    spaceAfter=2,
)

CARD_CAPTION_STYLE = ParagraphStyle(
    "CardCaption",
    parent=STYLES["Normal"],
    fontSize=8,
    leading=10,
    textColor=colors.HexColor("#6b7280"),
)

TABLE_BODY_STYLE = ParagraphStyle(
    "TableBody",
    parent=STYLES["Normal"],
    fontSize=8,
    leading=10,
    textColor=colors.HexColor("#0f172a"),
)

PAGE_WIDTH = A4[0] - 3 * cm

MetricDict = Dict[str, Any]


@dataclass
class SiteReport:
    """Structure de données normalisée pour un site."""

    site: str
    site_label: str
    metrics: MetricDict
    summary_df: pd.DataFrame
    equipment_summary: pd.DataFrame
    raw_blocks: pd.DataFrame
    pdc_summary: pd.DataFrame = field(default_factory=pd.DataFrame)


def _format_minutes(total_minutes: int) -> str:
    """Copie locale du formateur utilisé dans l'app."""

    minutes = int(total_minutes or 0)
    days, remainder = divmod(minutes, 1440)
    hours, mins = divmod(remainder, 60)

    parts: List[str] = []
    if days:
        parts.append(f"{days} {'jour' if days == 1 else 'jours'}")
    if hours:
        parts.append(f"{hours} {'heure' if hours == 1 else 'heures'}")
    if mins or not parts:
        parts.append(f"{mins} {'minute' if mins == 1 else 'minutes'}")
    return ", ".join(parts)


def _ensure_timezone(ts: datetime) -> pd.Timestamp:
    """Retourne un timestamp en Europe/Paris."""

    timestamp = pd.Timestamp(ts)
    if timestamp.tz is None:
        return timestamp.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
    return timestamp.tz_convert("Europe/Paris")


def _prepare_metrics_cards(metrics: MetricDict) -> List[Dict[str, str]]:
    availability = float(metrics.get("availability_pct", 0.0) or 0.0)
    downtime_minutes = int(metrics.get("downtime_minutes", 0) or 0)
    reference_minutes = int(metrics.get("reference_minutes", 0) or 0)
    coverage_pct = float(metrics.get("coverage_pct", 0.0) or 0.0)
    window_minutes = int(metrics.get("window_minutes", 0) or 0)

    cards = [
        {
            "label": "Disponibilité estimée",
            "value": f"{availability:.2f} %",
            "caption": "",
        },
        {
            "label": "Indisponibilité réelle",
            "value": _format_minutes(downtime_minutes),
            "caption": "",
        },
        {
            "label": "Temps analysé",
            "value": _format_minutes(reference_minutes),
            "caption": f"Couverture {_format_number(coverage_pct)} % / {_format_minutes(window_minutes)}",
        },
    ]
    return cards


def _format_number(value: float) -> str:
    return f"{value:.1f}".replace(".", ",")


NUMBER_PATTERN = re.compile(r"-?\d+")


@lru_cache(maxsize=None)
def _get_cached_config(equipement_id: str) -> Optional[Dict[str, Any]]:
    try:
        return get_equip_config(equipement_id)
    except Exception:
        return None


def _extract_ic_pc_tokens(cause: str) -> Tuple[Optional[int], Optional[int]]:
    text = str(cause or "")
    ic_match = re.search(r"IC\s*:?\s*(-?\d+)", text)
    pc_match = re.search(r"PC\s*:?\s*(-?\d+)", text)

    ic_val: Optional[int] = int(ic_match.group(1)) if ic_match else None
    pc_val: Optional[int] = int(pc_match.group(1)) if pc_match else None

    numbers = NUMBER_PATTERN.findall(text)
    if numbers:
        if ic_val is None and numbers:
            ic_val = int(numbers[0])
        if pc_val is None and len(numbers) > 1:
            pc_val = int(numbers[1])

    return ic_val, pc_val


def _translate_cause_label(cause: str, equipement_id: str) -> str:
    if not cause or str(cause).strip() == "" or cause == "Non spécifié":
        return "Cause non spécifiée"

    config = _get_cached_config(str(equipement_id))
    if not config:
        return str(cause)

    ic_val, pc_val = _extract_ic_pc_tokens(str(cause))
    if ic_val is None and pc_val is None:
        return str(cause)

    translated = translate_ic_pc(
        ic_val,
        pc_val,
        config.get("ic_map", {}),
        config.get("pc_map", {}),
    )
    return translated or str(cause)


def _card_flowables(cards: List[Dict[str, str]]) -> List[Any]:
    if not cards:
        return []

    cells: List[Any] = []
    for card in cards:
        rows: List[List[Any]] = [
            [Paragraph(escape(card["label"]).upper(), CARD_LABEL_STYLE)],
            [Paragraph(escape(card["value"]), CARD_VALUE_STYLE)],
        ]
        caption = card.get("caption")
        if caption:
            rows.append([Paragraph(escape(caption), CARD_CAPTION_STYLE)])

        card_table = Table(rows, hAlign="LEFT")
        card_table.setStyle(
            TableStyle(
                [
                    ("LEFTPADDING", (0, 0), (-1, -1), 0),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                    ("TOPPADDING", (0, 0), (-1, -1), 0),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ]
            )
        )

        cells.append(card_table)

    table = Table([cells], hAlign="LEFT")
    style_cmds = [
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]
    border_color = colors.HexColor("#d7dde5")
    for idx in range(len(cells)):
        style_cmds.append(("BACKGROUND", (idx, 0), (idx, 0), CARD_BACKGROUND))
        style_cmds.append(("BOX", (idx, 0), (idx, 0), 0.25, border_color))

    style_cmds.append(("INNERGRID", (0, 0), (-1, -1), 0.25, border_color))
    table.setStyle(TableStyle(style_cmds))

    return [table, Spacer(1, 0.4 * cm)]


def _prepare_summary_rows(summary_df: pd.DataFrame) -> List[Dict[str, str]]:
    if summary_df is None or summary_df.empty:
        return []

    display = summary_df.copy()
    if "Durée_Minutes" in display.columns:
        display["Durée"] = display["Durée_Minutes"].apply(lambda m: _format_minutes(int(m)))
    if "Temps_Analysé_Minutes" in display.columns:
        display["Temps analysé"] = display["Temps_Analysé_Minutes"].apply(lambda m: _format_minutes(int(m)))

    columns = [
        col
        for col in ["Condition", "Durée", "Temps analysé"]
        if col in display.columns
    ]
    if not columns:
        return []

    return [
        {col: str(row.get(col, "")) for col in columns}
        for _, row in display[columns].iterrows()
    ]


def _summary_table(rows: List[Dict[str, str]]) -> List[Any]:
    if not rows:
        return []

    header = [
        Paragraph("Condition", STYLES["Heading4"]),
        Paragraph("Durée", STYLES["Heading4"]),
        Paragraph("Temps analysé", STYLES["Heading4"]),
    ]

    data: List[List[Any]] = [header]
    for row in rows:
        data.append(
            [
                Paragraph(escape(row.get("Condition", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Durée", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Temps analysé", "")), TABLE_BODY_STYLE),
            ]
        )

    column_widths = [PAGE_WIDTH * 0.5, PAGE_WIDTH * 0.25, PAGE_WIDTH * 0.25]
    table = Table(data, colWidths=column_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), THEME_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), TABLE_ROW_ALT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    return [table, Spacer(1, 0.3 * cm)]


def _prepare_equipment_rows(equipment_df: pd.DataFrame) -> List[Dict[str, str]]:
    if equipment_df is None or equipment_df.empty:
        return []

    columns = [
        "Équipement",
        "Disponibilité Brute (%)",
        "Disponibilité Avec Exclusions (%)",
        "Durée Totale",
        "Temps Disponible",
        "Temps Indisponible",
    ]

    rows: List[Dict[str, str]] = []
    for _, row in equipment_df.iterrows():
        data = {}
        for col in columns:
            if col in row.index:
                data[col] = str(row[col])
        rows.append(data)
    return rows


def _equipment_table(rows: List[Dict[str, str]], title: Optional[str] = None) -> List[Any]:
    if not rows:
        return []

    headers = [
        "Équipement",
        "Disponibilité brute (%)",
        "Disponibilité avec exclusions (%)",
        "Durée totale",
        "Temps disponible",
        "Temps indisponible",
    ]

    data: List[List[Any]] = [
        [Paragraph(title, STYLES["Heading4"]) for title in headers]
    ]
    for row in rows:
        data.append(
            [
                Paragraph(escape(row.get("Équipement", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Disponibilité Brute (%)", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Disponibilité Avec Exclusions (%)", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Durée Totale", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Temps Disponible", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Temps Indisponible", "")), TABLE_BODY_STYLE),
            ]
        )

    col_ratios = [0.18, 0.14, 0.14, 0.14, 0.14, 0.14, 0.12]
    col_widths = [PAGE_WIDTH * ratio for ratio in col_ratios]

    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), THEME_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), TABLE_ROW_ALT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    return [table, Spacer(1, 0.3 * cm)]


def _prepare_pdc_rows(raw_blocks: pd.DataFrame) -> List[Dict[str, str]]:
    if raw_blocks is None or raw_blocks.empty:
        return []

    df = raw_blocks.copy()
    if "equipement_id" not in df.columns:
        return []

    df["equipement_id"] = df["equipement_id"].astype(str)
    pdc_df = df[df["equipement_id"].str.upper().str.startswith("PDC")].copy()
    if pdc_df.empty:
        return []

    for col in ["duration_minutes", "est_disponible", "is_excluded"]:
        if col not in pdc_df.columns:
            pdc_df[col] = 0
        pdc_df[col] = pd.to_numeric(pdc_df[col], errors="coerce").fillna(0).astype(int)
    if "exclusion_mode" in pdc_df.columns:
        pdc_df["exclusion_mode"] = (
            pdc_df["exclusion_mode"].fillna(EXCLUSION_MODE_NONE).astype(str)
        )
        pdc_df.loc[pdc_df["is_excluded"] == 0, "exclusion_mode"] = EXCLUSION_MODE_NONE
    else:
        pdc_df["exclusion_mode"] = EXCLUSION_MODE_NONE

    rows: List[Dict[str, str]] = []
    for equipement, group in pdc_df.groupby("equipement_id"):
        total = int(group["duration_minutes"].sum())
        missing = int(
            group.loc[
                (group["est_disponible"] == -1) & (group["is_excluded"] == 0),
                "duration_minutes",
            ].sum()
        )
        excluded_available = int(
            group.loc[
                (group["est_disponible"] == -1)
                & (group["is_excluded"] == 1)
                & (group["exclusion_mode"] == EXCLUSION_MODE_AVAILABLE),
                "duration_minutes",
            ].sum()
        )
        excluded_unavailable = int(
            group.loc[
                (group["est_disponible"] == -1)
                & (group["is_excluded"] == 1)
                & (group["exclusion_mode"] == EXCLUSION_MODE_UNAVAILABLE),
                "duration_minutes",
            ].sum()
        )
        excluded_unavail_blocks = int(
            group.loc[
                (group["est_disponible"] == 0) & (group["is_excluded"] == 1),
                "duration_minutes",
            ].sum()
        )
        available = int(
            group.loc[group["est_disponible"] == 1, "duration_minutes"].sum()
            + excluded_available
            + excluded_unavail_blocks
        )
        unavailable = int(
            group.loc[
                (group["est_disponible"] == 0) & (group["is_excluded"] == 0),
                "duration_minutes",
            ].sum()
            + excluded_unavailable
        )
        excluded = int(group.loc[group["is_excluded"] == 1, "duration_minutes"].sum())
        dominant_mode_series = group["exclusion_mode"].mode()
        dominant_mode = (
            dominant_mode_series.iloc[0]
            if not dominant_mode_series.empty
            else EXCLUSION_MODE_NONE
        )

        reference = max(total - missing, 0)
        availability_pct = (available / reference * 100) if reference else 0.0
        coverage_pct = (reference / total * 100) if total else 0.0

        rows.append(
            {
                "PDC": equipement,
                "Disponibilité (%)": f"{_format_number(availability_pct)} %",
                "Couverture (%)": f"{_format_number(coverage_pct)} %",
                "Temps disponible": _format_minutes(available),
                "Temps indisponible": _format_minutes(unavailable),
                "Données manquantes": _format_minutes(missing),
                "Durée exclue": _format_minutes(excluded),
                "Exclusions": _format_exclusion_label(
                    1 if group["is_excluded"].any() else 0, dominant_mode
                ) if not group.empty else "Non",
            }
        )

    return rows


def _pdc_table(rows: List[Dict[str, str]]) -> List[Any]:
    if not rows:
        return []

    headers = [
        "PDC",
        "Disponibilité (%)",
        "Couverture (%)",
        "Temps disponible",
        "Temps indisponible",
        "Données manquantes",
        "Durée exclue",
        "Exclusions",
    ]

    data: List[List[Any]] = [
        [Paragraph(title, STYLES["Heading4"]) for title in headers]
    ]

    for row in rows:
        data.append(
            [
                Paragraph(escape(row.get("PDC", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Disponibilité (%)", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Couverture (%)", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Temps disponible", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Temps indisponible", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Données manquantes", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Durée exclue", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Exclusions", "")), TABLE_BODY_STYLE),
            ]
        )

    col_ratios = [0.13, 0.11, 0.11, 0.15, 0.15, 0.12, 0.11, 0.12]
    col_widths = [PAGE_WIDTH * ratio for ratio in col_ratios]

    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), THEME_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), TABLE_ROW_ALT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    flow: List[Any] = []
    if title:
        flow.append(Paragraph(escape(title), STYLES["Heading4"]))
        flow.append(Spacer(1, 0.15 * cm))

    flow.append(table)
    flow.append(Spacer(1, 0.3 * cm))
    return flow


def _prepare_translated_cause_rows(raw_blocks: pd.DataFrame) -> Dict[str, List[Dict[str, str]]]:
    if raw_blocks is None or raw_blocks.empty:
        return {}

    df = raw_blocks.copy()
    required_columns = {"equipement_id", "duration_minutes", "est_disponible", "cause"}
    if not required_columns.issubset(df.columns):
        return {}

    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce").fillna(0).astype(int)
    df = df.loc[df["est_disponible"] == 0].copy()
    if df.empty:
        return {}

    df["equipement_id"] = df["equipement_id"].astype(str)
    df["cause"] = df["cause"].fillna("Non spécifié")
    df["translated"] = df.apply(
        lambda row: _translate_cause_label(row["cause"], row.get("equipement_id", "")),
        axis=1,
    )

    grouped: Dict[str, List[Dict[str, str]]] = {}
    for equipement, group in df.groupby("equipement_id"):
        agg = (
            group.groupby(["cause", "translated"], dropna=False)["duration_minutes"]
            .agg(total="sum", occurrences="count")
            .reset_index()
            .sort_values("total", ascending=False)
        )

        rows: List[Dict[str, str]] = []
        for record in agg.itertuples():
            rows.append(
                {
                    "Cause": str(record.cause) if record.cause else "Non spécifié",
                    "Cause traduite": str(record.translated) if record.translated else "Cause non spécifiée",
                    "Occurrences": str(int(record.occurrences)),
                    "Durée (min)": _format_minutes(str(int(record.total))),
                }
            )

        if rows:
            grouped[str(equipement)] = rows

    return grouped


def _translated_causes_table(
    rows: List[Dict[str, str]],
    equipment_label: Optional[str] = None,
) -> List[Any]:
    if not rows:
        return []

    headers = [
        "Cause brute",
        "Cause traduite",
        "Occurrences",
        "Durée (min)",
    ]

    data: List[List[Any]] = [
        [Paragraph(title, STYLES["Heading4"]) for title in headers]
    ]

    for row in rows:
        data.append(
            [
                Paragraph(escape(row.get("Cause", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Cause traduite", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Occurrences", "")), TABLE_BODY_STYLE),
                Paragraph(escape(row.get("Durée (min)", "")), TABLE_BODY_STYLE),
            ]
        )

    col_ratios = [0.34, 0.36, 0.14, 0.16]
    col_widths = [PAGE_WIDTH * ratio for ratio in col_ratios]

    table = Table(data, colWidths=col_widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), THEME_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("ALIGN", (0, 0), (-1, 0), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d1d5db")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), TABLE_ROW_ALT),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]
        )
    )

    flow: List[Any] = []
    if equipment_label:
        flow.append(Paragraph(escape(equipment_label), STYLES["Heading4"]))
        flow.append(Spacer(1, 0.15 * cm))

    flow.append(table)
    flow.append(Spacer(1, 0.3 * cm))
    return flow


def _translated_causes_tables(grouped_rows: Dict[str, List[Dict[str, str]]]) -> List[Any]:
    if not grouped_rows:
        return []

    flow: List[Any] = []
    for equipment in sorted(grouped_rows.keys()):
        rows = grouped_rows.get(equipment) or []
        if not rows:
            continue
        flow.extend(_translated_causes_table(rows, equipment))
    return flow


def _render_site_block(report: SiteReport, start_dt: datetime, end_dt: datetime) -> List[Any]:
    cards = _prepare_metrics_cards(report.metrics)
    summary_rows = _prepare_summary_rows(report.summary_df)
    equipment_rows = _prepare_equipment_rows(report.equipment_summary)
    pdc_summary_rows = _prepare_equipment_rows(report.pdc_summary)
    pdc_detail_rows = _prepare_pdc_rows(report.raw_blocks)
    translated_groups = _prepare_translated_cause_rows(report.raw_blocks)

    flow: List[Any] = [
        Paragraph(f"{escape(report.site_label)} ({escape(report.site)})", SITE_TITLE_STYLE),
        Spacer(1, 0.2 * cm),
    ]

    flow.extend(_card_flowables(cards))

    if summary_rows:
        flow.append(Paragraph("Conditions critiques".upper(), SECTION_STYLE))
        flow.extend(_summary_table(summary_rows))

    if equipment_rows or pdc_summary_rows:
        flow.append(Paragraph("Indicateurs clés par équipement".upper(), SECTION_STYLE))
        if equipment_rows:
            flow.extend(_equipment_table(equipment_rows, title="Équipements principaux (AC/DC)"))
        if pdc_summary_rows:
            flow.extend(_equipment_table(pdc_summary_rows, title="Points de charge (PDC)"))

    if pdc_detail_rows:
        flow.append(Paragraph("Détails des points de charge".upper(), SECTION_STYLE))
        flow.extend(_pdc_table(pdc_detail_rows))

    if translated_groups:
        flow.append(Paragraph("Causes traduites d'indisponibilité".upper(), SECTION_STYLE))
        flow.extend(_translated_causes_tables(translated_groups))

    return flow


def generate_statistics_pdf(
    reports: Iterable[SiteReport],
    start_dt: datetime,
    end_dt: datetime,
    title: str = "rapport mensuel de disponibilité",
) -> bytes:
    """Construit un PDF A4 avec les statistiques pour chaque site."""

    reports_list = list(reports)
    if not reports_list:
        raise ValueError("Aucun site à exporter")

    start_label = _ensure_timezone(start_dt).strftime("%d/%m/%Y")
    end_label = _ensure_timezone(end_dt).strftime("%d/%m/%Y")

    story: List[Any] = []
    story.append(Paragraph(escape(title.upper()), TITLE_STYLE))
    story.append(Spacer(1, 0.35 * cm))
    story.append(Paragraph(
        escape(f"Période : {start_label} → {end_label}"),
        META_STYLE,
    ))
    story.append(Spacer(1, 0.2 * cm))

    for idx, report in enumerate(reports_list):
        story.extend(_render_site_block(report, start_dt, end_dt))
        if idx < len(reports_list) - 1:
            story.append(PageBreak())

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=1.5 * cm,
        rightMargin=1.5 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )
    doc.build(story)
    return buffer.getvalue()