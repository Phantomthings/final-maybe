"""Contract availability computation utilities."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from bisect import bisect_right

import pandas as pd


@dataclass
class AvailabilityTimeline:
    """Represents availability states on a time range."""

    intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp, int, int]]

    def __post_init__(self) -> None:
        self._starts = [start for start, *_ in self.intervals]

    def status_at(self, ts: pd.Timestamp) -> Tuple[bool, bool, bool]:
        """Return (is_available, is_excluded, has_data) for a timestamp."""
        if not self.intervals:
            return False, False, False

        idx = bisect_right(self._starts, ts) - 1
        if idx < 0:
            return False, False, False

        start, end, available, is_excluded = self.intervals[idx]
        if start <= ts < end:
            return available == 1, is_excluded == 1, True
        return False, False, False

    def has_data(self) -> bool:
        return bool(self.intervals)


@dataclass
class IntervalCollection:
    intervals: Sequence[Tuple[pd.Timestamp, pd.Timestamp]]

    def covers(self, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        for interval_start, interval_end in self.intervals:
            if interval_start <= start and interval_end >= end:
                return True
        return False


EquipmentTimelineLoader = Callable[[str, datetime, datetime], Dict[str, AvailabilityTimeline]]
PdcTimelineLoader = Callable[[str, datetime, datetime], List[AvailabilityTimeline]]
ExclusionLoader = Callable[[str, datetime, datetime], IntervalCollection]


def localize_to_paris(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        return ts.tz_localize("Europe/Paris", nonexistent="shift_forward", ambiguous="NaT")
    return ts.tz_convert("Europe/Paris")


def build_timeline(
    df: Optional[pd.DataFrame],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> AvailabilityTimeline:
    if df is None or df.empty:
        return AvailabilityTimeline([])

    records: List[Tuple[pd.Timestamp, pd.Timestamp, int, int]] = []
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
        start_clipped = max(start_ts, start)
        end_clipped = min(end_ts, end)
        if pd.isna(start_clipped) or pd.isna(end_clipped) or start_clipped >= end_clipped:
            continue
        available = int(row.get("est_disponible", 0))
        is_excluded = int(row.get("is_excluded", 0))
        records.append((start_clipped, end_clipped, available, is_excluded))

    records.sort(key=lambda item: item[0])
    return AvailabilityTimeline(records)


class ContractCalculator:
    """Encapsulates the contract availability computation rules."""

    def __init__(
        self,
        load_equipment_timelines: EquipmentTimelineLoader,
        load_pdc_timelines: PdcTimelineLoader,
        load_exclusion_intervals: ExclusionLoader,
        t2_mode: str = "data_driven",  # "calendar" ou "data_driven"
    ) -> None:
        self._load_equipment_timelines = load_equipment_timelines
        self._load_pdc_timelines = load_pdc_timelines
        self._load_exclusion_intervals = load_exclusion_intervals
        self._t2_mode = t2_mode

    def calculate_monthly(
        self,
        site: str,
        start_dt: datetime,
        end_dt: datetime,
    ) -> Tuple[pd.DataFrame, List[str]]:
        start = localize_to_paris(start_dt)
        end = localize_to_paris(end_dt)

        if pd.isna(start) or pd.isna(end) or end <= start:
            return pd.DataFrame(), ["Plage temporelle invalide"]

        equipment_timelines = self._load_equipment_timelines(site, start_dt, end_dt)
        pdc_timelines = self._load_pdc_timelines(site, start_dt, end_dt)
        exclusion_intervals = self._load_exclusion_intervals(site, start_dt, end_dt)

        warnings: List[str] = []
        if not pdc_timelines:
            warnings.append("Aucun point de charge disponible pour le calcul contractuel.")

        missing = [name for name, timeline in equipment_timelines.items() if not timeline.has_data()]
        if missing:
            warnings.append("Données manquantes pour : " + ", ".join(missing))

        step_duration = pd.Timedelta(minutes=10)
        cursor = start

        aggregates: Dict[pd.Timestamp, Dict[str, float]] = defaultdict(
            lambda: {"T2": 0, "T3": 0, "T_sum": 0.0}
        )

        def checkpoints(step_start: pd.Timestamp) -> List[pd.Timestamp]:
            # 10 points par pas (au milieu de chaque minute)
            return [step_start + timedelta(minutes=i, seconds=30) for i in range(10)]

        while cursor + step_duration <= end:
            step_start = cursor
            step_end = cursor + step_duration
            month_key = step_start.to_period("M").to_timestamp()

            # 1) T3 prioritaire : si l'exclusion couvre 100% du pas, il compte dans T2 et T3
            if exclusion_intervals.covers(step_start, step_end):
                aggregates[month_key]["T2"] += 1
                aggregates[month_key]["T3"] += 1
                cursor = step_end
                continue

            cks = checkpoints(step_start)

            # 2) T2 data-driven (non-T3) : ignorer le pas s'il n'y a AUCUNE donnée sur tous les checkpoints
            if self._t2_mode == "data_driven":
                def any_has_data(ts: pd.Timestamp) -> bool:
                    _, _, ac_has = equipment_timelines.get("AC", AvailabilityTimeline([])).status_at(ts)
                    _, _, d1_has = equipment_timelines.get("DC1", AvailabilityTimeline([])).status_at(ts)
                    _, _, d2_has = equipment_timelines.get("DC2", AvailabilityTimeline([])).status_at(ts)
                    pdc_has_any = any(tl.status_at(ts)[2] for tl in pdc_timelines)
                    return ac_has or d1_has or d2_has or pdc_has_any

                if not any(any_has_data(ts) for ts in cks):
                    cursor = step_end
                    continue

            # 3) Ici, le pas entre dans T2
            aggregates[month_key]["T2"] += 1

            # 4) Gate AC : indispo ou pas de données à un checkpoint => pas = 0
            ac_timeline = equipment_timelines.get("AC", AvailabilityTimeline([]))
            if any(not ac_timeline.status_at(ts)[0] for ts in cks):
                cursor = step_end
                continue

            # 5) Gate Batteries : si DC1 ET DC2 indispo/no-data à un checkpoint => pas = 0
            dc1_timeline = equipment_timelines.get("DC1", AvailabilityTimeline([]))
            dc2_timeline = equipment_timelines.get("DC2", AvailabilityTimeline([]))
            batteries_fail = any(
                (not dc1_timeline.status_at(ts)[0]) and (not dc2_timeline.status_at(ts)[0])
                for ts in cks
            )
            if batteries_fail:
                cursor = step_end
                continue

            # 6) Règle PDC : blocage si >=3 indispo simultanées ; sinon prorata (up/6) par checkpoint
            total_pdc = max(len(pdc_timelines), 6)
            minute_ratios: List[float] = []
            station_blocked = False
            for ts in cks:
                up = 0
                for timeline in pdc_timelines:
                    avail, _, has = timeline.status_at(ts)
                    if avail and has:
                        up += 1
                if (total_pdc - up) >= 3:
                    station_blocked = True
                    break
                minute_ratios.append(up / total_pdc)

            if station_blocked or not minute_ratios:
                cursor = step_end
                continue

            # 7) Contribution du pas
            step_value = sum(minute_ratios) / len(minute_ratios)
            aggregates[month_key]["T_sum"] += step_value

            cursor = step_end

        # Sortie mensuelle
        rows: List[Dict[str, float]] = []
        for month, values in sorted(aggregates.items()):
            t2 = values["T2"]
            if t2 == 0:
                continue
            t3 = values["T3"]
            tsum = values["T_sum"]
            availability_pct = ((tsum + t3) / t2) * 100
            rows.append(
                {
                    "Mois": month.strftime("%Y-%m"),
                    "T2": int(t2),
                    "T3": int(t3),
                    "T(11..16)": round(tsum, 2),
                    "Disponibilité (%)": round(availability_pct, 2),
                }
            )

        result = pd.DataFrame(rows)
        if not result.empty:
            result = result.sort_values("Mois").reset_index(drop=True)

        return result, warnings
