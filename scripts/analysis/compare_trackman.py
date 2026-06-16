"""Pair OpenFlight shots with Trackman shots and emit a side-by-side
comparison CSV.

Usage::

    uv run python scripts/analysis/compare_trackman.py \\
        --openflight session_logs/session_20260506_*.jsonl \\
        --trackman ~/Downloads/TrackMan_2026-05-06.csv \\
        --output ~/openflight_sessions/comparison_20260506.csv

Pairing strategy
----------------
Both systems number shots sequentially but their numbering doesn't
necessarily align (one of them may miss a shot, or you may have
hit a couple of "warm up" shots only one was watching). To be
robust:

1. Group both systems' shots by club.
2. Within each club, pair shots in chronological order.
3. Reject pairs whose ball-speed delta exceeds ``--ball-speed-tol``
   mph — flags missing/extra shots so you can fix the alignment by
   hand.

The output CSV has both systems' values + deltas + a
``match_quality`` column (``good``, ``ball_speed_mismatch``,
``unmatched_openflight``, ``unmatched_trackman``).

Trackman CSV column handling
----------------------------
Different Trackman exports (TPS, Range, TM4) use slightly different
column headers. We map known aliases to canonical names; anything
unknown is preserved verbatim in the output.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Trackman CSV column aliases
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HeaderAlias:
    """Trackman header match rule."""

    text: str
    exact: bool = False


# Each canonical field maps to a list of header rules. Most fields use
# substring matching because Trackman exports add unit suffixes like
# "(mph)". Timestamp aliases intentionally include exact-only rules for
# generic names like "date" and "time" so "Last data Point - Time" does
# not steal the session timestamp from a real "Date" column.
_TM_ALIAS_CONFIG: Dict[str, List[str | HeaderAlias]] = {
    "ball_speed_mph":          ["ball speed", "ballspeed"],
    "club_speed_mph":          ["club speed", "clubspeed", "club head speed", "clubheadspeed"],
    "smash_factor":            ["smash factor", "smashfactor"],
    "launch_angle_vertical":   ["launch angle v", "launch angle (v)", "vertical launch", "launch angle"],
    "launch_angle_horizontal": ["launch direction", "launch angle h", "side angle", "azimuth"],
    "spin_rpm":                ["spin rate", "total spin", "spinrate"],
    "carry_yards":             ["carry distance", "carry"],
    "club":                    ["club type", "club name", "club"],
    "shot_number":             ["shot number", "shotnumber", "shot"],
    "timestamp":               [
        "date/time",
        "datetime",
        "timestamp",
        HeaderAlias("date", exact=True),
        HeaderAlias("time", exact=True),
    ],
}


_TM_ALIASES: Dict[str, List[HeaderAlias]] = {
    field_name: [
        alias if isinstance(alias, HeaderAlias) else HeaderAlias(alias)
        for alias in aliases
    ]
    for field_name, aliases in _TM_ALIAS_CONFIG.items()
}


def _canon_header(h: str) -> str:
    return h.strip().lower().replace("_", " ")


def _detect_units(headers: List[str]) -> Dict[str, str]:
    """Return a unit hint per canonical field by inspecting the header
    suffix. Currently only used for ball/club speed (mph vs km/h vs m/s)
    and carry (yards vs metres).
    """
    units: Dict[str, str] = {}
    for h in headers:
        ch = _canon_header(h)
        if "ball speed" in ch or "club speed" in ch or "club head speed" in ch:
            if "kph" in ch or "km/h" in ch or "kmh" in ch:
                units.setdefault("speed", "kph")
            elif "m/s" in ch:
                units.setdefault("speed", "mps")
            else:
                units.setdefault("speed", "mph")
        if "carry" in ch:
            if "metre" in ch or "meter" in ch or "(m)" in ch or " m " in ch:
                units.setdefault("carry", "m")
            else:
                units.setdefault("carry", "yards")
    return units


def _build_column_map(headers: List[str]) -> Dict[str, str]:
    """Map canonical-field-name → actual-header-string from the CSV."""
    col_map: Dict[str, str] = {}
    canon = {_canon_header(h): h for h in headers}
    # Iterate aliases in declared order so longer / more specific
    # aliases are matched first within a single field.
    for field_name, aliases in _TM_ALIASES.items():
        for alias in aliases:
            for canon_h, raw_h in canon.items():
                matches = canon_h == alias.text if alias.exact else alias.text in canon_h
                if matches and field_name not in col_map:
                    col_map[field_name] = raw_h
                    break
            if field_name in col_map:
                break
    return col_map


# ---------------------------------------------------------------------------
# Club name normalization
# ---------------------------------------------------------------------------

# A loose normalizer: strip case/whitespace/dashes, accept "7i" / "7iron"
# / "7-iron" / "iron 7" etc. Both systems should agree after normalization.
_CLUB_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\b(driver|drv|1w|1-wood)\b"),     "driver"),
    (re.compile(r"\b(\d)\s*-?\s*(wood|w)\b"),        r"\1-wood"),
    (re.compile(r"\b(\d)\s*-?\s*(hybrid|h|hy)\b"),   r"\1-hybrid"),
    (re.compile(r"\b(\d)\s*-?\s*(iron|i)\b"),        r"\1-iron"),
    (re.compile(r"\biron\s*-?\s*(\d)\b"),            r"\1-iron"),
    (re.compile(r"\bpitching\s*wedge\b|\bpw\b"),     "pw"),
    (re.compile(r"\bgap\s*wedge\b|\bgw\b"),          "gw"),
    (re.compile(r"\bsand\s*wedge\b|\bsw\b"),         "sw"),
    (re.compile(r"\blob\s*wedge\b|\blw\b"),          "lw"),
]


def normalize_club(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    s = str(raw).strip().lower()
    if not s:
        return ""
    for pat, repl in _CLUB_PATTERNS:
        m = pat.search(s)
        if m:
            return pat.sub(repl, m.group(0)).strip()
    return s


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@dataclass
class Shot:
    """Canonical per-shot record used by both sources."""
    source: str  # "of" or "tm"
    shot_number: Optional[int]
    timestamp: Optional[datetime]
    club: str
    ball_speed_mph: Optional[float] = None
    club_speed_mph: Optional[float] = None
    smash_factor: Optional[float] = None
    launch_angle_vertical: Optional[float] = None
    launch_angle_horizontal: Optional[float] = None
    spin_rpm: Optional[float] = None
    spin_confidence: Optional[float] = None
    spin_quality: Optional[str] = None
    spin_snr: Optional[float] = None
    spin_candidate_rpm: Optional[float] = None
    spin_rejection_reason: Optional[str] = None
    carry_yards: Optional[float] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def _to_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _to_int(v: Any) -> Optional[int]:
    f = _to_float(v)
    return int(f) if f is not None else None


def _parse_timestamp(v: Any) -> Optional[datetime]:
    if v is None or v == "":
        return None
    s = str(v).strip()
    # Try ISO first (OpenFlight), then a few common Trackman formats.
    candidates = [
        None,  # let fromisoformat try
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y %I:%M:%S %p",
        "%d/%m/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in candidates:
        try:
            if fmt is None:
                return datetime.fromisoformat(s)
            return datetime.strptime(s, fmt)
        except (ValueError, TypeError):
            continue
    return None


def load_openflight(path: Path) -> List[Shot]:
    """Load `shot_detected` entries from an OpenFlight JSONL session log."""
    shots: List[Shot] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("type") != "shot_detected":
                continue
            data = entry.get("data", entry)  # tolerate flat or nested
            shots.append(Shot(
                source="of",
                shot_number=_to_int(data.get("shot_number")),
                # OpenFlight uses ``ts`` at the entry level. Older logs
                # may use ``timestamp`` instead. Try both.
                timestamp=_parse_timestamp(
                    entry.get("ts")
                    or entry.get("timestamp")
                    or data.get("timestamp")),
                club=normalize_club(data.get("club")),
                ball_speed_mph=_to_float(data.get("ball_speed_mph")),
                club_speed_mph=_to_float(data.get("club_speed_mph")),
                smash_factor=_to_float(data.get("smash_factor")),
                launch_angle_vertical=_to_float(
                    data.get("launch_angle_vertical")),
                launch_angle_horizontal=_to_float(
                    data.get("launch_angle_horizontal")),
                spin_rpm=_to_float(data.get("spin_rpm")),
                spin_confidence=_to_float(data.get("spin_confidence")),
                spin_quality=data.get("spin_quality"),
                spin_snr=_to_float(data.get("spin_snr")),
                spin_candidate_rpm=_to_float(
                    data.get("spin_candidate_rpm")),
                spin_rejection_reason=data.get("spin_rejection_reason"),
                carry_yards=_to_float(
                    data.get("carry_spin_adjusted")
                    or data.get("estimated_carry_yards")),
                raw=data,
            ))
    return shots


def _looks_like_units_row(row: Dict[str, Any]) -> bool:
    """Trackman's "Normalized" exports include a row immediately under
    the header containing units in brackets, e.g. ``[mph]``, ``[deg]``,
    ``[rpm]``. The Date/Club/etc. cells in this row are blank. Detect
    by looking for any cell wrapped in ``[...]`` and no plausible
    numeric values."""
    bracketed = 0
    has_plausible_value = False
    for v in row.values():
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            continue
        if s.startswith("[") and s.endswith("]"):
            bracketed += 1
        elif _to_float(s) is not None:
            # If anything looks like a real number, it's not a units row.
            has_plausible_value = True
    return bracketed >= 1 and not has_plausible_value


def load_trackman(path: Path) -> List[Shot]:
    """Load shots from a Trackman CSV export. Tolerant of header
    variations (TPS / Range / TM4 / Normalized export)."""
    shots: List[Shot] = []
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        # Strip an optional ``sep=,`` Excel-hint preamble (Trackman
        # Normalized exports include one as the first line, which would
        # otherwise be parsed as the header row).
        # Note: utf-8-sig only strips ONE BOM and Trackman exports have
        # been seen with TWO, so strip any leading BOMs explicitly.
        first_pos = fh.tell()
        first_line = fh.readline().lstrip("\ufeff").lstrip()
        if not first_line.lower().startswith("sep="):
            fh.seek(first_pos)
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            return shots
        # Defensively strip BOMs from header field names too.
        cleaned_headers = [h.lstrip("\ufeff").strip()
                           for h in reader.fieldnames]
        reader.fieldnames = cleaned_headers
        col_map = _build_column_map(cleaned_headers)
        units = _detect_units(cleaned_headers)
        speed_unit = units.get("speed", "mph")
        carry_unit = units.get("carry", "yards")

        def _get(row: Dict[str, str], canon: str) -> Any:
            col = col_map.get(canon)
            return row.get(col) if col else None

        for row in reader:
            # Skip Trackman's units row (e.g. row of "[mph]", "[deg]"
            # under the header in Normalized exports).
            if _looks_like_units_row(row):
                continue
            ball = _to_float(_get(row, "ball_speed_mph"))
            club_sp = _to_float(_get(row, "club_speed_mph"))
            if speed_unit == "kph":
                if ball is not None:
                    ball /= 1.609344
                if club_sp is not None:
                    club_sp /= 1.609344
            elif speed_unit == "mps":
                if ball is not None:
                    ball *= 2.236936
                if club_sp is not None:
                    club_sp *= 2.236936
            carry = _to_float(_get(row, "carry_yards"))
            if carry_unit == "m" and carry is not None:
                carry *= 1.093613

            shots.append(Shot(
                source="tm",
                shot_number=_to_int(_get(row, "shot_number")),
                timestamp=_parse_timestamp(_get(row, "timestamp")),
                club=normalize_club(_get(row, "club")),
                ball_speed_mph=ball,
                club_speed_mph=club_sp,
                smash_factor=_to_float(_get(row, "smash_factor")),
                launch_angle_vertical=_to_float(
                    _get(row, "launch_angle_vertical")),
                launch_angle_horizontal=_to_float(
                    _get(row, "launch_angle_horizontal")),
                spin_rpm=_to_float(_get(row, "spin_rpm")),
                carry_yards=carry,
                raw=dict(row),
            ))
    return shots


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------

@dataclass
class Pair:
    of: Optional[Shot]
    tm: Optional[Shot]
    match_quality: str  # "good" | "ball_speed_mismatch" | "unmatched_openflight" | "unmatched_trackman"
    notes: str = ""


def _sort_for_pairing(shots: Iterable[Shot]) -> List[Shot]:
    """Order shots for chronological pairing. Prefer timestamp; fall
    back to shot_number; preserve input order otherwise (stable)."""
    def key(s: Shot, idx: int) -> Tuple:
        ts = s.timestamp.timestamp() if s.timestamp else float("inf")
        sn = s.shot_number if s.shot_number is not None else float("inf")
        return (ts, sn, idx)
    indexed = list(enumerate(shots))
    indexed.sort(key=lambda item: key(item[1], item[0]))
    return [s for _, s in indexed]


def pair_shots(
    of_shots: List[Shot],
    tm_shots: List[Shot],
    ball_speed_tol_mph: float = 5.0,
    club_filter: Optional[List[str]] = None,
) -> List[Pair]:
    """Pair OpenFlight ↔ Trackman shots in chronological order within
    each club. Pairs whose ball-speed delta exceeds the tolerance are
    flagged ``ball_speed_mismatch`` (still emitted so you can fix
    alignment by hand).
    """
    pairs: List[Pair] = []

    of_by_club: Dict[str, List[Shot]] = {}
    tm_by_club: Dict[str, List[Shot]] = {}
    for s in of_shots:
        of_by_club.setdefault(s.club, []).append(s)
    for s in tm_shots:
        tm_by_club.setdefault(s.club, []).append(s)

    clubs = set(of_by_club) | set(tm_by_club)
    if club_filter:
        wanted = {normalize_club(c) for c in club_filter}
        clubs &= wanted

    # Stable, friendly ordering — driver first, then irons by number, then wedges.
    def club_sort_key(c: str) -> Tuple:
        if c == "driver":
            return (0, 0, c)
        m = re.match(r"(\d+)-(wood|hybrid|iron)", c)
        if m:
            ord_map = {"wood": 1, "hybrid": 2, "iron": 3}
            return (ord_map[m.group(2)], int(m.group(1)), c)
        if c in ("pw", "gw", "sw", "lw"):
            return (4, ["pw","gw","sw","lw"].index(c), c)
        return (5, 0, c)

    for club in sorted(clubs, key=club_sort_key):
        of_seq = _sort_for_pairing(of_by_club.get(club, []))
        tm_seq = _sort_for_pairing(tm_by_club.get(club, []))
        i = j = 0
        while i < len(of_seq) and j < len(tm_seq):
            of_s, tm_s = of_seq[i], tm_seq[j]
            mq = "good"
            notes = ""
            if (of_s.ball_speed_mph is not None
                    and tm_s.ball_speed_mph is not None):
                d = abs(of_s.ball_speed_mph - tm_s.ball_speed_mph)
                if d > ball_speed_tol_mph:
                    mq = "ball_speed_mismatch"
                    notes = (f"ball-speed delta {d:.1f} mph "
                             f"exceeds tol {ball_speed_tol_mph} mph; "
                             f"check pairing manually")
            pairs.append(Pair(of=of_s, tm=tm_s,
                              match_quality=mq, notes=notes))
            i += 1
            j += 1

        # Trailing unmatched shots from either side — keep so the user
        # can see where alignment ran off.
        for k in range(i, len(of_seq)):
            pairs.append(Pair(of=of_seq[k], tm=None,
                              match_quality="unmatched_openflight",
                              notes="no Trackman shot for this club position"))
        for k in range(j, len(tm_seq)):
            pairs.append(Pair(of=None, tm=tm_seq[k],
                              match_quality="unmatched_trackman",
                              notes="no OpenFlight shot for this club position"))

    return pairs


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_OUTPUT_FIELDS = [
    "shot_number_of", "shot_number_tm",
    "timestamp_of", "timestamp_tm",
    "club",
    "ball_speed_of", "ball_speed_tm", "ball_speed_delta",
    "club_speed_of", "club_speed_tm", "club_speed_delta",
    "smash_of", "smash_tm", "smash_delta",
    "launch_v_of", "launch_v_tm", "launch_v_delta",
    "launch_h_of", "launch_h_tm", "launch_h_delta",
    "spin_of", "spin_tm", "spin_delta",
    "spin_candidate_of", "spin_confidence_of", "spin_quality_of",
    "spin_snr_of", "spin_rejection_of",
    "carry_of", "carry_tm", "carry_delta",
    "match_quality", "notes",
]


def _delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return round(a - b, 3)


def _row(pair: Pair) -> Dict[str, Any]:
    of, tm = pair.of, pair.tm
    def f(val: Any) -> Any:
        if isinstance(val, float):
            return round(val, 3)
        return val
    return {
        "shot_number_of":   f(of.shot_number) if of else None,
        "shot_number_tm":   f(tm.shot_number) if tm else None,
        "timestamp_of":     of.timestamp.isoformat() if of and of.timestamp else None,
        "timestamp_tm":     tm.timestamp.isoformat() if tm and tm.timestamp else None,
        "club":             (of or tm).club if (of or tm) else "",
        "ball_speed_of":    f(of.ball_speed_mph) if of else None,
        "ball_speed_tm":    f(tm.ball_speed_mph) if tm else None,
        "ball_speed_delta": _delta(of.ball_speed_mph if of else None,
                                   tm.ball_speed_mph if tm else None),
        "club_speed_of":    f(of.club_speed_mph) if of else None,
        "club_speed_tm":    f(tm.club_speed_mph) if tm else None,
        "club_speed_delta": _delta(of.club_speed_mph if of else None,
                                   tm.club_speed_mph if tm else None),
        "smash_of":         f(of.smash_factor) if of else None,
        "smash_tm":         f(tm.smash_factor) if tm else None,
        "smash_delta":      _delta(of.smash_factor if of else None,
                                   tm.smash_factor if tm else None),
        "launch_v_of":      f(of.launch_angle_vertical) if of else None,
        "launch_v_tm":      f(tm.launch_angle_vertical) if tm else None,
        "launch_v_delta":   _delta(of.launch_angle_vertical if of else None,
                                   tm.launch_angle_vertical if tm else None),
        "launch_h_of":      f(of.launch_angle_horizontal) if of else None,
        "launch_h_tm":      f(tm.launch_angle_horizontal) if tm else None,
        "launch_h_delta":   _delta(of.launch_angle_horizontal if of else None,
                                   tm.launch_angle_horizontal if tm else None),
        "spin_of":          f(of.spin_rpm) if of else None,
        "spin_tm":          f(tm.spin_rpm) if tm else None,
        "spin_delta":       _delta(of.spin_rpm if of else None,
                                   tm.spin_rpm if tm else None),
        "spin_candidate_of": f(of.spin_candidate_rpm) if of else None,
        "spin_confidence_of": f(of.spin_confidence) if of else None,
        "spin_quality_of":  of.spin_quality if of else None,
        "spin_snr_of":      f(of.spin_snr) if of else None,
        "spin_rejection_of": of.spin_rejection_reason if of else None,
        "carry_of":         f(of.carry_yards) if of else None,
        "carry_tm":         f(tm.carry_yards) if tm else None,
        "carry_delta":      _delta(of.carry_yards if of else None,
                                   tm.carry_yards if tm else None),
        "match_quality":    pair.match_quality,
        "notes":            pair.notes,
    }


def write_comparison_csv(pairs: List[Pair], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        for p in pairs:
            writer.writerow(_row(p))


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

_DELTA_LABELS = [
    ("ball_speed_delta", "ball speed",  "mph"),
    ("club_speed_delta", "club speed",  "mph"),
    ("launch_v_delta",   "launch V",    "deg"),
    ("launch_h_delta",   "launch H",    "deg"),
    ("spin_delta",       "spin",        "rpm"),
    ("carry_delta",      "carry",       "yds"),
]


def _fit_linear(x: List[float], y: List[float]) -> Tuple[float, float, float]:
    """Least-squares fit y = slope*x + intercept. Returns
    (slope, intercept, residual_stddev). Uses pure stdlib so the
    comparison tool stays free of a numpy dependency."""
    n = len(x)
    if n < 2:
        return 1.0, 0.0, 0.0
    sx = sum(x)
    sy = sum(y)
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    denom = n * sxx - sx * sx
    if denom == 0:
        return 1.0, 0.0, 0.0
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    resid = [yi - (slope * xi + intercept) for xi, yi in zip(x, y)]
    mean_r = sum(resid) / n
    sd = (sum((r - mean_r) ** 2 for r in resid) / n) ** 0.5
    return slope, intercept, sd


def _fit_proportional(x: List[float], y: List[float]) -> Tuple[float, float]:
    """Least-squares fit y = slope*x (no intercept). Returns
    (slope, residual_stddev)."""
    n = len(x)
    if n < 1:
        return 1.0, 0.0
    sxx = sum(xi * xi for xi in x)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    if sxx == 0:
        return 1.0, 0.0
    slope = sxy / sxx
    resid = [yi - slope * xi for xi, yi in zip(x, y)]
    mean_r = sum(resid) / n
    sd = (sum((r - mean_r) ** 2 for r in resid) / n) ** 0.5
    return slope, sd


def _correlation(x: List[float], y: List[float]) -> float:
    """Return Pearson correlation, or 0 when either side is constant."""
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    numerator = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    denom_x = sum((xi - mx) ** 2 for xi in x)
    denom_y = sum((yi - my) ** 2 for yi in y)
    denom = (denom_x * denom_y) ** 0.5
    if denom == 0:
        return 0.0
    return numerator / denom


def _metric_pairs(
    pairs: List[Pair],
    field_name: str,
) -> Tuple[List[float], List[float]]:
    """Collect OpenFlight/Trackman values for one metric from good pairs."""
    of: List[float] = []
    tm: List[float] = []
    for p in pairs:
        if p.match_quality != "good" or p.of is None or p.tm is None:
            continue
        a = getattr(p.of, field_name)
        b = getattr(p.tm, field_name)
        if a is None or b is None:
            continue
        of.append(a)
        tm.append(b)
    return of, tm


def _rmse(values: List[float]) -> float:
    if not values:
        return 0.0
    return (sum(v * v for v in values) / len(values)) ** 0.5


def print_ball_speed_calibration(pairs: List[Pair]) -> None:
    """Fit ball-speed-correction models from OF/TM pairs and print
    recommended calibration constants. Lets the user decide whether to
    wire them into the live processor.

    Two models, both reported:

    * Slope+offset: ``corrected_of = slope * raw_of + offset`` (best fit
      when there's both a multiplicative and additive systematic).
    * Slope-only:   ``corrected_of = slope * raw_of`` (one knob, simpler
      to maintain; gives up a small amount of residual fit quality).

    The least-squares regressors run on the existing good-pair set.
    For applying the correction, the inverse mapping is what matters:
    we want ``corrected_of`` to predict ``tm`` from ``raw_of``. So the
    regression is ``tm = slope*of + offset`` and the recommended live
    correction is exactly those constants.
    """
    of: List[float] = []
    tm: List[float] = []
    for p in pairs:
        if p.match_quality != "good":
            continue
        if p.of is None or p.tm is None:
            continue
        a, b = p.of.ball_speed_mph, p.tm.ball_speed_mph
        if a is None or b is None:
            continue
        of.append(a)
        tm.append(b)

    if len(of) < 3:
        print()
        print("  (not enough good ball-speed pairs to recommend a "
              f"calibration — need >=3, got {len(of)})")
        return

    slope_lin, offset_lin, sd_lin = _fit_linear(of, tm)
    slope_prop, sd_prop = _fit_proportional(of, tm)
    raw_resid_sd = (
        sum((b - a) ** 2 for a, b in zip(of, tm)) / len(of)
    ) ** 0.5

    print()
    print("=" * 72)
    print("  BALL-SPEED CALIBRATION RECOMMENDATION")
    print("=" * 72)
    print(f"  Fit on {len(of)} good ball-speed pairs.")
    print()
    print(f"  Uncorrected residual (TM - OF) stddev:    {raw_resid_sd:6.3f} mph")
    print()
    print( "  Two-parameter model (slope + offset):")
    print(f"    corrected_of_mph = {slope_lin:.5f} * raw_of_mph "
          f"{offset_lin:+.4f}")
    print(f"    residual stddev: {sd_lin:6.3f} mph")
    print()
    print( "  One-parameter model (slope only):")
    print(f"    corrected_of_mph = {slope_prop:.5f} * raw_of_mph")
    print(f"    residual stddev: {sd_prop:6.3f} mph")
    print()
    print("  Caveats:")
    print("    - Calibration depends on the radar unit and mounting")
    print("      geometry. Re-fit per setup, not per session.")
    print("    - One session of ~20 shots is a useful start but light")
    print("      on data — collect more before wiring this into the")
    print("      live processor as a tuned default.")
    print("    - Apply the correction at the FFT-peak-to-mph step in")
    print("      rolling_buffer/processor.py, not at the UI layer, so")
    print("      that downstream metrics (smash, carry, etc.) benefit.")
    print("=" * 72)


def print_launch_angle_calibration(pairs: List[Pair]) -> None:
    """Print Trackman-based diagnostics for vertical and horizontal launch.

    Angle calibration is less straightforward than ball speed. A useful
    report needs to show both the simple bias and whether OpenFlight has
    shot-to-shot correlation with Trackman. Low correlation means an offset
    may improve average error but should not be treated as a reliable
    measured-angle calibration.
    """
    axes = [
        ("Vertical", "launch_angle_vertical", "launch_v"),
        ("Horizontal", "launch_angle_horizontal", "launch_h"),
    ]

    print()
    print("=" * 72)
    print("  LAUNCH-ANGLE CALIBRATION DIAGNOSTICS")
    print("=" * 72)

    any_axis = False
    for label, field_name, short_name in axes:
        of, tm = _metric_pairs(pairs, field_name)
        if len(of) < 3:
            print(
                f"  {label}: not enough good paired values "
                f"(need >=3, got {len(of)})"
            )
            continue

        any_axis = True
        deltas = [a - b for a, b in zip(of, tm)]
        bias = statistics.fmean(deltas)
        mae = statistics.fmean(abs(d) for d in deltas)
        raw_rmse = _rmse(deltas)

        offset = -bias
        offset_resid = [b - (a + offset) for a, b in zip(of, tm)]
        offset_rmse = _rmse(offset_resid)

        slope, intercept, fit_sd = _fit_linear(of, tm)
        corr = _correlation(of, tm)

        print()
        print(f"  {label} launch ({short_name}) — {len(of)} good paired values")
        print(f"    raw OF-TM bias:         {bias:+6.2f} deg")
        print(f"    raw MAE / RMSE:         {mae:6.2f} / {raw_rmse:6.2f} deg")
        print(f"    offset-only correction: corrected = raw {offset:+.2f} deg")
        print(f"    offset-only RMSE:       {offset_rmse:6.2f} deg")
        print(
            f"    linear correction:      corrected = {slope:.4f} * raw "
            f"{intercept:+.3f}"
        )
        print(f"    linear residual stddev: {fit_sd:6.2f} deg")
        print(f"    OF↔TM correlation:      {corr:+6.2f}")
        if abs(corr) < 0.5:
            print(
                "    note: weak shot-to-shot correlation; prefer using this "
                "to tune gating/fallbacks before applying a live correction."
            )

    if not any_axis:
        print("  (no launch-angle calibration available from this comparison)")

    print("=" * 72)


def _format_metric_summary_line(label: str, vals: List[float],
                                unit: str) -> str:
    """Format a single per-metric summary row. With one sample we
    suppress the (meaningless) stddev rather than printing 0.00 next
    to a single value, which previously misled the user."""
    n = len(vals)
    if n == 1:
        return (f"    {label:<12}  {vals[0]:>+12.2f} {unit:<3}  "
                f"{'(n=1)':>9}  {vals[0]:>+7.2f}")
    mean = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    mx = max(vals, key=abs)
    return (f"    {label:<12}  {mean:>+12.2f} {unit:<3}  "
            f"{sd:>9.2f}  {mx:>+7.2f}")


def _format_pair_short(row: Dict[str, Any]) -> str:
    """One-line summary of a pair, used in the rejected/unmatched
    detail sections."""
    of_n = row.get("shot_number_of") or "-"
    tm_n = row.get("shot_number_tm") or "-"
    of_t = row.get("timestamp_of") or ""
    tm_t = row.get("timestamp_tm") or ""
    of_b = row.get("ball_speed_of")
    tm_b = row.get("ball_speed_tm")
    of_b_s = f"{of_b:.1f} mph" if of_b is not None else "(no OF)"
    tm_b_s = f"{tm_b:.1f} mph" if tm_b is not None else "(no TM)"
    notes = row.get("notes") or ""
    return (f"    OF #{of_n} ({of_t})  ↔  TM #{tm_n} ({tm_t})    "
            f"OF: {of_b_s}    TM: {tm_b_s}    {notes}")


def print_summary(pairs: List[Pair]) -> None:
    rows = [_row(p) for p in pairs]
    by_club: Dict[str, List[Dict[str, Any]]] = {}
    counts = {"good": 0, "ball_speed_mismatch": 0,
              "unmatched_openflight": 0, "unmatched_trackman": 0}
    for r in rows:
        counts[r["match_quality"]] = counts.get(r["match_quality"], 0) + 1
        by_club.setdefault(r["club"] or "(no club)", []).append(r)

    print()
    print("=" * 72)
    print("  COMPARISON SUMMARY")
    print("=" * 72)
    print(f"  Total pairs:           {len(rows)}")
    print(f"  Good:                  {counts['good']}")
    print(f"  Ball-speed mismatch:   {counts['ball_speed_mismatch']}")
    print(f"  Unmatched OpenFlight:  {counts['unmatched_openflight']}")
    print(f"  Unmatched Trackman:    {counts['unmatched_trackman']}")
    print()

    for club in sorted(by_club):
        good = [r for r in by_club[club] if r["match_quality"] == "good"]
        if not good:
            print(f"  {club}: no good pairs ({len(by_club[club])} total)")
            continue
        print(f"  {club} — {len(good)} good pair(s) "
              f"(of {len(by_club[club])} total)")
        print(f"    {'metric':<12}  {'mean Δ (of-tm)':>15}  "
              f"{'stddev':>10}  {'|max|':>8}")
        for key, label, unit in _DELTA_LABELS:
            vals = [r[key] for r in good if r[key] is not None]
            if not vals:
                continue
            print(_format_metric_summary_line(label, vals, unit))
        print()

    # Detail sections for non-good pairs so the user can see which
    # specific shots got rejected and decide whether to manually fix
    # the alignment or ignore them.
    rejected = [r for r in rows if r["match_quality"] == "ball_speed_mismatch"]
    if rejected:
        print(f"  Ball-speed-mismatch pairs ({len(rejected)}) "
              "— excluded from per-club stats:")
        for r in rejected:
            print(_format_pair_short(r))
        print()

    unmatched = [r for r in rows
                 if r["match_quality"].startswith("unmatched_")]
    if unmatched:
        print(f"  Unmatched pairs ({len(unmatched)}):")
        for r in unmatched:
            print(_format_pair_short(r))
        print()

    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Pair OpenFlight ↔ Trackman shots and emit a "
                    "side-by-side comparison CSV.",
    )
    parser.add_argument("--openflight", required=True, type=Path,
                        help="OpenFlight session JSONL file")
    parser.add_argument("--trackman", required=True, type=Path,
                        help="Trackman CSV export")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output comparison CSV")
    parser.add_argument("--ball-speed-tol", type=float, default=5.0,
                        help="Max ball-speed delta (mph) before a pair "
                             "is flagged as a likely mis-pair (default 5.0)")
    parser.add_argument("--club-filter", default=None,
                        help="Comma-separated list of clubs to include "
                             "(e.g. '7-iron,driver'). Default: all.")
    args = parser.parse_args(argv)

    if not args.openflight.exists():
        print(f"OpenFlight log not found: {args.openflight}", file=sys.stderr)
        return 2
    if not args.trackman.exists():
        print(f"Trackman CSV not found: {args.trackman}", file=sys.stderr)
        return 2

    of_shots = load_openflight(args.openflight)
    tm_shots = load_trackman(args.trackman)
    print(f"Loaded {len(of_shots)} OpenFlight shots, "
          f"{len(tm_shots)} Trackman shots")

    club_filter = (
        [c.strip() for c in args.club_filter.split(",") if c.strip()]
        if args.club_filter else None
    )
    pairs = pair_shots(of_shots, tm_shots,
                       ball_speed_tol_mph=args.ball_speed_tol,
                       club_filter=club_filter)
    write_comparison_csv(pairs, args.output)
    print(f"Wrote {len(pairs)} pair rows to {args.output}")
    print_summary(pairs)
    print_ball_speed_calibration(pairs)
    print_launch_angle_calibration(pairs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
