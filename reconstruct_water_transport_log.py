#!/usr/bin/env python3
"""
Reconstruct water_transport_log.csv (and water_transport_events.csv) from an
existing autorun.log for runs completed before the 2026-04-24 per-step logging
was added.

Usage:
    python reconstruct_water_transport_log.py                  # all sites
    python reconstruct_water_transport_log.py 2022_T3_25m      # one site
    python reconstruct_water_transport_log.py 2022_T3_25m 2023_CP_25m  # several
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent

# ── Regex patterns ───────────────────────────────────────────────────────────

# Step completion timestamps (Python-side prints, all represent t1 = step end)
_STEP_PATTERNS = [
    # "2023-06-13 22:00:00: updated initial_profile.sno"
    # "2023-06-13 22:00:00: adjustments (wet-drain/no-obs) — SNO written..."
    # "2023-06-13 22:00:00: 47 layer temps → SETTEMPS"
    # "2023-06-13 22:00:00: wrote raw=..., adjusted=..."
    re.compile(
        r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): "
        r"(?:updated|adjustments|[0-9]+ layer temps|wrote raw)"
    ),
    # "[daemon] CHECKPOINT 2023-06-13T22:00:00"
    re.compile(r"^\[daemon\] CHECKPOINT (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"),
]

# Scheme transitions (all timestamps are step START times, t0, unless noted)
#
# "Stabilization complete at 2023-06-13 21:00:00 — switching to RICHARDSEQUATION"
#   t0 = step start; step t0→t0+h runs as RE; first RE step ends at t0+h
_RE_STAB = re.compile(
    r"Stabilization complete at (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
)

# "2023-06-13 22:00:00: RE SafeMode convergence failure; switching to BUCKET..."
#   t1 = step END; THIS step was RE (ok, but SafeMode); NEXT step starts BUCKET
_RE_SAFEMODE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): RE SafeMode convergence failure"
)

# "2023-06-13T21:00:00→2023-06-13T22:00:00: RE timed out ... retrying with BUCKET"
#   t1 (after →) = step END; this step was retried and ran as BUCKET
_RE_TIMEOUT = re.compile(
    r"→(\d{4}-\d{2}-\d{2}[T ]?\d{2}:\d{2}:\d{2}): RE timed out"
)

# "2023-06-14 21:00:00: RE fallback period over — switching back to RICHARDSEQUATION"
#   t0 = step START; this step and beyond run as RE
_RE_RESTORE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}): RE fallback period over"
)

# "Resume: INI set to RICHARDSEQUATION" / "Resume: INI set to BUCKET"
_RE_RESUME_INI = re.compile(r"^Resume: INI set to (\w+)")

# "Fresh start: ..." — marks where the current run begins in the log
_RE_FRESH = re.compile(r"^Fresh start:")

# "Spawning SNOWPACK daemon: ... -b YYYY-MM-DDTHH:MM ..." — run start marker
_RE_SPAWN = re.compile(r"Spawning SNOWPACK daemon:.*-b (\d{4}-\d{2}-\d{2}T\d{2}:\d{2})")


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s.replace("T", " "))


def reconstruct(site_id: str) -> None:
    log_path = BASE / site_id / "autorun.log"
    out_dir  = BASE / site_id / "output"

    if not log_path.exists():
        print(f"{site_id}: no autorun.log — skipping")
        return
    if not out_dir.exists():
        print(f"{site_id}: no output/ directory — skipping")
        return

    lines = log_path.read_text(errors="replace").splitlines()

    # ── Find where the last run starts in the log ─────────────────────────── #
    # Multiple runs may be concatenated if the site was restarted from scratch
    # (fresh start). Split only on "Fresh start: ... cleared" which marks a
    # true restart from step 0, discarding all prior output. Daemon respawns
    # (scheme switches) also print "Spawning SNOWPACK daemon" but are NOT run
    # boundaries — do not split on those.
    last_run_start_line = 0
    for i, line in enumerate(lines):
        if _RE_FRESH.match(line.strip()):
            last_run_start_line = i
    lines = lines[last_run_start_line:]

    # ── Collect step completion timestamps ────────────────────────────────── #
    step_times: set[pd.Timestamp] = set()
    for line in lines:
        s = line.strip()
        for p in _STEP_PATTERNS:
            m = p.match(s)
            if m:
                step_times.add(_ts(m.group(1)))

    if not step_times:
        print(f"{site_id}: no step timestamps found — skipping")
        return

    steps = sorted(step_times)
    n = len(steps)

    # Infer step interval from median of first 50 consecutive gaps
    sample = min(50, n - 1)
    gaps = [(steps[i + 1] - steps[i]).total_seconds() for i in range(sample)]
    interval = pd.Timedelta(seconds=sorted(gaps)[len(gaps) // 2])

    # ── Collect scheme-transition events ─────────────────────────────────── #
    # Each event is (step_end_timestamp, new_scheme):
    # "from this step end onward, scheme = new_scheme"
    initial_scheme = "BUCKET"   # adaptive default for fresh starts
    raw_events: list[tuple[pd.Timestamp, str]] = []

    for line in lines:
        s = line.strip()

        m = _RE_RESUME_INI.match(s)
        if m:
            initial_scheme = m.group(1)
            continue

        m = _RE_STAB.search(s)
        if m:
            # t0 = step start → first RE step ends at t0+interval
            raw_events.append((_ts(m.group(1)) + interval, "RICHARDSEQUATION"))
            continue

        m = _RE_SAFEMODE.match(s)
        if m:
            # t1 = step END (this step was RE); NEXT step starts BUCKET
            raw_events.append((_ts(m.group(1)) + interval, "BUCKET"))
            continue

        m = _RE_TIMEOUT.search(s)
        if m:
            # t1 = step END; this step ran as BUCKET (retried); mark from t1
            raw_events.append((_ts(m.group(1)), "BUCKET"))
            continue

        m = _RE_RESTORE.match(s)
        if m:
            # t0 = step start → first RE step after restore ends at t0+interval
            raw_events.append((_ts(m.group(1)) + interval, "RICHARDSEQUATION"))
            continue

    # ── Build sorted events list ──────────────────────────────────────────── #
    # Prepend an anchor at the start so every step gets assigned a scheme
    anchor = steps[0] - pd.Timedelta(hours=24)
    events: list[tuple[pd.Timestamp, str]] = [(anchor, initial_scheme)] + sorted(raw_events)
    # Remove duplicates at the same timestamp (keep last)
    seen: dict[pd.Timestamp, str] = {}
    for t, s in events:
        seen[t] = s
    events = sorted(seen.items())

    # ── Write sparse events file ──────────────────────────────────────────── #
    events_path = out_dir / "water_transport_events.csv"
    with open(events_path, "w", encoding="utf-8") as f:
        f.write("datetime,scheme\n")
        for t, s in events:
            f.write(f"{t.isoformat()},{s}\n")

    # ── Expand to dense per-step log ─────────────────────────────────────── #
    ev_idx = 0
    cur_scheme = events[0][1]
    rows: list[str] = []
    for step_t in steps:
        while ev_idx + 1 < len(events) and events[ev_idx + 1][0] <= step_t:
            ev_idx += 1
            cur_scheme = events[ev_idx][1]
        rows.append(f"{step_t.isoformat()},{cur_scheme}\n")

    log_path_out = out_dir / "water_transport_log.csv"
    with open(log_path_out, "w", encoding="utf-8") as f:
        f.write("datetime,scheme\n")
        f.writelines(rows)

    bucket = sum(1 for r in rows if r.endswith(",BUCKET\n"))
    re_count = n - bucket
    print(
        f"{site_id}: {n} steps — {re_count} RE ({re_count*100//n}%), "
        f"{bucket} BUCKET ({bucket*100//n}%)"
    )
    if len(events) > 1:
        print(f"  transitions: {[(str(t), s) for t, s in events[1:]]}")


def main() -> None:
    if len(sys.argv) > 1:
        sites = sys.argv[1:]
    else:
        sites = [
            d.name for d in sorted(BASE.iterdir())
            if d.is_dir()
            and re.match(r"^\d{4}_.+_\d+m$", d.name)
            and (d / "autorun.log").exists()
        ]

    for sid in sites:
        reconstruct(sid)


if __name__ == "__main__":
    main()
