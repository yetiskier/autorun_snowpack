"""
SNOWPACK autorun — Streamlit GUI
Run from the autorun_snowpack/ directory:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import json
import math
import streamlit as st
import streamlit.components.v1 as components
import tomlkit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR       = Path(__file__).resolve().parent   # autorun_snowpack/
SETTINGS_FILE = APP_DIR / "settings.toml"
SCRIPT        = APP_DIR / "autorun_snowpack.py"
VIZ_SCRIPT    = APP_DIR / "visualize_pro.py"
PYTHON        = "python3"

TEMP_DIR  = APP_DIR / "AllCoreDataCommonFormat" / "Concatenated_Temperature_files"
PROM_DIR  = APP_DIR / "AllCoreDataCommonFormat" / "Depth_change_estimate" / "PROMICE"
DEN_DIR   = APP_DIR / "AllCoreDataCommonFormat" / "CoreDataEGIG"

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SNOWPACK autorun",
    page_icon="❄️",
    layout="wide",
)

st.title("❄️ SNOWPACK autorun")

# ---------------------------------------------------------------------------
# Core discovery
# ---------------------------------------------------------------------------
_SITE_RE = re.compile(r"^(\d{4})_(.+?)_(\d+)m_Tempconcatenated\.csv$")


def _temp_date_range(path: Path) -> tuple[str, str]:
    """Read the first and last timestamp from a Tempconcatenated CSV.
    The file has 4 header lines; timestamps are in column 0."""
    first = last = ""
    with open(path, errors="replace") as fh:
        for _ in range(4):          # skip 3 metadata rows + column-header row
            fh.readline()
        for line in fh:
            ts = line.split(",", 1)[0].strip()
            if not ts:
                continue
            if not first:
                first = ts[:10]     # keep date part only (YYYY-MM-DD)
            last = ts[:10]
    return first, last


def discover_cores() -> list[tuple[int, str, int, str, str]]:
    """Return list of (year, site, depth, date_start, date_end) for cores
    that have all three required data files."""
    cores = []
    if not TEMP_DIR.exists():
        return cores
    for tf in sorted(TEMP_DIR.glob("*_Tempconcatenated.csv")):
        m = _SITE_RE.match(tf.name)
        if not m:
            continue
        year, site, depth = int(m.group(1)), m.group(2), int(m.group(3))
        sid = f"{year}_{site}_{depth}m"
        prom = PROM_DIR / f"{sid}_daily_PROMICE_snowfall.csv"
        den  = DEN_DIR  / f"{sid}_den.csv"
        if prom.exists() and den.exists():
            d0, d1 = _temp_date_range(tf)
            cores.append((year, site, depth, d0, d1))
    return cores


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def site_id(year: int, site: str, depth: int) -> str:
    return f"{year}_{site}_{depth}m"


def project_dir(sid: str) -> Path:
    return APP_DIR / sid


def log_path(sid: str) -> Path:
    return project_dir(sid) / "autorun.log"


def pid_file(sid: str) -> Path:
    return project_dir(sid) / ".autorun.pid"


def read_pid(sid: str) -> int | None:
    pf = pid_file(sid)
    if pf.exists():
        try:
            return int(pf.read_text().strip())
        except ValueError:
            return None
    return None


def write_pid(sid: str, pid: int) -> None:
    pid_file(sid).write_text(str(pid))


def clear_pid(sid: str) -> None:
    pf = pid_file(sid)
    if pf.exists():
        pf.unlink()


def is_running(sid: str) -> bool:
    pid = read_pid(sid)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)   # signal 0 = existence check
    except (ProcessLookupError, PermissionError):
        clear_pid(sid)
        return False
    # A zombie passes the kill(0) check but the process has actually finished.
    # Read /proc/<pid>/status to detect this case.
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("State:"):
                if "Z" in line:          # zombie — process done, not yet reaped
                    clear_pid(sid)
                    return False
                break
    except OSError:
        # /proc entry vanished between the kill check and here — process is gone
        clear_pid(sid)
        return False
    return True


def kill_run(sid: str) -> None:
    pid = read_pid(sid)
    if pid:
        try:
            # Kill the entire process group (shell + python + grep)
            os.killpg(os.getpgid(pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        clear_pid(sid)


# Lines from SNOWPACK's Richards-equation solver that clutter the log.
_LOG_FILTER = (
    # Richards-equation convergence warning blocks (header line has no leading whitespace)
    r"Richards-Equation solver"
    r"|SafeMode was used"
    # Indented body lines of the warning block
    r"|^\s*(layer \[|upper boundary \[|SAFE MODE|Estimated mass balance|[-]{10}"
    r"|POSSIBLE SOLUTIONS|={5,}|LB_COND_WATERFLUX|WATERTRANSPORTMODEL"
    r"|SOLVER DUMP|surfacefluxrate|wet state\.|of the ini-file\."
    r"|Verify that the soil|If the snow|If the soil is|Try bucket scheme)"
)

def launch_run(year: int, site: str, depth: int, run_until: str) -> int:
    sid = site_id(year, site, depth)
    log = log_path(sid)
    log.parent.mkdir(parents=True, exist_ok=True)

    py_args = [PYTHON, "-u", str(SCRIPT),
               "--site", site, "--year", str(year), "--depth", str(depth)]
    if run_until.strip():
        py_args += ["--run-until", run_until.strip()]

    # Pipe through grep to strip SNOWPACK's verbose Richards-equation diagnostics
    py_cmd   = " ".join(shlex.quote(a) for a in py_args)
    shell_cmd = f"{py_cmd} 2>&1 | grep -vE {shlex.quote(_LOG_FILTER)}"

    with open(log, "w") as fh:
        proc = subprocess.Popen(
            ["bash", "-c", shell_cmd],
            stdout=fh, stderr=fh,
            cwd=str(APP_DIR),
            start_new_session=True,
        )
    write_pid(sid, proc.pid)
    return proc.pid


def load_settings() -> tomlkit.TOMLDocument:
    return tomlkit.parse(SETTINGS_FILE.read_text())


def save_settings(doc: tomlkit.TOMLDocument) -> None:
    SETTINGS_FILE.write_text(tomlkit.dumps(doc))


# Matches anything that looks like a file-system path (absolute, relative, or ~)
_PATH_IN_LINE = re.compile(r'(?:^|[\s\'",(=])(?:~|\.{0,2})?/[/\w.\-]+')

def _strip_path_lines(text: str) -> str:
    """Remove lines that contain a file-system path."""
    return "\n".join(
        line for line in text.splitlines()
        if not _PATH_IN_LINE.search(line)
    )


def read_log_tail(sid: str, n_lines: int = 60) -> str:
    lp = log_path(sid)
    if not lp.exists():
        return "(no log yet)"
    lines = lp.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def get_pro_current_time(sid: str) -> "pd.Timestamp | None":
    """Read the most recent timestep from the tail of the PRO file."""
    pro = find_pro_file(sid)
    if pro is None or not pro.exists():
        return None
    try:
        with open(pro, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 3000))
            tail = fh.read().decode("utf-8", errors="replace")
        for line in reversed(tail.splitlines()):
            if line.startswith("0500,"):
                return pd.to_datetime(line[5:].strip(), dayfirst=True)
    except Exception:
        pass
    return None


@st.cache_data(show_spinner=False)
def get_expected_date_range(sid: str) -> "tuple[pd.Timestamp | None, pd.Timestamp | None]":
    """Return (start, end) dates from the Tempconcatenated CSV for this sid."""
    csv = TEMP_DIR / f"{sid}_Tempconcatenated.csv"
    if not csv.exists():
        return None, None
    try:
        d0, d1 = _temp_date_range(csv)
        return pd.to_datetime(d0), pd.to_datetime(d1)
    except Exception:
        return None, None


def output_figures(sid: str) -> list[Path]:
    out = project_dir(sid) / "output"
    if not out.exists():
        return []
    return sorted(out.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)


def sites_with_results() -> list[str]:
    """Return site IDs that have at least one .pro file in their output dir."""
    results = []
    for d in sorted(APP_DIR.iterdir()):
        if not d.is_dir():
            continue
        if not re.match(r"^\d{4}_.+_\d+m$", d.name):
            continue
        out = d / "output"
        if out.exists() and any(out.glob("*.pro")):
            results.append(d.name)
    return results


def find_pro_file(sid: str) -> Path | None:
    out = project_dir(sid) / "output"
    pros = list(out.glob("*.pro")) if out.exists() else []
    if not pros:
        return None
    # Prefer files without "backup" or "Copy" in the name; among those pick
    # the most recently modified so we always show the latest run output.
    primary = [p for p in pros if "backup" not in p.name.lower() and "copy" not in p.name.lower()]
    candidates = primary if primary else pros
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# Swiss grain type catalog (same as visualize_pro.py)
# ---------------------------------------------------------------------------
MK_CATALOG: dict[int, tuple[str, str]] = {
    220: ("#d4b96a", "220 DF"),          230: ("#c9a84c", "230 DF mixed"),
    231: ("#b89030", "231 DF small"),
    321: ("#a8d8ea", "321 RG small"),    330: ("#5dade2", "330 RG"),
    341: ("#1a6fa8", "341 RG large"),
    430: ("#f4a460", "430 FC"),          431: ("#e8874a", "431 FC mixed"),
    440: ("#d4622a", "440 FC/RG"),       470: ("#ffd580", "470 FCsf"),
    471: ("#ffbe33", "471 FCsf mixed"),  472: ("#e6a000", "472 FCsf large"),
    490: ("#c06820", "490 FC mixed"),
    550: ("#e74c3c", "550 DH"),          572: ("#c0392b", "572 DH mixed"),
    591: ("#922b21", "591 DH cup"),
    751: ("#d7a8e0", "751 MFr"),         752: ("#bb72cc", "752 MFr large"),
    770: ("#8e44ad", "770 MFcr"),        772: ("#6c3483", "772 MFcr mixed"),
    791: ("#f1948a", "791 MF wet"),      792: ("#e74c8b", "792 MF wet large"),
    880: ("#7f8c8d", "880 IF/ice"),
    951: ("#2ecc71", "951 FC/DH mixed"), 990: ("#95a5a6", "990 mixed"),
}

# Hand-hardness index (1=fist … 6=ice) by grain-type family (first digit)
_HARDNESS_BY_FAMILY = {2: 1.5, 3: 2.5, 4: 3.5, 5: 3.0, 7: 4.5, 8: 6.0, 9: 2.5}

def _code_hardness(code: float) -> float:
    return _HARDNESS_BY_FAMILY.get(int(code) // 100, 2.5) if code > 0 else 0.0

def _code_color(code: float) -> str:
    return MK_CATALOG.get(int(round(code)), ("#cccccc", "unknown"))[0]

def _code_name(code: float) -> str:
    return MK_CATALOG.get(int(round(code)), ("#cccccc", "unknown"))[1]

# Plotly discrete colorscale for the grain-type heatmap
_ALL_CODES  = sorted(MK_CATALOG.keys())
_N          = len(_ALL_CODES)
_CODE_IDX   = {c: i for i, c in enumerate(_ALL_CODES)}
# Paired entries create a true step function (no interpolation between colours).
# Each colour i occupies the band [i/_N, (i+1)/_N] in the normalised range.
_MK_COLORSCALE = []
for _i, _c in enumerate(_ALL_CODES):
    _col = MK_CATALOG[_c][0]
    _MK_COLORSCALE.append([_i / _N, _col])
    _MK_COLORSCALE.append([(_i + 1) / _N, _col])


# ---------------------------------------------------------------------------
# PRO parser (minimal, reused from visualize_pro.py)
# ---------------------------------------------------------------------------
_WANTED     = {501, 502, 503, 506, 509, 510, 512, 513, 515}     # main PRO codes
_SAT_WANTED = {501, 502, 515}                                     # codes for residual saturation only

def _parse_pro(path: Path) -> dict:
    data, current, times, in_data = {c: [] for c in _WANTED}, {}, [], False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("[DATA]"):
                in_data = True; continue
            if not in_data:
                continue
            code_str, _, rest = line.partition(",")
            try:
                code = int(code_str)
            except ValueError:
                continue
            if code == 500:
                if current:
                    for c in _WANTED:
                        data[c].append(current.get(c, np.array([])))
                current = {}
                times.append(pd.to_datetime(rest.strip(), dayfirst=True))
            elif code in _WANTED:
                vals = rest.split(",")
                current[code] = np.array([float(v) for v in vals[1:]])
    if current:
        for c in _WANTED:
            data[c].append(current.get(c, np.array([])))
    return {"times": times, **data}


def _to_grid(times, heights_list, values_list, depth_grid, kind="linear"):
    grid = np.full((len(times), len(depth_grid)), np.nan)
    for i, (h, v) in enumerate(zip(heights_list, values_list)):
        if len(h) < 2 or len(v) < 2:
            continue
        dm = (h.max() - h) / 100.0
        order = np.argsort(dm)
        dm, v = dm[order], v[order]
        _, u = np.unique(dm, return_index=True)
        dm, v = dm[u], v[u]
        if len(dm) < 2:
            continue
        grid[i] = interp1d(dm, v, kind=kind, bounds_error=False,
                           fill_value=np.nan)(depth_grid)
    return grid


def _to_grid_stepfn(times, heights_list, values_list, depth_grid):
    """Assign each depth cell to the SNOWPACK element that contains it.

    Heights (code 501) are element TOPS in cm above the base, sorted ascending.
    Element j spans from heights[j-1] to heights[j].  Each depth cell gets the
    value of the element whose vertical extent covers that depth — no
    interpolation, so finite-element boundaries are preserved exactly.
    """
    grid = np.full((len(times), len(depth_grid)), np.nan)
    for i, (h, v) in enumerate(zip(heights_list, values_list)):
        if len(h) < 2 or len(v) < 2:
            continue
        order  = np.argsort(h)
        h_s    = h[order]           # element tops ascending (cm above base)
        v_s    = v[order]
        surface = h_s[-1]           # surface height (cm)
        # Convert depth grid (m from surface) → height from base (cm)
        h_at_d  = surface - depth_grid * 100.0
        # searchsorted 'left': for h_at_d in (h_s[j-1], h_s[j]] → returns j
        idx = np.searchsorted(h_s, h_at_d, side='left')
        valid = (idx < len(v_s)) & (h_at_d >= h_s[0])
        grid[i, valid] = v_s[idx[valid]]
    return grid


@st.cache_data(show_spinner="Parsing PRO file…")
def load_pro(pro_path_str: str, _mtime: float = 0.0) -> dict:
    """Parse PRO file and build gridded arrays. v4 (density-filtered refreezing)"""
    path = Path(pro_path_str)
    pro  = _parse_pro(path)
    times = pd.DatetimeIndex(pro["times"])
    soil_depth_m = np.array([h.max() / 100.0 for h in pro[501]])
    max_depth    = float(np.nanmax(soil_depth_m)) if len(soil_depth_m) else 30.0
    depth_grid   = np.arange(0, max_depth + 0.05, 0.05)

    T_grid   = _to_grid(times, pro[501], pro[503], depth_grid)
    # Step function: each depth cell gets the value of the element containing it,
    # preserving finite-element layer boundaries (no interpolation).
    MK_raw   = _to_grid_stepfn(times, pro[501], pro[513], depth_grid)
    Den_grid = _to_grid_stepfn(times, pro[501], pro[502], depth_grid)
    LWC_grid = _to_grid_stepfn(times, pro[501], pro[506], depth_grid)
    Ice_grid = _to_grid_stepfn(times, pro[501], pro[515], depth_grid)  # % by volume

    # Map grain-type codes to index for colour scale
    MK_idx = np.full_like(MK_raw, np.nan)
    for code, idx in _CODE_IDX.items():
        MK_idx = np.where(np.abs(MK_raw - code) < 0.5, float(idx), MK_idx)

    # Mask below soil
    for ti, sd in enumerate(soil_depth_m):
        below = depth_grid > sd
        MK_idx[ti, below]    = np.nan
        MK_raw[ti, below]    = np.nan
        Den_grid[ti, below]  = np.nan
        LWC_grid[ti, below]  = np.nan
        Ice_grid[ti, below]  = np.nan

    # Cumulative refreezing (mm w.e. = kg/m²)
    # Use raw per-element data (not gridded) to avoid Lagrangian→Eulerian
    # interpolation artifacts that create spurious ice-fraction jumps.
    # Refreezing is the only process that simultaneously increases ice mass
    # AND decreases LWC mass, so: refreezing = min(Δice+, ΔLWC-) per step.
    n_times = len(times)
    ice_col = np.zeros(n_times)   # firn-only column ice  mass (kg/m²)
    lwc_col = np.zeros(n_times)   # firn-only column LWC  mass (kg/m²)
    for ti in range(n_times):
        h   = np.asarray(pro[501][ti], dtype=float)
        ice = np.asarray(pro[515][ti], dtype=float)
        lwc = np.asarray(pro[506][ti], dtype=float)
        den = np.asarray(pro[502][ti], dtype=float)
        n = min(len(h), len(ice), len(lwc), len(den))
        if n < 2:
            continue
        h = h[:n]; ice = ice[:n]; lwc = lwc[:n]; den = den[:n]
        order = np.argsort(h)
        h = h[order]; ice = ice[order]; lwc = lwc[order]; den = den[order]
        # Exclude soil and glacial-ice elements:
        #   h > 0  → above the soil/ice interface (reference level)
        #   den < 900  → not pure glacial ice (firn ≤ ~880, ice ≈ 917 kg/m³)
        mask = (h > 0) & (den < 900.0)
        if mask.sum() < 2:
            continue
        h = h[mask]; ice = ice[mask]; lwc = lwc[mask]
        nf = len(h)
        thick = np.empty(nf)
        thick[1:] = (h[1:] - h[:-1]) / 100.0   # m
        thick[0]  = h[0] / 100.0                 # bottom firn element → soil at h=0
        ice_col[ti] = np.nansum(ice / 100.0 * thick * 917.0)
        lwc_col[ti] = np.nansum(lwc / 100.0 * thick * 1000.0)
    d_ice = np.diff(ice_col)
    d_lwc = np.diff(lwc_col)
    # Count all refreezing as permanent: once ice refreezes below ~4 cm depth
    # it does not melt again, so no credit/debit system is needed.
    refreezing_per_step = np.minimum(
        np.where(d_ice > 0, d_ice, 0.0),
        np.where(d_lwc < 0, -d_lwc, 0.0),
    )
    cumul_refreezing = np.concatenate([[0.0], np.cumsum(refreezing_per_step)])

    return {
        "times":             times,
        "depth_grid":        depth_grid,
        "soil_depth_m":      soil_depth_m,
        "T_grid":            T_grid,
        "MK_raw":            MK_raw,
        "MK_idx":            MK_idx,
        "Den_grid":          Den_grid,
        "LWC_grid":          LWC_grid,
        "cumul_refreezing":  cumul_refreezing,   # kg/m² = mm w.e., len = n_times
        "raw":               pro,
    }


@st.cache_data(show_spinner="Computing residual saturation…")
def load_sat_grid(pro_path_str: str, _mtime: float = 0.0) -> dict:
    """Coléou & Lesaffre (1998) residual saturation for the top 5 m only.

    Parses only codes 501 (heights), 502 (density), 516 (air vol %) and grids
    to a 5 m depth axis so this stays fast even for multi-year PRO files.
    """
    path = Path(pro_path_str)
    data: dict = {c: [] for c in _SAT_WANTED}
    current: dict = {}
    times: list = []
    in_data = False
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("[DATA]"):
                in_data = True; continue
            if not in_data:
                continue
            code_str, _, rest = line.partition(",")
            try:
                code = int(code_str)
            except ValueError:
                continue
            if code == 500:
                if current:
                    for c in _SAT_WANTED:
                        data[c].append(current.get(c, np.array([])))
                current = {}
                times.append(pd.to_datetime(rest.strip(), dayfirst=True))
            elif code in _SAT_WANTED:
                vals = rest.split(",")
                current[code] = np.array([float(v) for v in vals[1:]])
    if current:
        for c in _SAT_WANTED:
            data[c].append(current.get(c, np.array([])))

    times_idx = pd.DatetimeIndex(times)
    depth_5m  = np.arange(0.0, 5.05, 0.05)   # 101 depth points, much less than full grid

    Den5 = _to_grid_stepfn(times_idx, data[501], data[502], depth_5m)
    Ice5 = _to_grid_stepfn(times_idx, data[501], data[515], depth_5m)

    # Mask below the firn/soil surface
    for ti, h_arr in enumerate(data[501]):
        if len(h_arr):
            sd = h_arr.max() / 100.0
            below = depth_5m > sd
            Den5[ti, below] = np.nan
            Ice5[ti, below] = np.nan

    # C&L (1998): θ_irr = 0.0264*φ + 0.0099  (vol fraction)
    # φ = 1 - θ_ice: total pore space (water + air), i.e. everything that is not ice
    with np.errstate(invalid="ignore", divide="ignore"):
        phi       = np.where(np.isfinite(Ice5), np.clip(1.0 - Ice5 / 100.0, 0.0, 1.0), np.nan)
        theta_irr = 0.0264 * phi + 0.0099
        Sat5 = np.where(
            np.isfinite(Den5) & (Den5 > 0),
            np.clip(theta_irr * (1000.0 / Den5) * 100.0, 0.0, None),
            np.nan,
        )

    return {"times": times_idx, "depth_5m": depth_5m, "Sat_grid": Sat5}


# ---------------------------------------------------------------------------
# Interactive charts
# ---------------------------------------------------------------------------

def _mk_name_grid(MK_raw: np.ndarray) -> np.ndarray:
    names = np.full(MK_raw.shape, "", dtype=object)
    for code in _ALL_CODES:
        names[np.abs(MK_raw - code) < 0.5] = MK_CATALOG[code][1]
    return names


@st.cache_data(show_spinner="Loading observed temperatures…")
def load_observed_temp(sid: str):
    """Parse Tempconcatenated CSV → (times, depths_m, T_array).

    Returns None if the file is not found.
    Keeps only non-negative depth columns (subsurface sensors).
    Resamples to hourly means.
    """
    csv = TEMP_DIR / f"{sid}_Tempconcatenated.csv"
    if not csv.exists():
        return None
    df = pd.read_csv(csv, skiprows=4, index_col=0, parse_dates=True)
    valid: dict[float, str] = {}
    for col in df.columns:
        try:
            d = float(col)
            if d >= 0.0:
                valid[d] = col
        except (ValueError, TypeError):
            pass
    if not valid:
        return None
    depths = np.array(sorted(valid.keys()))
    cols   = [valid[d] for d in depths]
    data   = df[cols].apply(pd.to_numeric, errors="coerce")
    data   = data.resample("1h").mean()
    times  = pd.DatetimeIndex(data.index)
    arr    = data.to_numpy(dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return times, depths, arr


def _profile_for_timestep(raw: dict, ti: int) -> dict | None:
    """Compute one column-profile data dict from raw PRO arrays."""
    heights = raw[501][ti]; mk = raw[513][ti]
    temp    = raw[503][ti]; rg = raw[512][ti]
    n = min(len(heights), len(mk), len(temp), len(rg))
    if n < 2:
        return None
    heights = heights[:n]; mk = mk[:n]; temp = temp[:n]; rg = rg[:n]
    order   = np.argsort(heights)
    heights = heights[order]; mk = mk[order]; temp = temp[order]; rg = rg[order]
    surface  = float(heights[-1])
    dtop     = (surface - heights) / 100.0
    hbase    = np.empty_like(heights)
    hbase[1:] = heights[:-1]
    hbase[0]  = heights[0] - (heights[1] - heights[0])
    dbot      = (surface - hbase) / 100.0
    thick     = np.clip(dbot - dtop, 0.001, None)
    dmid      = (dtop + dbot) / 2.0
    t_max     = min(max(float(np.abs(temp).max()) * 1.05, 2.0), 30.0)
    return {
        "hard":      [round(float(_code_hardness(c)), 2) for c in mk],
        "temp":      [round(float(v), 2)  for v in temp],
        "depth_mid": [round(float(v), 4)  for v in dmid],
        "thickness": [round(float(v), 4)  for v in thick],
        "depth_top": [round(float(v), 4)  for v in dtop],
        "depth_bot": [round(float(v), 4)  for v in dbot],
        "colors":    [_code_color(c)       for c in mk],
        "names":     [_code_name(c)        for c in mk],
        "t_min":     round(-t_max, 1),
    }


def _density_profile_for_timestep(raw: dict, ti: int, smooth_cm: float = 50.0) -> dict | None:
    """Return smoothed density-vs-depth profile (50 cm box filter by default)."""
    heights = raw[501][ti]
    density = raw[502][ti]
    n = min(len(heights), len(density))
    if n < 2:
        return None
    heights = heights[:n]; density = density[:n]
    order   = np.argsort(heights)
    heights = heights[order]; density = density[order]
    surface  = float(heights[-1])

    # Regular 5 cm depth grid spanning the full profile
    grid_step = 5.0          # cm
    depth_max = (surface - heights[0]) / 100.0
    depth_grid = np.arange(0.0, depth_max + grid_step / 100.0, grid_step / 100.0)

    # Step-function assignment: each grid cell → containing element
    h_at_d = surface - depth_grid * 100.0
    idx    = np.searchsorted(heights, h_at_d, side="left")
    valid  = (idx < n) & (h_at_d >= heights[0])
    den    = np.full(len(depth_grid), np.nan)
    den[valid] = density[idx[valid]]

    # Box-filter smoothing over 50 cm (= 50/5 = 10 cells)
    w = max(1, int(round(smooth_cm / grid_step)))
    kernel  = np.ones(w) / w
    padded  = np.pad(den, w // 2, mode="edge")
    den_sm  = np.convolve(padded, kernel, mode="valid")[: len(den)]
    den_sm[~valid] = np.nan

    mask = np.isfinite(den_sm)
    xs = [round(float(v), 1) for v in den_sm[mask]]
    ys = [round(float(v), 4) for v in depth_grid[mask]]
    return {"x": xs, "y": ys} if xs else None


@st.cache_data(show_spinner="Preparing density data…")
def _build_density_payload(pro_path_str: str, _mtime: float = 0.0) -> str:
    """JSON payload for the density heatmap + hover profiles component. v2-smooth"""
    d   = load_pro(pro_path_str, _mtime=_mtime)
    raw = d["raw"]
    times      = d["times"]
    depth_grid = d["depth_grid"]
    Den_grid   = d["Den_grid"]
    soil_depth = d["soil_depth_m"]

    n_times = len(times)
    step = max(1, n_times // 365)
    idx  = list(range(0, n_times, step))

    t_labels  = [str(times[i])[:16] for i in idx]
    den_sub   = Den_grid[idx]
    soil_sub  = [round(float(v), 3) for v in soil_depth[idx]]
    depth_list = [round(float(v), 4) for v in depth_grid]

    def _mat(arr):
        out = []
        for di in range(arr.shape[1]):
            row = []
            for ti in range(arr.shape[0]):
                v = arr[ti, di]
                row.append(None if math.isnan(v) else round(float(v), 1))
            out.append(row)
        return out

    profiles = [_density_profile_for_timestep(raw, ti) for ti in idx]
    first_p  = _density_profile_for_timestep(raw, 0)
    last_p   = _density_profile_for_timestep(raw, n_times - 1)

    payload = {
        "heatmap": {
            "x":        t_labels,
            "y":        depth_list,
            "z":        _mat(den_sub),
            "soil":     soil_sub,
            "maxDepth": float(depth_grid.max()),
        },
        "profiles": profiles,
        "first":    first_p,
        "last":     last_p,
        "t_first":  str(times[0])[:16],
        "t_last":   str(times[-1])[:16],
    }
    return json.dumps(payload, allow_nan=False)


@st.cache_data(show_spinner="Preparing hover data…")
def _build_hover_payload(pro_path_str: str, _mtime: float = 0.0) -> str:
    """Return JSON string with downsampled heatmap + all column profiles.

    Cached so we only rebuild when the PRO file changes.
    Everything is serialised once; the browser handles hover entirely in JS.
    """
    d   = load_pro(pro_path_str, _mtime=_mtime)
    raw = d["raw"]
    times      = d["times"]
    depth_grid = d["depth_grid"]
    MK_raw     = d["MK_raw"]
    MK_idx     = d["MK_idx"]
    soil_depth = d["soil_depth_m"]

    n_times = len(times)
    # Downsample to ≈daily (≤400 columns) so heatmap JSON stays small
    step = max(1, n_times // 365)
    idx  = list(range(0, n_times, step))

    t_labels   = [str(times[i])[:16] for i in idx]
    mk_idx_sub = MK_idx[idx]         # (n_sub, n_depth)
    mk_raw_sub = MK_raw[idx]
    soil_sub   = [round(float(v), 3) for v in soil_depth[idx]]
    depth_list = [round(float(v), 4) for v in depth_grid]

    def _mat(arr):
        """(n_depth, n_sub) nested list with NaN→null."""
        out = []
        for di in range(arr.shape[1]):
            row = []
            for ti in range(arr.shape[0]):
                v = arr[ti, di]
                row.append(None if math.isnan(v) else round(float(v), 3))
            out.append(row)
        return out

    # Precompute profiles only for the downsampled timesteps used by the heatmap.
    # The hover handler indexes profiles by the heatmap column index directly,
    # so we only need len(idx) profiles instead of all n_times hourly profiles.
    profiles = [_profile_for_timestep(raw, ti) for ti in idx]

    payload = {
        "heatmap": {
            "x":          t_labels,
            "y":          depth_list,
            "z":          _mat(mk_idx_sub),
            "customdata": _mat(mk_raw_sub),
            "soil":       soil_sub,
            "colorscale": _MK_COLORSCALE,
            "zmin":       -0.5,
            "zmax":       float(_N) - 0.5,
            "maxDepth":   float(depth_grid.max()),
        },
        # One profile per heatmap column (indexed by subIdx, same as heatmap x)
        "profiles": profiles,
    }
    return json.dumps(payload, allow_nan=False)


@st.cache_data(show_spinner="Preparing heatmap data…")
def _build_timeseries_payload(pro_path_str: str, sid: str, _mtime: float = 0.0) -> str:
    """Downsampled T, LWC, refreezing + optional observed-T payload for JS charts. v1"""
    d          = load_pro(pro_path_str, _mtime=_mtime)
    times      = d["times"]
    depth_grid = d["depth_grid"]
    T_grid     = d["T_grid"]
    LWC_grid   = d["LWC_grid"]
    cumul_rf   = d["cumul_refreezing"]
    soil_depth = d["soil_depth_m"]

    n_times = len(times)
    step    = max(1, n_times // 365)
    idx     = list(range(0, n_times, step))

    t_labels   = [str(times[i])[:16] for i in idx]
    depth_list = [round(float(v), 4) for v in depth_grid]
    soil_sub   = [round(float(v), 3) for v in soil_depth[idx]]

    def _mat(arr_sub):
        out = []
        for di in range(arr_sub.shape[1]):
            row = []
            for ti in range(arr_sub.shape[0]):
                v = arr_sub[ti, di]
                row.append(None if math.isnan(v) else round(float(v), 3))
            out.append(row)
        return out

    T_sub   = T_grid[idx]
    LWC_sub = LWC_grid[idx]

    obs = load_observed_temp(sid)
    if obs is not None:
        zmin_T = max(float(np.nanmin(obs[2])) - 1.0, -30.0)
    else:
        zmin_T = max(float(np.nanmin(T_grid)) - 1.0, -30.0)

    # Refreezing at full resolution (1D — small enough)
    rf_t = [str(times[i])[:16] for i in range(n_times)]
    rf_y = [None if (v != v) else round(float(v), 3) for v in cumul_rf]

    payload: dict = {
        "T": {
            "x": t_labels, "y": depth_list, "z": _mat(T_sub),
            "soil": soil_sub, "maxDepth": float(depth_grid.max()),
            "zmin": round(float(zmin_T), 2),
        },
        "LWC": {
            "x": t_labels, "y": depth_list, "z": _mat(LWC_sub),
            "soil": soil_sub, "maxDepth": float(depth_grid.max()),
        },
        "rf":     {"x": rf_t, "y": rf_y},
        "hasObs": obs is not None,
    }

    if obs is not None:
        obs_times, obs_depths, obs_T_data = obs
        n_obs     = len(obs_times)
        obs_step  = max(1, n_obs // 365)
        obs_idx   = list(range(0, n_obs, obs_step))
        obs_t     = [str(obs_times[i])[:16] for i in obs_idx]
        obs_dep   = [round(float(v), 4) for v in obs_depths]
        obs_T_sub = obs_T_data[obs_idx]
        obs_z: list = []
        for di in range(obs_T_sub.shape[1]):
            row = []
            for ti in range(obs_T_sub.shape[0]):
                v = obs_T_sub[ti, di]
                row.append(None if (v != v) else round(float(v), 3))
            obs_z.append(row)
        payload["obsT"] = {
            "x": obs_t, "y": obs_dep, "z": obs_z,
            "zmin": round(float(zmin_T), 2),
        }

    return json.dumps(payload, allow_nan=False)


def show_interactive_charts(sid: str) -> None:
    pro_path = find_pro_file(sid)
    if pro_path is None:
        st.info("No .pro output file found — run the model first.")
        return

    mtime = pro_path.stat().st_mtime

    # ── Plot selector ──────────────────────────────────────────────────── #
    _CHECKBOXES = [
        ("Grain type",          ["Grain type"]),
        ("Temperature",         ["Modelled temperature", "Observed temperature"]),
        ("LWC & Refreezing",    ["Liquid water content", "Cumulative refreezing"]),
        ("Residual saturation", ["Residual saturation"]),
        ("Density",             ["Density"]),
    ]
    show = set()
    cols = st.columns(len(_CHECKBOXES))
    for col, (label, members) in zip(cols, _CHECKBOXES):
        if col.checkbox(label, value=False, key=f"plot_sel_{sid}_{label}"):
            show.update(members)
    if not show:
        return

    title_safe = sid.replace("'", "\\'")

    # ── Grain-type legend + HTML component ─────────────────────────────── #
    if "Grain type" in show:
        mk_payload = _build_hover_payload(str(pro_path), _mtime=mtime)
        swatches = "".join(
            f'<div style="display:flex;align-items:center;gap:4px;background:#f5f5f5;'
            f'border-radius:4px;padding:3px 8px;white-space:nowrap">'
            f'<div style="width:14px;height:14px;background:{color};border-radius:3px;'
            f'flex-shrink:0;border:1px solid rgba(0,0,0,0.15)"></div>'
            f'<span style="font-size:11px;color:#333">{name}</span></div>'
            for code in sorted(MK_CATALOG.keys())
            for color, name in [MK_CATALOG[code]]
        )
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:6px;padding:6px 0">'
            f'{swatches}</div>',
            unsafe_allow_html=True,
        )
        mk_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:sans-serif;overflow:hidden}}
#wrap{{display:flex;width:100%;height:550px;gap:4px}}
#mk-div{{flex:3;min-width:0}}#col-div{{flex:1;min-width:0}}</style>
</head><body>
<div id="wrap"><div id="mk-div"></div><div id="col-div"></div></div>
<script>
var PL={mk_payload};
var hd=PL.heatmap,profiles=PL.profiles;
Plotly.newPlot('mk-div',[
  {{type:'heatmap',x:hd.x,y:hd.y,z:hd.z,customdata:hd.customdata,
    colorscale:hd.colorscale,zmin:hd.zmin,zmax:hd.zmax,showscale:false,
    hovertemplate:'%{{x}}<br>%{{y:.2f}} m  code %{{customdata:.0f}}<extra></extra>'}},
  {{type:'scatter',x:hd.x,y:hd.soil,mode:'lines',
    line:{{color:'black',width:1.5}},hoverinfo:'skip',showlegend:false}},
  {{type:'scatter',x:[hd.x[0],hd.x[0]],y:[0,hd.maxDepth],mode:'lines',
    line:{{color:'white',width:2,dash:'dot'}},hoverinfo:'skip',showlegend:false}}
],{{title:{{text:'{title_safe} — Swiss grain type (hover for profile)',font:{{size:13}}}},
  yaxis:{{title:'Depth (m)',autorange:'reversed'}},xaxis:{{title:'Date'}},
  height:550,margin:{{l:60,r:10,t:40,b:55}},plot_bgcolor:'white'
}},{{responsive:true,displayModeBar:false}});
function colTraces(p){{
  var cd=p.names.map(function(n,i){{return [n,p.depth_top[i],p.depth_bot[i],p.temp[i],p.hard[i]];}});
  return [
    {{type:'bar',orientation:'h',x:p.hard,y:p.depth_mid,width:p.thickness,
      marker:{{color:p.colors,line:{{color:'rgba(0,0,0,0.2)',width:0.5}}}},customdata:cd,
      hovertemplate:'<b>%{{customdata[0]}}</b><br>%{{customdata[1]:.2f}}–%{{customdata[2]:.2f}} m<br>T: %{{customdata[3]:.2f}}°C  Hard: %{{customdata[4]:.1f}}<extra></extra>',
      showlegend:false,xaxis:'x'}},
    {{type:'scatter',x:p.temp,y:p.depth_mid,mode:'lines',
      line:{{color:'crimson',width:2}},xaxis:'x2',yaxis:'y',name:'Temp',
      hovertemplate:'T: %{{x:.2f}}°C @ %{{y:.2f}} m<extra></extra>'}}
  ];
}}
function colLayout(p,label){{return {{
  title:{{text:label||'',font:{{size:11}}}},
  xaxis:{{title:'Hardness',range:[0,6.5],side:'bottom',tickvals:[1,2,3,4,5,6],
          ticktext:['fist','4f','1f','pen','knife','ice']}},
  xaxis2:{{title:'T (°C)',side:'top',overlaying:'x',range:[p.t_min,0],showgrid:false,tickformat:'.0f'}},
  yaxis:{{title:'Depth (m)',autorange:'reversed'}},
  legend:{{x:0,y:-0.09,orientation:'h',font:{{size:10}}}},
  height:550,margin:{{l:55,r:10,t:50,b:65}},bargap:0,plot_bgcolor:'white'
}};}}
var p0=null,i0=0;
for(var i=0;i<profiles.length;i++){{if(profiles[i]){{p0=profiles[i];i0=i;break;}}}}
if(p0) Plotly.newPlot('col-div',colTraces(p0),colLayout(p0,hd.x[i0]),{{responsive:true,displayModeBar:false}});
var curSub=-1;
document.getElementById('mk-div').on('plotly_hover',function(data){{
  if(!data||!data.points||!data.points.length) return;
  var subIdx=Array.isArray(data.points[0].pointIndex)?data.points[0].pointIndex[1]:0;
  if(subIdx===curSub) return; curSub=subIdx;
  var p=profiles[subIdx]; if(!p) return;
  Plotly.restyle('mk-div',{{x:[[hd.x[subIdx],hd.x[subIdx]]]}},2);
  Plotly.react('col-div',colTraces(p),colLayout(p,hd.x[subIdx]),{{responsive:true,displayModeBar:false}});
}});
</script></body></html>"""
        components.html(mk_html, height=570, scrolling=False)

    # ── Plotly subplot: T / obs_T / LWC / refreezing (shared x-axis) ──── #
    _PLOTLY_PLOTS = ["Modelled temperature", "Observed temperature",
                     "Liquid water content", "Cumulative refreezing",
                     "Residual saturation"]
    active_plotly = [p for p in _PLOTLY_PLOTS if p in show]

    if active_plotly:
        # Load pro grids (cached — only once regardless of which plots are shown)
        d          = load_pro(str(pro_path), _mtime=mtime)
        times      = d["times"]
        depth_grid = d["depth_grid"]
        T_grid     = d["T_grid"]
        LWC_grid   = d["LWC_grid"]
        cumul_rf   = d["cumul_refreezing"]
        soil_depth = d["soil_depth_m"]
        t_dt       = times.to_pydatetime()

        # Log scale bounds for residual saturation coloraxis
        _SAT_FLOOR   = 0.1          # % mass — floor for log transform
        _SAT_MAX     = 6.0          # % mass — colorbar maximum
        _SAT_LOG_MIN = np.log10(_SAT_FLOOR)
        _SAT_LOG_MAX = np.log10(_SAT_MAX)

        # Stride: at least hourly, but capped so heatmap never exceeds ~1000 columns
        MAX_PLOT_TIMES = 1000
        n_times = len(times)
        if n_times > 1:
            dt_sec   = (t_dt[1] - t_dt[0]).total_seconds()
            raw_step = max(1, round(3600.0 / dt_sec))
        else:
            raw_step = 1
        step = max(raw_step, n_times // MAX_PLOT_TIMES)
        idx  = list(range(0, n_times, step))
        t_sub    = [t_dt[i] for i in idx]
        soil_sub = [float(soil_depth[i]) for i in idx]

        # Observed T
        want_obs = "Observed temperature" in show
        obs = load_observed_temp(sid) if want_obs else None
        has_obs = obs is not None

        if has_obs:
            zmin_T = max(float(np.nanmin(obs[2])) - 1.0, -30.0)
        else:
            zmin_T = max(float(np.nanmin(T_grid)) - 1.0, -30.0)

        # Build ordered row list (only selected plots, obs only if data exists)
        plotly_rows = []
        if "Modelled temperature" in show:
            plotly_rows.append("mod_T")
        if has_obs:
            plotly_rows.append("obs_T")
        if "Liquid water content" in show:
            plotly_rows.append("LWC")
        if "Cumulative refreezing" in show:
            plotly_rows.append("RF")
        if "Residual saturation" in show:
            plotly_rows.append("RS")

        row_map = {name: i + 1 for i, name in enumerate(plotly_rows)}
        n_rows  = len(plotly_rows)
        rh      = [1.0 / n_rows] * n_rows
        # Give RF row a bit less height than heatmap rows
        if n_rows > 1 and "RF" in row_map:
            rf_frac = 0.55 / n_rows
            hm_frac = (1.0 - rf_frac) / (n_rows - 1)
            rh = [rf_frac if plotly_rows[i] == "RF" else hm_frac
                  for i in range(n_rows)]
        # RS (saturation heatmap) is shallower (5 m) so give it slightly less height
        elif n_rows > 1 and "RS" in row_map and "RF" not in row_map:
            rs_frac = 0.75 / n_rows
            hm_frac = (1.0 - rs_frac) / (n_rows - 1)
            rh = [rs_frac if plotly_rows[i] == "RS" else hm_frac
                  for i in range(n_rows)]
        fig_h = max(300, 260 * n_rows)

        titles = []
        for p in plotly_rows:
            if p == "mod_T":  titles.append(f"{sid} — Modelled temperature")
            elif p == "obs_T": titles.append(f"{sid} — Observed temperature")
            elif p == "LWC":  titles.append(f"{sid} — Liquid water content")
            elif p == "RF":   titles.append(f"{sid} — Cumulative refreezing")
            elif p == "RS":   titles.append(f"{sid} — Residual saturation (Coléou & Lesaffre 1998)")

        _LWC_CS = [
            [0.00, "#ffffff"], [0.01, "#f7fbff"], [0.20, "#c6dbef"],
            [0.50, "#6baed6"], [0.80, "#2171b5"], [0.99, "#08306b"], [1.00, "red"],
        ]

        fig = make_subplots(
            rows=n_rows, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=rh,
            subplot_titles=titles,
        )

        max_depth = float(depth_grid.max())

        def _zero_contour(x, y, z):
            return go.Contour(
                x=x, y=y, z=np.asarray(z, dtype=float),
                showscale=False,
                contours=dict(start=-0.2, end=-0.1, size=1,
                              coloring="lines", showlabels=False),
                colorscale=[[0, "white"], [1, "white"]],
                line=dict(width=2.5),
                hoverinfo="skip", showlegend=False, name="-0.2 °C",
            )

        if "mod_T" in row_map:
            T_sub = T_grid[idx]
            r = row_map["mod_T"]
            fig.add_trace(go.Heatmap(
                x=t_sub, y=depth_grid, z=T_sub.T,
                coloraxis="coloraxis",
                hovertemplate="Date: %{x}<br>Depth: %{y:.2f} m<br>T: %{z:.2f} °C<extra></extra>",
            ), row=r, col=1)
            fig.add_trace(_zero_contour(t_sub, depth_grid, T_sub.T), row=r, col=1)
            fig.add_trace(go.Scatter(
                x=t_sub, y=soil_sub, mode="lines",
                line=dict(color="black", width=1), hoverinfo="skip", showlegend=False,
            ), row=r, col=1)
            fig.update_yaxes(title_text="Depth (m)", range=[max_depth, 0], row=r)

        if "obs_T" in row_map:
            obs_times, obs_depths, obs_T_arr = obs
            obs_tdt  = obs_times.to_pydatetime()
            n_obs    = len(obs_tdt)
            obs_step = max(1, n_obs // MAX_PLOT_TIMES)
            obs_idx  = list(range(0, n_obs, obs_step))
            obs_t_sub   = [obs_tdt[i] for i in obs_idx]
            obs_T_arr   = obs_T_arr[obs_idx]
            r = row_map["obs_T"]
            fig.add_trace(go.Heatmap(
                x=obs_t_sub, y=obs_depths, z=obs_T_arr.T,
                coloraxis="coloraxis",
                hovertemplate="Date: %{x}<br>Depth: %{y:.2f} m<br>T: %{z:.2f} °C<extra></extra>",
            ), row=r, col=1)
            fig.add_trace(_zero_contour(obs_t_sub, obs_depths, obs_T_arr.T),
                          row=r, col=1)
            fig.update_yaxes(title_text="Depth (m)", range=[max_depth, 0], row=r)

        if "LWC" in row_map:
            LWC_sub = LWC_grid[idx]
            r = row_map["LWC"]
            fig.add_trace(go.Heatmap(
                x=t_sub, y=depth_grid, z=LWC_sub.T,
                coloraxis="coloraxis2",
                hovertemplate="Date: %{x}<br>Depth: %{y:.2f} m<br>LWC: %{z:.2f}%<extra></extra>",
            ), row=r, col=1)
            fig.add_trace(go.Scatter(
                x=t_sub, y=soil_sub, mode="lines",
                line=dict(color="black", width=1), hoverinfo="skip", showlegend=False,
            ), row=r, col=1)
            fig.update_yaxes(title_text="Depth (m)", autorange="reversed", row=r)

        if "RF" in row_map:
            r = row_map["RF"]
            fig.add_trace(go.Scatter(
                x=t_dt, y=cumul_rf, mode="lines",
                line=dict(color="#2980b9", width=2),
                fill="tozeroy", fillcolor="rgba(41,128,185,0.15)",
                hovertemplate="Date: %{x}<br>Cumul. refreezing: %{y:.1f} mm w.e.<extra></extra>",
            ), row=r, col=1)
            fig.update_yaxes(title_text="mm w.e.", rangemode="tozero", row=r)

        if "RS" in row_map:
            sat_d    = load_sat_grid(str(pro_path), _mtime=mtime)
            depth_5m = sat_d["depth_5m"]
            Sat_raw  = sat_d["Sat_grid"][idx]
            # Log10-transform for display; floor at _SAT_FLOOR to avoid log(0)
            Sat_log  = np.log10(np.clip(Sat_raw, _SAT_FLOOR, None))
            # Isotherm from main T_grid (already loaded), clipped to 5 m
            d5m      = depth_grid <= 5.0
            T_sub_5m = T_grid[idx][:, d5m]
            r = row_map["RS"]
            fig.add_trace(go.Heatmap(
                x=t_sub, y=depth_5m, z=Sat_log.T,
                customdata=Sat_raw.T,
                coloraxis="coloraxis3",
                hovertemplate="Date: %{x}<br>Depth: %{y:.2f} m<br>W_irr: %{customdata:.2f} % mass<extra></extra>",
            ), row=r, col=1)
            fig.add_trace(_zero_contour(t_sub, depth_5m, T_sub_5m.T), row=r, col=1)
            fig.update_yaxes(title_text="Depth (m)", range=[5.0, 0], row=r)

        fig.update_xaxes(title_text="Date", row=n_rows)

        # Colorbar y positions: centre on each heatmap row group
        t_rows   = [row_map[k] for k in ("mod_T", "obs_T") if k in row_map]
        lwc_rows = [row_map["LWC"]] if "LWC" in row_map else []
        rs_rows  = [row_map["RS"]]  if "RS"  in row_map else []
        cb_T_y   = 1.0 - (np.mean(t_rows)   - 0.5) / n_rows if t_rows   else 0.75
        cb_LWC_y = 1.0 - (np.mean(lwc_rows) - 0.5) / n_rows if lwc_rows else 0.25
        cb_SAT_y = 1.0 - (np.mean(rs_rows)  - 0.5) / n_rows if rs_rows  else 0.10

        fig.update_layout(
            height=fig_h,
            margin=dict(l=70, r=90, t=35, b=55),
            showlegend=False,
            coloraxis=dict(
                colorscale="Turbo",
                cmin=zmin_T, cmax=0,
                colorbar=dict(title="°C", thickness=12, x=1.02,
                              len=0.45, y=cb_T_y),
            ),
            coloraxis2=dict(
                colorscale=_LWC_CS,
                cmin=0, cmax=10,
                colorbar=dict(
                    title="LWC %", thickness=12, x=1.02,
                    len=0.30, y=cb_LWC_y,
                    tickvals=[0, 2, 4, 6, 8, 10],
                    ticktext=["0", "2", "4", "6", "8", "≥10"],
                ),
            ),
            coloraxis3=dict(
                colorscale="Blues",
                cmin=_SAT_LOG_MIN, cmax=_SAT_LOG_MAX,
                colorbar=dict(
                    title="W_irr %", thickness=12, x=1.02,
                    len=0.30, y=cb_SAT_y,
                    tickvals=[np.log10(v) for v in [0.1, 0.3, 1.0, 3.0, 6.0]],
                    ticktext=["0.1", "0.3", "1", "3", "6"],
                ),
            ),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"ts_chart_{sid}")

    # ── Density heatmap + profile (self-contained HTML iframe) ────────── #
    if "Density" not in show:
        return
    den_payload = _build_density_payload(str(pro_path), _mtime=mtime)
    den_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:sans-serif;overflow:hidden}}
#wrap{{display:flex;width:100%;height:450px;gap:4px}}
#den-hm{{flex:3;min-width:0}}#den-prof{{flex:1;min-width:0}}</style>
</head><body>
<div id="wrap"><div id="den-hm"></div><div id="den-prof"></div></div>
<script>
var PL={den_payload};
var hd=PL.heatmap,profiles=PL.profiles,firstP=PL.first,lastP=PL.last;
Plotly.newPlot('den-hm',[
  {{type:'heatmap',x:hd.x,y:hd.y,z:hd.z,
    colorscale:'Greys',reversescale:true,zmin:0,zmax:900,
    colorbar:{{title:'kg/m³',thickness:12}},
    hovertemplate:'%{{x}}<br>%{{y:.2f}} m  %{{z:.0f}} kg/m³<extra></extra>'}},
  {{type:'scatter',x:hd.x,y:hd.soil,mode:'lines',
    line:{{color:'black',width:1.5}},hoverinfo:'skip',showlegend:false}},
  {{type:'scatter',x:[hd.x[0],hd.x[0]],y:[0,hd.maxDepth],mode:'lines',
    line:{{color:'#e74c3c',width:2,dash:'dot'}},hoverinfo:'skip',showlegend:false}}
],{{title:{{text:'{title_safe} — Firn density (hover for profile)',font:{{size:13}}}},
  yaxis:{{title:'Depth (m)',autorange:'reversed'}},xaxis:{{title:'Date'}},
  height:450,margin:{{l:60,r:10,t:40,b:55}},plot_bgcolor:'white'
}},{{responsive:true,displayModeBar:false}});
function profTraces(p,label){{
  var tr=[];
  if(firstP) tr.push({{type:'scatter',x:firstP.x,y:firstP.y,mode:'lines',name:PL.t_first,
    line:{{color:'#aaaaaa',width:1.5,dash:'dash'}},
    hovertemplate:'%{{x:.0f}} kg/m³ @ %{{y:.2f}} m<extra>first</extra>'}});
  if(lastP) tr.push({{type:'scatter',x:lastP.x,y:lastP.y,mode:'lines',name:PL.t_last,
    line:{{color:'#e67e22',width:1.5,dash:'dash'}},
    hovertemplate:'%{{x:.0f}} kg/m³ @ %{{y:.2f}} m<extra>last</extra>'}});
  if(p) tr.push({{type:'scatter',x:p.x,y:p.y,mode:'lines',name:label||'',
    line:{{color:'#2c3e50',width:2}},
    hovertemplate:'%{{x:.0f}} kg/m³ @ %{{y:.2f}} m<extra>hover</extra>'}});
  return tr;
}}
var profLayout={{xaxis:{{title:'Density (kg/m³)',range:[200,920]}},
  yaxis:{{title:'Depth (m)',autorange:'reversed'}},
  legend:{{x:0,y:-0.12,orientation:'h',font:{{size:9}}}},
  height:450,margin:{{l:55,r:10,t:40,b:75}},plot_bgcolor:'white'}};
var p0=null,i0=0;
for(var i=0;i<profiles.length;i++){{if(profiles[i]){{p0=profiles[i];i0=i;break;}}}}
if(p0) Plotly.newPlot('den-prof',profTraces(p0,hd.x[i0]),profLayout,{{responsive:true,displayModeBar:false}});
var curSub=-1;
document.getElementById('den-hm').on('plotly_hover',function(data){{
  if(!data||!data.points||!data.points.length) return;
  var subIdx=Array.isArray(data.points[0].pointIndex)?data.points[0].pointIndex[1]:0;
  if(subIdx===curSub) return; curSub=subIdx;
  var p=profiles[subIdx]; if(!p) return;
  Plotly.restyle('den-hm',{{x:[[hd.x[subIdx],hd.x[subIdx]]]}},2);
  Plotly.react('den-prof',profTraces(p,hd.x[subIdx]),profLayout);
}});
</script></body></html>"""
    components.html(den_html, height=470, scrolling=False)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_run, tab_settings, tab_results, tab_ovm = st.tabs(["▶ Run", "⚙ Settings", "📊 Results", "🌡 Obs vs Model"])

# ============================================================
# TAB 1 — Run
# ============================================================
with tab_run:
    col_params, col_status = st.columns([1, 2])

    with col_params:
        st.subheader("Run parameters")

        # --- Core selection ---
        available_cores = discover_cores()

        def _label(y, s, d, d0, d1):
            return f"{y} · {s} · {d} m    {d0} → {d1}"

        def _core_status(y, s, d):
            """Return 'running', 'incomplete', or 'fresh'."""
            sid_ = site_id(y, s, d)
            if is_running(sid_):
                return "running"
            if find_pro_file(sid_) is not None:
                t_cur = get_pro_current_time(sid_)
                _, t_end_ = get_expected_date_range(sid_)
                if t_cur is not None and t_end_ is not None and t_cur < t_end_:
                    return "incomplete"
            return "fresh"

        cores_running    = [(y, s, d, d0, d1) for y, s, d, d0, d1 in available_cores if _core_status(y, s, d) == "running"]
        cores_incomplete = [(y, s, d, d0, d1) for y, s, d, d0, d1 in available_cores if _core_status(y, s, d) == "incomplete"]
        cores_all        = available_cores  # all cores for a fresh / re-run

        use_custom = st.checkbox("Enter site manually", value=not bool(available_cores))

        if not use_custom and available_cores:
            _group_options = [
                f"All cores ({len(cores_all)})",
                f"Incomplete / crashed ({len(cores_incomplete)})",
                f"In progress ({len(cores_running)})",
            ]
            _group = st.radio("", _group_options, horizontal=True,
                              index=0, key="core_group_radio",
                              label_visibility="collapsed")

            if "In progress" in _group:
                _pool = cores_running
            elif "Incomplete" in _group:
                _pool = cores_incomplete
            else:
                _pool = cores_all

            if not _pool:
                st.info("No cores in this category.")
                year, site, depth = cores_all[0][:3] if cores_all else (2022, "T3", 25)
            else:
                _pool_labels = [_label(*c) for c in _pool]
                chosen_label = st.selectbox("Core", _pool_labels,
                                            label_visibility="collapsed")
                idx = _pool_labels.index(chosen_label)
                year, site, depth = _pool[idx][:3]

        else:
            if not available_cores:
                st.info("No cores found in AllCoreDataCommonFormat/ — enter manually.")
            year  = st.number_input("Year drilled", value=2022, step=1, format="%d")
            site  = st.text_input("Site name", value="T3")
            depth = st.number_input("String depth (m)", value=25, step=1, format="%d")

        run_until = st.text_input(
            "Run until (optional)",
            value="",
            placeholder="YYYY-MM-DD HH:MM  — leave blank for full record",
        )

        sid     = site_id(int(year), site, int(depth))
        running = is_running(sid)

        col_launch, col_kill = st.columns(2)
        with col_launch:
            launch = st.button("▶ Launch", disabled=running, use_container_width=True,
                               type="primary")
        with col_kill:
            kill = st.button("⏹ Kill", disabled=not running, use_container_width=True)

        if launch:
            pid = launch_run(int(year), site, int(depth), run_until)
            st.success(f"Started PID {pid}")
            time.sleep(1)
            st.rerun()

        if kill:
            kill_run(sid)
            st.warning("Process terminated.")
            time.sleep(1)
            st.rerun()

    with col_status:
        st.subheader(f"Status — {sid}")

        crashed = False
        if running:
            st.status("Running…", state="running")
        else:
            pid = read_pid(sid)
            if pid is None and log_path(sid).exists():
                last = read_log_tail(sid, 5)
                if "Done." in last:
                    st.status("Finished", state="complete")
                else:
                    st.status("Crashed / stopped", state="error")
                    crashed = True
            else:
                st.status("Not started", state="error")

        # Progress bar
        t_start, t_end = get_expected_date_range(sid)
        t_cur = get_pro_current_time(sid)
        if t_start and t_end and t_cur:
            total_s   = (t_end   - t_start).total_seconds()
            elapsed_s = (t_cur   - t_start).total_seconds()
            frac = float(max(0.0, min(1.0, elapsed_s / total_s))) if total_s > 0 else 0.0
            label = (
                f"{t_cur.strftime('%Y-%m-%d')} / {t_end.strftime('%Y-%m-%d')}"
                f"  ({frac * 100:.0f}%)"
            )
            st.progress(frac, text=label)
        elif running:
            st.progress(0.0, text="Starting…")

        # Only show log when the run has crashed
        if crashed:
            st.markdown("**Run log (crash output):**")
            st.code(_strip_path_lines(read_log_tail(sid, 100)), language="text")

        if running:
            time.sleep(3)
            st.rerun()

# ============================================================
# TAB 2 — Settings
# ============================================================
with tab_settings:
    st.subheader(f"Settings — `{SETTINGS_FILE}`")

    if not SETTINGS_FILE.exists():
        st.error(f"Settings file not found: {SETTINGS_FILE}")
    else:
        doc = load_settings()

        with st.form("settings_form"):

            with st.expander("Paths", expanded=False):
                c1, c2 = st.columns(2)
                doc["paths"]["snowpack_exe"] = c1.text_input(
                    "SNOWPACK executable", value=str(doc["paths"]["snowpack_exe"]))
                doc["paths"]["data_root"] = c2.text_input(
                    "Data root (blank = auto)", value=str(doc["paths"].get("data_root", "")))

            with st.expander("Run", expanded=True):
                c1, c2, c3 = st.columns(3)
                doc["run"]["run_until"] = c1.text_input(
                    "Stop at (blank = full record)",
                    value=str(doc["run"].get("run_until", "")))
                _wt_options = ["adaptive", "BUCKET", "RICHARDSEQUATION"]
                _wt_current = str(doc["run"].get("water_transport", "adaptive"))
                doc["run"]["water_transport"] = c2.selectbox(
                    "Water transport",
                    options=_wt_options,
                    index=_wt_options.index(_wt_current) if _wt_current in _wt_options else 0)
                doc["run"]["assimilation_interval_h"] = c3.number_input(
                    "Assimilation interval (hours)",
                    value=int(doc["run"].get("assimilation_interval_h", 1)),
                    min_value=1, max_value=24, step=1,
                    help="Hours between temperature assimilation steps. Higher = faster runs, less frequent nudging.")
                c1, c2, c3, c4 = st.columns(4)
                doc["run"]["keep_last_n_sno"] = c1.number_input(
                    "Keep last N .sno", value=int(doc["run"]["keep_last_n_sno"]),
                    min_value=1, step=1)
                doc["run"]["use_ramdisk"] = c2.checkbox(
                    "RAM disk (/dev/shm)",
                    value=bool(doc["run"].get("use_ramdisk", False)),
                    help="Run .sno and config files from RAM for faster I/O. Output .pro stays on disk.")
                doc["run"]["keep_hourly_archives"] = c3.checkbox(
                    "Keep hourly .sno archives",
                    value=bool(doc["run"]["keep_hourly_archives"]))

            with st.expander("Model & assimilation", expanded=False):
                c1, c2, c3 = st.columns(3)
                doc["model"]["calculation_step_length_min"] = c1.number_input(
                    "Step length (min)", value=float(doc["model"]["calculation_step_length_min"]),
                    step=1.0)
                doc["model"]["alpha"] = c2.number_input(
                    "Assimilation alpha", value=float(doc["model"]["alpha"]),
                    step=0.01, format="%.2f")
                doc["model"]["default_elevation_m"] = c3.number_input(
                    "Default elevation (m)", value=float(doc["model"]["default_elevation_m"]),
                    step=10.0)
                c1, c2, c3, c4 = st.columns(4)
                doc["assimilation"]["temp_min_c"] = c1.number_input(
                    "T min (°C)", value=float(doc["assimilation"]["temp_min_c"]), step=1.0)
                doc["assimilation"]["temp_max_c"] = c2.number_input(
                    "T max (°C)", value=float(doc["assimilation"]["temp_max_c"]), step=1.0)
                doc["assimilation"]["max_adjust_per_hour_c"] = c3.number_input(
                    "Max adjust/hr (°C)", value=float(doc["assimilation"]["max_adjust_per_hour_c"]),
                    step=0.01, format="%.2f")
                doc["assimilation"]["min_obs_for_adjust"] = c4.number_input(
                    "Min obs for adjust", value=int(doc["assimilation"]["min_obs_for_adjust"]),
                    step=1)

            with st.expander("Basal boundary", expanded=False):
                c1, c2, c3 = st.columns(3)
                doc["basal"]["tsg_mode"] = c1.selectbox(
                    "TSG mode",
                    options=["profile_gradient", "zero", "soil_temp"],
                    index=["profile_gradient", "zero", "soil_temp"].index(
                        str(doc["basal"]["tsg_mode"])))
                doc["basal"]["tsg_lookback_m"] = c2.number_input(
                    "TSG lookback (m)", value=float(doc["basal"]["tsg_lookback_m"]), step=0.5)
                doc["basal"]["basal_temp_min_c"] = c3.number_input(
                    "Basal T min (°C)", value=float(doc["basal"]["basal_temp_min_c"]), step=1.0)

            with st.expander("ERA5 forcing adjustments", expanded=False):
                st.caption("physical = ERA5 × multiplier + offset  (applied via SMET `units_multiplier` / `units_offset`)")
                if "forcing_adjustments" not in doc:
                    doc.add("forcing_adjustments", tomlkit.table())
                fa = doc["forcing_adjustments"]
                _fa_vars = [
                    ("ta",   "Air temp (°C)",      "TA"),
                    ("rh",   "Rel. humidity (–)",  "RH"),
                    ("vw",   "Wind speed (m/s)",   "VW"),
                    ("iswr", "Shortwave (W m⁻²)",  "ISWR"),
                    ("ilwr", "Longwave (W m⁻²)",   "ILWR"),
                    ("psum", "Precipitation (mm)", "PSUM"),
                ]
                _hdr = st.columns([2, 1, 1])
                _hdr[0].markdown("Variable")
                _hdr[1].markdown("Multiplier")
                _hdr[2].markdown("Offset")
                for _key, _label, _field in _fa_vars:
                    _c0, _c1, _c2 = st.columns([2, 1, 1])
                    _c0.markdown(_label)
                    fa[f"{_key}_multiplier"] = _c1.number_input(
                        f"{_field} ×", label_visibility="collapsed",
                        value=float(fa.get(f"{_key}_multiplier", 1.0)),
                        step=0.05, format="%.3f", key=f"fa_{_key}_mult")
                    fa[f"{_key}_offset"] = _c2.number_input(
                        f"{_field} +", label_visibility="collapsed",
                        value=float(fa.get(f"{_key}_offset", 0.0)),
                        step=0.1, format="%.3f", key=f"fa_{_key}_off")

            saved = st.form_submit_button("💾 Save settings", type="primary")

        if saved:
            save_settings(doc)
            st.success("Settings saved.")

# ============================================================
# TAB 3 — Results
# ============================================================
with tab_results:
    result_sites = sites_with_results()

    if not result_sites:
        st.info("No results yet — run a simulation first.")
    else:
        results_sid = st.selectbox(
            "Site",
            options=result_sites,
            key="results_sid_select",
        )

        st.subheader(f"Interactive plots — {results_sid}")
        show_interactive_charts(results_sid)

        figs = output_figures(results_sid)
        if figs:
            st.markdown("---")
            st.subheader("Static figures")
            for fig_path in figs:
                st.image(str(fig_path), caption=fig_path.name, width="stretch")

# ============================================================
# TAB 4 — Obs vs Model (all sites)
# ============================================================
with tab_ovm:
    import subprocess, sys
    _OVM_SITES = [
        ("T2 (2007)",         "T2_2007_obs_vs_model.png",        "T2_2007"),
        ("T2 bucket (2007)", "T2_2007_bucket_obs_vs_model.png", "T2_2007_bucket"),
        ("T2− (2019)",   "T2minus_obs_vs_model.png",  "T2minus"),
        ("T3 (1794 m)",  "T3_obs_vs_model.png",       "T3"),
        ("T4 (1873 m)",  "T4_obs_vs_model.png",       "T4"),
        ("CP (1998 m)",  "CP_obs_vs_model.png",       "CP"),
        ("UP18 (2109 m)","UP18_obs_vs_model.png",     "UP18"),
    ]
    _ovm_labels = [s[0] for s in _OVM_SITES]
    _ovm_choice = st.radio("Site", _ovm_labels, horizontal=True)
    _ovm_site   = next(s for s in _OVM_SITES if s[0] == _ovm_choice)
    _ovm_fig    = APP_DIR / _ovm_site[1]
    _ovm_key    = _ovm_site[2]

    if _ovm_fig.exists():
        st.image(str(_ovm_fig),
                 caption=f"{_ovm_choice} — observed vs modelled firn temperature",
                 width="stretch")
    else:
        st.info("Figure not generated yet.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Regenerate this site", key="ovm_regen"):
            with st.spinner(f"Generating {_ovm_choice} …"):
                result = subprocess.run(
                    [sys.executable, str(APP_DIR / "plot_obs_vs_model.py"), _ovm_key],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("Done.")
                st.rerun()
            else:
                st.error(result.stderr)
    with col2:
        if st.button("Regenerate all sites", key="ovm_regen_all"):
            with st.spinner("Generating all sites …"):
                result = subprocess.run(
                    [sys.executable, str(APP_DIR / "plot_obs_vs_model.py")],
                    capture_output=True, text=True
                )
            if result.returncode == 0:
                st.success("All figures updated.")
                st.rerun()
            else:
                st.error(result.stderr)
