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
_LOG_FILTER = r"^\s*(layer \[|upper boundary \[|SAFE MODE|Estimated mass balance|[-]{10})"

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
_WANTED = {501, 503, 506, 509, 510, 512, 513}

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
    path = Path(pro_path_str)
    pro  = _parse_pro(path)
    times = pd.DatetimeIndex(pro["times"])
    soil_depth_m = np.array([h.max() / 100.0 for h in pro[501]])
    max_depth    = float(np.nanmax(soil_depth_m)) if len(soil_depth_m) else 30.0
    depth_grid   = np.arange(0, max_depth + 0.05, 0.05)

    T_grid  = _to_grid(times, pro[501], pro[503], depth_grid)
    # Step function: each depth cell gets the value of the element containing it,
    # preserving finite-element layer boundaries (no interpolation).
    MK_raw  = _to_grid_stepfn(times, pro[501], pro[513], depth_grid)

    # Map grain-type codes to index for colour scale
    MK_idx = np.full_like(MK_raw, np.nan)
    for code, idx in _CODE_IDX.items():
        MK_idx = np.where(np.abs(MK_raw - code) < 0.5, float(idx), MK_idx)

    # Mask below soil
    for ti, sd in enumerate(soil_depth_m):
        MK_idx[ti, depth_grid > sd] = np.nan
        MK_raw[ti, depth_grid > sd] = np.nan

    return {
        "times":        times,
        "depth_grid":   depth_grid,
        "soil_depth_m": soil_depth_m,
        "T_grid":       T_grid,
        "MK_raw":       MK_raw,
        "MK_idx":       MK_idx,
        "raw":          pro,          # raw per-timestep layer arrays
    }


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

    # Precompute column profiles for every original hourly timestep
    profiles = []
    for ti in range(n_times):
        p = _profile_for_timestep(raw, ti)
        profiles.append(p)  # None if empty

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
        # Mapping from downsampled index → original hourly index
        "idx":      idx,
        # All hourly profiles (None entries for empty timesteps)
        "profiles": profiles,
        "t_all":    [str(t)[:16] for t in times],
    }
    return json.dumps(payload, allow_nan=False)


def show_interactive_charts(sid: str) -> None:
    pro_path = find_pro_file(sid)
    if pro_path is None:
        st.info("No .pro output file found — run the model first.")
        return

    mtime   = pro_path.stat().st_mtime
    payload = _build_hover_payload(str(pro_path), _mtime=mtime)

    # ------------------------------------------------------------------ #
    # Self-contained HTML component — grain-type heatmap + column profile
    # Hover on heatmap → column updates entirely in the browser (no rerun)
    # ------------------------------------------------------------------ #
    title_safe = sid.replace("'", "\\'")
    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:transparent;font-family:sans-serif;overflow:hidden}}
#wrap{{display:flex;width:100%;height:650px;gap:4px}}
#mk-div{{flex:3;min-width:0}}
#col-div{{flex:1;min-width:0}}
</style>
</head>
<body>
<div id="wrap"><div id="mk-div"></div><div id="col-div"></div></div>
<script>
var PL = {payload};
var hd = PL.heatmap;
var profiles = PL.profiles;
var idx = PL.idx;          // downsampled → hourly index map
var t_all = PL.t_all;      // all hourly labels

/* ── Grain-type heatmap ── */
Plotly.newPlot('mk-div',[
  {{type:'heatmap',x:hd.x,y:hd.y,z:hd.z,
    customdata:hd.customdata,
    colorscale:hd.colorscale,zmin:hd.zmin,zmax:hd.zmax,
    showscale:false,
    hovertemplate:'%{{x}}<br>%{{y:.2f}} m  code %{{customdata:.0f}}<extra></extra>'}},
  {{type:'scatter',x:hd.x,y:hd.soil,mode:'lines',
    line:{{color:'black',width:1.5}},hoverinfo:'skip',showlegend:false}},
  {{type:'scatter',x:[hd.x[0],hd.x[0]],y:[0,hd.maxDepth],mode:'lines',
    line:{{color:'white',width:2,dash:'dot'}},hoverinfo:'skip',showlegend:false}}
],{{
  title:{{text:'{title_safe} — Swiss grain type (hover for profile)',font:{{size:13}}}},
  yaxis:{{title:'Depth (m)',autorange:'reversed'}},
  xaxis:{{title:'Date'}},
  height:650,margin:{{l:60,r:10,t:40,b:55}},
  plot_bgcolor:'white'
}},{{responsive:true,displayModeBar:false}});

/* ── Column profile helpers ── */
function colTraces(p){{
  var cd=p.names.map(function(n,i){{
    return [n,p.depth_top[i],p.depth_bot[i],p.temp[i],p.hard[i]];
  }});
  return [
    {{type:'bar',orientation:'h',
      x:p.hard,y:p.depth_mid,width:p.thickness,
      marker:{{color:p.colors,line:{{color:'rgba(0,0,0,0.2)',width:0.5}}}},
      customdata:cd,
      hovertemplate:'<b>%{{customdata[0]}}</b><br>%{{customdata[1]:.2f}}–%{{customdata[2]:.2f}} m<br>T: %{{customdata[3]:.2f}}°C  Hard: %{{customdata[4]:.1f}}<extra></extra>',
      showlegend:false,xaxis:'x'}},
    {{type:'scatter',x:p.temp,y:p.depth_mid,mode:'lines',
      line:{{color:'crimson',width:2}},xaxis:'x2',yaxis:'y',
      name:'Temp',
      hovertemplate:'T: %{{x:.2f}}°C @ %{{y:.2f}} m<extra></extra>'}}
  ];
}}
function colLayout(p,label){{
  return {{
    title:{{text:label||'',font:{{size:11}}}},
    xaxis:{{title:'Hardness',range:[0,6.5],side:'bottom',
            tickvals:[1,2,3,4,5,6],
            ticktext:['fist','4f','1f','pen','knife','ice']}},
    xaxis2:{{title:'T (°C)',side:'top',overlaying:'x',
             range:[p.t_min,0],showgrid:false,tickformat:'.0f'}},
    yaxis:{{title:'Depth (m)',autorange:'reversed'}},
    legend:{{x:0,y:-0.09,orientation:'h',font:{{size:10}}}},
    height:650,margin:{{l:55,r:10,t:50,b:65}},
    bargap:0,plot_bgcolor:'white'
  }};
}}

/* Initial render — first non-null profile */
var p0=null,i0=0;
for(var i=0;i<profiles.length;i++){{if(profiles[i]){{p0=profiles[i];i0=i;break;}}}}
if(p0) Plotly.newPlot('col-div',colTraces(p0),colLayout(p0,t_all[i0]),
                      {{responsive:true,displayModeBar:false}});

/* ── Hover handler ── */
var curHourly=-1;
document.getElementById('mk-div').on('plotly_hover',function(data){{
  if(!data||!data.points||!data.points.length) return;
  var pt=data.points[0];
  // pointIndex for heatmap: [depth_idx, time_idx]
  var subIdx=Array.isArray(pt.pointIndex)?pt.pointIndex[1]:0;
  // Map downsampled index → nearest hourly index
  var hourly=idx[subIdx]||0;
  if(hourly===curHourly) return;
  curHourly=hourly;
  var p=profiles[hourly];
  if(!p) return;
  // Move cursor line in heatmap (trace index 2)
  Plotly.restyle('mk-div',{{x:[[hd.x[subIdx],hd.x[subIdx]]]}},2);
  // Update column
  Plotly.react('col-div',colTraces(p),colLayout(p,t_all[hourly]));
}});
</script>
</body></html>"""

    components.html(html, height=670, scrolling=False)

    # ------------------------------------------------------------------ #
    # Swiss grain-type legend
    # ------------------------------------------------------------------ #
    swatches = "".join(
        f'<div style="display:flex;align-items:center;gap:4px;'
        f'background:#f5f5f5;border-radius:4px;padding:3px 8px;white-space:nowrap">'
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

    # ------------------------------------------------------------------ #
    # Modelled temperature heatmap (full width — click to pin cursor)
    # ------------------------------------------------------------------ #
    d          = load_pro(str(pro_path), _mtime=mtime)
    times      = d["times"]
    depth_grid = d["depth_grid"]
    T_grid     = d["T_grid"]
    t_dt       = times.to_pydatetime()

    key_ti = f"ti_{sid}"
    if key_ti not in st.session_state:
        st.session_state[key_ti] = 0
    ti = st.session_state[key_ti]

    # Compute shared colorscale limits from observed data (falls back to modelled)
    obs = load_observed_temp(sid)
    if obs is not None:
        zmin_shared = max(float(np.nanmin(obs[2])) - 1.0, -30.0)
    else:
        zmin_shared = max(float(np.nanmin(T_grid)) - 1.0, -30.0)

    fig_T = go.Figure(go.Heatmap(
        x=t_dt, y=depth_grid, z=T_grid.T,
        colorscale="RdYlBu_r", zmin=zmin_shared, zmax=0,
        colorbar=dict(title="°C", thickness=12),
        hovertemplate="Date: %{x}<br>Depth: %{y:.2f} m<br>T: %{z:.2f} °C<extra></extra>",
    ))
    fig_T.add_vline(x=t_dt[ti], line=dict(color="white", width=2, dash="dot"))
    fig_T.update_layout(
        title=f"{sid} — Modelled firn temperature (click to pin cursor)",
        yaxis=dict(title="Depth (m)", autorange="reversed"),
        xaxis_title="Date", height=320,
        margin=dict(l=55, r=10, t=35, b=55),
        clickmode="event+select",
    )
    T_event = st.plotly_chart(fig_T, width="stretch",
                              on_select="rerun", key=f"T_chart_{sid}")
    if T_event and T_event.selection and T_event.selection.points:
        cx = T_event.selection.points[0].get("x", "")
        try:
            cdt = (pd.Timestamp(int(cx), unit="ms") if isinstance(cx, (int, float))
                   else pd.Timestamp(str(cx)[:16]))
            nti = int(np.argmin(np.abs(times - cdt)))
            if nti != st.session_state[key_ti]:
                st.session_state[key_ti] = nti
                st.rerun()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Observed temperature heatmap (full width)
    # ------------------------------------------------------------------ #
    if obs is not None:
        obs_times, obs_depths, obs_T = obs
        fig_obs = go.Figure(go.Heatmap(
            x=obs_times.to_pydatetime(),
            y=obs_depths,
            z=obs_T.T,
            colorscale="RdYlBu_r",
            zmin=zmin_shared, zmax=0,
            colorbar=dict(title="°C", thickness=12),
            hovertemplate=(
                "Date: %{x}<br>Depth: %{y:.2f} m<br>T: %{z:.2f} °C<extra></extra>"
            ),
        ))
        fig_obs.update_layout(
            title=f"{sid} — Observed firn temperature",
            yaxis=dict(title="Depth (m)", autorange="reversed"),
            xaxis_title="Date", height=320,
            margin=dict(l=55, r=10, t=35, b=55),
        )
        st.plotly_chart(fig_obs, width="stretch", key=f"obs_T_chart_{sid}")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_run, tab_settings, tab_results = st.tabs(["▶ Run", "⚙ Settings", "📊 Results"])

# ============================================================
# TAB 1 — Run
# ============================================================
with tab_run:
    col_params, col_status = st.columns([1, 2])

    with col_params:
        st.subheader("Run parameters")

        # --- Core selection ---
        available_cores = discover_cores()
        core_labels = [
            f"{y} · {s} · {d} m    {d0} → {d1}"
            for y, s, d, d0, d1 in available_cores
        ]

        use_custom = st.checkbox("Enter site manually", value=not bool(available_cores))

        if not use_custom and available_cores:
            chosen_label = st.selectbox(
                "Available cores (all required data files present)",
                options=core_labels,
            )
            idx   = core_labels.index(chosen_label)
            year, site, depth, _, _ = available_cores[idx]
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
            st.markdown("**Paths**")
            c1, c2 = st.columns(2)
            doc["paths"]["snowpack_exe"] = c1.text_input(
                "SNOWPACK executable", value=str(doc["paths"]["snowpack_exe"]))
            doc["paths"]["data_root"] = c2.text_input(
                "Data root (blank = auto)", value=str(doc["paths"].get("data_root", "")))

            st.markdown("**Run**")
            c1, c2 = st.columns(2)
            doc["run"]["run_until"] = c1.text_input(
                "Default run_until (blank = full record)",
                value=str(doc["run"].get("run_until", "")))
            doc["run"]["keep_last_n_sno"] = c2.number_input(
                "Keep last N .sno files", value=int(doc["run"]["keep_last_n_sno"]),
                min_value=1, step=1)
            doc["run"]["keep_hourly_archives"] = st.checkbox(
                "Keep hourly .sno archives",
                value=bool(doc["run"]["keep_hourly_archives"]))

            st.markdown("**Model**")
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

            st.markdown("**Assimilation safeguards**")
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

            st.markdown("**Basal boundary**")
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
