"""
SNOWPACK autorun — Streamlit GUI
Run from the autorun_snowpack/ directory:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
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
import streamlit as st
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
        return True
    except (ProcessLookupError, PermissionError):
        clear_pid(sid)
        return False


def kill_run(sid: str) -> None:
    pid = read_pid(sid)
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        clear_pid(sid)


def launch_run(year: int, site: str, depth: int, run_until: str) -> int:
    sid = site_id(year, site, depth)
    log = log_path(sid)
    log.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON, "-u", str(SCRIPT),
        "--site",  site,
        "--year",  str(year),
        "--depth", str(depth),
    ]
    if run_until.strip():
        cmd += ["--run-until", run_until.strip()]

    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=fh,
                                cwd=str(APP_DIR),
                                start_new_session=True)
    write_pid(sid, proc.pid)
    return proc.pid


def load_settings() -> tomlkit.TOMLDocument:
    return tomlkit.parse(SETTINGS_FILE.read_text())


def save_settings(doc: tomlkit.TOMLDocument) -> None:
    SETTINGS_FILE.write_text(tomlkit.dumps(doc))


def read_log_tail(sid: str, n_lines: int = 60) -> str:
    lp = log_path(sid)
    if not lp.exists():
        return "(no log yet)"
    lines = lp.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n_lines:])


def output_figures(sid: str) -> list[Path]:
    out = project_dir(sid) / "output"
    if not out.exists():
        return []
    return sorted(out.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)


def find_pro_file(sid: str) -> Path | None:
    out = project_dir(sid) / "output"
    pros = sorted(out.glob("*.pro")) if out.exists() else []
    return pros[0] if pros else None


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
_MK_COLORSCALE = [
    [i / (_N - 1), MK_CATALOG[c][0]] for i, c in enumerate(_ALL_CODES)
]


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


@st.cache_data(show_spinner="Parsing PRO file…")
def load_pro(pro_path_str: str) -> dict:
    path = Path(pro_path_str)
    pro  = _parse_pro(path)
    times = pd.DatetimeIndex(pro["times"])
    soil_depth_m = np.array([h.max() / 100.0 for h in pro[501]])
    max_depth    = float(np.nanmax(soil_depth_m)) if len(soil_depth_m) else 30.0
    depth_grid   = np.arange(0, min(max_depth, 30) + 0.05, 0.05)

    T_grid  = _to_grid(times, pro[501], pro[503], depth_grid)
    MK_raw  = _to_grid(times, pro[501], pro[513], depth_grid, kind="nearest")

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


def show_interactive_charts(sid: str) -> None:
    pro_path = find_pro_file(sid)
    if pro_path is None:
        st.info("No .pro output file found — run the model first.")
        return

    d = load_pro(str(pro_path))
    times      = d["times"]
    depth_grid = d["depth_grid"]
    T_grid     = d["T_grid"]
    MK_raw     = d["MK_raw"]
    MK_idx     = d["MK_idx"]
    raw        = d["raw"]
    soil_depth = d["soil_depth_m"]

    t_str = [str(t)[:16] for t in times]   # "YYYY-MM-DD HH:MM"

    # ------------------------------------------------------------------ #
    # Heatmap 1 — Temperature
    # ------------------------------------------------------------------ #
    fig_T = go.Figure(go.Heatmap(
        x=t_str,
        y=depth_grid,
        z=T_grid.T,
        colorscale="RdYlBu_r",
        zmin=-24, zmax=0,
        colorbar=dict(title="°C", thickness=12),
        hovertemplate=(
            "Date: %{x}<br>"
            "Depth: %{y:.2f} m<br>"
            "T: %{z:.2f} °C"
            "<extra></extra>"
        ),
    ))
    fig_T.update_layout(
        title=f"{sid} — Modelled firn temperature",
        yaxis=dict(title="Depth (m)", autorange="reversed"),
        xaxis_title="Date",
        height=380,
        margin=dict(l=60, r=20, t=40, b=60),
    )

    # ------------------------------------------------------------------ #
    # Heatmap 2 — Swiss grain type
    # ------------------------------------------------------------------ #
    mk_names = _mk_name_grid(MK_raw)
    fig_MK = go.Figure(go.Heatmap(
        x=t_str,
        y=depth_grid,
        z=MK_idx.T,
        colorscale=_MK_COLORSCALE,
        zmin=-0.5, zmax=_N - 0.5,
        showscale=False,
        customdata=np.stack([MK_raw.T, mk_names.T], axis=-1),
        hovertemplate=(
            "Date: %{x}<br>"
            "Depth: %{y:.2f} m<br>"
            "Code: %{customdata[0]:.0f}<br>"
            "Type: %{customdata[1]}"
            "<extra></extra>"
        ),
    ))
    # Soil interface line
    fig_MK.add_trace(go.Scatter(
        x=t_str, y=soil_depth,
        mode="lines", line=dict(color="black", width=1.5),
        name="Soil interface", hoverinfo="skip",
    ))
    fig_MK.update_layout(
        title=f"{sid} — Swiss grain type (mk 513)",
        yaxis=dict(title="Depth (m)", autorange="reversed"),
        xaxis_title="Date",
        height=380,
        margin=dict(l=60, r=20, t=40, b=60),
    )

    st.plotly_chart(fig_T,  width="stretch")
    st.plotly_chart(fig_MK, width="stretch")

    # ------------------------------------------------------------------ #
    # Column plot — profile at selected timestep
    # ------------------------------------------------------------------ #
    st.markdown("---")
    st.subheader("Stratigraphic column")

    ti = st.slider("Select timestep", min_value=0, max_value=len(times) - 1, value=0)
    st.caption(t_str[ti])

    heights = raw[501][ti]   # cm, bottom of each layer
    if len(heights) == 0:
        st.warning("No layer data for this timestep.")
        return

    mk_vals  = raw[513][ti]
    temp_vals = raw[503][ti]
    nc_vals  = raw[510][ti]

    # Layer geometry (depth below surface, top→bottom)
    surface_cm = heights.max()
    depth_bot  = (surface_cm - heights)       / 100.0   # m
    h_prev     = np.concatenate([[surface_cm], heights[:-1]])
    depth_top  = (surface_cm - h_prev)        / 100.0
    thickness  = depth_bot - depth_top
    depth_mid  = (depth_top + depth_bot) / 2.0

    hardness = np.array([_code_hardness(c) for c in mk_vals])
    colors   = [_code_color(c) for c in mk_vals]
    names    = [_code_name(c)  for c in mk_vals]

    col_profile, col_legend = st.columns([2, 1])

    with col_profile:
        fig_col = go.Figure()
        fig_col.add_trace(go.Bar(
            orientation="h",
            x=hardness,
            y=depth_mid,
            width=thickness,
            marker=dict(
                color=colors,
                line=dict(color="rgba(0,0,0,0.2)", width=0.4),
            ),
            customdata=np.stack([temp_vals, names, depth_top, depth_bot], axis=-1),
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Depth: %{customdata[2]:.2f}–%{customdata[3]:.2f} m<br>"
                "T: %{customdata[0]:.2f} °C<br>"
                "Hardness: %{x:.1f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))
        fig_col.update_layout(
            title=f"Profile at {t_str[ti]}",
            xaxis=dict(title="Hand hardness (1=fist … 6=ice)", range=[0, 6.5]),
            yaxis=dict(title="Depth (m)", autorange="reversed"),
            height=600,
            margin=dict(l=60, r=20, t=40, b=60),
            bargap=0,
        )
        st.plotly_chart(fig_col, width="stretch")

    with col_legend:
        st.markdown("**Grain types at this timestep**")
        seen = {}
        for code, name, color in zip(mk_vals, names, colors):
            k = int(round(code))
            if k not in seen and k in MK_CATALOG:
                seen[k] = (color, name)
        for color, name in seen.values():
            st.markdown(
                f'<span style="background:{color};display:inline-block;'
                f'width:14px;height:14px;margin-right:6px;border-radius:2px;'
                f'vertical-align:middle"></span>{name}',
                unsafe_allow_html=True,
            )


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
        st.subheader(f"Log — {sid}")

        if running:
            st.status("Running…", state="running")
        else:
            pid = read_pid(sid)
            if pid is None and log_path(sid).exists():
                last = read_log_tail(sid, 3)
                if "Done." in last:
                    st.status("Finished", state="complete")
                else:
                    st.status("Stopped / not started", state="error")
            else:
                st.status("Not running", state="error")

        log_text = read_log_tail(sid, 80)
        st.code(log_text, language="text")

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
    st.subheader(f"Interactive plots — {sid}")
    show_interactive_charts(sid)

    figs = output_figures(sid)
    if figs:
        st.markdown("---")
        st.subheader("Static figures")
        for fig_path in figs:
            st.image(str(fig_path), caption=fig_path.name, width="stretch")
