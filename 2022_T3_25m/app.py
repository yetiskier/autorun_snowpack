"""
SNOWPACK autorun — Streamlit GUI
Run from the project directory:
    streamlit run app.py
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import tomllib
from pathlib import Path

import streamlit as st
import tomlkit

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
APP_DIR      = Path(__file__).resolve().parent   # 2022_T3_25m/
BASE_DIR     = APP_DIR.parent                    # autorun_snowpack/
SCRIPT       = APP_DIR / "autorun_snowpack.py"
SETTINGS_FILE = BASE_DIR / "settings.toml"
PYTHON       = "python3"

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
# Helpers
# ---------------------------------------------------------------------------

def site_id(year: int, site: str, depth: int) -> str:
    return f"{year}_{site}_{depth}m"


def project_dir(sid: str) -> Path:
    return BASE_DIR / sid


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
        year  = st.number_input("Year drilled", value=2022, step=1, format="%d")
        site  = st.text_input("Site name", value="T3")
        depth = st.number_input("String depth (m)", value=25, step=1, format="%d")
        run_until = st.text_input(
            "Run until (optional)",
            value="",
            placeholder="YYYY-MM-DD HH:MM  — leave blank for full record",
        )

        sid = site_id(int(year), site, int(depth))
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
    sid_results = site_id(int(year), site, int(depth))
    figs = output_figures(sid_results)

    if not figs:
        st.info(f"No figures found in `{project_dir(sid_results) / 'output'}`.")
    else:
        st.subheader(f"Output figures — {sid_results}")
        for fig_path in figs:
            st.image(str(fig_path), caption=fig_path.name, use_container_width=True)
