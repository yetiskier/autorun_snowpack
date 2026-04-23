"""
Observed vs modelled firn temperature — all completed model runs.
Generates one 2-panel PNG per site (observed top, modelled bottom).
Usage:
    python plot_obs_vs_model.py              # all sites
    python plot_obs_vs_model.py T3 CP        # specific sites by key
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

SCRIPT_DIR = Path(__file__).parent
TEMP_DIR   = SCRIPT_DIR / "AllCoreDataCommonFormat/Concatenated_Temperature_files"
PROM_DIR   = SCRIPT_DIR / "AllCoreDataCommonFormat/Depth_change_estimate/PROMICE"

SITES = [
    dict(key="T2_2007",        sid="2007_T2_10m", pro="2007_T2_10m/output/2007-T2-10m_TEMP_ASSIM_RUN.pro",               label="T2 (2007)",        out="T2_2007_obs_vs_model.png"),
    dict(key="T2_2007_bucket", sid="2007_T2_10m", pro="2007_T2_10m_bucket/output/2007-T2-10m_TEMP_ASSIM_RUN.pro",       label="T2 bucket (2007)", out="T2_2007_bucket_obs_vs_model.png"),
    dict(key="T2minus",   sid="2019_T2minus_32m",  pro="2019_T2minus_32m/output/2019-T2minus-32m_TEMP_ASSIM_RUN.pro", label="T2− (2019)",    out="T2minus_obs_vs_model.png"),
    dict(key="T3",        sid="2022_T3_25m",       pro="2022_T3_25m/output/2022-T3-25m_TEMP_ASSIM_RUN.pro",           label="T3 (1794 m)",   out="T3_obs_vs_model.png"),
    dict(key="T4",        sid="2022_T4_25m",       pro="2022_T4_25m/output/2022-T4-25m_TEMP_ASSIM_RUN.pro",           label="T4 (1873 m)",   out="T4_obs_vs_model.png"),
    dict(key="CP",        sid="2023_CP_25m",       pro="2023_CP_25m/output/2023-CP-25m_TEMP_ASSIM_RUN.pro",           label="CP (1998 m)",   out="CP_obs_vs_model.png"),
    dict(key="UP18",      sid="2023_UP18_25m",     pro="2023_UP18_25m/output/2023-UP18-25m_TEMP_ASSIM_RUN.pro",       label="UP18 (2109 m)", out="UP18_obs_vs_model.png"),
]

MAX_DEPTH  = 10.0
DEPTH_GRID = np.arange(0.0, MAX_DEPTH + 0.05, 0.05)
FS         = 40

boundaries = list(np.arange(-20, -0.05, 2)) + [-0.05]
n_bins     = len(boundaries) - 1
cmap       = matplotlib.colormaps["turbo"].resampled(n_bins).copy()
cmap.set_over("gray")
norm       = mcolors.BoundaryNorm(boundaries, ncolors=n_bins)


def load_surface_change(sid):
    df = pd.read_csv(PROM_DIR / f"{sid}_daily_PROMICE_snowfall.csv",
                     parse_dates=["timestamp"], index_col="timestamp")
    s  = df["cumulative_surface_total_change_cm"] / 100.0
    s.index = s.index.normalize()
    return s


def load_observed(sid):
    csv = TEMP_DIR / f"{sid}_Tempconcatenated.csv"
    df  = pd.read_csv(csv, skiprows=4, index_col=0, parse_dates=True)
    valid = {}
    for col in df.columns:
        try:
            valid[float(col)] = col
        except (ValueError, TypeError):
            pass
    nominal_depths = np.array(sorted(valid.keys()))
    data = df[[valid[d] for d in nominal_depths]].apply(pd.to_numeric, errors="coerce")
    data = data.resample("1h").mean()
    arr  = data.to_numpy(dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    times = pd.DatetimeIndex(data.index)

    surf = load_surface_change(sid).reindex(times.normalize()).ffill().fillna(0.0).to_numpy()

    grid = np.full((len(times), len(DEPTH_GRID)), np.nan)
    for ti in range(len(times)):
        adj = nominal_depths + surf[ti]
        row = arr[ti]
        vm  = np.isfinite(row)
        if vm.sum() < 2:
            continue
        f = interp1d(adj[vm], row[vm], kind="linear",
                     bounds_error=False, fill_value=np.nan)
        grid[ti] = f(DEPTH_GRID)

    sensor_depths = nominal_depths[:, None] + surf[None, :]
    return times, grid, sensor_depths


def parse_pro(path):
    times, pro = [], {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("0500,"):
                raw = line[5:].strip()
                if raw == "Date":
                    continue
                try:
                    times.append(pd.to_datetime(raw, dayfirst=True))
                except Exception:
                    continue
            elif "," in line:
                parts = line.split(",")
                try:
                    code = int(parts[0])
                except ValueError:
                    continue
                for target in (501, 503):
                    if code == target:
                        try:
                            n = int(parts[1])
                        except ValueError:
                            continue
                        pro.setdefault(code, []).append(
                            np.array([float(v) for v in parts[2:2+n] if v.strip()]))
    return pd.DatetimeIndex(times), pro


def load_modelled(pro_path):
    times, pro = parse_pro(pro_path)
    grid = np.full((len(times), len(DEPTH_GRID)), np.nan)
    for ti in range(len(times)):
        h = pro[501][ti]
        t = pro[503][ti]
        n = min(len(h), len(t))
        if n < 2:
            continue
        order   = np.argsort(h[:n])
        h_s     = h[:n][order]
        t_s     = t[:n][order]
        surface = h_s[-1]
        dm      = (surface - h_s) / 100.0
        dm_rev  = dm[::-1]
        t_rev   = t_s[::-1]
        f = interp1d(dm_rev, t_rev, kind="linear",
                     bounds_error=False, fill_value=np.nan)
        grid[ti] = f(DEPTH_GRID)
    return times, grid


def make_figure(site):
    sid   = site["sid"]
    label = site["label"]
    out   = SCRIPT_DIR / site["out"]

    print(f"  Loading observed ({sid}) …")
    obs_times, obs_grid, sensor_depths = load_observed(sid)
    print(f"  Loading modelled …")
    mod_times, mod_grid = load_modelled(SCRIPT_DIR / site["pro"])

    # Auto-detect overlapping time window
    t_start = max(obs_times[0],  mod_times[0])
    t_end   = min(obs_times[-1], mod_times[-1])

    obs_mask = (obs_times >= t_start) & (obs_times <= t_end)
    mod_mask = (mod_times >= t_start) & (mod_times <= t_end)
    obs_times_w = obs_times[obs_mask];  obs_grid_w = obs_grid[obs_mask]
    mod_times_w = mod_times[mod_mask];  mod_grid_w = mod_grid[mod_mask]

    sd_times_full = pd.DatetimeIndex(
        pd.read_csv(TEMP_DIR / f"{sid}_Tempconcatenated.csv",
                    skiprows=4, index_col=0, parse_dates=True)
        .resample("1h").mean().index
    )
    sd_mask = (sd_times_full >= t_start) & (sd_times_full <= t_end)
    sensor_depths_w = sensor_depths[:, obs_mask]
    sd_t_num = matplotlib.dates.date2num(sd_times_full[sd_mask].to_pydatetime())

    obs_t = matplotlib.dates.date2num(obs_times_w.to_pydatetime())
    mod_t = matplotlib.dates.date2num(mod_times_w.to_pydatetime())
    x_min = min(obs_t[0],  mod_t[0])
    x_max = max(obs_t[-1], mod_t[-1])

    fig, axes = plt.subplots(2, 1, figsize=(20, 16),
                             sharex=True,
                             gridspec_kw={"hspace": 0.08})

    panel_titles = ["Observed", "Modelled (SNOWPACK)"]
    grids = [(obs_t, obs_grid_w), (mod_t, mod_grid_w)]

    for ax, title, (t_num, grid) in zip(axes, panel_titles, grids):
        Z = np.ma.masked_invalid(grid.T)
        ax.contourf(t_num, DEPTH_GRID, Z,
                    levels=boundaries, cmap=cmap, norm=norm, extend="max")
        ax.contour(t_num, DEPTH_GRID, Z,
                   levels=[-0.05], colors="white", linewidths=1.5)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(MAX_DEPTH, 0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        ax.tick_params(axis="y", labelsize=FS * 0.56)
        ax.set_ylabel("Depth (m)", fontsize=FS * 0.8)
        ax.text(0.01, 0.03, title, transform=ax.transAxes,
                fontsize=FS, fontweight="bold", va="bottom", ha="left",
                color="white", zorder=10,
                bbox=dict(boxstyle="round,pad=0.1",
                          facecolor="black", alpha=0.35, linewidth=0))

    for sd in sensor_depths_w:
        d = np.where((sd >= 0) & (sd <= MAX_DEPTH), sd, np.nan)
        axes[0].plot(sd_t_num, d, color="black", linewidth=0.5,
                     linestyle="-", alpha=0.4, zorder=5)

    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    axes[-1].xaxis.set_minor_locator(matplotlib.dates.WeekdayLocator())
    axes[-1].tick_params(axis="x", labelsize=FS * 0.7, rotation=30)

    fig.suptitle(f"{label}  —  observed vs modelled firn temperature",
                 fontsize=FS, y=1.01)

    h_iso    = mlines.Line2D([], [], color="white", linewidth=1.5, label="−0.05 °C isotherm")
    h_sensor = mlines.Line2D([], [], color="black", linewidth=0.5, alpha=0.4, label="Sensor depth")
    axes[0].legend(handles=[h_iso, h_sensor], fontsize=FS * 0.5,
                   loc="center left", bbox_to_anchor=(1.01, 0.5),
                   framealpha=0.7, facecolor="dimgray", labelcolor="white",
                   edgecolor="none")

    cbar = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=axes, orientation="vertical", fraction=0.015, pad=0.02,
        extend="max", ticks=boundaries,
    )
    cbar.ax.tick_params(labelsize=FS * 0.7)
    cbar.set_label("Temperature (°C)", fontsize=FS * 0.8)
    cbar.set_ticklabels([f"{b:.0f}" if b != -0.05 else "−0.05" for b in boundaries])

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {out}")


def site_from_sid(sid: str) -> "dict | None":
    """Build a site dict from a run directory sid (e.g. '2023_UP18_25m').

    Discovers the .pro file automatically; used when autorun_snowpack.py
    calls this script at the end of a completed run.
    """
    run_dir = SCRIPT_DIR / sid
    if not run_dir.exists():
        return None
    out_dir = run_dir / "output"
    pros = sorted(out_dir.glob("*.pro"), key=lambda p: p.stat().st_size, reverse=True) if out_dir.exists() else []
    if not pros:
        return None
    pro = pros[0]  # largest .pro = most complete
    # Derive human label from sid: "2023_UP18_25m" → "UP18 25 m (2023)"
    parts = sid.split("_")
    year  = parts[0] if parts[0].isdigit() else ""
    depth = parts[-1] if parts[-1].endswith("m") else ""
    name  = "_".join(p for p in parts[1:-1] if p != depth)
    label = f"{name} {depth} ({year})" if year else sid
    return dict(
        key=sid,
        sid=sid,
        pro=str(pro.relative_to(SCRIPT_DIR)),
        label=label,
        out=f"{sid}_obs_vs_model.png",
    )


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        targets = list(SITES)
    else:
        targets = []
        for k in args:
            match = next((s for s in SITES if s["key"] == k), None)
            if match:
                targets.append(match)
            else:
                # Try as a raw sid (e.g. "2023_UP18_25m")
                site = site_from_sid(k)
                if site:
                    targets.append(site)
                else:
                    print(f"Unknown key/sid '{k}'. Known keys: {[s['key'] for s in SITES]}")
                    sys.exit(1)

    for site in targets:
        print(f"\n── {site['label']} ──")
        try:
            make_figure(site)
        except Exception as e:
            print(f"  ERROR: {e}")
