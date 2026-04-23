"""
Compare two SNOWPACK RE runs for 2019_T2minus_32m:
  - fixed theta_r  (cap = 0.02)
  - dynamic theta_r (Coléou & Lesaffre 1998, density-dependent)

Both .pro files must exist in 2019_T2minus_32m/output/.
Usage:
    python compare_RE_theta_r.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker

SCRIPT_DIR = Path(__file__).parent
OUT_DIR    = SCRIPT_DIR / "2019_T2minus_32m" / "output"

FIXED_PRO   = OUT_DIR / "2019-T2minus-32m_TEMP_ASSIM_RUN_RE_fixed_theta_r.pro"
DYNAMIC_PRO = OUT_DIR / "2019-T2minus-32m_TEMP_ASSIM_RUN.pro"

MAX_DEPTH  = 10.0
DEPTH_GRID = np.arange(0.0, MAX_DEPTH + 0.05, 0.05)

boundaries = list(np.arange(-20, -0.05, 2)) + [-0.05]
n_bins     = len(boundaries) - 1
cmap_T     = matplotlib.colormaps["turbo"].resampled(n_bins).copy()
cmap_T.set_over("gray")
norm_T     = mcolors.BoundaryNorm(boundaries, ncolors=n_bins)

LWC_LEVELS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]
cmap_lwc   = matplotlib.colormaps["Blues"]
norm_lwc   = mcolors.BoundaryNorm(LWC_LEVELS, ncolors=len(LWC_LEVELS)-1)


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
                for target in (501, 503, 535):  # height, temp, LWC
                    if code == target:
                        try:
                            n = int(parts[1])
                        except ValueError:
                            continue
                        pro.setdefault(code, []).append(
                            np.array([float(v) for v in parts[2:2+n] if v.strip()]))
    return pd.DatetimeIndex(times), pro


def build_grids(pro_path, depth_grid=None):
    if depth_grid is None:
        depth_grid = DEPTH_GRID
    times, pro = parse_pro(pro_path)
    T_grid   = np.full((len(times), len(depth_grid)), np.nan)
    lwc_grid = np.full((len(times), len(depth_grid)), np.nan)
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
        fT = interp1d(dm_rev, t_rev, kind="linear",
                      bounds_error=False, fill_value=np.nan)
        T_grid[ti] = fT(depth_grid)

        if 535 in pro and ti < len(pro[535]):
            lwc = pro[535][ti]
            n2  = min(len(h), len(lwc))
            if n2 >= 2:
                lwc_s = lwc[:n2][order[:n2]]
                fl = interp1d(dm_rev[:n2], lwc_s[::-1], kind="linear",
                              bounds_error=False, fill_value=np.nan)
                lwc_grid[ti] = fl(depth_grid)

    return times, T_grid, lwc_grid


def plot_comparison():
    print("Parsing fixed theta_r …")
    t_fix, T_fix, lwc_fix = build_grids(FIXED_PRO)
    print("Parsing dynamic theta_r …")
    t_dyn, T_dyn, lwc_dyn = build_grids(DYNAMIC_PRO)

    # Common time window
    t_start = max(t_fix[0],  t_dyn[0])
    t_end   = min(t_fix[-1], t_dyn[-1])
    m_fix   = (t_fix >= t_start) & (t_fix <= t_end)
    m_dyn   = (t_dyn >= t_start) & (t_dyn <= t_end)
    t_fix_w = t_fix[m_fix];  T_fix_w = T_fix[m_fix];  lwc_fix_w = lwc_fix[m_fix]
    t_dyn_w = t_dyn[m_dyn];  T_dyn_w = T_dyn[m_dyn];  lwc_dyn_w = lwc_dyn[m_dyn]

    t_fix_n = matplotlib.dates.date2num(t_fix_w.to_pydatetime())
    t_dyn_n = matplotlib.dates.date2num(t_dyn_w.to_pydatetime())
    x_min   = min(t_fix_n[0],  t_dyn_n[0])
    x_max   = max(t_fix_n[-1], t_dyn_n[-1])

    FS = 28
    fig, axes = plt.subplots(4, 1, figsize=(20, 22),
                             sharex=True, gridspec_kw={"hspace": 0.10})

    panels = [
        (axes[0], t_fix_n, T_fix_w,   "Temperature — RE fixed θᵣ=0.02",           "T"),
        (axes[1], t_dyn_n, T_dyn_w,   "Temperature — RE dynamic θᵣ (Coléou 1998)", "T"),
        (axes[2], t_fix_n, lwc_fix_w, "LWC — RE fixed θᵣ=0.02",                   "lwc"),
        (axes[3], t_dyn_n, lwc_dyn_w, "LWC — RE dynamic θᵣ (Coléou 1998)",         "lwc"),
    ]

    for ax, t_num, grid, title, kind in panels:
        Z = np.ma.masked_invalid(grid.T)
        if kind == "T":
            ax.contourf(t_num, DEPTH_GRID, Z,
                        levels=boundaries, cmap=cmap_T, norm=norm_T, extend="max")
            ax.contour(t_num, DEPTH_GRID, Z,
                       levels=[-0.05], colors="white", linewidths=1.5)
        else:
            # log-scale LWC
            Z_log = np.ma.log10(np.ma.masked_less_equal(Z, 0))
            ax.contourf(t_num, DEPTH_GRID, Z_log,
                        levels=np.log10(LWC_LEVELS), cmap=cmap_lwc, extend="both")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(MAX_DEPTH, 0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        ax.tick_params(axis="y", labelsize=FS * 0.56)
        ax.set_ylabel("Depth (m)", fontsize=FS * 0.8)
        ax.text(0.01, 0.03, title, transform=ax.transAxes,
                fontsize=FS * 0.75, fontweight="bold", va="bottom", ha="left",
                color="white", zorder=10,
                bbox=dict(boxstyle="round,pad=0.1",
                          facecolor="black", alpha=0.40, linewidth=0))

    # Colorbars
    cbar_T = fig.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm_T, cmap=cmap_T),
        ax=axes[:2], orientation="vertical", fraction=0.015, pad=0.02,
        extend="max", ticks=boundaries,
    )
    cbar_T.ax.tick_params(labelsize=FS * 0.7)
    cbar_T.set_label("Temperature (°C)", fontsize=FS * 0.8)
    cbar_T.set_ticklabels([f"{b:.0f}" if b != -0.05 else "−0.05" for b in boundaries])

    lwc_sm  = matplotlib.cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=np.log10(LWC_LEVELS[0]),
                               vmax=np.log10(LWC_LEVELS[-1])),
        cmap=cmap_lwc)
    cbar_lwc = fig.colorbar(lwc_sm, ax=axes[2:], orientation="vertical",
                             fraction=0.015, pad=0.02)
    cbar_lwc.ax.tick_params(labelsize=FS * 0.7)
    cbar_lwc.set_label("LWC (kg m⁻²)", fontsize=FS * 0.8)
    tick_vals  = [l for l in LWC_LEVELS]
    tick_log   = [np.log10(l) for l in tick_vals]
    tick_labels = [str(l) if l >= 0.1 else str(l) for l in tick_vals]
    cbar_lwc.set_ticks(tick_log)
    cbar_lwc.set_ticklabels(tick_labels)

    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    axes[-1].xaxis.set_minor_locator(matplotlib.dates.WeekdayLocator())
    axes[-1].tick_params(axis="x", labelsize=FS * 0.7, rotation=30)

    fig.suptitle("T2− (2019) — RE fixed vs dynamic residual saturation",
                 fontsize=FS, y=1.005)

    out = SCRIPT_DIR / "T2minus_RE_theta_r_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


if __name__ == "__main__":
    for p in (FIXED_PRO, DYNAMIC_PRO):
        if not p.exists():
            print(f"Missing: {p}")
            raise SystemExit(1)
    plot_comparison()
