"""
LWC comparison: RE fixed theta_r vs RE dynamic theta_r (Coléou 1998)
3-panel: fixed | dynamic | dynamic - fixed difference
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
import matplotlib.dates

SCRIPT_DIR  = Path(__file__).parent
OUT_DIR     = SCRIPT_DIR / "2019_T2minus_32m" / "output"
FIXED_PRO   = OUT_DIR / "2019-T2minus-32m_TEMP_ASSIM_RUN_RE_fixed_theta_r.pro"
DYNAMIC_PRO = OUT_DIR / "2019-T2minus-32m_TEMP_ASSIM_RUN.pro"
OUT_FILE    = SCRIPT_DIR / "T2minus_LWC_comparison.png"

MAX_DEPTH  = 10.0
DEPTH_GRID = np.arange(0.0, MAX_DEPTH + 0.05, 0.05)

LWC_FLOOR = 1e-4   # kg m⁻² minimum for log scale
LWC_TICKS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def parse_lwc(path):
    times, h_list, lwc_list = [], [], []
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
                if code == 501:
                    try:
                        n = int(parts[1])
                        h_list.append(np.array([float(v) for v in parts[2:2+n] if v.strip()]))
                    except ValueError:
                        pass
                elif code == 535:
                    try:
                        n = int(parts[1])
                        lwc_list.append(np.array([float(v) for v in parts[2:2+n] if v.strip()]))
                    except ValueError:
                        pass
    times = pd.DatetimeIndex(times)
    grid = np.full((len(times), len(DEPTH_GRID)), np.nan)
    n = min(len(times), len(h_list), len(lwc_list))
    for ti in range(n):
        h = h_list[ti]; lwc = lwc_list[ti]
        nl = min(len(h), len(lwc))
        if nl < 2:
            continue
        order   = np.argsort(h[:nl])
        h_s     = h[:nl][order]
        lwc_s   = lwc[:nl][order]
        surface = h_s[-1]
        dm      = (surface - h_s) / 100.0
        f = interp1d(dm[::-1], lwc_s[::-1], kind="linear",
                     bounds_error=False, fill_value=np.nan)
        grid[ti] = f(DEPTH_GRID)
    return times, grid


def plot():
    print("Parsing fixed …")
    t_fix, lwc_fix = parse_lwc(FIXED_PRO)
    print("Parsing dynamic …")
    t_dyn, lwc_dyn = parse_lwc(DYNAMIC_PRO)

    t_start = max(t_fix[0],  t_dyn[0])
    t_end   = min(t_fix[-1], t_dyn[-1])
    mf = (t_fix >= t_start) & (t_fix <= t_end)
    md = (t_dyn >= t_start) & (t_dyn <= t_end)
    t_fix_w  = t_fix[mf];  lwc_fix_w  = lwc_fix[mf]
    t_dyn_w  = t_dyn[md];  lwc_dyn_w  = lwc_dyn[md]

    # Regrid dynamic onto fixed time axis for difference
    lwc_dyn_on_fix = np.full_like(lwc_fix_w, np.nan)
    t_fix_n = matplotlib.dates.date2num(t_fix_w.to_pydatetime())
    t_dyn_n = matplotlib.dates.date2num(t_dyn_w.to_pydatetime())
    for di, dg in enumerate(DEPTH_GRID):
        col_dyn = lwc_dyn_w[:, di]
        col_fix = lwc_fix_w[:, di]
        valid = np.isfinite(col_dyn)
        if valid.sum() >= 2:
            f = interp1d(t_dyn_n[valid], col_dyn[valid], kind="linear",
                         bounds_error=False, fill_value=np.nan)
            lwc_dyn_on_fix[:, di] = f(t_fix_n)
        else:
            lwc_dyn_on_fix[:, di] = np.nan

    diff = lwc_dyn_on_fix - lwc_fix_w   # positive = more LWC in dynamic run

    x_min = min(t_fix_n[0],  t_dyn_n[0])
    x_max = max(t_fix_n[-1], t_dyn_n[-1])

    FS = 30

    # ── Colour maps ──────────────────────────────────────────────────────── #
    log_min = np.log10(LWC_FLOOR)
    log_max = np.log10(LWC_TICKS[-1])
    cmap_lwc = matplotlib.colormaps["Blues"]
    norm_abs = mcolors.Normalize(vmin=log_min, vmax=log_max)

    diff_abs = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95) if np.any(np.isfinite(diff)) else 1.0
    diff_abs = max(diff_abs, 0.05)
    cmap_diff = matplotlib.colormaps["RdBu_r"]
    norm_diff = mcolors.Normalize(vmin=-diff_abs, vmax=diff_abs)

    fig, axes = plt.subplots(3, 1, figsize=(20, 16),
                             sharex=True, gridspec_kw={"hspace": 0.10})

    def plot_abs(ax, t_num, lwc_grid, title):
        Z = np.ma.masked_invalid(lwc_grid.T)
        Z_log = np.ma.log10(np.ma.masked_less_equal(Z, LWC_FLOOR))
        im = ax.pcolormesh(t_num, DEPTH_GRID, Z_log,
                           cmap=cmap_lwc, norm=norm_abs,
                           shading="auto", rasterized=True)
        ax.set_ylim(MAX_DEPTH, 0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        ax.tick_params(axis="y", labelsize=FS * 0.56)
        ax.set_ylabel("Depth (m)", fontsize=FS * 0.8)
        ax.set_xlim(x_min, x_max)
        ax.text(0.01, 0.04, title, transform=ax.transAxes,
                fontsize=FS * 0.75, fontweight="bold", va="bottom", ha="left",
                color="white", zorder=10,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black",
                          alpha=0.4, linewidth=0))
        return im

    def plot_diff(ax, t_num, diff_grid, title):
        Z = np.ma.masked_invalid(diff_grid.T)
        im = ax.pcolormesh(t_num, DEPTH_GRID, Z,
                           cmap=cmap_diff, norm=norm_diff,
                           shading="auto", rasterized=True)
        ax.set_ylim(MAX_DEPTH, 0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        ax.tick_params(axis="y", labelsize=FS * 0.56)
        ax.set_ylabel("Depth (m)", fontsize=FS * 0.8)
        ax.set_xlim(x_min, x_max)
        ax.text(0.01, 0.04, title, transform=ax.transAxes,
                fontsize=FS * 0.75, fontweight="bold", va="bottom", ha="left",
                color="k", zorder=10,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="white",
                          alpha=0.5, linewidth=0))
        return im

    im0 = plot_abs(axes[0], t_fix_n, lwc_fix_w,
                   "LWC — RE fixed θᵣ = 0.02")
    im1 = plot_abs(axes[1], t_dyn_n, lwc_dyn_w,
                   "LWC — RE dynamic θᵣ (Coléou 1998)")
    im2 = plot_diff(axes[2], t_fix_n, diff,
                    "Difference: dynamic − fixed (kg m⁻²)")

    # Colorbars
    cb0 = fig.colorbar(im0, ax=axes[:2], orientation="vertical",
                       fraction=0.015, pad=0.02)
    cb0.ax.tick_params(labelsize=FS * 0.7)
    cb0.set_label("LWC (kg m⁻²)", fontsize=FS * 0.8)
    tick_log = [np.log10(t) for t in LWC_TICKS]
    cb0.set_ticks(tick_log)
    cb0.set_ticklabels([str(t) for t in LWC_TICKS])

    cb2 = fig.colorbar(im2, ax=axes[2], orientation="vertical",
                       fraction=0.015, pad=0.02)
    cb2.ax.tick_params(labelsize=FS * 0.7)
    cb2.set_label("ΔLWC (kg m⁻²)", fontsize=FS * 0.8)

    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    axes[-1].xaxis.set_minor_locator(matplotlib.dates.WeekdayLocator())
    axes[-1].tick_params(axis="x", labelsize=FS * 0.7, rotation=30)

    fig.suptitle("T2− (2019) — LWC: RE fixed vs dynamic residual saturation",
                 fontsize=FS, y=1.005)
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    for p in (FIXED_PRO, DYNAMIC_PRO):
        if not p.exists():
            raise SystemExit(f"Missing: {p}")
    plot()
