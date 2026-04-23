"""
Check whether blue regions in the LWC difference map (dynamic - fixed)
correlate with high-density / high-ice-fraction layers.

Plots three panels:
  top:    density from the dynamic run
  middle: ice volume fraction from the dynamic run
  bottom: ΔLWC (dynamic − fixed)
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
OUT_FILE    = SCRIPT_DIR / "T2minus_density_vs_LWC_diff.png"

MAX_DEPTH  = 10.0
DEPTH_GRID = np.arange(0.0, MAX_DEPTH + 0.05, 0.05)
LWC_FLOOR  = 1e-4


def parse_fields(path, codes):
    """Parse multiple field codes from a .pro file into time-indexed grids."""
    times, data = [], {c: [] for c in codes}
    h_buf = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("0500,"):
                raw = line[5:].strip()
                if raw == "Date":
                    continue
                try:
                    times.append(pd.to_datetime(raw, dayfirst=True))
                    h_buf = None
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
                        h_buf = np.array([float(v) for v in parts[2:2+n] if v.strip()])
                    except ValueError:
                        pass
                elif code in codes:
                    try:
                        n = int(parts[1])
                        data[code].append(
                            np.array([float(v) for v in parts[2:2+n] if v.strip()]))
                    except ValueError:
                        pass

    times = pd.DatetimeIndex(times)
    grids = {}
    for code in codes:
        vals = data[code]
        # Need heights — re-parse paired with 501
        grids[code] = np.full((len(times), len(DEPTH_GRID)), np.nan)

    return times, grids, data


def build_grids(path, codes):
    """Return (times, {code: grid}) for all requested codes plus 501 heights."""
    times_list = []
    buf = {}   # code -> list of arrays per timestep
    h_list = []

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("0500,"):
                raw = line[5:].strip()
                if raw == "Date":
                    continue
                try:
                    times_list.append(pd.to_datetime(raw, dayfirst=True))
                except Exception:
                    continue
            elif "," in line:
                parts = line.split(",")
                try:
                    code = int(parts[0])
                except ValueError:
                    continue
                if code == 501 or code in codes:
                    try:
                        n = int(parts[1])
                        arr = np.array([float(v) for v in parts[2:2+n] if v.strip()])
                    except ValueError:
                        continue
                    if code == 501:
                        h_list.append(arr)
                    else:
                        buf.setdefault(code, []).append(arr)

    times = pd.DatetimeIndex(times_list)
    grids = {code: np.full((len(times), len(DEPTH_GRID)), np.nan) for code in codes}

    n_steps = min(len(times), len(h_list))
    for code in codes:
        vals = buf.get(code, [])
        n_steps_c = min(n_steps, len(vals))
        for ti in range(n_steps_c):
            h   = h_list[ti]
            v   = vals[ti]
            nl  = min(len(h), len(v))
            if nl < 2:
                continue
            order   = np.argsort(h[:nl])
            h_s     = h[:nl][order]
            v_s     = v[:nl][order]
            surface = h_s[-1]
            dm      = (surface - h_s) / 100.0
            f = interp1d(dm[::-1], v_s[::-1], kind="linear",
                         bounds_error=False, fill_value=np.nan)
            grids[code][ti] = f(DEPTH_GRID)

    return times, grids


def main():
    print("Parsing dynamic run …")
    t_dyn, grids_dyn = build_grids(DYNAMIC_PRO, codes={502, 515, 535})

    print("Parsing fixed run (LWC only) …")
    t_fix, grids_fix = build_grids(FIXED_PRO, codes={535})

    # Common window
    t_start = max(t_dyn[0], t_fix[0])
    t_end   = min(t_dyn[-1], t_fix[-1])
    md = (t_dyn >= t_start) & (t_dyn <= t_end)
    mf = (t_fix >= t_start) & (t_fix <= t_end)

    t_dyn_w = t_dyn[md]
    den_w   = grids_dyn[502][md]
    ice_w   = grids_dyn[515][md]   # ice volume fraction (%)
    lwc_dyn_w = grids_dyn[535][md]

    t_fix_w   = t_fix[mf]
    lwc_fix_w = grids_fix[535][mf]

    # Interpolate dynamic LWC onto fixed time axis for difference
    t_dyn_n = matplotlib.dates.date2num(t_dyn_w.to_pydatetime())
    t_fix_n = matplotlib.dates.date2num(t_fix_w.to_pydatetime())
    lwc_dyn_on_fix = np.full_like(lwc_fix_w, np.nan)
    for di in range(len(DEPTH_GRID)):
        col = lwc_dyn_w[:, di]
        valid = np.isfinite(col)
        if valid.sum() >= 2:
            f = interp1d(t_dyn_n[valid], col[valid], kind="linear",
                         bounds_error=False, fill_value=np.nan)
            lwc_dyn_on_fix[:, di] = f(t_fix_n)

    diff = lwc_dyn_on_fix - lwc_fix_w

    x_min = min(t_fix_n[0], t_dyn_n[0])
    x_max = max(t_fix_n[-1], t_dyn_n[-1])

    FS = 30

    # Colour maps
    cmap_den  = matplotlib.colormaps["viridis"]
    norm_den  = mcolors.Normalize(vmin=300, vmax=917)

    cmap_ice  = matplotlib.colormaps["Greys"]
    norm_ice  = mcolors.Normalize(vmin=0, vmax=100)

    diff_abs  = np.nanpercentile(np.abs(diff[np.isfinite(diff)]), 95) if np.any(np.isfinite(diff)) else 0.1
    diff_abs  = max(diff_abs, 0.01)
    cmap_diff = matplotlib.colormaps["RdBu_r"]
    norm_diff = mcolors.Normalize(vmin=-diff_abs, vmax=diff_abs)

    fig, axes = plt.subplots(3, 1, figsize=(20, 16),
                             sharex=True, gridspec_kw={"hspace": 0.10})

    def base_ax(ax, title, label_color="white"):
        ax.set_ylim(MAX_DEPTH, 0)
        ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
        ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
        ax.tick_params(axis="y", labelsize=FS * 0.56)
        ax.set_ylabel("Depth (m)", fontsize=FS * 0.8)
        ax.set_xlim(x_min, x_max)
        ax.text(0.01, 0.04, title, transform=ax.transAxes,
                fontsize=FS * 0.72, fontweight="bold", va="bottom", ha="left",
                color=label_color, zorder=10,
                bbox=dict(boxstyle="round,pad=0.1", facecolor="black" if label_color == "white" else "white",
                          alpha=0.4, linewidth=0))

    im0 = axes[0].pcolormesh(t_dyn_n, DEPTH_GRID, np.ma.masked_invalid(den_w.T),
                              cmap=cmap_den, norm=norm_den, shading="auto", rasterized=True)
    base_ax(axes[0], "Density — dynamic run (kg m⁻³)")

    im1 = axes[1].pcolormesh(t_dyn_n, DEPTH_GRID, np.ma.masked_invalid(ice_w.T),
                              cmap=cmap_ice, norm=norm_ice, shading="auto", rasterized=True)
    base_ax(axes[1], "Ice volume fraction — dynamic run (%)")

    im2 = axes[2].pcolormesh(t_fix_n, DEPTH_GRID, np.ma.masked_invalid(diff.T),
                              cmap=cmap_diff, norm=norm_diff, shading="auto", rasterized=True)
    base_ax(axes[2], "ΔLWC: dynamic − fixed (kg m⁻²)", label_color="k")

    cb0 = fig.colorbar(im0, ax=axes[0], orientation="vertical", fraction=0.015, pad=0.02)
    cb0.ax.tick_params(labelsize=FS * 0.65); cb0.set_label("Density (kg m⁻³)", fontsize=FS * 0.75)

    cb1 = fig.colorbar(im1, ax=axes[1], orientation="vertical", fraction=0.015, pad=0.02)
    cb1.ax.tick_params(labelsize=FS * 0.65); cb1.set_label("Ice fraction (%)", fontsize=FS * 0.75)

    cb2 = fig.colorbar(im2, ax=axes[2], orientation="vertical", fraction=0.015, pad=0.02)
    cb2.ax.tick_params(labelsize=FS * 0.65); cb2.set_label("ΔLWC (kg m⁻²)", fontsize=FS * 0.75)

    axes[-1].xaxis_date()
    axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    axes[-1].xaxis.set_minor_locator(matplotlib.dates.WeekdayLocator())
    axes[-1].tick_params(axis="x", labelsize=FS * 0.7, rotation=30)

    fig.suptitle("T2− (2019) — Density & ice fraction vs ΔLWC", fontsize=FS, y=1.005)
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
