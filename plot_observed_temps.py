"""
Plot observed borehole temperature heatmaps for six 25 m cores, 2022-2025.
Stacked single-column, lowest elevation (T3) at top to highest (UP18) at bottom.
Usage: python plot_observed_temps.py
Output: observed_temps_25m_2022-2025.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

TEMP_DIR = Path(__file__).parent / "AllCoreDataCommonFormat/Concatenated_Temperature_files"

# Ordered lowest → highest elevation
SITES = [
    ("2022_T3_25m",   "T3"),
    ("2022_T4_25m",   "T4"),
    ("2023_T5_25m",   "T5"),
    ("2023_CP_25m",   "CP"),
    ("2024_UP10_25m", "UP10"),
    ("2023_UP18_25m", "UP18"),
]

T_START   = pd.Timestamp("2022-01-01")
T_END     = pd.Timestamp("2025-12-31")
MAX_DEPTH = 5.0  # m

FS = 50   # base font size (5× original ~10)


def load_site(sid: str):
    csv = TEMP_DIR / f"{sid}_Tempconcatenated.csv"
    df  = pd.read_csv(csv, skiprows=4, index_col=0, parse_dates=True)
    valid = {}
    for col in df.columns:
        try:
            d = float(col)
            if 0.0 <= d <= MAX_DEPTH:
                valid[d] = col
        except (ValueError, TypeError):
            pass
    depths = np.array(sorted(valid.keys()))
    cols   = [valid[d] for d in depths]
    data   = df[cols].apply(pd.to_numeric, errors="coerce")
    data   = data.resample("1D").mean()
    data   = data.loc[T_START:T_END]
    arr    = data.to_numpy(dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    return pd.DatetimeIndex(data.index), depths, arr


T_MIN, T_MAX = -20.0, -0.05
cmap = matplotlib.colormaps["turbo"].copy()
cmap.set_over("gray")
norm = mcolors.Normalize(vmin=T_MIN, vmax=T_MAX)

n = len(SITES)
fig, axes = plt.subplots(n, 1, figsize=(20, 20),
                         sharex=True,
                         gridspec_kw={"hspace": 0.08})
fig.subplots_adjust(left=0.10)

# Pre-load all sites to find the true last measurement date
site_data = {}
for sid, label in SITES:
    try:
        site_data[(sid, label)] = load_site(sid)
    except FileNotFoundError:
        site_data[(sid, label)] = None

last_time = max(
    d[0][-1] for d in site_data.values() if d is not None
)

x_min = matplotlib.dates.date2num(T_START.to_pydatetime())
x_max = matplotlib.dates.date2num(last_time.to_pydatetime())

for ax, (sid, label) in zip(axes, SITES):
    loaded = site_data[(sid, label)]
    if loaded is None:
        ax.text(0.5, 0.5, f"{label} — no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS)
        ax.set_ylabel(label, fontsize=FS, fontweight="bold")
        continue

    times, depths, arr = loaded
    t_num = matplotlib.dates.date2num(times.to_pydatetime())
    if len(t_num) > 1:
        dt = t_num[1] - t_num[0]
        t_edges = np.append(t_num - dt / 2, t_num[-1] + dt / 2)
    else:
        t_edges = np.array([t_num[0] - 0.5, t_num[0] + 0.5])

    d_edges = np.zeros(len(depths) + 1)
    d_edges[1:-1] = (depths[:-1] + depths[1:]) / 2
    d_edges[0]    = max(0.0, depths[0] - (depths[1] - depths[0]) / 2)
    d_edges[-1]   = min(MAX_DEPTH, depths[-1] + (depths[-1] - depths[-2]) / 2)

    ax.pcolormesh(t_edges, d_edges, arr.T,
                  cmap=cmap, norm=norm,
                  shading="flat", rasterized=True)

    # -0.05 °C isotherm
    ax.contour(t_num, depths, arr.T,
               levels=[-0.05], colors="white", linewidths=2.0)

    ax.set_ylim(MAX_DEPTH, 0)
    ax.set_xlim(x_min, x_max)
    ax.text(0.01, 0.97, label, transform=ax.transAxes,
            fontsize=FS, fontweight="bold", va="top", ha="left",
            color="white", bbox=dict(boxstyle="round,pad=0.1",
                                     facecolor="black", alpha=0.35, linewidth=0))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(0.5))
    ax.tick_params(axis="y", labelsize=FS * 0.28)

# x-axis ticks on bottom panel only (sharex handles the rest)
axes[-1].xaxis_date()
axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
axes[-1].xaxis.set_major_locator(matplotlib.dates.YearLocator())
axes[-1].xaxis.set_minor_locator(matplotlib.dates.MonthLocator(bymonth=[4, 7, 10]))
axes[-1].tick_params(axis="x", labelsize=FS)

# Common y-axis label
fig.text(0.01, 0.5, "Depth (m)", va="center", rotation="vertical", fontsize=FS)

# Colourbar
cbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes, orientation="vertical", fraction=0.018, pad=0.02,
    label="Temperature (°C)", extend="max",
)
cbar.ax.tick_params(labelsize=FS * 0.8)
cbar.set_label("Temperature (°C)", fontsize=FS)

out = Path(__file__).parent / "observed_temps_25m_2022-2025.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
