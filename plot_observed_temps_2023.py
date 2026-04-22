"""
Observed borehole temperature heatmaps — 2023 summer/autumn window.
Time range: 2023-05-29 to 2023-12-31. Five sites (no UP10).
Discrete colorscale at 2 °C intervals.
Usage: python plot_observed_temps_2023.py
Output: observed_temps_25m_2023.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

TEMP_DIR = Path(__file__).parent / "AllCoreDataCommonFormat/Concatenated_Temperature_files"

# Lowest → highest elevation; UP10 excluded (no data this window)
SITES = [
    ("2022_T3_25m",   "T3"),
    ("2022_T4_25m",   "T4"),
    ("2023_T5_25m",   "T5"),
    ("2023_CP_25m",   "CP"),
    ("2023_UP18_25m", "UP18"),
]

T_START   = pd.Timestamp("2023-05-29")
T_END     = pd.Timestamp("2023-12-31")
MAX_DEPTH = 5.0   # m
FS        = 50    # base font size


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


# Discrete colorscale: 2 °C bins from -20 to -0.1, gray above
boundaries = list(np.arange(-20, -0.05, 2)) + [-0.05]
n_bins     = len(boundaries) - 1
base_cmap  = matplotlib.colormaps["turbo"].resampled(n_bins)
cmap       = base_cmap.copy()
cmap.set_over("gray")
norm       = mcolors.BoundaryNorm(boundaries, ncolors=n_bins)

# Pre-load to find actual last timestamp
site_data = {}
for sid, label in SITES:
    try:
        site_data[(sid, label)] = load_site(sid)
    except FileNotFoundError:
        site_data[(sid, label)] = None

last_time = max(d[0][-1] for d in site_data.values() if d is not None)
# Cap at T_END
last_time = min(last_time, T_END)

x_min = matplotlib.dates.date2num(T_START.to_pydatetime())
x_max = matplotlib.dates.date2num(last_time.to_pydatetime())

n = len(SITES)
fig, axes = plt.subplots(n, 1, figsize=(20, 20),
                         sharex=True,
                         gridspec_kw={"hspace": 0.08})
fig.subplots_adjust(left=0.10)

for ax, (sid, label) in zip(axes, SITES):
    loaded = site_data[(sid, label)]
    if loaded is None:
        ax.text(0.5, 0.5, f"{label} — no data", transform=ax.transAxes,
                ha="center", va="center", fontsize=FS)
        continue

    times, depths, arr = loaded
    t_num = matplotlib.dates.date2num(times.to_pydatetime())

    Z = np.ma.masked_invalid(arr.T)   # (n_depths, n_times)
    ax.contourf(t_num, depths, Z,
                levels=boundaries, cmap=cmap, norm=norm,
                extend="max")

    ax.contour(t_num, depths, Z,
               levels=[-0.05], colors="white", linewidths=2.0)

    ax.set_ylim(MAX_DEPTH, 0)
    ax.set_xlim(x_min, x_max)
    ax.text(0.01, 0.97, label, transform=ax.transAxes,
            fontsize=FS, fontweight="bold", va="top", ha="left",
            color="white", bbox=dict(boxstyle="round,pad=0.1",
                                     facecolor="black", alpha=0.35, linewidth=0))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax.tick_params(axis="y", labelsize=FS * 0.56)

# x-axis on bottom panel
axes[-1].xaxis_date()
axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator())
axes[-1].xaxis.set_minor_locator(matplotlib.dates.WeekdayLocator())
axes[-1].tick_params(axis="x", labelsize=FS * 0.6, rotation=30)

# Common y-axis label
fig.text(0.01, 0.5, "Depth (m)", va="center", rotation="vertical", fontsize=FS)

# Discrete colourbar with a tick at every boundary
cbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes, orientation="vertical", fraction=0.018, pad=0.02,
    label="Temperature (°C)", extend="max",
    ticks=boundaries,
)
cbar.ax.tick_params(labelsize=FS * 0.8)
cbar.set_label("Temperature (°C)", fontsize=FS)
cbar.set_ticklabels([f"{b:.0f}" if b != -0.05 else "−0.05" for b in boundaries])

out = Path(__file__).parent / "observed_temps_25m_2023_contourf.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
