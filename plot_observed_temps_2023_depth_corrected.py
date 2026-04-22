"""
Observed borehole temperature heatmaps — 2023 summer/autumn window.
Sensor depths corrected using PROMICE cumulative surface height change.
Time range: 2023-05-29 to 2023-12-31. Five sites (no UP10).
Discrete colorscale at 2 °C intervals.
Usage: python plot_observed_temps_2023_depth_corrected.py
Output: observed_temps_25m_2023_depth_corrected.png
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.lines as mlines

TEMP_DIR  = Path(__file__).parent / "AllCoreDataCommonFormat/Concatenated_Temperature_files"
PROM_DIR  = Path(__file__).parent / "AllCoreDataCommonFormat/Depth_change_estimate/PROMICE"
LOC_FILE  = Path(__file__).parent / "hole_locations_2025.csv"

# Load elevations; normalise name to match site keys
_loc = pd.read_csv(LOC_FILE)
_loc["key"] = _loc["name"].str.strip().str.lower().str.replace(" 2025", "", regex=False)
_ELEV = {row["key"]: int(row["ele"]) for _, row in _loc.iterrows()}

def _elev_label(site_key: str) -> str:
    """Return 'XXXX m a.s.l.' label from site id like '2022_t3_25m'."""
    short = site_key.split("_")[1].lower()   # e.g. 't3', 'up18', 'cp'
    return f"{_ELEV[short] - 131} m"

# Lowest → highest elevation; UP10 excluded
SITES = [
    ("2022_T3_25m",   _elev_label("2022_t3_25m")),
    ("2022_T4_25m",   _elev_label("2022_t4_25m")),
    ("2023_T5_25m",   _elev_label("2023_t5_25m")),
    ("2023_CP_25m",   _elev_label("2023_cp_25m")),
    ("2023_UP18_25m", _elev_label("2023_up18_25m")),
]

T_START   = pd.Timestamp("2023-05-29")
T_END     = pd.Timestamp("2023-10-01")
MAX_DEPTH  = 5.0       # m from current surface
DEPTH_GRID = np.arange(0.0, MAX_DEPTH + 0.05, 0.05)
FS         = 50


def load_surface_change(sid: str) -> pd.Series:
    """Return daily cumulative surface total change in metres, indexed by date."""
    csv = PROM_DIR / f"{sid}_daily_PROMICE_snowfall.csv"
    df  = pd.read_csv(csv, parse_dates=["timestamp"], index_col="timestamp")
    s   = df["cumulative_surface_total_change_cm"] / 100.0  # → metres
    s.index = s.index.normalize()   # strip time component → date-level
    return s


def load_site_corrected(sid: str):
    """Load temp data and reproject sensor depths onto current-surface reference."""
    # ── Temperature ──────────────────────────────────────────────────────────
    csv = TEMP_DIR / f"{sid}_Tempconcatenated.csv"
    df  = pd.read_csv(csv, skiprows=4, index_col=0, parse_dates=True)
    valid = {}
    for col in df.columns:
        try:
            d = float(col)
            valid[d] = col        # keep all sensors, including above-surface negatives
        except (ValueError, TypeError):
            pass
    nominal_depths = np.array(sorted(valid.keys()))
    cols = [valid[d] for d in nominal_depths]
    data = df[cols].apply(pd.to_numeric, errors="coerce")
    data = data.loc[T_START:T_END]
    arr  = data.to_numpy(dtype=float)
    arr[~np.isfinite(arr)] = np.nan
    times = pd.DatetimeIndex(data.index)

    # ── PROMICE surface change ────────────────────────────────────────────────
    surf  = load_surface_change(sid)
    # Reindex to the temperature timestamps; forward-fill gaps, fallback to 0
    surf  = surf.reindex(times.normalize()).ffill().fillna(0.0)
    surf  = surf.to_numpy()   # shape (n_times,)

    # Linear interpolation onto fixed depth grid at each timestep
    n_t  = len(times)
    grid = np.full((n_t, len(DEPTH_GRID)), np.nan)
    for ti in range(n_t):
        adj = nominal_depths + surf[ti]
        row = arr[ti]
        vm  = np.isfinite(row)
        if vm.sum() < 2:
            continue
        f = interp1d(adj[vm], row[vm], kind="linear",
                     bounds_error=False, fill_value=np.nan)
        grid[ti] = f(DEPTH_GRID)

    # Shape (n_sensors, n_times): adjusted depth of every sensor over time
    sensor_depths = nominal_depths[:, None] + surf[None, :]
    return times, DEPTH_GRID, grid, sensor_depths


# ── Colorscale ────────────────────────────────────────────────────────────────
boundaries = list(np.arange(-20, -0.05, 2)) + [-0.05]
n_bins     = len(boundaries) - 1
base_cmap  = matplotlib.colormaps["turbo"].resampled(n_bins)
cmap       = base_cmap.copy()
cmap.set_over("gray")
norm       = mcolors.BoundaryNorm(boundaries, ncolors=n_bins)

# ── Pre-load ──────────────────────────────────────────────────────────────────
site_data = {}
for sid, label in SITES:
    try:
        site_data[(sid, label)] = load_site_corrected(sid)
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        site_data[(sid, label)] = None

last_time = max(d[0][-1] for d in site_data.values() if d is not None)
last_time = min(last_time, T_END)

x_min = matplotlib.dates.date2num(T_START.to_pydatetime())
x_max = matplotlib.dates.date2num(last_time.to_pydatetime())

# ── Figure ────────────────────────────────────────────────────────────────────
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

    times, depth_grid, grid, sensor_depths = loaded
    t_num = matplotlib.dates.date2num(times.to_pydatetime())

    Z = np.ma.masked_invalid(grid.T)
    ax.contourf(t_num, depth_grid, Z,
                levels=boundaries, cmap=cmap, norm=norm, extend="max")
    ax.contour(t_num, depth_grid, Z,
               levels=[-0.05], colors="white", linewidths=2.0)

    for sd in sensor_depths:
        d = np.where((sd >= 0) & (sd <= MAX_DEPTH), sd, np.nan)
        ax.plot(t_num, d, color="black", linewidth=0.6,
                linestyle="-", alpha=0.5, zorder=5)

    if ax is axes[0]:
        h_iso    = mlines.Line2D([], [], color="white", linewidth=2.0,
                                 label="−0.05 °C isotherm")
        h_sensor = mlines.Line2D([], [], color="black", linewidth=0.6,
                                 alpha=0.5, label="Sensor depth")
        ax.legend(handles=[h_iso, h_sensor], fontsize=FS * 0.4,
                  loc="center left", bbox_to_anchor=(1.01, 0.5),
                  framealpha=0.7, facecolor="dimgray", labelcolor="white",
                  edgecolor="none")

    ax.set_ylim(MAX_DEPTH, 0)
    ax.set_xlim(x_min, x_max)
    ax.text(0.01, 0.97, label, transform=ax.transAxes,
            fontsize=FS, fontweight="bold", va="top", ha="left",
            color="white", zorder=10,
            bbox=dict(boxstyle="round,pad=0.1",
                      facecolor="black", alpha=0.35, linewidth=0))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(2))
    ax.yaxis.set_minor_locator(matplotlib.ticker.MultipleLocator(1))
    ax.tick_params(axis="y", labelsize=FS * 0.56)

axes[-1].xaxis_date()
axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator())
axes[-1].xaxis.set_minor_locator(matplotlib.dates.WeekdayLocator())
axes[-1].tick_params(axis="x", labelsize=FS * 0.6, rotation=30)

fig.text(0.01, 0.5, "Depth (m)", va="center", rotation="vertical", fontsize=FS)

cbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes, orientation="vertical", fraction=0.018, pad=0.02,
    label="Temperature (°C)", extend="max",
    ticks=boundaries,
)
cbar.ax.tick_params(labelsize=FS * 0.8)
cbar.set_label("Temperature (°C)", fontsize=FS)
cbar.set_ticklabels([f"{b:.0f}" if b != -0.05 else "−0.05" for b in boundaries])

out = Path(__file__).parent / "observed_temps_25m_2023_depth_corrected.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
