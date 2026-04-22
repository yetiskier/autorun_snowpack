"""
Observed vs modelled temperature for 2023_UP18_25m.
Observed: Tempconcatenated CSV with PROMICE depth correction.
Modelled: SNOWPACK .pro output, depth from surface.
Output: UP18_obs_vs_model.png
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

SCRIPT_DIR = Path(__file__).parent
TEMP_DIR   = SCRIPT_DIR / "AllCoreDataCommonFormat/Concatenated_Temperature_files"
PROM_DIR   = SCRIPT_DIR / "AllCoreDataCommonFormat/Depth_change_estimate/PROMICE"
PRO_PATH   = SCRIPT_DIR / "2023_UP18_25m/output/2023-UP18-25m_TEMP_ASSIM_RUN.pro"

SID        = "2023_UP18_25m"
MAX_DEPTH  = 10.0
DEPTH_GRID = np.arange(0.0, MAX_DEPTH + 0.05, 0.05)
FS         = 40

# ── Colorscale ────────────────────────────────────────────────────────────────
boundaries = list(np.arange(-20, -0.05, 2)) + [-0.05]
n_bins     = len(boundaries) - 1
cmap       = matplotlib.colormaps["turbo"].resampled(n_bins).copy()
cmap.set_over("gray")
norm       = mcolors.BoundaryNorm(boundaries, ncolors=n_bins)


# ── Load observed (with PROMICE depth correction) ─────────────────────────────
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
    return times, grid, sensor_depths, arr


# ── Parse PRO file ────────────────────────────────────────────────────────────
def parse_pro(path):
    times, pro = [], {}
    current_time = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("0500,"):
                raw = line[5:].strip()
                if raw == "Date":
                    continue
                try:
                    current_time = pd.to_datetime(raw, dayfirst=True)
                    times.append(current_time)
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
                    except ValueError:
                        continue
                    pro.setdefault(code, []).append(
                        np.array([float(v) for v in parts[2:2+n]]))
                elif code == 503:
                    try:
                        n = int(parts[1])
                    except ValueError:
                        continue
                    pro.setdefault(code, []).append(
                        np.array([float(v) for v in parts[2:2+n]]))
    return pd.DatetimeIndex(times), pro

def load_modelled():
    times, pro = parse_pro(PRO_PATH)
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


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading observed …")
obs_times, obs_grid, sensor_depths, obs_arr = load_observed(SID)
print("Loading modelled …")
mod_times, mod_grid = load_modelled()

# Zoom to same window as the multi-site figure
t_start = pd.Timestamp("2023-05-29")
t_end   = pd.Timestamp("2023-10-01")
obs_mask = (obs_times >= t_start) & (obs_times <= t_end)
mod_mask = (mod_times >= t_start) & (mod_times <= t_end)
obs_times = obs_times[obs_mask];  obs_grid = obs_grid[obs_mask];  obs_arr = obs_arr[obs_mask]
mod_times = mod_times[mod_mask];  mod_grid = mod_grid[mod_mask]

# Sensor depth time mask to match
sd_times_full = pd.DatetimeIndex(
    pd.read_csv(TEMP_DIR / f"{SID}_Tempconcatenated.csv",
                skiprows=4, index_col=0, parse_dates=True)
    .resample("1h").mean().index
)
sd_mask = (sd_times_full >= t_start) & (sd_times_full <= t_end)
sensor_depths = sensor_depths[:, sd_mask]
sd_t_num = matplotlib.dates.date2num(sd_times_full[sd_mask].to_pydatetime())

obs_t = matplotlib.dates.date2num(obs_times.to_pydatetime())
mod_t = matplotlib.dates.date2num(mod_times.to_pydatetime())
x_min = min(obs_t[0],  mod_t[0])
x_max = max(obs_t[-1], mod_t[-1])

# ── Figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(20, 16),
                         sharex=True,
                         gridspec_kw={"hspace": 0.08})

titles = ["Observed", "Modelled (SNOWPACK)"]
grids  = [(obs_t, obs_grid), (mod_t, mod_grid)]

for ax, title, (t_num, grid) in zip(axes, titles, grids):
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

axes[-1].xaxis_date()
axes[-1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %Y"))
axes[-1].xaxis.set_major_locator(matplotlib.dates.MonthLocator())
axes[-1].xaxis.set_minor_locator(matplotlib.dates.WeekdayLocator())
axes[-1].tick_params(axis="x", labelsize=FS * 0.7, rotation=30)

# Sensor depth lines on interpolated observed panel only
for sd in sensor_depths:
    d = np.where((sd >= 0) & (sd <= MAX_DEPTH), sd, np.nan)
    axes[0].plot(sd_t_num, d, color="black", linewidth=0.5,
                 linestyle="-", alpha=0.4, zorder=5)

fig.suptitle("UP18 (2109 m)  —  observed vs modelled firn temperature",
             fontsize=FS, y=1.01)

# Legend
h_iso    = mlines.Line2D([], [], color="white", linewidth=1.5, label="−0.05 °C isotherm")
h_sensor = mlines.Line2D([], [], color="black", linewidth=0.5, alpha=0.4, label="Sensor depth")
axes[0].legend(handles=[h_iso, h_sensor], fontsize=FS * 0.5,
               loc="center left", bbox_to_anchor=(1.01, 0.5),
               framealpha=0.7, facecolor="dimgray", labelcolor="white",
               edgecolor="none")

# Shared colourbar
cbar = fig.colorbar(
    matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes, orientation="vertical", fraction=0.015, pad=0.02,
    label="Temperature (°C)", extend="max", ticks=boundaries,
)
cbar.ax.tick_params(labelsize=FS * 0.7)
cbar.set_label("Temperature (°C)", fontsize=FS * 0.8)
cbar.set_ticklabels([f"{b:.0f}" if b != -0.05 else "−0.05" for b in boundaries])

out = SCRIPT_DIR / "UP18_obs_vs_model.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved → {out}")
