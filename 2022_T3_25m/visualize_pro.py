"""
Visualize SNOWPACK PRO output for 2022-T3-25m.

Four panels:
  1. Modelled firn temperature  (depth–time curtain)
  2. Observed firn temperature  (same colour scale)
  3. Model − Observation bias   (diverging scale centred at 0)
  4. Modelled liquid-water content (shows why T=0°C regions appear)

Run from the project directory:
    python visualize_pro.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import BoundaryNorm, TwoSlopeNorm, LogNorm
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent
PRO_FILE    = PROJECT_DIR / "output" / "2022-T3-25m_TEMP_ASSIM_RUN.pro"
OBS_FILE    = (PROJECT_DIR.parent.parent
               / "AllCoreDataCommonFormat"
               / "Concatenated_Temperature_files"
               / "2022_T3_25m_Tempconcatenated.csv")
OUT_FIG     = PROJECT_DIR / "output" / "2022-T3-25m_visualization.png"

# ---------------------------------------------------------------------------
# PRO parser
# ---------------------------------------------------------------------------
WANTED_CODES = {501, 503, 506}   # heights, temperature, LWC

def parse_pro(path: Path) -> dict:
    data   = {c: [] for c in WANTED_CODES}
    times  = []
    current = {}
    in_data = False
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("[DATA]"):
                in_data = True
                continue
            if not in_data:
                continue
            code_str, _, rest = line.partition(",")
            try:
                code = int(code_str)
            except ValueError:
                continue
            if code == 500:
                if current:
                    for c in WANTED_CODES:
                        data[c].append(current.get(c, np.array([])))
                current = {}
                times.append(pd.to_datetime(rest.strip(), dayfirst=True))
            elif code in WANTED_CODES:
                vals = rest.split(",")
                current[code] = np.array([float(v) for v in vals[1:]])
    if current:
        for c in WANTED_CODES:
            data[c].append(current.get(c, np.array([])))
    return {"times": times, **data}


# ---------------------------------------------------------------------------
# Interpolate ragged profiles onto a fixed depth grid
# ---------------------------------------------------------------------------
def to_regular_grid(times, heights_list, values_list, depth_grid):
    nt, nd = len(times), len(depth_grid)
    grid   = np.full((nt, nd), np.nan)
    for i, (h, v) in enumerate(zip(heights_list, values_list)):
        if len(h) == 0 or len(v) == 0:
            continue
        depth_m = (h.max() - h) / 100.0
        order   = np.argsort(depth_m)
        d, v    = depth_m[order], v[order]
        _, uniq = np.unique(d, return_index=True)
        d, v    = d[uniq], v[uniq]
        if len(d) < 2:
            continue
        f = interp1d(d, v, kind="linear", bounds_error=False, fill_value=np.nan)
        grid[i] = f(depth_grid)
    return grid


# ---------------------------------------------------------------------------
# Load observations
# Observation columns are labelled by installation depth (m below surface at
# drill time).  We use them as-is; sensor depths relative to the evolving
# surface are not corrected here.
# ---------------------------------------------------------------------------
def load_observations(path: Path):
    obs = pd.read_csv(path, skiprows=3, index_col=0, parse_dates=True)
    obs.columns = pd.to_numeric(obs.columns, errors="coerce")
    obs = obs.loc[:, obs.columns >= 0]      # drop above-surface sensors
    return obs


# ---------------------------------------------------------------------------
# Interpolate observations onto the same depth grid
# ---------------------------------------------------------------------------
def obs_to_grid(obs: pd.DataFrame, depth_grid: np.ndarray,
                time_grid: pd.DatetimeIndex) -> np.ndarray:
    """Interpolate obs (irregular depth columns) onto depth_grid,
    then resample to hourly time_grid by linear interpolation."""
    obs_depths = obs.columns.values.astype(float)
    nt = len(time_grid)
    nd = len(depth_grid)
    grid = np.full((nt, nd), np.nan)

    # First: interpolate each obs row onto depth_grid
    obs_on_depth = np.full((len(obs), nd), np.nan)
    for i, row in enumerate(obs.values):
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        f = interp1d(obs_depths[mask], row[mask].astype(float),
                     kind="linear", bounds_error=False, fill_value=np.nan)
        obs_on_depth[i] = f(depth_grid)

    obs_df = pd.DataFrame(obs_on_depth, index=obs.index)

    # Reindex + interpolate to the model's hourly time axis
    combined = obs_df.reindex(obs_df.index.union(time_grid)).sort_index()
    combined = combined.interpolate(method="time", limit=3)   # fill ≤3-h gaps
    grid = combined.reindex(time_grid).values
    return grid


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Parsing PRO file …")
    pro   = parse_pro(PRO_FILE)
    times = pd.DatetimeIndex(pro["times"])
    print(f"  {len(times)} timesteps  {times[0]} → {times[-1]}")

    depth_grid = np.arange(0, 25.05, 0.05)   # 0–25 m, 5 cm steps

    print("Interpolating model onto regular grid …")
    T_mod  = to_regular_grid(times, pro[501], pro[503], depth_grid)
    LWC    = to_regular_grid(times, pro[501], pro[506], depth_grid)

    print("Loading & regridding observations …")
    obs    = load_observations(OBS_FILE) if OBS_FILE.exists() else None
    T_obs  = obs_to_grid(obs, depth_grid, times) if obs is not None else None

    # Bias: model − obs  (only where both exist)
    if T_obs is not None:
        bias = T_mod - T_obs
    else:
        bias = None

    # ---- Matplotlib dates ----
    t_mpl = mdates.date2num(times.to_pydatetime())
    t_edges = np.concatenate([[t_mpl[0] - 0.5*(t_mpl[1]-t_mpl[0])],
                               0.5*(t_mpl[:-1]+t_mpl[1:]),
                               [t_mpl[-1] + 0.5*(t_mpl[-1]-t_mpl[-2])]])
    d_edges = np.concatenate([[depth_grid[0]],
                               0.5*(depth_grid[:-1]+depth_grid[1:]),
                               [depth_grid[-1]]])

    # ---- Colour scales ----
    # Temperature: -24 to 0 °C, centred on -12 °C
    T_levels = np.arange(-24, 1, 2)
    cmap_T   = plt.cm.RdYlBu      # blue=cold, red=warm; centred white ≈ -12
    norm_T   = BoundaryNorm(T_levels, cmap_T.N)

    # Bias: ±10 °C, diverging
    bias_levels = np.arange(-10, 11, 2)
    cmap_bias   = plt.cm.RdBu_r
    norm_bias   = BoundaryNorm(bias_levels, cmap_bias.N, extend="both")

    # LWC: log scale 0.01–10 %, values ≤ 0 masked before plotting
    cmap_lwc = plt.cm.Blues
    norm_lwc = LogNorm(vmin=0.01, vmax=10)

    # ---- Figure ----
    print("Plotting …")
    nrows = 4 if bias is not None else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(16, 5*nrows),
                             sharex=True, gridspec_kw={"hspace": 0.07})

    def curtain(ax, data, norm, cmap, title, cbar_label, extend="both", mask_nonpositive=False):
        masked = np.ma.masked_invalid(data)
        if mask_nonpositive:
            masked = np.ma.masked_where(masked <= 0, masked)
        pm = ax.pcolormesh(t_edges, d_edges, masked.T,
                           cmap=cmap, norm=norm, shading="flat")
        cb = fig.colorbar(pm, ax=ax, pad=0.01, fraction=0.018, extend=extend)
        cb.set_label(cbar_label, fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=10)
        ax.set_ylim(25, 0)
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.grid(axis="y", which="major", color="grey", alpha=0.3, linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        return pm

    # Panel 1 – modelled T
    curtain(axes[0], T_mod, norm_T, cmap_T,
            "Modelled firn temperature", "Temperature (°C)")
    # 0 °C isotherm
    axes[0].contour(t_mpl, depth_grid, np.ma.masked_invalid(T_mod).T,
                    levels=[0], colors="k", linewidths=1.2, linestyles="--")
    axes[0].text(0.01, 0.04, "dashed = 0 °C isotherm",
                 transform=axes[0].transAxes, fontsize=8, color="k")

    # Panel 2 – observed T
    if T_obs is not None:
        curtain(axes[1], T_obs, norm_T, cmap_T,
                "Observed firn temperature", "Temperature (°C)")
        axes[1].contour(t_mpl, depth_grid, np.ma.masked_invalid(T_obs).T,
                        levels=[0], colors="k", linewidths=1.2, linestyles="--")
    else:
        axes[1].text(0.5, 0.5, "Observation file not found",
                     ha="center", transform=axes[1].transAxes)

    # Panel 3 – bias
    if bias is not None:
        curtain(axes[2], bias, norm_bias, cmap_bias,
                "Model − Observation  (warm = model too warm)", "Bias (°C)")
        axes[2].contour(t_mpl, depth_grid, np.ma.masked_invalid(bias).T,
                        levels=[0], colors="k", linewidths=0.8, linestyles="-")

    # Panel 4 – LWC
    ax_lwc = axes[3] if bias is not None else axes[2]
    curtain(ax_lwc, LWC, norm_lwc, cmap_lwc,
            "Modelled liquid water content  (explains 0 °C saturation)", "LWC (%)",
            extend="max", mask_nonpositive=True)

    # ---- Shared x-axis ----
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(axes[-1].xaxis.get_majorticklabels(),
             rotation=30, ha="right", fontsize=9)
    for ax in axes:
        ax.set_xlim(t_mpl[0], t_mpl[-1])

    fig.suptitle("SNOWPACK – 2022 T3 25 m  (RICHARDSEQUATION scheme)",
                 fontsize=13, fontweight="bold", y=0.995)

    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FIG}")


if __name__ == "__main__":
    main()
