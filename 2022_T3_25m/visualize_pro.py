"""
Visualize SNOWPACK PRO output for 2022-T3-25m.

Figure 1 – Temperature (four panels):
  1. Modelled firn temperature  (depth–time curtain)
  2. Observed firn temperature  (same colour scale)
  3. Model − Observation bias   (diverging scale centred at 0)
  4. Modelled liquid-water content (shows why T=0°C regions appear)

Figure 2 – Microstructure (four panels):
  1. Grain radius rg  (mm)
  2. Dendricity dd    (grain shape, 0=rounded / 1=dendritic)
  3. Sphericity sp    (0–1)
  4. Coordination number nc  (bonds per grain, connectivity)

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
OUT_FIG        = PROJECT_DIR / "output" / "2022-T3-25m_visualization.png"
OUT_FIG_MICRO  = PROJECT_DIR / "output" / "2022-T3-25m_microstructure.png"
OUT_FIG_GRAINS = PROJECT_DIR / "output" / "2022-T3-25m_grain_type.png"

# ---------------------------------------------------------------------------
# PRO parser
# ---------------------------------------------------------------------------
WANTED_CODES = {501, 503, 506, 509, 510, 512, 513}   # heights, T, LWC, sp, nc, rg, mk

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
def to_regular_grid(times, heights_list, values_list, depth_grid, interp_kind="linear"):
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
        f = interp1d(d, v, kind=interp_kind, bounds_error=False, fill_value=np.nan)
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
    cmap_T   = plt.cm.RdYlBu_r    # red=warm, blue=cold
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

    # -----------------------------------------------------------------------
    # Figure 2 – Microstructure
    # -----------------------------------------------------------------------
    print("Interpolating microstructure onto regular grid …")
    RG = to_regular_grid(times, pro[501], pro[512], depth_grid)   # grain radius [mm]
    MK = to_regular_grid(times, pro[501], pro[513], depth_grid,
                         interp_kind="nearest")                   # Swiss grain type code
    SP = to_regular_grid(times, pro[501], pro[509], depth_grid)   # sphericity
    NC = to_regular_grid(times, pro[501], pro[510], depth_grid)   # coordination number

    fig2, axes2 = plt.subplots(4, 1, figsize=(16, 20),
                               sharex=True, gridspec_kw={"hspace": 0.07})

    def curtain2(ax, data, norm, cmap, title, cbar_label,
                 extend="both", mask_nonpositive=False):
        masked = np.ma.masked_invalid(data)
        if mask_nonpositive:
            masked = np.ma.masked_where(masked <= 0, masked)
        pm = ax.pcolormesh(t_edges, d_edges, masked.T,
                           cmap=cmap, norm=norm, shading="flat")
        cb = fig2.colorbar(pm, ax=ax, pad=0.01, fraction=0.018, extend=extend)
        cb.set_label(cbar_label, fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=10)
        ax.set_ylim(12, 0)
        ax.yaxis.set_major_locator(plt.MultipleLocator(2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
        ax.grid(axis="y", which="major", color="grey", alpha=0.3, linewidth=0.5)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
        return pm

    from matplotlib.colors import Normalize

    # Panel 1 – grain radius
    rg_max = np.nanpercentile(RG[np.isfinite(RG)], 98) if np.any(np.isfinite(RG)) else 3.0
    curtain2(axes2[0], RG,
             Normalize(vmin=0, vmax=max(rg_max, 0.1)),
             plt.cm.turbo,
             "Grain radius  rg", "rg (mm)", extend="max")

    # Panel 2 – Swiss grain type code (all unique 3-digit codes, grouped by family)
    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    # Family colours (standard snow-science palette)
    # 2xx DF, 3xx RG, 4xx FC, 5xx DH, 7xx MF, 8xx IF, 9xx mixed
    mk_catalog = {
        # DF – Decomposing/Fragmented (tan/yellow)
        220: ("#d4b96a", "220 DF"),
        230: ("#c9a84c", "230 DF mixed"),
        231: ("#b89030", "231 DF small"),
        # RG – Rounded Grains (blue)
        321: ("#a8d8ea", "321 RG small"),
        330: ("#5dade2", "330 RG"),
        341: ("#1a6fa8", "341 RG large"),
        # FC – Faceted Crystals (orange)
        430: ("#f4a460", "430 FC"),
        431: ("#e8874a", "431 FC mixed"),
        440: ("#d4622a", "440 FC/RG"),
        470: ("#ffd580", "470 FCsf"),
        471: ("#ffbe33", "471 FCsf mixed"),
        472: ("#e6a000", "472 FCsf large"),
        490: ("#c06820", "490 FC mixed"),
        # DH – Depth Hoar (red/maroon)
        550: ("#e74c3c", "550 DH"),
        572: ("#c0392b", "572 DH mixed"),
        591: ("#922b21", "591 DH cup"),
        # MF – Melt Forms (purple/pink)
        751: ("#d7a8e0", "751 MFr"),
        752: ("#bb72cc", "752 MFr large"),
        770: ("#8e44ad", "770 MFcr"),
        772: ("#6c3483", "772 MFcr mixed"),
        791: ("#f1948a", "791 MF wet"),
        792: ("#e74c8b", "792 MF wet large"),
        # IF – Ice Formations (grey)
        880: ("#7f8c8d", "880 IF/ice"),
        # mixed/other
        951: ("#2ecc71", "951 FC/DH mixed"),
        990: ("#95a5a6", "990 mixed"),
    }

    all_codes  = sorted(mk_catalog.keys())
    all_colors = [mk_catalog[c][0] for c in all_codes]
    all_labels = [mk_catalog[c][1] for c in all_codes]
    mk_cmap    = ListedColormap(all_colors)

    MK_idx = np.full_like(MK, np.nan)
    for i, code in enumerate(all_codes):
        MK_idx = np.where(np.abs(MK - code) < 0.5, float(i), MK_idx)

    axes2[1].pcolormesh(t_edges, d_edges,
                        np.ma.masked_invalid(MK_idx).T,
                        cmap=mk_cmap, vmin=-0.5, vmax=len(all_codes) - 0.5,
                        shading="flat")
    patches = [mpatches.Patch(color=all_colors[i], label=all_labels[i])
               for i in range(len(all_codes))]
    axes2[1].legend(handles=patches, loc="lower right", fontsize=7,
                    framealpha=0.8, ncol=2)
    axes2[1].set_ylabel("Depth (m)", fontsize=10)
    axes2[1].set_ylim(25, 0)
    axes2[1].yaxis.set_major_locator(plt.MultipleLocator(5))
    axes2[1].yaxis.set_minor_locator(plt.MultipleLocator(1))
    axes2[1].grid(axis="y", which="major", color="grey", alpha=0.3, linewidth=0.5)
    axes2[1].set_title("Swiss grain type code  mk  (513)", fontsize=11, fontweight="bold", loc="left")

    # Panel 3 – sphericity (0 = faceted, 1 = spherical)
    curtain2(axes2[2], SP,
             Normalize(vmin=0, vmax=1),
             plt.cm.RdPu,
             "Sphericity  sp  (0 = faceted / 1 = spherical)", "sp (–)", extend="neither")

    # Panel 4 – coordination number (connectivity)
    nc_max = np.nanpercentile(NC[np.isfinite(NC)], 98) if np.any(np.isfinite(NC)) else 20.0
    curtain2(axes2[3], NC,
             Normalize(vmin=0, vmax=max(nc_max, 1.0)),
             plt.cm.viridis,
             "Coordination number  nc  (bonds per grain, connectivity)", "nc (–)",
             extend="neither")

    axes2[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes2[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes2[-1].xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(axes2[-1].xaxis.get_majorticklabels(),
             rotation=30, ha="right", fontsize=9)
    for ax in axes2:
        ax.set_xlim(t_mpl[0], t_mpl[-1])

    fig2.suptitle("SNOWPACK – 2022 T3 25 m  microstructure  (RICHARDSEQUATION scheme)",
                  fontsize=13, fontweight="bold", y=0.995)

    fig2.savefig(OUT_FIG_MICRO, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FIG_MICRO}")

    # -----------------------------------------------------------------------
    # Figure 3 – Grain type only, with accurate absolute depths
    # Surface is flat at top (depth = 0); soil appears at varying depth
    # at the bottom, revealing true snowpack thickness changes.
    # -----------------------------------------------------------------------
    print("Building grain type figure with accurate depths …")

    # Per-timestep total snowpack depth (cm → m)
    soil_depth_m = np.array([h.max() / 100.0 for h in pro[501]])

    # Deep enough grid to cover all timesteps (round up to next 0.5 m)
    max_depth = np.ceil(soil_depth_m.max() * 2) / 2.0   # e.g. 29.5 m
    depth_grid_full = np.arange(0, max_depth + 0.05, 0.05)

    MK_full = to_regular_grid(times, pro[501], pro[513], depth_grid_full,
                              interp_kind="nearest")

    # Mask each column below its soil depth
    for ti in range(len(times)):
        below_soil = depth_grid_full > soil_depth_m[ti]
        MK_full[ti, below_soil] = np.nan

    # Map to colour indices (same catalog as before)
    MK_full_idx = np.full_like(MK_full, np.nan)
    for i, code in enumerate(all_codes):
        MK_full_idx = np.where(np.abs(MK_full - code) < 0.5,
                               float(i), MK_full_idx)

    # Edges for pcolormesh
    d_edges_full = np.concatenate([
        [depth_grid_full[0]],
        0.5 * (depth_grid_full[:-1] + depth_grid_full[1:]),
        [depth_grid_full[-1]],
    ])

    fig3, ax3 = plt.subplots(figsize=(16, 7))
    fig3.subplots_adjust(right=0.72)   # leave room for legend on the right

    ax3.pcolormesh(t_edges, d_edges_full,
                   np.ma.masked_invalid(MK_full_idx).T,
                   cmap=mk_cmap, vmin=-0.5, vmax=len(all_codes) - 0.5,
                   shading="flat")

    # Soil interface line
    ax3.plot(t_mpl, soil_depth_m, color="k", linewidth=1.5)

    ax3.set_ylabel("Depth below surface (m)", fontsize=11)
    ax3.set_ylim(12, 0)
    ax3.yaxis.set_major_locator(plt.MultipleLocator(2))
    ax3.yaxis.set_minor_locator(plt.MultipleLocator(0.5))
    ax3.grid(axis="y", which="major", color="grey", alpha=0.3, linewidth=0.5)
    ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax3.xaxis.set_minor_locator(mdates.WeekdayLocator())
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)
    ax3.set_xlim(t_mpl[0], t_mpl[-1])
    ax3.set_title("Swiss grain type  mk  (513) — surface flat at top, "
                  "soil depth variation at bottom",
                  fontsize=11, fontweight="bold", loc="left")

    # Legend outside axes, grouped by grain type family with full names
    # Family headers use a blank patch as a spacer
    family_groups = [
        ("DF – Decomposing & Fragmented particles", [220, 230, 231]),
        ("RG – Rounded Grains",                     [321, 330, 341]),
        ("FC – Faceted Crystals",                   [430, 431, 440, 470, 471, 472, 490]),
        ("DH – Depth Hoar",                         [550, 572, 591]),
        ("MF – Melt Forms",                         [751, 752, 770, 772, 791, 792]),
        ("IF – Ice Formations",                     [880]),
        ("Mixed / other",                           [951, 990]),
    ]

    legend_handles = [
        plt.Line2D([0], [0], color="k", linewidth=1.5, label="— Soil interface")
    ]
    for family_name, codes in family_groups:
        legend_handles.append(
            mpatches.Patch(color="none", label=f"\n{family_name}")
        )
        for code in codes:
            if code in mk_catalog:
                color, label = mk_catalog[code]
                legend_handles.append(mpatches.Patch(color=color, label=f"  {label}"))

    ax3.legend(handles=legend_handles, loc="upper left",
               bbox_to_anchor=(1.01, 1.0), borderaxespad=0,
               fontsize=8, framealpha=0.9, handlelength=1.2)

    fig3.suptitle("SNOWPACK – 2022 T3 25 m  grain type  (RICHARDSEQUATION scheme)",
                  fontsize=13, fontweight="bold")

    fig3.savefig(OUT_FIG_GRAINS, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FIG_GRAINS}")


if __name__ == "__main__":
    main()
