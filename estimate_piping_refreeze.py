"""
Estimate the mass of water refrozen by piping events at depths below the
SNOWPACK-predicted wetting front.

Physics: water arriving via pipes freezes below the SNOWPACK wetting front,
releasing latent heat and producing a positive temperature anomaly
    ΔT(z,t) = T_obs(z,t) − T_model(z,t)
at those depths that persists until the heat diffuses away.

PRIMARY ESTIMATE — seasonal pre/post comparison:
  For each melt season, compare the mean model-obs temperature anomaly at
  depths below the maximum SNOWPACK wetting front:
    • before melt season onset  (pre-melt window)
    • after melt season ends and firn has refrozen  (post-melt window)
  The increase δ(z) = ΔT_post(z) − ΔT_pre(z) at those depths represents
  latent heat from piping-delivered water.

      m_refreeze = Σ_z  ρ(z) · c_p · max(δ(z), 0) · dz  /  L_f   [kg m⁻²]

  This automatically removes the systematic sensor/model bias (it cancels in
  the difference) and is measured after full refreezing, so there is no
  partial-refreeze ambiguity for seasons where observations continue through
  winter.

DIAGNOSTIC — instantaneous heat anomaly Q(t):
  Provides a real-time view of excess heat stored below the wetting front
  (useful for identifying individual events and checking the seasonal estimate).

Note on partial refreezing: only a concern where observations end during the
melt season.  For all other seasons the comparison is valid.

Usage:
    python estimate_piping_refreeze.py --site T3 --year 2022 --depth 25
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import TwoSlopeNorm
from scipy.interpolate import interp1d

# ── Physical constants ──────────────────────────────────────────────────────
L_F           = 334_000.0   # latent heat of fusion            [J kg⁻¹]
C_P_ICE       = 2_090.0     # specific heat of ice             [J kg⁻¹ K⁻¹]
LWC_THRESHOLD = 0.1         # minimum LWC (% vol) to count as wetted  [%]
MIN_WF_DEPTH  = 2.0         # minimum wetting-front depth (m) for diagnostic Q(t)
WINDOW_STEPS     = 14 * 48   # pre/post averaging window target: 14 days × 48 half-hours
MIN_WINDOW_STEPS =  7 * 48   # minimum acceptable window length (7 days); shorter → skip

SCRIPT_DIR = Path(__file__).resolve().parent


# ── Path helpers ─────────────────────────────────────────────────────────────
def build_paths(year: int, site: str, depth: int) -> dict:
    site_id  = f"{year}_{site}_{depth}m"
    pro_name = f"{year}-{site}-{depth}m_TEMP_ASSIM_RUN.pro"
    out_dir  = SCRIPT_DIR / site_id / "output"
    obs_dir  = SCRIPT_DIR / "AllCoreDataCommonFormat" / "Concatenated_Temperature_files"
    return {
        "pro_file":   out_dir / pro_name,
        "obs_file":   obs_dir / f"{site_id}_Tempconcatenated.csv",
        "out_csv":    out_dir / f"{site_id}_piping_refreeze.csv",
        "out_fig":    out_dir / f"{site_id}_piping_refreeze.png",
        "site_label": f"{year} {site} {depth} m",
        "out_dir":    out_dir,
    }


# ── PRO parser ───────────────────────────────────────────────────────────────
CODES_NEEDED = {501, 502, 503, 506}   # heights, density, temperature, LWC

def parse_pro(path: Path) -> dict:
    data    = {c: [] for c in CODES_NEEDED}
    times   = []
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
                    for c in CODES_NEEDED:
                        data[c].append(current.get(c, np.array([])))
                current = {}
                times.append(pd.to_datetime(rest.strip(), dayfirst=True))
            elif code in CODES_NEEDED:
                vals = rest.split(",")
                current[code] = np.array([float(v) for v in vals[1:]])
    if current:
        for c in CODES_NEEDED:
            data[c].append(current.get(c, np.array([])))
    return {"times": times, **data}


# ── Regridding ───────────────────────────────────────────────────────────────
def to_regular_grid(times, heights_list, values_list, depth_grid):
    nt, nd = len(times), len(depth_grid)
    grid   = np.full((nt, nd), np.nan)
    for i, (h, v) in enumerate(zip(heights_list, values_list)):
        if len(h) == 0 or len(v) == 0:
            continue
        depth_m = (h.max() - h) / 100.0
        order   = np.argsort(depth_m)
        d, v_   = depth_m[order], v[order]
        _, uniq = np.unique(d, return_index=True)
        d, v_   = d[uniq], v_[uniq]
        if len(d) < 2:
            continue
        f       = interp1d(d, v_, bounds_error=False, fill_value=np.nan)
        grid[i] = f(depth_grid)
    return grid


def load_observations(path: Path) -> pd.DataFrame:
    obs = pd.read_csv(path, skiprows=3, index_col=0, parse_dates=True)
    obs.columns = pd.to_numeric(obs.columns, errors="coerce")
    obs = obs.loc[:, obs.columns >= 0]
    return obs


def obs_to_grid(obs: pd.DataFrame, depth_grid: np.ndarray,
                time_grid: pd.DatetimeIndex) -> np.ndarray:
    obs_depths   = obs.columns.values.astype(float)
    nd           = len(depth_grid)
    obs_on_depth = np.full((len(obs), nd), np.nan)
    for i, row in enumerate(obs.values):
        mask = np.isfinite(row)
        if mask.sum() < 2:
            continue
        f = interp1d(obs_depths[mask], row[mask].astype(float),
                     bounds_error=False, fill_value=np.nan)
        obs_on_depth[i] = f(depth_grid)
    obs_df   = pd.DataFrame(obs_on_depth, index=obs.index)
    combined = obs_df.reindex(obs_df.index.union(time_grid)).sort_index()
    combined = combined.interpolate(method="time", limit=3)
    return combined.reindex(time_grid).values


# ── Wetting-front depth ───────────────────────────────────────────────────────
def wetting_front_depth(lwc_row: np.ndarray, depth_grid: np.ndarray) -> float:
    wet = np.where(lwc_row > LWC_THRESHOLD)[0]
    return depth_grid[wet[-1]] if len(wet) > 0 else np.nan


# ── Diagnostic: instantaneous Q(t) below wetting front ───────────────────────
def compute_Q_timeseries(T_mod, T_obs, LWC, RHO, depth_grid, dz):
    """
    Compute the baseline-corrected anomalous heat Q(t) stored below the
    wetting front at each timestep.  Used for visualization and event detection.
    """
    # Baseline: median ΔT during non-melt periods (no LWC anywhere)
    no_melt  = np.all((LWC <= LWC_THRESHOLD) | ~np.isfinite(LWC), axis=1)
    dT_all   = T_obs - T_mod
    baseline = np.nanmedian(dT_all[no_melt, :], axis=0) if no_melt.sum() >= 10 \
               else np.zeros(len(depth_grid))
    dT_corr  = dT_all - baseline[np.newaxis, :]

    nt       = T_mod.shape[0]
    wf_arr   = np.full(nt, np.nan)
    Q        = np.zeros(nt)

    for i in range(nt):
        wf = wetting_front_depth(LWC[i], depth_grid)
        wf_arr[i] = wf
        if np.isnan(wf) or wf < MIN_WF_DEPTH:
            continue
        below = depth_grid > wf
        dT    = dT_corr[i, below]
        rho   = RHO[i, below]
        good  = np.isfinite(dT) & np.isfinite(rho) & (dT > 0)
        if np.any(good):
            Q[i] = np.nansum(rho[good] * C_P_ICE * dT[good] * dz)

    return Q, wf_arr, dT_corr, baseline


# ── Primary estimate: seasonal pre/post comparison ───────────────────────────
def compute_seasonal_refreeze(T_mod, T_obs, LWC, RHO, depth_grid, dz, times):
    """
    Per-season piping refreeze estimate.

    For each year in the record, find:
      • pre-melt window:  up to WINDOW_STEPS half-hours before first LWC onset
      • post-melt window: starting after last LWC disappears

    The increase in (T_obs − T_model) between post and pre, at depths below
    the season's maximum wetting-front depth, represents latent heat deposited
    by piping events.

    Returns a list of per-season result dicts.
    """
    dT_all = T_obs - T_mod
    years  = sorted(set(times.year.tolist()))
    seasons = []

    for year in years:
        yr_idx  = np.where(times.year == year)[0]
        yr_lwc  = LWC[yr_idx]

        # Melt-active timesteps for this year
        melt_active = np.any(yr_lwc > LWC_THRESHOLD, axis=1)
        melt_steps  = np.where(melt_active)[0]
        if len(melt_steps) == 0:
            continue

        first_melt = melt_steps[0]
        last_melt  = melt_steps[-1]

        # ── Pre-melt window ───────────────────────────────────────────────
        pre_end   = yr_idx[first_melt]
        pre_start = max(0, pre_end - WINDOW_STEPS)
        n_pre = pre_end - pre_start
        if n_pre < MIN_WINDOW_STEPS:
            note = (f"{year}: pre-melt window only {n_pre} steps ({n_pre//48} days) "
                    f"— likely a drill-year with no pre-melt data; season skipped.")
            print(f"  Note: {note}")
            seasons.append({"year": year, "skipped": True, "note": note})
            continue

        # ── Post-melt window ──────────────────────────────────────────────
        post_start = yr_idx[last_melt] + 1
        post_end   = min(len(times), post_start + WINDOW_STEPS)
        n_post     = post_end - post_start
        if n_post < MIN_WINDOW_STEPS:
            note = (f"{year}: post-melt window only {n_post} steps ({n_post//48} days) "
                    f"— observation record ends during/shortly after melt; season skipped.")
            print(f"  Note: {note}")
            seasons.append({"year": year, "skipped": True, "note": note})
            continue

        post_lwc_any = np.any(LWC[post_start:post_end] > LWC_THRESHOLD)
        if post_lwc_any:
            note = f"{year}: post-melt window still has LWC; firn not fully refrozen — season flagged."
            print(f"  Warning: {note}")
            partial = True
        else:
            partial = False
            note    = ""

        # ── Mean anomaly in each window ───────────────────────────────────
        mean_dT_pre  = np.nanmean(dT_all[pre_start:pre_end],   axis=0)
        mean_dT_post = np.nanmean(dT_all[post_start:post_end], axis=0)
        delta_dT     = mean_dT_post - mean_dT_pre   # change due to melt season

        # ── Maximum wetting-front depth this season ───────────────────────
        season_wf = [wetting_front_depth(yr_lwc[i], depth_grid)
                     for i in range(len(yr_idx))]
        max_wf = np.nanmax([v for v in season_wf if not np.isnan(v)], initial=0.0)

        # ── Integrate piping heat below max wetting front ─────────────────
        below    = depth_grid > max_wf
        rho_mean = np.nanmean(RHO[yr_idx], axis=0)
        pipe_dT  = np.where(below & (delta_dT > 0) & np.isfinite(delta_dT) &
                             np.isfinite(rho_mean), delta_dT, 0.0)

        Q_season = float(np.sum(rho_mean * C_P_ICE * pipe_dT * dz))
        m_season = Q_season / L_F   # kg m⁻² = mm w.e.

        seasons.append({
            "year":           year,
            "skipped":        False,
            "partial":        partial,
            "note":           note,
            "max_wf_depth_m": float(max_wf),
            "Q_J_m2":         Q_season,
            "m_mm_we":        m_season,
            "delta_dT":       delta_dT,
            "mean_dT_pre":    mean_dT_pre,
            "mean_dT_post":   mean_dT_post,
            "pre_window":     (times[pre_start], times[pre_end - 1]),
            "post_window":    (times[post_start], times[min(post_end, len(times)) - 1]),
        })

        pre_str  = f"{times[pre_start].date()} to {times[pre_end-1].date()}"
        post_str = f"{times[post_start].date()} to {times[min(post_end,len(times))-1].date()}"
        flag     = " [PARTIAL]" if partial else ""
        print(f"  {year}: max wf = {max_wf:.1f} m | pre: {pre_str} | post: {post_str}"
              f" | m = {m_season:.1f} mm w.e.{flag}")

    return seasons


# ── Plotting ──────────────────────────────────────────────────────────────────
def make_figure(times, depth_grid, T_mod, T_obs, LWC,
                Q, wf_arr, dT_corr, seasons, site_label, out_path):

    # Baseline-corrected piping anomaly below wetting front
    dT_pipe = np.full_like(dT_corr, np.nan)
    for i, wf in enumerate(wf_arr):
        if np.isnan(wf) or wf < MIN_WF_DEPTH:
            continue
        below = depth_grid > wf
        pos   = np.where(below)[0]
        vals  = dT_corr[i, pos]
        dT_pipe[i, pos] = np.where(vals > 0, vals, np.nan)

    t_mpl   = mdates.date2num(times.to_pydatetime())
    t_edges = np.concatenate([[t_mpl[0] - 0.5*(t_mpl[1]-t_mpl[0])],
                               0.5*(t_mpl[:-1]+t_mpl[1:]),
                               [t_mpl[-1] + 0.5*(t_mpl[-1]-t_mpl[-2])]])
    d_edges = np.concatenate([[depth_grid[0]],
                               0.5*(depth_grid[:-1]+depth_grid[1:]),
                               [depth_grid[-1]]])

    fig, axes = plt.subplots(3, 1, figsize=(16, 14),
                             gridspec_kw={"hspace": 0.10,
                                         "height_ratios": [2, 2, 1]})

    # ── Panel 1: baseline-corrected full bias map ────────────────────────
    ax  = axes[0]
    vmax = max(np.nanpercentile(np.abs(dT_corr[np.isfinite(dT_corr)]), 98)
               if np.any(np.isfinite(dT_corr)) else 5.0, 0.5)
    pm1 = ax.pcolormesh(t_edges, d_edges,
                        np.ma.masked_invalid(dT_corr).T,
                        cmap="RdBu_r",
                        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax),
                        shading="flat")
    ax.plot(t_mpl, wf_arr, color="lime", lw=1.5, label="SNOWPACK wetting front")
    fig.colorbar(pm1, ax=ax, pad=0.01, fraction=0.018, extend="both"
                 ).set_label("ΔT (baseline-corrected) (°C)", fontsize=9)
    ax.set_ylabel("Depth (m)", fontsize=10)
    ax.set_ylim(depth_grid[-1], 0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis="y", color="grey", alpha=0.3, lw=0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("T_obs − T_model (winter-baseline-corrected)", fontsize=11,
                 fontweight="bold", loc="left")

    # ── Panel 2: piping anomaly below wetting front ─────────────────────
    ax  = axes[1]
    vmax2 = max(np.nanpercentile(dT_pipe[np.isfinite(dT_pipe)], 98)
                if np.any(np.isfinite(dT_pipe)) else 2.0, 0.1)
    pm2 = ax.pcolormesh(t_edges, d_edges,
                        np.ma.masked_invalid(dT_pipe).T,
                        cmap="hot_r", vmin=0, vmax=vmax2, shading="flat")
    ax.plot(t_mpl, wf_arr, color="cyan", lw=1.5, label="SNOWPACK wetting front")
    fig.colorbar(pm2, ax=ax, pad=0.01, fraction=0.018, extend="max"
                 ).set_label("Piping anomaly\n(°C above baseline)", fontsize=9)

    # Mark pre/post windows for each season
    for s in seasons:
        if s.get("skipped"):
            continue
        for t0, t1 in [s["pre_window"], s["post_window"]]:
            ax.axvspan(mdates.date2num(t0), mdates.date2num(t1),
                       color="white", alpha=0.25)
        ax.axhline(s["max_wf_depth_m"], xmin=0.0, xmax=1.0,
                   color="yellow", lw=0.8, ls="--", alpha=0.6)

    ax.set_ylabel("Depth (m)", fontsize=10)
    ax.set_ylim(depth_grid[-1], 0)
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
    ax.grid(axis="y", color="grey", alpha=0.3, lw=0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Piping signature below wetting front  "
                 "(white bands = pre/post windows; dashed = max wf depth per season)",
                 fontsize=11, fontweight="bold", loc="left")

    # ── Panel 3: Q(t) diagnostic + seasonal bar chart ────────────────────
    ax  = axes[2]
    ax2 = ax.twinx()
    ax.fill_between(t_mpl, Q / 1e6, alpha=0.3, color="steelblue")
    ax.plot(t_mpl, Q / 1e6, color="steelblue", lw=1.0, label="Q anomaly (MJ m⁻²)")
    # Seasonal estimates as horizontal bars
    for s in seasons:
        if s.get("skipped") or s["m_mm_we"] <= 0:
            continue
        t_post0 = mdates.date2num(s["post_window"][0])
        t_post1 = mdates.date2num(s["post_window"][1])
        ax2.barh(s["m_mm_we"] / 2, t_post1 - t_post0, left=t_post0,
                 height=s["m_mm_we"], color="crimson", alpha=0.5)
        ax2.text((t_post0 + t_post1) / 2, s["m_mm_we"],
                 f"{s['year']}: {s['m_mm_we']:.1f} mm",
                 ha="center", va="bottom", fontsize=8, color="crimson")
    ax.set_ylabel("Q  (MJ m⁻²)", fontsize=10, color="steelblue")
    ax2.set_ylabel("Seasonal refreeze  (mm w.e.)", fontsize=10, color="crimson")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax.grid(axis="x", color="grey", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Diagnostic Q(t) & seasonal pre/post refreeze estimate",
                 fontsize=11, fontweight="bold", loc="left")

    for ax_ in axes:
        ax_.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.setp(ax_.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=9)
        ax_.set_xlim(t_mpl[0], t_mpl[-1])

    fig.suptitle(f"Piping refreeze estimate — {site_label}",
                 fontsize=13, fontweight="bold", y=0.998)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Estimate piping-event refreezing below the SNOWPACK wetting front.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--site",  default="T3",  help="Site name, e.g. T3")
    p.add_argument("--year",  type=int, default=2022, help="Drill year")
    p.add_argument("--depth", type=int, default=25,   help="String depth (m)")
    args = p.parse_args()

    paths = build_paths(args.year, args.site, args.depth)
    if not paths["pro_file"].exists():
        raise FileNotFoundError(f"PRO file not found: {paths['pro_file']}")
    if not paths["obs_file"].exists():
        raise FileNotFoundError(f"Observation file not found: {paths['obs_file']}")

    print("Parsing PRO file …")
    pro   = parse_pro(paths["pro_file"])
    times = pd.DatetimeIndex(pro["times"])
    print(f"  {len(times)} timesteps  {times[0]} → {times[-1]}")

    dz         = 0.05
    depth_grid = np.arange(0, args.depth + dz / 2, dz)

    print("Regridding model fields …")
    T_mod = to_regular_grid(times, pro[501], pro[503], depth_grid)
    LWC   = to_regular_grid(times, pro[501], pro[506], depth_grid)
    RHO   = to_regular_grid(times, pro[501], pro[502], depth_grid)

    print("Loading and regridding observations …")
    obs   = load_observations(paths["obs_file"])
    T_obs = obs_to_grid(obs, depth_grid, times)

    print("Computing Q(t) diagnostic …")
    Q, wf_arr, dT_corr, baseline = compute_Q_timeseries(
        T_mod, T_obs, LWC, RHO, depth_grid, dz)

    print("Computing per-season refreeze estimates …")
    seasons = compute_seasonal_refreeze(T_mod, T_obs, LWC, RHO, depth_grid, dz, times)

    total = sum(s["m_mm_we"] for s in seasons if not s.get("skipped", True))
    n_seasons = sum(1 for s in seasons if not s.get("skipped", True))
    print(f"\n  Total piping refreeze over {n_seasons} season(s): {total:.1f} mm w.e.")
    print(f"  Mean per season: {total/max(n_seasons,1):.1f} mm w.e.\n")

    # ── Save CSV ──────────────────────────────────────────────────────────
    seas_rows = []
    for s in seasons:
        row = {"year": s["year"], "skipped": s.get("skipped", False),
               "note": s.get("note", "")}
        if not s.get("skipped"):
            row.update({
                "max_wf_depth_m":    s["max_wf_depth_m"],
                "Q_J_m2":            s["Q_J_m2"],
                "m_mm_we":           s["m_mm_we"],
                "partial_refreeze":  s.get("partial", False),
                "pre_window_start":  s["pre_window"][0],
                "pre_window_end":    s["pre_window"][1],
                "post_window_start": s["post_window"][0],
                "post_window_end":   s["post_window"][1],
            })
        seas_rows.append(row)

    pd.DataFrame(seas_rows).to_csv(paths["out_csv"], index=False, float_format="%.4f")
    print(f"Saved CSV → {paths['out_csv']}")

    print("Plotting …")
    make_figure(times, depth_grid, T_mod, T_obs, LWC,
                Q, wf_arr, dT_corr, seasons,
                paths["site_label"], paths["out_fig"])


if __name__ == "__main__":
    main()
