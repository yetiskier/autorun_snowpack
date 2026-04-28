"""
Estimate piping-event refreezing from borehole temperature observations
using the Saito et al. (2024) Crank-Nicolson thermal diffusion method.

Reference:
  Saito, J., Harper, J., & Humphrey, N. (2024). Uptake and transfer of heat
  within the firn layer of Greenland Ice Sheet's percolation zone.
  Journal of Geophysical Research: Earth Surface, 129, e2024JF007667.
  https://doi.org/10.1029/2024JF007667

Method:
  For each consecutive pair of observation timesteps [t, t+Δt] where both
  have complete (NaN-free) temperature observations across the full domain
  [DOMAIN_TOP_M, string depth]:

  1. Interpolate T_obs and SNOWPACK density to a uniform DZ-spaced grid.

  2. Compute thermal conductivity k(z) using Calonne et al. (2019):
       k(ρ, T) = k_2011(ρ) × k_ice(T) / k_ice(0°C)
     where  k_2011  = 2.5×10⁻⁶ρ² − 1.23×10⁻⁴ρ + 0.024  [W m⁻¹ K⁻¹]
     and    k_ice(T) = 9.828 exp(−5.7×10⁻³ T_K)          [W m⁻¹ K⁻¹]

  3. Crank-Nicolson forward step: predict T_cond(z, t+Δt) from conduction
     alone.  Dirichlet BCs from observed T at both ends of the domain.

  4. Latent heat residual at each interior depth:
       Q_lh(z) = ρ(z) · Cₚ · [T_obs(z, t+Δt) − T_cond(z, t+Δt)]   [J m⁻³]
     Positive Q_lh means heat arrived that conduction alone cannot explain.

  5. Obs-based wetting front z_wf: deepest depth in the connected warm zone
     (T_obs ≥ −0.05°C scanning down from DOMAIN_TOP_M; stops at first cold
     layer to exclude isolated deep piping warmings from the wf definition).

  6. Piping signal: Q_lh > 0 strictly below z_wf.

  7. Per-interval piping refreeze:
       m_step = Σ_z max(Q_lh_pipe(z), 0) · dz  /  L_f        [kg m⁻²]

  Density comes from the SNOWPACK PRO file (code 502), which is the evolving
  obs-constrained density profile.  Only observation timesteps are used; the
  model is never advanced during gaps in the temperature record.

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
from scipy.linalg import solve_banded

# ── Physical constants ──────────────────────────────────────────────────────
L_F      = 334_000.0   # latent heat of fusion          [J kg⁻¹]
CP_ICE   = 2_097.0     # specific heat of firn/ice       [J kg⁻¹ K⁻¹]

# ── Algorithm parameters ────────────────────────────────────────────────────
DOMAIN_TOP_M = 1.0     # shallowest depth in diffusion domain  [m]
DZ           = 0.5     # diffusion grid spacing                [m]
WF_THRESHOLD = -0.05   # wetting-front temperature threshold   [°C]

SCRIPT_DIR = Path(__file__).resolve().parent


# ── Path helpers ─────────────────────────────────────────────────────────────
def build_paths(year: int, site: str, depth: int) -> dict:
    site_id  = f"{year}_{site}_{depth}m"
    pro_name = f"{year}-{site}-{depth}m_TEMP_ASSIM_RUN.pro"
    out_dir  = SCRIPT_DIR / site_id / "output"
    obs_dir  = (SCRIPT_DIR / "AllCoreDataCommonFormat"
                / "Concatenated_Temperature_files")
    return {
        "pro_file":   out_dir / pro_name,
        "obs_file":   obs_dir / f"{site_id}_Tempconcatenated.csv",
        "out_csv":    out_dir / f"{site_id}_piping_refreeze.csv",
        "out_fig":    out_dir / f"{site_id}_piping_refreeze.png",
        "site_label": f"{year} {site} {depth} m",
        "out_dir":    out_dir,
    }


# ── PRO file parser ───────────────────────────────────────────────────────────
_PRO_CODES = {501, 502, 506}   # heights, density, LWC

def parse_pro(path: Path) -> dict:
    """Parse a SNOWPACK .pro file; return dict of ragged-array time series."""
    data    = {c: [] for c in _PRO_CODES}
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
                    for c in _PRO_CODES:
                        data[c].append(current.get(c, np.array([])))
                current = {}
                times.append(pd.to_datetime(rest.strip(), dayfirst=True))
            elif code in _PRO_CODES:
                vals = rest.split(",")
                current[code] = np.array([float(v) for v in vals[1:]])
    if current:
        for c in _PRO_CODES:
            data[c].append(current.get(c, np.array([])))
    return {"times": times, **data}


def _pro_to_grid(times, heights_list, values_list, depth_grid: np.ndarray) -> np.ndarray:
    """Regrid ragged per-timestep PRO profiles onto a fixed depth grid."""
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
        f = interp1d(d, v_, bounds_error=False, fill_value=np.nan)
        grid[i] = f(depth_grid)
    return grid


# ── Thermal conductivity — Calonne et al. (2019) ────────────────────────────
def k_calonne2019(rho: np.ndarray, T_celsius: np.ndarray) -> np.ndarray:
    """
    Effective thermal conductivity [W m⁻¹ K⁻¹] after Calonne et al. (2019).

    Extends the Calonne 2011 density parameterisation with temperature
    dependence through the ice conductivity ratio:

        k(ρ, T) = k_2011(ρ) × k_ice(T) / k_ice(0°C)

    where k_2011(ρ) = 2.5×10⁻⁶ρ² − 1.23×10⁻⁴ρ + 0.024  [W m⁻¹ K⁻¹]
    and   k_ice(T)  = 9.828 exp(−5.7×10⁻³ T_K)            [W m⁻¹ K⁻¹]

    Typical range: ~12% higher conductivity at −20°C vs 0°C.
    """
    T_K      = np.asarray(T_celsius) + 273.15
    k_ice    = 9.828 * np.exp(-5.7e-3 * T_K)
    k_ice_0c = 9.828 * np.exp(-5.7e-3 * 273.15)   # ≈ 2.07 W/(m·K) at 0°C
    k_2011   = 2.5e-6 * rho**2 - 1.23e-4 * rho + 0.024
    return k_2011 * k_ice / k_ice_0c


# ── Crank-Nicolson 1D heat diffusion ─────────────────────────────────────────
def crank_nicolson_step(
    T_curr: np.ndarray,
    rho: np.ndarray,
    k: np.ndarray,
    dz: float,
    dt_sec: float,
    T_top_next: float,
    T_bot_next: float,
) -> np.ndarray:
    """
    One Crank-Nicolson step for 1D heat diffusion with variable k.

    Solves:  ρ Cₚ ∂T/∂t = ∂/∂z(k ∂T/∂z)

    Parameters
    ----------
    T_curr     : temperature at t, shape (N,); T_curr[0] and T_curr[-1]
                 are the current Dirichlet boundary values
    rho        : density [kg m⁻³], shape (N,)
    k          : thermal conductivity [W m⁻¹ K⁻¹], shape (N,)
    dz         : uniform grid spacing [m]
    dt_sec     : time step [s]
    T_top_next : top Dirichlet BC at t+Δt (observed temperature at 1 m)
    T_bot_next : bottom Dirichlet BC at t+Δt (observed temperature at string bottom)

    Returns
    -------
    T_next_interior : predicted temperatures at interior nodes, shape (N-2,)
    """
    # Interface conductivities at half-points for all interior nodes
    kp = 0.5 * (k[1:-1] + k[2:])    # k_{i+½}, shape N-2
    km = 0.5 * (k[:-2]  + k[1:-1])  # k_{i-½}, shape N-2

    Ap = kp / dz**2   # shape N-2
    Am = km / dz**2   # shape N-2
    r  = rho[1:-1] * CP_ICE / dt_sec   # shape N-2

    # Tridiagonal coefficients
    diag  =  r + 0.5 * (Ap + Am)    # main diagonal,  shape N-2
    upper = -0.5 * Ap[:-1]           # superdiagonal,  shape N-3
    lower = -0.5 * Am[1:]            # subdiagonal,    shape N-3

    # Explicit (current-time) RHS
    rhs = ((r - 0.5 * (Ap + Am)) * T_curr[1:-1]
           + 0.5 * Ap * T_curr[2:]
           + 0.5 * Am * T_curr[:-2])

    # Add future Dirichlet BC contributions to boundary-adjacent rows
    rhs[0]  += 0.5 * Am[0]  * T_top_next
    rhs[-1] += 0.5 * Ap[-1] * T_bot_next

    # Solve tridiagonal system via scipy banded solver
    ab = np.zeros((3, len(diag)))
    ab[0, 1:]  = upper   # superdiagonal
    ab[1, :]   = diag    # main diagonal
    ab[2, :-1] = lower   # subdiagonal
    return solve_banded((1, 1), ab, rhs)


# ── Observation-based wetting front ──────────────────────────────────────────
def obs_wetting_front(T_profile: np.ndarray, depth_grid: np.ndarray,
                      threshold: float = WF_THRESHOLD) -> float:
    """
    Deepest depth of the connected warm zone at/above threshold.

    Scans from DOMAIN_TOP_M downward and stops at the first layer that drops
    below threshold.  This distinguishes the main wetting front (connected
    to the surface) from isolated deep warmings caused by piping events.

    Returns NaN when the shallowest domain node is below threshold
    (no active melt front in the domain).
    """
    if T_profile[0] < threshold:
        return np.nan
    wf = depth_grid[0]
    for i in range(1, len(depth_grid)):
        if T_profile[i] >= threshold:
            wf = depth_grid[i]
        else:
            break
    return wf


# ── Load observations ─────────────────────────────────────────────────────────
def load_observations(path: Path) -> pd.DataFrame:
    obs = pd.read_csv(path, skiprows=3, index_col=0, parse_dates=True)
    obs.columns = pd.to_numeric(obs.columns, errors="coerce")
    obs = obs.loc[:, obs.columns >= 0]
    return obs


# ── Core computation ──────────────────────────────────────────────────────────
def compute_piping_refreeze(obs: pd.DataFrame, pro: dict,
                             string_depth_m: int) -> tuple:
    """
    Run the Saito 2024 diffusion-based piping refreeze estimation.

    The CN diffusion is run on **daily-mean** temperature profiles, matching
    Saito et al. (2024) who explicitly use 24-hour time steps.  At sub-hourly
    resolution the temperature change per step (~0.001°C) is smaller than
    sensor noise (~0.008°C), so Q_lh = ρCₚ(T_obs − T_cond) would be
    dominated by measurement noise rather than latent heat.  Daily averaging
    reduces noise by ~1/√48 ≈ 7× while the physical signal (daily conductive
    change ~0.01–0.1°C) remains intact.

    Gap policy: a calendar day is valid only if every sub-hourly observation
    within it has no NaN at any depth in the diffusion domain.  A single gap
    anywhere in the 24-hour window invalidates that day's CN step.

    Returns
    -------
    results      : DataFrame with one row per daily interval
    diff_grid    : depth grid used for diffusion [m]
    T_daily_grid : daily-mean T_obs on diff_grid, shape (n_days, nd)
    Q_lh_grid    : Q_lh profiles, shape (n_days-1, nd); NaN where invalid
    """
    diff_grid = np.arange(DOMAIN_TOP_M, string_depth_m + DZ / 2, DZ)
    nd = len(diff_grid)

    # ── SNOWPACK density → regrid to diffusion depth grid ────────────────
    pro_times = pd.DatetimeIndex(pro["times"])
    rho_pro   = _pro_to_grid(pro_times, pro[501], pro[502], diff_grid)

    # ── Observations: restrict to domain depths ───────────────────────────
    obs_depths = obs.columns.values.astype(float)
    in_domain  = obs_depths >= DOMAIN_TOP_M
    obs_dom    = obs.loc[:, in_domain]
    depths_dom = obs_depths[in_domain]
    obs_times  = obs.index

    # ── Interpolate sub-hourly obs to diffusion depth grid ────────────────
    # Keep original time resolution here; daily averaging happens below.
    nt = len(obs_times)
    T_subhourly = np.full((nt, nd), np.nan)
    for i in range(nt):
        row  = obs_dom.iloc[i].values.astype(float)
        good = np.isfinite(row)
        if good.sum() < 2:
            continue
        f = interp1d(depths_dom[good], row[good],
                     bounds_error=False, fill_value=np.nan)
        T_subhourly[i] = f(diff_grid)

    T_sh_df = pd.DataFrame(T_subhourly, index=obs_times)

    # ── Gap check: any NaN in a calendar day → day is invalid ────────────
    # Use min (not mean) so a single NaN propagates to the daily validity flag
    day_has_nan = T_sh_df.resample("D").apply(
        lambda x: bool(np.any(~np.isfinite(x.values)))
    ).any(axis=1)    # True = this day has at least one NaN anywhere in domain

    # ── Daily means for CN diffusion (Saito et al. use 24-h steps) ───────
    T_daily_df  = T_sh_df.resample("D").mean()
    daily_times = T_daily_df.index
    n_days      = len(daily_times)
    T_daily_grid = T_daily_df.values   # shape (n_days, nd)

    # ── SNOWPACK density → daily mean on diffusion grid ───────────────────
    rho_df     = pd.DataFrame(rho_pro, index=pro_times)
    rho_daily  = (rho_df
                  .reindex(rho_df.index.union(daily_times))
                  .sort_index()
                  .interpolate(method="time")
                  .reindex(daily_times)
                  .values)   # shape (n_days, nd)

    # ── Main loop: one CN step per consecutive valid daily pair ───────────
    Q_lh_grid = np.full((n_days - 1, nd), np.nan)
    rows      = []
    m_cumul   = 0.0
    DT_SEC    = 86400.0   # 24-hour step

    for i in range(n_days - 1):
        day_i   = daily_times[i]
        day_ip1 = daily_times[i + 1]

        # Skip if either day has a gap anywhere in the domain
        gap_i   = day_has_nan.get(day_i,   True)
        gap_ip1 = day_has_nan.get(day_ip1, True)
        if gap_i or gap_ip1:
            rows.append({"datetime": day_i, "valid": False,
                         "wf_depth_m": np.nan, "Q_lh_J_m2": np.nan,
                         "m_step_mm_we": np.nan, "m_cumul_mm_we": m_cumul})
            continue

        T_curr = T_daily_grid[i]
        T_next = T_daily_grid[i + 1]

        if np.any(~np.isfinite(T_curr)) or np.any(~np.isfinite(T_next)):
            rows.append({"datetime": day_i, "valid": False,
                         "wf_depth_m": np.nan, "Q_lh_J_m2": np.nan,
                         "m_step_mm_we": np.nan, "m_cumul_mm_we": m_cumul})
            continue

        rho = rho_daily[i]
        if np.any(~np.isfinite(rho)):
            rows.append({"datetime": day_i, "valid": False,
                         "wf_depth_m": np.nan, "Q_lh_J_m2": np.nan,
                         "m_step_mm_we": np.nan, "m_cumul_mm_we": m_cumul})
            continue

        # Thermal conductivity from Calonne 2019 at current daily mean T
        k = k_calonne2019(rho, T_curr)

        # Crank-Nicolson 24-hour forward step
        T_cond_int = crank_nicolson_step(
            T_curr, rho, k, DZ, DT_SEC, T_next[0], T_next[-1])
        T_cond = np.concatenate([[T_next[0]], T_cond_int, [T_next[-1]]])

        # Latent heat residual [J m⁻³]
        Q_lh = rho * CP_ICE * (T_next - T_cond)
        Q_lh[[0, -1]] = 0.0   # BCs are exact by construction
        Q_lh_grid[i]  = Q_lh

        # Obs wetting front from daily-mean profile
        wf = obs_wetting_front(T_curr, diff_grid)

        # Piping signal: positive Q_lh strictly below wetting front
        if np.isnan(wf):
            Q_pipe = np.zeros(nd)
        else:
            below  = diff_grid > wf
            Q_pipe = np.where(below & (Q_lh > 0), Q_lh, 0.0)

        Q_total  = float(np.sum(Q_pipe) * DZ)   # J m⁻²
        m_step   = Q_total / L_F                 # kg m⁻² = mm w.e.
        m_cumul += m_step

        rows.append({"datetime": day_i, "valid": True,
                     "wf_depth_m": wf, "Q_lh_J_m2": Q_total,
                     "m_step_mm_we": m_step, "m_cumul_mm_we": m_cumul})

    return pd.DataFrame(rows), diff_grid, T_daily_grid, Q_lh_grid


# ── Melt sanity check ────────────────────────────────────────────────────────
def melt_sanity_check(pro: dict, obs_times: pd.DatetimeIndex,
                      string_depth_m: int) -> dict:
    """
    Compare SNOWPACK max wetting-front depth (obs-constrained timesteps only)
    to the obs-based max wetting-front depth.

    A large discrepancy flags that SNOWPACK drifted during an obs gap
    (e.g. T3 2023: SNOWPACK wf = 11.8 m while obs wf never exceeded ~4 m).
    """
    pro_times = pd.DatetimeIndex(pro["times"])
    diff_grid = np.arange(DOMAIN_TOP_M, string_depth_m + DZ / 2, DZ)
    LWC_pro   = _pro_to_grid(pro_times, pro[501], pro[506], diff_grid)

    # Restrict to timesteps within the obs record
    obs_start, obs_end = obs_times.min(), obs_times.max()
    in_obs = (pro_times >= obs_start) & (pro_times <= obs_end)

    max_wf_model = np.nan
    for i in np.where(in_obs)[0]:
        row = LWC_pro[i]
        wet = np.where(row > 0.1)[0]
        if len(wet):
            d = diff_grid[wet[-1]]
            if np.isnan(max_wf_model) or d > max_wf_model:
                max_wf_model = d

    return {"max_wf_model_m": max_wf_model}


# ── Figure ────────────────────────────────────────────────────────────────────
def make_figure(results: pd.DataFrame, diff_grid: np.ndarray,
                T_on_grid: np.ndarray, Q_lh_grid: np.ndarray,
                obs_times: pd.DatetimeIndex, site_label: str,
                out_path: Path) -> None:
    """
    Three-panel figure:
      1. T_obs curtain with obs wetting front overlay
      2. Q_lh curtain (latent heat residual — all depths)
      3. Q_lh below wetting front only (piping signature) + cumulative refreeze
    """
    nd = len(diff_grid)
    nt = len(obs_times)

    # obs_times are the interval-start dates (one per curtain row).
    # T_on_grid[:-1] and Q_lh_grid both have len(obs_times) rows.
    t_mpl   = mdates.date2num(obs_times.to_pydatetime())   # N cell centres
    t_edges = np.concatenate([[t_mpl[0] - 0.5 * (t_mpl[1] - t_mpl[0])],
                               0.5 * (t_mpl[:-1] + t_mpl[1:]),
                               [t_mpl[-1] + 0.5 * (t_mpl[-1] - t_mpl[-2])]])
    # → N+1 edges for N cells ✓
    t_mpl_all = t_mpl   # used for xlim
    d_edges = np.concatenate([[diff_grid[0] - DZ / 2],
                               0.5 * (diff_grid[:-1] + diff_grid[1:]),
                               [diff_grid[-1] + DZ / 2]])

    # Wetting-front line: one value per interval (aligned with cell centres)
    wf_line = results["wf_depth_m"].values   # shape N (one per interval)
    wf_x    = t_mpl                          # N cell-centre mpl dates

    # Q_lh below wetting front only (piping signal)
    Q_pipe = np.full_like(Q_lh_grid, np.nan)
    for i, row in results.iterrows():
        if not row.get("valid", False) or np.isnan(row["wf_depth_m"]):
            continue
        idx = int(i)
        if idx >= Q_lh_grid.shape[0]:
            continue
        below = diff_grid > row["wf_depth_m"]
        Q_pipe[idx] = np.where(below & (Q_lh_grid[idx] > 0),
                                Q_lh_grid[idx], np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(16, 15),
                             gridspec_kw={"hspace": 0.10,
                                          "height_ratios": [2, 2, 1]})

    def _curtain(ax, data, cmap, norm, label, extend="both"):
        masked = np.ma.masked_invalid(data)
        pm = ax.pcolormesh(t_edges, d_edges, masked.T,
                           cmap=cmap, norm=norm, shading="flat")
        fig.colorbar(pm, ax=ax, pad=0.01, fraction=0.018,
                     extend=extend).set_label(label, fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=10)
        ax.set_ylim(diff_grid[-1], diff_grid[0])
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.grid(axis="y", color="grey", alpha=0.3, lw=0.5)
        return pm

    # ── Panel 1: T_obs ────────────────────────────────────────────────────
    T_vmax = max(abs(np.nanpercentile(T_on_grid, [2, 98])))
    _curtain(axes[0], T_on_grid[:-1], "RdYlBu_r",
             TwoSlopeNorm(vmin=-T_vmax, vcenter=-10, vmax=0),
             "T_obs (°C)", extend="both")
    wf_x = wf_x   # already in mpl date float format
    axes[0].plot(wf_x, wf_line, color="lime", lw=1.5,
                 label="Obs wetting front (−0.05°C)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_title("Observed firn temperature", fontsize=11,
                      fontweight="bold", loc="left")

    # ── Panel 2: Q_lh full column ─────────────────────────────────────────
    qlh_abs = np.nanpercentile(np.abs(Q_lh_grid[np.isfinite(Q_lh_grid)]), 98) \
              if np.any(np.isfinite(Q_lh_grid)) else 1e4
    qlh_abs = max(qlh_abs, 100.0)
    _curtain(axes[1], Q_lh_grid, "RdBu_r",
             TwoSlopeNorm(vmin=-qlh_abs, vcenter=0, vmax=qlh_abs),
             "Q_lh (J m⁻³)", extend="both")
    axes[1].plot(wf_x, wf_line, color="lime", lw=1.5,
                 label="Obs wetting front")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].set_title(
        "Latent heat residual Q_lh = ρCₚ(T_obs − T_cond)  [full column]",
        fontsize=11, fontweight="bold", loc="left")

    # ── Panel 3: piping signal + cumulative refreeze ──────────────────────
    ax  = axes[2]
    ax2 = ax.twinx()

    # Integrated piping Q_lh [MJ m⁻²] — sum over depths below wf
    valid = results[results["valid"] == True].copy()
    if len(valid):
        t_valid = mdates.date2num(valid["datetime"].values.astype("datetime64[ms]")
                                  .astype(object))
        ax.fill_between(t_valid, valid["Q_lh_J_m2"] / 1e6,
                        alpha=0.35, color="steelblue")
        ax.plot(t_valid, valid["Q_lh_J_m2"] / 1e6,
                color="steelblue", lw=0.8, label="Q_pipe (MJ m⁻²)")
        ax2.plot(t_valid, valid["m_cumul_mm_we"],
                 color="crimson", lw=1.8, label="Cumulative refreeze (mm w.e.)")

    ax.set_ylabel("Integrated piping Q  (MJ m⁻²)", fontsize=10,
                  color="steelblue")
    ax2.set_ylabel("Cumulative piping refreeze  (mm w.e.)", fontsize=10,
                   color="crimson")
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax.grid(axis="x", color="grey", alpha=0.3)
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, loc="upper left", fontsize=9)
    ax.set_title("Piping heat below obs wetting front & cumulative refreeze",
                 fontsize=11, fontweight="bold", loc="left")

    for ax_ in axes:
        ax_.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        ax_.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax_.xaxis.set_minor_locator(mdates.MonthLocator())
        plt.setp(ax_.xaxis.get_majorticklabels(),
                 rotation=30, ha="right", fontsize=9)
        ax_.set_xlim(t_mpl_all[0], t_mpl_all[-1])

    fig.suptitle(f"Piping refreeze estimate (Saito 2024 method) — {site_label}",
                 fontsize=13, fontweight="bold", y=0.999)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Estimate piping refreezing via Saito et al. (2024) method.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--site",  default="T3",  help="Site name, e.g. T3")
    p.add_argument("--year",  type=int, default=2022, help="Drill year")
    p.add_argument("--depth", type=int, default=25,   help="String depth (m)")
    args = p.parse_args()

    paths = build_paths(args.year, args.site, args.depth)
    if not paths["pro_file"].exists():
        raise FileNotFoundError(f"PRO not found: {paths['pro_file']}")
    if not paths["obs_file"].exists():
        raise FileNotFoundError(f"Obs not found: {paths['obs_file']}")

    print("Parsing PRO file …")
    pro       = parse_pro(paths["pro_file"])
    pro_times = pd.DatetimeIndex(pro["times"])
    print(f"  {len(pro_times)} model timesteps  "
          f"{pro_times[0]} → {pro_times[-1]}")

    print("Loading observations …")
    obs = load_observations(paths["obs_file"])
    print(f"  {len(obs)} obs timesteps  "
          f"{obs.index[0]} → {obs.index[-1]}")
    print(f"  Sensor depths: {obs.columns.min():.2f} – "
          f"{obs.columns.max():.2f} m  ({len(obs.columns)} sensors)")

    print("Running CN diffusion piping estimation …")
    results, diff_grid, T_daily_grid, Q_lh_grid = compute_piping_refreeze(
        obs, pro, args.depth)

    n_valid   = results["valid"].sum()
    n_total   = len(results)
    total_mwe = results["m_cumul_mm_we"].iloc[-1]
    daily_times = pd.DatetimeIndex(results["datetime"])
    print(f"  Daily intervals: {n_valid}/{n_total} valid "
          f"({100*n_valid/n_total:.1f}%)")
    print(f"\n  Total piping refreeze: {total_mwe:.2f} mm w.e.")

    # Melt sanity check
    sanity = melt_sanity_check(pro, obs.index, args.depth)
    print(f"  SNOWPACK max wetting front (within obs period): "
          f"{sanity['max_wf_model_m']:.1f} m")

    # Per-year summary
    print("\n  Per-year piping refreeze:")
    valid = results[results["valid"]].copy()
    valid["year"] = pd.to_datetime(valid["datetime"]).dt.year
    for yr, grp in valid.groupby("year"):
        print(f"    {yr}: {grp['m_step_mm_we'].sum():.2f} mm w.e.  "
              f"({len(grp)} days, max wf "
              f"{grp['wf_depth_m'].max():.1f} m)")

    # Save CSV
    results.to_csv(paths["out_csv"], index=False, float_format="%.5f")
    print(f"\nSaved CSV → {paths['out_csv']}")

    # Plot
    print("Plotting …")
    make_figure(results, diff_grid, T_daily_grid, Q_lh_grid,
                daily_times, paths["site_label"], paths["out_fig"])


if __name__ == "__main__":
    main()
