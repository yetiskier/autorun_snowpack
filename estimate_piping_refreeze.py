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
_PRO_CODES = {501, 502, 506, 515}   # heights, density, LWC, ice vol fraction

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

    The CN diffusion runs at the native observation timestep (sub-hourly).
    A step is valid only when both consecutive observation rows are NaN-free
    across the full diffusion domain AND the interval is ≤ MAX_OBS_GAP_SEC
    (otherwise a gap in the record would produce a spuriously large dt step).

    SNOWPACK refreezing is attributed to each obs-step interval by
    interpolating the obs-gated hourly cumulative refreezing series and taking
    the per-interval increment; it is zeroed on invalid steps.

    Returns
    -------
    results    : DataFrame with one row per obs interval (sub-hourly)
    diff_grid  : depth grid for diffusion [m]
    T_obs_grid : T_obs on diff_grid at obs times, shape (nt, nd)
    Q_lh_grid  : Q_lh profiles, shape (nt-1, nd); NaN on invalid intervals
    """
    MAX_OBS_GAP_SEC = 7200   # skip interval if obs timestamps > 2 h apart

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
    nt = len(obs_times)

    # ── Interpolate obs to diffusion depth grid at each obs timestep ─────
    T_obs_grid = np.full((nt, nd), np.nan)
    for i in range(nt):
        row  = obs_dom.iloc[i].values.astype(float)
        good = np.isfinite(row)
        if good.sum() < 2:
            continue
        f = interp1d(depths_dom[good], row[good],
                     bounds_error=False, fill_value=np.nan)
        T_obs_grid[i] = f(diff_grid)

    # ── SNOWPACK density → interpolated to obs times ─────────────────────
    rho_df = pd.DataFrame(rho_pro, index=pro_times)
    rho_at_obs = (rho_df
                  .reindex(rho_df.index.union(obs_times))
                  .sort_index()
                  .interpolate(method="time")
                  .reindex(obs_times)
                  .values)   # shape (nt, nd)

    # ── SNOWPACK column refreezing — same method as app.py `load_pro` ─────
    # Compute hourly refreezing; gate to obs-constrained periods using the
    # same ≥3-sensor validity criterion as load_pro v5.
    n_pro   = len(pro_times)
    ice_col = np.zeros(n_pro)
    lwc_col = np.zeros(n_pro)
    for ti in range(n_pro):
        h   = np.asarray(pro[501][ti], dtype=float)
        ice = np.asarray(pro[515][ti], dtype=float)
        lwc = np.asarray(pro[506][ti], dtype=float)
        den = np.asarray(pro[502][ti], dtype=float)
        n   = min(len(h), len(ice), len(lwc), len(den))
        if n < 2:
            continue
        h = h[:n]; ice = ice[:n]; lwc = lwc[:n]; den = den[:n]
        order = np.argsort(h)
        h=h[order]; ice=ice[order]; lwc=lwc[order]; den=den[order]
        mask = (h > 0) & (den < 900.0)
        if mask.sum() < 2:
            continue
        h_f=h[mask]; ice_f=ice[mask]; lwc_f=lwc[mask]
        nf = len(h_f)
        thick = np.empty(nf)
        thick[1:] = (h_f[1:] - h_f[:-1]) / 100.0
        thick[0]  = h_f[0] / 100.0
        ice_col[ti] = np.nansum(ice_f / 100.0 * thick * 917.0)
        lwc_col[ti] = np.nansum(lwc_f / 100.0 * thick * 1000.0)

    d_ice = np.diff(ice_col)
    d_lwc = np.diff(lwc_col)
    sp_hourly = np.concatenate([[0.0], np.minimum(
        np.where(d_ice > 0, d_ice, 0.0),
        np.where(d_lwc < 0, -d_lwc, 0.0),
    )])

    # Gate SNOWPACK refreezing to obs-constrained hours (≥3 valid sensors)
    obs_at_pro = obs.reindex(pro_times, method="nearest",
                              tolerance=pd.Timedelta("30min"))
    obs_valid_pro = obs_at_pro.notna().sum(axis=1).values >= 3
    sp_hourly_gated = np.where(obs_valid_pro, sp_hourly, 0.0)

    # Build cumulative series on PRO times → interpolate to obs times → per-step diff
    sp_cumul_pro = np.cumsum(sp_hourly_gated)
    sp_cumul_ser = pd.Series(sp_cumul_pro, index=pro_times)
    sp_cumul_at_obs = (sp_cumul_ser
                       .reindex(sp_cumul_ser.index.union(obs_times))
                       .sort_index()
                       .interpolate(method="time")
                       .reindex(obs_times)
                       .values)
    # Per obs-step SNOWPACK refreezing (diff of cumulative, zeroed on NaN)
    sp_step_arr = np.diff(sp_cumul_at_obs, prepend=sp_cumul_at_obs[0])
    sp_step_arr = np.where(np.isfinite(sp_step_arr), sp_step_arr, 0.0)

    # ── Main loop: one CN step per consecutive obs interval ──────────────
    Q_lh_grid = np.full((nt - 1, nd), np.nan)
    rows      = []
    m_cumul   = 0.0
    sp_cumul  = 0.0

    for i in range(nt - 1):
        t_i   = obs_times[i]
        t_ip1 = obs_times[i + 1]
        dt_sec = (t_ip1 - t_i).total_seconds()

        # Skip if timestep is too large (obs gap) or non-positive
        if dt_sec <= 0 or dt_sec > MAX_OBS_GAP_SEC:
            rows.append({"datetime": t_i, "valid": False,
                         "wf_depth_m": np.nan, "Q_lh_J_m2": np.nan,
                         "m_step_mm_we": np.nan, "m_cumul_mm_we": m_cumul,
                         "sp_refreeze_mm_we": np.nan, "sp_cumul_mm_we": sp_cumul})
            continue

        T_curr = T_obs_grid[i]
        T_next = T_obs_grid[i + 1]

        # Both endpoints must be gap-free across the full domain
        if np.any(~np.isfinite(T_curr)) or np.any(~np.isfinite(T_next)):
            rows.append({"datetime": t_i, "valid": False,
                         "wf_depth_m": np.nan, "Q_lh_J_m2": np.nan,
                         "m_step_mm_we": np.nan, "m_cumul_mm_we": m_cumul,
                         "sp_refreeze_mm_we": np.nan, "sp_cumul_mm_we": sp_cumul})
            continue

        rho = rho_at_obs[i]
        if np.any(~np.isfinite(rho)):
            rows.append({"datetime": t_i, "valid": False,
                         "wf_depth_m": np.nan, "Q_lh_J_m2": np.nan,
                         "m_step_mm_we": np.nan, "m_cumul_mm_we": m_cumul,
                         "sp_refreeze_mm_we": np.nan, "sp_cumul_mm_we": sp_cumul})
            continue

        # SNOWPACK refreezing for this obs step (obs-gated, from cumulative diff)
        sp_step   = float(sp_step_arr[i])
        sp_cumul += sp_step

        # Thermal conductivity (Calonne 2019) at current obs profile
        k = k_calonne2019(rho, T_curr)

        # Crank-Nicolson forward step at native obs timestep
        T_cond_int = crank_nicolson_step(
            T_curr, rho, k, DZ, dt_sec, T_next[0], T_next[-1])
        T_cond = np.concatenate([[T_next[0]], T_cond_int, [T_next[-1]]])

        # Latent heat residual [J m⁻³]
        Q_lh = rho * CP_ICE * (T_next - T_cond)
        Q_lh[[0, -1]] = 0.0
        Q_lh_grid[i]  = Q_lh

        # Obs wetting front from current obs profile
        wf = obs_wetting_front(T_curr, diff_grid)

        # Piping: positive Q_lh strictly below obs wetting front
        if np.isnan(wf):
            Q_pipe = np.zeros(nd)
        else:
            below  = diff_grid > wf
            Q_pipe = np.where(below & (Q_lh > 0), Q_lh, 0.0)

        Q_total  = float(np.sum(Q_pipe) * DZ)
        m_step   = Q_total / L_F
        m_cumul += m_step

        rows.append({"datetime": t_i, "valid": True,
                     "wf_depth_m": wf, "Q_lh_J_m2": Q_total,
                     "m_step_mm_we": m_step, "m_cumul_mm_we": m_cumul,
                     "sp_refreeze_mm_we": sp_step, "sp_cumul_mm_we": sp_cumul})

    return pd.DataFrame(rows), diff_grid, T_obs_grid, Q_lh_grid


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
    Four-panel figure (shared x-axis, tick labels on bottom panel only):
      1. T_obs depth-time curtain with obs wetting front
      2. Positive latent heat input Q_lh [J m⁻³] depth-time curtain
         (negative values discarded — per Saito et al. these are small
         spatial-heterogeneity artefacts, not physical heat loss)
      3. Daily refreezing rate [mm w.e. day⁻¹]:
         SNOWPACK column refreeze (blue) and piping estimate (red)
      4. Cumulative refreezing [mm w.e.]:
         SNOWPACK (blue), piping (red), total (black dashed)
    """
    # ── Aggregate sub-hourly obs to daily for legible curtain display ─────
    # The CN computation runs at obs timestep resolution; plotting every step
    # at sub-hourly resolution would produce illegibly thin stripes and huge
    # file sizes.  We resample T_obs and Q_lh to daily means/max for display.
    res_dt   = pd.DatetimeIndex(results["datetime"])
    nd       = len(diff_grid)

    T_df     = pd.DataFrame(T_on_grid[:-1], index=res_dt)
    Q_df     = pd.DataFrame(
        np.where(Q_lh_grid > 0, Q_lh_grid, np.nan),   # positive only
        index=res_dt)

    T_daily  = T_df.resample("D").mean()
    Q_daily  = Q_df.resample("D").max()                # max preserves event peaks

    # Wetting front: deepest valid value each day
    wf_ser   = results.set_index("datetime")["wf_depth_m"]
    wf_daily = wf_ser.resample("D").max()

    daily_idx = T_daily.index
    n_disp    = len(daily_idx)

    t_mpl   = mdates.date2num(daily_idx.to_pydatetime())
    t_edges = np.concatenate([[t_mpl[0] - 0.5 * (t_mpl[1] - t_mpl[0])],
                               0.5 * (t_mpl[:-1] + t_mpl[1:]),
                               [t_mpl[-1] + 0.5 * (t_mpl[-1] - t_mpl[-2])]])
    d_edges = np.concatenate([[diff_grid[0] - DZ / 2],
                               0.5 * (diff_grid[:-1] + diff_grid[1:]),
                               [diff_grid[-1] + DZ / 2]])

    wf_line  = wf_daily.values
    T_disp   = T_daily.values    # shape (n_disp, nd)
    Q_pos    = Q_daily.values    # shape (n_disp, nd)

    # ── Figure layout ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        4, 1, figsize=(16, 18), sharex=True,
        gridspec_kw={"height_ratios": [2.5, 2.5, 1, 1]},
        constrained_layout=True,
    )

    def _curtain(ax, data, cmap, norm, ylabel_cb, extend="max"):
        masked = np.ma.masked_invalid(data)
        pm = ax.pcolormesh(t_edges, d_edges, masked.T,
                           cmap=cmap, norm=norm, shading="flat")
        cb = fig.colorbar(pm, ax=ax, pad=0.01, fraction=0.018, extend=extend)
        cb.set_label(ylabel_cb, fontsize=9)
        ax.set_ylabel("Depth (m)", fontsize=10)
        ax.set_ylim(diff_grid[-1], diff_grid[0])
        ax.yaxis.set_major_locator(plt.MultipleLocator(5))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(1))
        ax.grid(axis="y", color="grey", alpha=0.25, lw=0.5)
        return pm

    # ── Panel 1: observed firn temperature ───────────────────────────────
    T_vmax = max(abs(np.nanpercentile(T_disp[np.isfinite(T_disp)], [1, 99])))
    _curtain(axes[0], T_disp, "RdYlBu_r",
             TwoSlopeNorm(vmin=-T_vmax, vcenter=-10, vmax=0),
             "T_obs (°C)", extend="both")
    axes[0].plot(t_mpl, wf_line, color="lime", lw=1.5,
                 label="Obs wetting front (−0.05°C)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_title("Observed firn temperature", fontsize=10,
                      fontweight="bold", loc="left", pad=4)

    # ── Panel 2: positive latent heat input ──────────────────────────────
    # vmin set to 500 J/m³ so noise-floor variations (sub-physical at depth,
    # likely from density/conductivity model imperfection or numerical residual)
    # do not draw the eye away from the real latent-heat signal.
    from matplotlib.colors import LogNorm as _LogNorm
    q_vals = Q_pos[np.isfinite(Q_pos)]
    if len(q_vals) and q_vals.max() > 0:
        q_vmax = float(np.nanpercentile(q_vals, 99))
        q_vmin = 500.0
        q_norm = _LogNorm(vmin=q_vmin, vmax=max(q_vmax, q_vmin * 10))
    else:
        q_norm = _LogNorm(vmin=500, vmax=1e5)
    _curtain(axes[1], Q_pos, "hot_r", q_norm, "Q_lh  (J m⁻³)", extend="max")
    axes[1].plot(t_mpl, wf_line, color="cyan", lw=1.5,
                 label="Obs wetting front")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].set_title(
        "Latent heat input  Q_lh = ρCₚ(T_obs − T_cond),  positive values only",
        fontsize=10, fontweight="bold", loc="left", pad=4)

    # ── Panel 3: daily refreezing rate ────────────────────────────────────
    ax3 = axes[2]
    t_all = mdates.date2num(
        pd.DatetimeIndex(results["datetime"]).to_pydatetime())

    sp  = results["sp_refreeze_mm_we"].fillna(0).values
    pip = results["m_step_mm_we"].fillna(0).values

    ax3.bar(t_all, sp,  width=0.8, color="steelblue", alpha=0.7,
            label="SNOWPACK refreeze")
    ax3.bar(t_all, pip, width=0.8, color="crimson", alpha=0.8,
            bottom=sp, label="Piping refreeze")
    ax3.set_ylabel("mm w.e. day⁻¹", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9, ncol=2)
    ax3.grid(axis="y", color="grey", alpha=0.25, lw=0.5)
    ax3.set_title("Daily refreezing rate", fontsize=10,
                  fontweight="bold", loc="left", pad=4)

    # ── Panel 4: cumulative refreezing ────────────────────────────────────
    ax4 = axes[3]
    sp_cumul  = results["sp_cumul_mm_we"].ffill().fillna(0).values
    pip_cumul = results["m_cumul_mm_we"].ffill().fillna(0).values
    tot_cumul = sp_cumul + pip_cumul

    ax4.fill_between(t_all, sp_cumul, alpha=0.25, color="steelblue")
    ax4.fill_between(t_all, sp_cumul, tot_cumul, alpha=0.25, color="crimson")
    ax4.plot(t_all, sp_cumul,  color="steelblue", lw=1.5,
             label=f"SNOWPACK  ({sp_cumul[-1]:.1f} mm)")
    ax4.plot(t_all, pip_cumul, color="crimson",   lw=1.5,
             label=f"Piping     ({pip_cumul[-1]:.1f} mm)")
    ax4.plot(t_all, tot_cumul, color="black",     lw=1.8, ls="--",
             label=f"Total      ({tot_cumul[-1]:.1f} mm)")
    ax4.set_ylabel("mm w.e.", fontsize=10)
    ax4.legend(loc="upper left", fontsize=9)
    ax4.grid(axis="y", color="grey", alpha=0.25, lw=0.5)
    ax4.set_title("Cumulative refreezing", fontsize=10,
                  fontweight="bold", loc="left", pad=4)

    # ── Shared x-axis (bottom panel only shows tick labels) ───────────────
    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axes[-1].xaxis.set_minor_locator(mdates.MonthLocator())
    plt.setp(axes[-1].xaxis.get_majorticklabels(),
             rotation=30, ha="right", fontsize=9)
    axes[-1].set_xlim(t_mpl[0], t_mpl[-1])

    fig.suptitle(f"Piping refreeze estimate (Saito 2024 method) — {site_label}",
                 fontsize=12, fontweight="bold")
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
    results, diff_grid, T_obs_grid, Q_lh_grid = compute_piping_refreeze(
        obs, pro, args.depth)

    n_valid   = results["valid"].sum()
    n_total   = len(results)
    total_mwe = results["m_cumul_mm_we"].iloc[-1]
    print(f"  Obs intervals: {n_valid}/{n_total} valid "
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
              f"({len(grp)} intervals, max wf "
              f"{grp['wf_depth_m'].max():.1f} m)")

    # Save CSV
    results.to_csv(paths["out_csv"], index=False, float_format="%.5f")
    print(f"\nSaved CSV → {paths['out_csv']}")

    # Plot
    print("Plotting …")
    make_figure(results, diff_grid, T_obs_grid, Q_lh_grid,
                pd.DatetimeIndex(results["datetime"]),
                paths["site_label"], paths["out_fig"])


if __name__ == "__main__":
    main()
