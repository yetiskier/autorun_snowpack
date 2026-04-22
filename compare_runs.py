"""
compare_runs.py  —  Compare two SNOWPACK .pro outputs side-by-side.

Usage:
    python compare_runs.py \
        --site-a 2007_T2_10m   --label-a "Adaptive (RE+BUCKET)" \
        --site-b 2007_T2_10m_bucket --label-b "BUCKET only"

Produces a single PNG with difference heatmaps for temperature and LWC.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
from scipy.interpolate import interp1d

SCRIPT_DIR = Path(__file__).resolve().parent


# ── PRO parser (mirrors load_pro in app.py, minimal version) ────────────────

def _parse_pro(pro_path: Path):
    """Return dict of {code: list-of-lists} and list of timestamps."""
    times, pro = [], {}
    current_time = None
    with open(pro_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("0500,"):
                raw = line[5:].strip()
                if raw == "Date":
                    continue
                current_time = pd.to_datetime(raw, dayfirst=True)
                times.append(current_time)
            elif "," in line:
                parts = line.split(",")
                try:
                    code = int(parts[0])
                except ValueError:
                    continue
                if code in (501, 502, 503, 506):
                    try:
                        n = int(parts[1])
                    except ValueError:
                        continue
                    vals = [float(v) for v in parts[2:2 + n]]
                    pro.setdefault(code, []).append(vals)
    return pd.DatetimeIndex(times), pro


def _to_grid(times, heights_list, values_list, depth_grid):
    """Lagrangian → Eulerian on a fixed depth-from-surface grid (step function)."""
    n_t = len(times)
    grid = np.full((n_t, len(depth_grid)), np.nan)
    for ti in range(n_t):
        h = np.asarray(heights_list[ti], dtype=float)
        v = np.asarray(values_list[ti], dtype=float)
        n = min(len(h), len(v))
        if n < 2:
            continue
        order = np.argsort(h[:n])
        h_s = h[:n][order]          # element tops ascending, cm above base
        v_s = v[:n][order]
        surface = h_s[-1]           # surface height, cm above base
        # Convert depth-from-surface grid (m) → height above base (cm)
        h_at_d = surface - depth_grid * 100.0
        idx = np.searchsorted(h_s, h_at_d, side="left")
        valid = (idx < n) & (h_at_d >= h_s[0])
        grid[ti, valid] = v_s[idx[valid]]
    return grid


def load_run(site_dir: Path):
    pro_files = list((site_dir / "output").glob("*.pro"))
    if not pro_files:
        raise FileNotFoundError(f"No .pro file in {site_dir / 'output'}")
    pro_path = pro_files[0]
    times, pro = _parse_pro(pro_path)
    if not (501 in pro and 503 in pro and 506 in pro):
        raise ValueError("Missing required PRO codes (501/503/506)")

    # Build common depth grid
    all_h = [h for hlist in pro[501] for h in hlist]
    depth_max = max(all_h) / 100.0
    depth_grid = np.arange(0.0, depth_max + 0.05, 0.05)

    T_grid   = _to_grid(times, pro[501], pro[503], depth_grid)
    LWC_grid = _to_grid(times, pro[501], pro[506], depth_grid)

    return times, depth_grid, T_grid, LWC_grid


def align_to_common_grid(times_a, dg_a, data_a,
                         times_b, dg_b, data_b):
    """Interpolate both runs onto a common (time, depth) grid."""
    # Common time axis: intersection
    t_start = max(times_a[0], times_b[0])
    t_end   = min(times_a[-1], times_b[-1])
    common_times = pd.date_range(t_start, t_end, freq="1h")

    # Common depth axis: shallower of the two maxima
    d_max = min(dg_a.max(), dg_b.max())
    common_depth = np.arange(0.0, d_max + 0.05, 0.05)

    def _interp(times, dg, data, ct, cd):
        # Time interpolation (nearest-neighbour on hourly grid is fine)
        tidx_a = np.array([np.argmin(np.abs(times - t)) for t in ct])
        d2 = data[tidx_a]          # (n_ct, n_depth_orig)
        # Depth interpolation (linear, NaN-aware)
        out = np.full((len(ct), len(cd)), np.nan)
        for ti in range(len(ct)):
            row = d2[ti]
            valid = ~np.isnan(row)
            if valid.sum() >= 2:
                f = interp1d(dg[valid], row[valid],
                             bounds_error=False, fill_value=np.nan)
                out[ti] = f(cd)
        return out

    A = _interp(times_a, dg_a, data_a, common_times, common_depth)
    B = _interp(times_b, dg_b, data_b, common_times, common_depth)
    return common_times, common_depth, A, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-a",  default="2007_T2_10m")
    ap.add_argument("--label-a", default="Adaptive (RE+BUCKET)")
    ap.add_argument("--site-b",  default="2007_T2_10m_bucket")
    ap.add_argument("--label-b", default="BUCKET only")
    ap.add_argument("--out",     default=None,
                    help="Output PNG path (default: <site-a>_vs_<site-b>_comparison.png)")
    args = ap.parse_args()

    dir_a = SCRIPT_DIR / args.site_a
    dir_b = SCRIPT_DIR / args.site_b

    print(f"Loading {args.site_a} …")
    ta, dga, T_a, LWC_a = load_run(dir_a)
    print(f"Loading {args.site_b} …")
    tb, dgb, T_b, LWC_b = load_run(dir_b)

    print("Aligning grids …")
    ct, cd, Ta, Tb = align_to_common_grid(ta, dga, T_a, tb, dgb, T_b)
    _,  _,  La, Lb = align_to_common_grid(ta, dga, LWC_a, tb, dgb, LWC_b)

    dT   = Tb - Ta          # BUCKET − adaptive
    dLWC = Lb - La

    out_path = Path(args.out) if args.out else \
        SCRIPT_DIR / f"{args.site_a}_vs_{args.site_b}_comparison.png"

    fig, axes = plt.subplots(2, 3, figsize=(18, 10),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.35})

    def _hm(ax, times, depth, data, title, cmap, vmin, vmax, unit):
        c = ax.pcolormesh(times, depth, data.T,
                          cmap=cmap, vmin=vmin, vmax=vmax,
                          shading="auto", rasterized=True)
        ax.invert_yaxis()
        ax.set_xlabel("Date")
        ax.set_ylabel("Depth (m)")
        ax.set_title(title, fontsize=10)
        plt.colorbar(c, ax=ax, label=unit, pad=0.02)

    T_vmin = np.nanpercentile(np.concatenate([Ta.ravel(), Tb.ravel()]), 2)
    T_vmax = 0.0
    L_vmax = min(10.0, np.nanpercentile(np.concatenate([La.ravel(), Lb.ravel()]), 98))

    _hm(axes[0, 0], ct, cd, Ta,   f"T — {args.label_a}",  "turbo", T_vmin, T_vmax, "°C")
    _hm(axes[0, 1], ct, cd, Tb,   f"T — {args.label_b}",  "turbo", T_vmin, T_vmax, "°C")
    dT_lim = max(0.5, np.nanpercentile(np.abs(dT), 99))
    _hm(axes[0, 2], ct, cd, dT,   f"ΔT (BUCKET − adaptive)",
        "RdBu_r", -dT_lim, dT_lim, "°C")

    _hm(axes[1, 0], ct, cd, La,   f"LWC — {args.label_a}", "Blues", 0, L_vmax, "%")
    _hm(axes[1, 1], ct, cd, Lb,   f"LWC — {args.label_b}", "Blues", 0, L_vmax, "%")
    dL_lim = max(0.5, np.nanpercentile(np.abs(dLWC), 99))
    _hm(axes[1, 2], ct, cd, dLWC, f"ΔLWC (BUCKET − adaptive)",
        "RdBu_r", -dL_lim, dL_lim, "%")

    fig.suptitle(f"{args.site_a}  vs  {args.site_b}", fontsize=13, y=1.01)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
