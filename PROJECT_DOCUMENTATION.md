# SNOWPACK Autorun — Project Documentation

> **For Claude Code sessions**: Read this file at the start of any session on this project to quickly get up to speed. The working directory is `/home/yeti/Documents/autorun_snowpack/` (or a site subdirectory). The primary files to understand are `autorun_snowpack.py`, `app.py`, and `compare_runs.py`.
>
> **At the end of every session, append a new dated section** (format: `## Update — YYYY-MM-DD`) summarising any new work: algorithms changed, bugs fixed, runs started/finished, design decisions made. Keep it concise but complete enough that the next session needs no extra context.
>
> **Commit and push to GitHub after every meaningful change** to `autorun_snowpack.py` or `app.py`.

---

## 1. Project Purpose

Automated end-to-end pipeline for running the SNOWPACK snow/firn model on borehole temperature datasets from glaciological field sites. The pipeline:

1. Reads hourly temperature observations from boreholes (CSV).
2. Downloads ERA5 meteorological forcing (via CDS API).
3. Writes SNOWPACK input files (`.sno` initial profile, `.ini` configuration).
4. Runs SNOWPACK hour-by-hour with temperature assimilation — nudging the modelled firn temperature profile toward observations.
5. Outputs a `.pro` timeseries file of modelled snow/firn properties.
6. Visualises results interactively via a Streamlit web app.

**SNOWPACK binary**: `/home/yeti/snowmodel/snowpack-master/bin/snowpack`

---

## 2. Directory Layout

```
/home/yeti/Documents/autorun_snowpack/
├── autorun_snowpack.py      # Main simulation runner
├── app.py                   # Streamlit interactive GUI
├── compare_runs.py          # Side-by-side PNG comparison of two runs
├── visualize_pro.py         # Standalone static plot helper
├── settings.toml            # Shared runtime settings (paths, physics, assimilation)
├── hole_locations_2025.csv  # Site lat/lon/elevation lookup table
├── AllCoreDataCommonFormat/ # Raw borehole temperature CSVs (all sites)
│   ├── Concatenated_Temperature_files/
│   └── Depth_change_estimate/PROMICE/   # Cumulative surface height change
│
├── plot_obs_vs_model.py     # Generate observed vs modelled figures for all sites
├── plot_T4_obs_vs_model.py  # T4-specific obs vs model (legacy, kept for app Regenerate button)
├── plot_UP18_obs_vs_model.py
├── plot_observed_temps.py               # Full 2022–2025 multi-site temperature heatmap
├── plot_observed_temps_2023.py          # 2023 summer window, no depth correction
├── plot_observed_temps_2023_depth_corrected.py  # 2023 summer, PROMICE depth correction
│
├── 2007_T2_10m/             # Site directory — canonical adaptive-RE run
├── 2007_T2_10m_bucket/      # Same site, BUCKET-only comparison run
├── 2019_T2minus_32m/
├── 2022_T3_25m/
├── 2022_T4_25m/
├── 2023_CP_25m/
├── 2023_UP18_25m/
└── 2022_2022_T3_25m_25m/    # Spurious duplicate directory — safe to delete
```

Each site directory mirrors `{year}_{site}_{depth}m`. Output `.pro` files are named `{year}-{site}-{depth}m_TEMP_ASSIM_RUN.pro`.

---

## 3. Site Naming and CLI Invocation

**CRITICAL**: The `site_id` is constructed internally as `f"{year}_{site}_{depth}m"`. Always pass the components separately:

```bash
# Correct
cd /home/yeti/Documents/autorun_snowpack
python autorun_snowpack.py --site T3 --year 2022 --depth 25

# WRONG — creates doubled site_id "2022_2022_T3_25m_25m"
cd /home/yeti/Documents/autorun_snowpack/2022_T3_25m
python autorun_snowpack.py --site 2022_T3_25m
```

**Key CLI flags**:

| Flag | Default | Purpose |
|------|---------|---------|
| `--site SITE` | required | e.g. `T3`, `T2`, `T2minus` |
| `--year YEAR` | required | e.g. `2022` |
| `--depth DEPTH` | required | Sensor depth in metres, e.g. `25` |
| `--run-tag TAG` | None | Appended to project_id for a parallel run without overwriting original data paths |
| `--water-transport` | None (uses settings.toml) | `adaptive` / `BUCKET` / `RICHARDSEQUATION` |
| `--run-until DATE` | `""` | Stop early, e.g. `"2023-12-31 00:00"` |

**`--run-tag` mechanics**: `project_id = f"{site_id}_{TAG}"`, so output goes to `2007_T2_10m_bucket/`. Data files (temperature CSV, ERA5 cache) still read from the canonical `site_id` directory. This allows running a different water transport scheme alongside the original.

---

## 4. Simulation Pipeline (`autorun_snowpack.py`)

### 4.1 Startup Sequence

1. Load `settings.toml`; CLI args override.
2. Build `site_id` and `project_id`.
3. Read borehole temperature CSV from `AllCoreDataCommonFormat/`.
4. Write `.sno` initial snow/firn profile from observations (`write_sno_file`).
5. Download or load ERA5 forcing cache.
6. Write SNOWPACK `.ini` configuration (`write_ini_file`).
7. **Checkpoint restore**: before `write_sno_file` runs, snapshot the existing `initial_profile.sno` bytes. After setup is complete, if the snapshot has a `ProfileDate` later than the new write, restore it — this preserves the resume point.
8. Call `cycle_hourly_snowpack_with_moving_profile`.

### 4.2 Resume Detection

At the start of the hourly loop, `i_start` is determined by reading `ProfileDate` from the existing `.sno` file and finding the matching timestamp in the forcing array. The loop fast-forwards to that index without re-running already-completed steps.

### 4.3 Hourly Loop

For each timestep:
- Advance the sensor depth (linear interpolation over the burial record).
- Assimilate temperature: compare modelled profile to observed, nudge layer temperatures toward observations (`alpha=0.1` blending coefficient).
- Run SNOWPACK for one 15-minute step (or the configured `calculation_step_length_min`).

### 4.4 Water Transport — Adaptive Mode

SNOWPACK supports two water transport schemes:

- **BUCKET**: simple bucket-fill, always stable, less physically accurate.
- **RICHARDSEQUATION (RE)**: solves Richards equation, more physically correct, but can fail to converge on wet firn/ice columns.

**Adaptive strategy** (default `water_transport = "adaptive"` in settings.toml):

1. First `stabilization_days` (default 15 days) always use BUCKET.
2. Switch to RE after stabilisation.
3. If RE reports `"Richards-Equation solver: no convergence"` (SafeMode or timeout):
   - Switch to BUCKET immediately.
   - Retry the same timestep with BUCKET (in case RE timed out).
   - Use BUCKET for the next 24 model hours.
   - Then automatically switch back to RE.

The `water_transport` setting lives in `settings.toml [run]` section and can be overridden by `--water-transport` CLI flag. It is also configurable from the app Settings tab.

**CRITICAL implementation detail**: The convergence check must occur *before* the `if not ok: raise RuntimeError` guard. When SNOWPACK times out (300 s), `ok=False`; if the convergence check came after, the RuntimeError would fire before the BUCKET fallback could trigger.

### 4.5 RE Warning Suppression

RE convergence failures generate verbose multi-line log blocks. These are suppressed in the GUI log viewer via `_LOG_FILTER` regex (in `app.py`) matching patterns like `"Richards-Equation solver"`, `"SafeMode was used"`, `"SAFE MODE"`, etc.

### 4.6 Temperature Assimilation

Each hour the modelled temperature profile is nudged toward observed borehole temperatures:

- Sensor depths are PROMICE-corrected (cumulative surface height change added to install depth).
- **Above-surface sensors are included** as interpolation anchors (removed `actual_depth_m >= 0.0` filter in `update_sno_temperatures_from_moving_profile`). This improves near-surface nudging during the early deployment period before sensors are buried.
- Observed temperatures are clipped to `[TEMP_MIN_C, 0.0°C]` before interpolation — above-surface sensors reading positive air temperatures are clamped to 0°C and cannot warm the firn above freezing.
- Linear interpolation (`np.interp`) between sensors is used to estimate temperature at every model layer midpoint.
- Only layers within the observed depth range (with 10 cm margin inward) are nudged.
- Wet layers are only drained when observed T < `wet_layer_drain_threshold_c` (default −1.0°C).

---

## 5. PRO File Format

SNOWPACK `.pro` files are the main output. Key record codes:

| Code | Quantity | Unit |
|------|----------|------|
| `0500` | Timestamp | `DD.MM.YYYY HH:MM:SS` |
| `501` | Layer heights (top of each element from bottom) | cm |
| `502` | Density | kg/m³ |
| `503` | Temperature | °C |
| `506` | Liquid water content (LWC) | % |
| `513` | Grain type | integer code |
| `515` | Ice volume fraction | — |

Layers are in Lagrangian coordinates (heights change as snow settles/melts). The app and `compare_runs.py` both regrid to a fixed Eulerian depth grid (`_to_grid`) using step-function nearest-layer assignment.

**Glacial ice artifact filter**: Dense basal ice layers (density ≥ 900 kg/m³ and `h > 0`) are excluded from the depth-profile plots to prevent them dominating the colour scale.

---

## 6. Streamlit App (`app.py`)

Run with:
```bash
cd /home/yeti/Documents/autorun_snowpack
streamlit run app.py
```

### 6.1 Tabs

| Tab | Contents |
|-----|----------|
| ▶ Run | Site selector, launch/stop controls, live log viewer |
| ⚙ Settings | Edit and save `settings.toml` (all sections including forcing adjustments and water transport) |
| 📊 Results | Per-site interactive plots (grain type, temperature, LWC, density) |
| 🌡 Obs vs Model | Static observed vs modelled temperature figures for all completed runs |

### 6.2 Obs vs Model Tab

Site radio selector with options: T2 (2007), T2 bucket (2007), T2− (2019), T3 (1794 m), T4 (1873 m), CP (1998 m), UP18 (2109 m).

- "Regenerate this site" button runs `plot_obs_vs_model.py <key>`.
- "Regenerate all sites" button runs `plot_obs_vs_model.py` (no args).
- Output PNGs are stored in the project root (e.g. `T4_obs_vs_model.png`, `CP_obs_vs_model.png`).

### 6.3 Results Tab — Plot Selector

Four checkboxes (none checked by default). Checking a box loads and renders the corresponding data:

| Checkbox | Plots shown |
|----------|-------------|
| Grain type | Grain type hover heatmap (HTML iframe, JS-rendered) |
| Temperature | Modelled temperature heatmap + optional observed T overlay |
| LWC & Refreezing | Liquid water content heatmap + cumulative refreezing line |
| Density | Density heatmap (HTML iframe, JS-rendered) |

### 6.4 Temperature Heatmap

- Colourmap: Turbo, capped at 0°C (`vmax=0`).
- White isotherm line drawn at exactly **−0.2°C** (not zero) using a Plotly Contour trace with `contours=dict(start=-0.2, end=-0.1, size=1)`.
- Observed temperature shown as an overplotted line series (one trace per sensor depth).
- Both modelled and observed use the same shared `coloraxis` for consistent scaling.

### 6.5 Cumulative Refreezing

Calculated in `load_pro` from simultaneous ice fraction increase + LWC decrease:

```python
d_ice = np.diff(ice_col)         # ice volume fraction change
d_lwc = np.diff(lwc_col)         # LWC change
refreezing_per_step = np.minimum(
    np.where(d_ice > 0, d_ice, 0.0),   # ice increasing
    np.where(d_lwc < 0, -d_lwc, 0.0),  # LWC decreasing
)
cumul_refreezing = np.concatenate([[0.0], np.cumsum(refreezing_per_step)])
```

**Design decision**: Refreezing below ~4 cm depth is considered permanent (ice at depth never re-melts in these firn columns). Therefore **no credit system** — simple cumulative sum is correct. Any refreezing event at any depth counts permanently.

### 6.6 X-axis Synchronisation

All Plotly subplots use `make_subplots(shared_xaxes=True)` so zoom/pan on any panel syncs all others.

### 6.7 Caching

`@st.cache_data` with a `_mtime` parameter (file modification timestamp) busts the cache when the `.pro` file changes during a live run.

### 6.8 PID Tracking

The app reads `{site_dir}/.autorun.pid` to determine if a run is active. Runs launched from the GUI write this file automatically. Runs launched from the terminal do **not** — you must write it manually:

```bash
echo <PID> > /home/yeti/Documents/autorun_snowpack/2022_T3_25m/.autorun.pid
```

---

## 7. Comparison Tool (`compare_runs.py`)

Produces a 2×3 matplotlib PNG comparing two runs:

```
[T_A]   [T_B]   [ΔT = B−A]
[LWC_A] [LWC_B] [ΔLWC = B−A]
```

Usage:
```bash
python compare_runs.py \
    --site-a 2007_T2_10m   --label-a "Adaptive (RE+BUCKET)" \
    --site-b 2007_T2_10m_bucket --label-b "BUCKET only"
```

Output: `2007_T2_10m_vs_2007_T2_10m_bucket_comparison.png` in the project root.

**Grid alignment**: both runs are interpolated onto the intersection of their time ranges (hourly, nearest-neighbour) and the shallower of their two depth maxima (linear depth interpolation).

---

## 8. Plot Conventions

- All SNOWPACK output plots use **depth from surface** on the y-axis: surface = 0 at top, depth increasing downward. The PRO file code 501 gives element-top heights in cm above the base — always convert: `depth_m = (surface_height_cm - element_height_cm) / 100.0`.
- **Isotherm**: all static figures draw a white contour at **−0.05°C**. The colorbar gray cutoff is also −0.05°C (values above are shown gray).
- Colorscale: discrete turbo, 2°C bins from −20°C to −0.05°C, gray above.
- Sensor depth lines (black, α=0.4, linewidth=0.5) on observed panels track PROMICE-corrected depth over time.

---

## 9. ERA5 Forcing and SMET File

### 9.1 Variables Used

| ERA5 field | SMET field | Conversion |
|-----------|-----------|------------|
| `t2m` (K) | `TA` (°C stored) | −273.15; SMET units_offset adds 273.15 back |
| `d2m` + `t2m` | `RH` (fraction) | Magnus formula |
| `u10`, `v10` | `VW`, `DW` | vector magnitude/direction |
| `ssrd` (J m⁻²) | `ISWR` (W m⁻²) | ÷3600 |
| `strd` (J m⁻²) | `ILWR` (W m⁻²) | ÷3600 |
| `tp` (m) | `PSUM` (mm) | incremental diff ×1000 |

### 9.2 Forcing Adjustments

Per-variable multiplicative and additive adjustments are stored in `settings.toml [forcing_adjustments]` and applied via the SMET header:

```
physical = stored_ERA5_value * multiplier + offset
```

For `TA`, the base offset of 273.15 (°C→K for SNOWPACK) is combined with the user additive: `smet_offset = 273.15 + ta_offset`. All other fields have base offset = 0.

**Old hardcoded factor removed**: precipitation was previously multiplied by 1.5 in Python before writing to SMET. This is now gone — use `psum_multiplier = 1.5` in `[forcing_adjustments]` instead (currently set to 1.5 to preserve the original behaviour).

### 9.3 Altitude

The SMET `altitude` field is derived from the ERA5 surface geopotential (`z` variable, m²/s²) at the nearest grid point: `altitude_m = z / 9.80665`. Falls back to `meta.elevation` from the temperature CSV header if the geopotential file is unavailable.

---

## 10. Configuration Reference (`settings.toml`)

Key sections and their purpose:

- `[paths]` — SNOWPACK binary path; data root (auto-detected if blank).
- `[run]` — `run_until` date, archiving options, `water_transport` scheme (`adaptive`/`BUCKET`/`RICHARDSEQUATION`).
- `[model]` — timestep (15 min), sensor height, assimilation `alpha` (0.1).
- `[physics]` — ice/water density, temperature tolerances.
- `[assimilation]` — T bounds, max hourly nudge, wet-layer drain threshold.
- `[basal]` — bottom boundary T estimation (`profile_gradient` mode, looks back 5 m, targets 15 m offset).
- `[era5]` — CDS API dataset names.
- `[forcing_adjustments]` — per-variable multiplier and offset for TA, RH, VW, ISWR, ILWR, PSUM. Applied via SMET header; no Python-side scaling.

All editable from the app's ⚙ Settings tab.

---

## 11. Key Algorithms Summary

### Temperature Assimilation
Each hour, the modelled temperature profile is nudged toward observed borehole temperatures using `alpha=0.1` (10% blend per hour). Limits: max 1.0°C adjustment per hour (changed from 0.1); observed T clamped to [−60, 0]°C. Above-surface sensors included as interpolation anchors (clamped to 0°C).

### Moving Sensor Depth
As firn settles, the physical depth of each thermistor changes. PROMICE cumulative surface height change is added to nominal install depth at each timestep.

### Basal Temperature
Estimated by extrapolating the gradient of the deepest 5 m of the observed profile to 15 m below the lowest thermistor. Prevents unconstrained drift in the basal layers.

### Refreezing Quantification
See §6.5. Uses simultaneous increase in ice volume fraction and decrease in LWC at the same layer and timestep. Simple cumulative sum — no credit system.

---

## 12. Workflow for a New Site

1. Place temperature CSV in `AllCoreDataCommonFormat/` following the existing naming convention.
2. Run:
   ```bash
   cd /home/yeti/Documents/autorun_snowpack
   python autorun_snowpack.py --site SITE --year YEAR --depth DEPTH
   ```
3. Monitor via `{site_dir}/autorun.log` or the Streamlit app.
4. If the run exits early, check the log for SNOWPACK errors; the adaptive fallback handles most RE convergence issues automatically.
5. To resume: just re-run the same command — resume detection reads `ProfileDate` from the existing `.sno` file.

---

## 13. Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Doubled site ID `2022_2022_T3_25m_25m` | `--site 2022_T3_25m` instead of `--site T3 --year 2022 --depth 25` | Always pass components separately |
| App shows "Crashed/stopped" for a terminal run | No `.autorun.pid` file | `echo <PID> > {site_dir}/.autorun.pid` |
| RE never triggers fallback | Convergence check placed after `if not ok: raise RuntimeError` | Check must come before the ok-guard |
| Resume point overwritten | `write_sno_file` runs before loop reads ProfileDate | Snapshot bytes before write, restore after setup |
| RE warning spam in log | RE SafeMode is very verbose | `_LOG_FILTER` regex in `app.py` suppresses it |
| CP obs vs model fails to load | Trailing comma in `.pro` file produces empty string | `float(v) for v in parts if v.strip()` guard in parse_pro |

---

## Update — 2026-04-21

### Temperature Assimilation — Above-Surface Sensors
- Removed `actual_depth_m >= 0.0` filter in `update_sno_temperatures_from_moving_profile`. Above-surface sensors now included as interpolation anchors. Observed values clamped to 0°C before interpolation, so air temperatures cannot warm firn above freezing. This improves near-surface nudging early in each deployment.

### ERA5 Forcing — Configurable Adjustments
- Removed hardcoded `*1.5` precipitation factor from Python code.
- Added `[forcing_adjustments]` section to `settings.toml` with per-variable `{ta,rh,vw,iswr,ilwr,psum}_{multiplier,offset}`. Applied via SMET `units_offset` / `units_multiplier` header — no Python-side scaling. Precipitation multiplier currently set to 1.5 in settings to preserve prior behaviour.
- ERA5 geopotential altitude (`z / 9.80665`) now used as SMET `altitude` instead of site elevation from CSV header. Falls back to site elevation if geopotential unavailable.

### Water Transport — Settings-Configurable
- Added `water_transport` to `settings.toml [run]` (default `"adaptive"`).
- CLI `--water-transport` default changed to `None`; settings.toml value used when flag absent.
- Selectbox added to app Settings tab.
- Current setting: `BUCKET` (user changed from `adaptive`).

### Obs vs Model Figures
- `plot_obs_vs_model.py`: single script generates 2-panel (observed top, modelled bottom) figures for all 6 completed sites. Time range auto-detected from data overlap. Sites: T2 (2007), T2 bucket (2007), T2− (2019), T3, T4, CP, UP18.
- T4 and UP18 individual scripts updated to 2-panel layout (removed raw-sensor bottom panel).
- App: consolidated two individual obs vs model tabs into single "🌡 Obs vs Model" tab with radio site selector and Regenerate buttons.

### Isotherm and Colorbar Update
- All temperature figures (observed multi-site and obs vs model) updated from −0.1°C to **−0.05°C** isotherm and gray colorbar cutoff.

### compare_runs.py Fixes (earlier session)
- `_parse_pro`: skip `"Date"` header line in `0500` records; wrap `int(parts[1])` in try/except for `nElems` header.
- `_to_grid`: fixed to use depth-from-surface (`surface - height`) instead of height-from-base as depth.

### max_adjust_per_hour_c
- Increased from 0.1°C to 1.0°C in settings (user changed via app).
