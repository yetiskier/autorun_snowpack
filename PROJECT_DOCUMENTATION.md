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

---

## Update — 2026-04-22 (Performance: assimilation interval, RAM disk, run status UI)

### assimilation_interval_h
- Added `assimilation_interval_h` to `settings.toml [run]` (default 1). Controls how many hours SNOWPACK runs per cycle. The assimilation temperature cap scales proportionally: `max_delta = MAX_ADJUST_PER_HOUR_C * n_hours`.
- `cycle_hourly_snowpack_with_moving_profile` now accepts `assimilation_interval_h` and steps the loop in chunks of that many hours.
- Exposed in app Settings tab.

### RAM disk (`use_ramdisk`)
- Added `use_ramdisk = true` to `settings.toml [run]`.
- `_setup_ramdisk(site_id, disk_project_dir)` creates `/dev/shm/snowpack_{site_id}/`, copies hot files (SNO, cfgfiles, current_snow) there, symlinks output/era_tmp/cache to disk.
- `configure()` redirects PROJECT_DIR / INPUT_DIR / CFG_DIR / CURRENT_SNOW_DIR to RAM paths when enabled. `DISK_INPUT_DIR` retains the real disk path for checkpoint syncs.
- After each assimilation step, modified SNO is synced back to disk.

### Run tab — core status UI
- `_core_status(y, s, d)` classifies each core as `fresh / complete / incomplete / running`.
- Four radio buttons always shown with counts: "Not yet run (N)", "Completed (N)", "Incomplete / crashed (N)", "In progress (N)".
- `get_pro_current_time` reads last 2 MB of .pro file (previously 3 KB, too small for ~125 KB/timestep blocks).
- `_temp_date_range` fixed to skip non-date rows (previously skipped first 4 rows but CSVs have 5 header rows, causing `pd.to_datetime("timestamp")` error → all cores falsely "fresh").
- `get_expected_date_range`: removed `@st.cache_data` to prevent stale (None, None) poisoning.
- `is_running`: added `/proc/*/cwd` fallback to detect runs launched from terminal (no PID file).

### SNOWPACK timeout
- Raised from 300 s to 900 s in `run_snowpack_one_step` (2022 T3 summer melt steps took up to 141 s).

### Settings tab reorganisation
- Settings tab uses 5 `st.expander` accordion sections: Paths, Run, Model & assimilation, Basal boundary, ERA5 forcing adjustments.

---

## Update — 2026-04-23 (Persistent SNOWPACK daemon)

### Problem
Each assimilation step spawned a fresh SNOWPACK process. The dominant startup cost was MeteoIO re-reading and pre-buffering the SMET forcing file on every respawn — O(record_length) per hour of simulation.

### Design
- Modified `SNOWPACK/applications/snowpack/Main.cc` to support `--daemon` mode.
- In daemon mode, SNOWPACK initialises once (reads SNO, pre-buffers SMET), then enters a command loop on stdin:
  - `RUN YYYY-MM-DDTHH:MM` — advance simulation to that date, write SNO checkpoint, print `CHECKPOINT <date>` to stdout, wait for next command.
  - `RELOAD_SNO` — re-read the (Python-modified) `.sno` file into memory; IOManager buffer stays warm.
  - `QUIT` — write final SNO and exit.
- The Hazard object and qr_Hdata arrays are reinitialised per chunk (duration changes). vecXdata, IOManager, SunObject persist across chunks.
- Recompiled: `/home/yeti/snowmodel/snowpack-master/bin/snowpack` (version 3.7.0, Apr 23 2026).

### Python changes (`autorun_snowpack.py`)
- `SnowpackDaemon` class manages the subprocess (spawn, wait_for_checkpoint, reload_and_run, respawn, quit).
- Daemon stdout is the pipe channel (CHECKPOINT lines); stderr is drained by a background thread and printed to console.
- `cycle_hourly_snowpack_with_moving_profile` accepts `use_daemon=False`. When True, spawns the daemon for the first chunk (which runs automatically on spawn), then uses `reload_and_run` for subsequent steps.
- Water-transport scheme switches (adaptive mode stabilisation switch, RE→BUCKET fallback) handled by `daemon.respawn()` — re-spawns with new INI, costs one cold start but only happens once per run.
- `USE_DAEMON` global loaded from `settings.toml [run]`; passed to `cycle_hourly_snowpack_with_moving_profile` from `main()`.
- Enabled by default: `use_daemon = true` in `settings.toml`.

### App changes (`app.py`)
- Added "Persistent daemon" checkbox to Settings tab Run section (alongside RAM disk checkbox).

---

## Update — 2026-04-23 (SETTEMPS — direct in-memory temperature injection)

### Problem
Even with the daemon, each assimilation step still required:
1. Python writing the adjusted `.sno` file (~5 ms)
2. C++ `RELOAD_SNO` parsing and re-loading the file (~50–100 ms)

### Design — SETTEMPS command
Added a new daemon stdin command: `SETTEMPS i0:T0,i1:T1,...`

- Indices are 0 = bottom layer, matching `Edata[]` order.
- Temperatures are in Kelvin (native SNOWPACK units).
- C++ applies `Xdata.Edata[idx].Te = T_K` directly to the in-memory state.
- Responds `READY\n` when done; Python then sends `RUN <date>`.
- Eliminates the SNO file round-trip for temperature-only adjustments.

### C++ changes (`Main.cc`)
- Added `SETTEMPS` branch in the daemon command loop (between `RELOAD_SNO` and `RUN` handlers).
- Parses `"SETTEMPS i:T,i:T,..."` with `std::istringstream` + `std::stoul`/`std::stod`.
- Uses `Xdata.getNumberOfElements()` (public accessor) to bounds-check indices.
- Responds `READY\n` then continues waiting for `RUN`.
- Recompiled: `/home/yeti/snowmodel/snowpack-master/bin/snowpack` (Apr 23 2026, 2491296 bytes).

### Python changes (`autorun_snowpack.py`)
- `SnowpackDaemon.settemps_and_run(adjustments, new_end)`:
  - Sends `SETTEMPS i0:T0,...\n`, waits for `READY`, sends `RUN <date>\n`, waits for CHECKPOINT.
  - Returns `(checkpoint_str, log_text)` like `reload_and_run`.
- `update_sno_temperatures_from_moving_profile` adds `return_adjustments: bool = False`:
  - When `True`: skips SNO write; returns `(adjustments, needs_reload)`.
  - `adjustments = [(layer_idx_from_bottom, T_K), ...]` for all layers.
  - `needs_reload=True` when wet-layer draining occurred (volume fraction changes needed) — SNO is written and caller must use `RELOAD_SNO` instead of SETTEMPS.
  - Early-return branch (< `MIN_OBS_FOR_ADJUST` obs): always writes SNO, returns `([], True)`.
- `SETTEMPS_SNO_WRITE_INTERVAL = 24`: daemon uses SETTEMPS for up to 24 consecutive steps, then falls back to full SNO write + RELOAD_SNO for crash recovery.
- Cycle loop tracks `_pending_adjustments` (set each step) and `_settemps_steps_since_sno_write` counter.
- Both are reset to `None / 0` after any `daemon.respawn()` call.
- Disk-sync (RAM disk → real disk) skipped when SETTEMPS was used (SNO not written that step).

---

## Update — 2026-04-23 (App improvements: completion date, LWC log scale, obs vs model tab)

### Run completion date in dropdown
- `_run_completed_date(sid)` uses `log_path(sid).stat().st_mtime` (wall-clock time the log was last written) rather than the last model timestep in the .pro file.
- `_label()` updated to accept `t_done` kwarg; appends `" (completed YYYY-MM-DD)"` suffix to dropdown labels for completed runs.

### LWC interactive plot — log scale
- LWC data pre-transformed with `np.log10(np.clip(..., _LWC_FLOOR, None))` before passing to Plotly.
- `customdata` carries the raw (linear) values for hover display.
- Explicit tick labels at 0.01, 0.1, 1, 5, ≥10 kg m⁻².

### Obs vs Model tab — auto-discovery and controls
- `_OVM_HARDCODED`: list of 7 known sites with hand-written labels.
- Auto-discovery loop appends any `sites_with_results()` entries not already in the hardcoded list as `_ovm_extra`.
- `_OVM_ALL = _OVM_HARDCODED + _ovm_extra`: radio selector covers all completed runs.
- "Regenerate this site" expander has three columns: start date, end date, max depth (default 10 m). Passes `--start`, `--end`, `--max-depth` to `plot_obs_vs_model.py` subprocess.
- Auto-generation at end of run: `autorun_snowpack.py` calls `plot_obs_vs_model.py <sid>` via subprocess after `main()` completes.

### Obs vs Model figure layout
- `figsize` reduced from `(20, 16)` to `(20, 11.2)` (−30% height per subplot).
- Legend repositioned to `bbox_to_anchor=(1.01, 1.0)` so top of legend aligns with top of upper subplot.
- `make_figure()` signature extended: `make_figure(site, t_start_override, t_end_override, max_depth)`.
- `load_observed` and `load_modelled` both accept `depth_grid` kwarg built from `max_depth`.
- `site_from_sid(sid)`: auto-discovers the largest `.pro` file in `output/`, derives human label from sid string. Used when autorun calls the script at end of run.

---

## Update — 2026-04-23 (SNOWPACK patched source files in repo; RE density-dependent θᵣ)

### Patched source files now tracked in repo
- `snowpack/Main.cc` — SETTEMPS + daemon patch (already present).
- `snowpack/vanGenuchten.cc` — new: density-dependent residual saturation (see below).
- `snowpack/build_snowpack.sh` — updated to copy both files into the SNOWPACK source tree before building.

### RE density-dependent residual saturation (`vanGenuchten.cc`)
Modified `SetVGParamsSnow()` in `vanGenuchten.cc` to replace the hard-coded `0.02` cap on `theta_r` with `ElementData::snowResidualWaterContent(EMS->theta[ICE])` — the same Coléou & Lesaffre (1998) formula BUCKET already uses. Added `#include <snowpack/DataClasses.h>`.

**Effect by density:**
- Low-density firn (~400 kg/m³): θᵣ ≈ 0.05–0.08 (was capped at 0.02)
- High-density ice (>850 kg/m³): θᵣ ≈ 0.027, but clamped to `θ_water − ε` so effectively near zero for dry/dense layers

Recompiled: `/home/yeti/snowmodel/snowpack-master/bin/snowpack`.

---

## Update — 2026-04-23 (T2minus RE comparison runs and LWC analysis)

### Comparison runs for 2019_T2minus_32m
Two adaptive RE runs exist side-by-side:
- **Fixed θᵣ:** `output/2019-T2minus-32m_TEMP_ASSIM_RUN_RE_fixed_theta_r.pro`
- **Dynamic θᵣ:** `output/2019-T2minus-32m_TEMP_ASSIM_RUN.pro` (current, Coléou)
Checkpoint backup: `current_snow_backup_fixed_theta_r/`, `input/initial_profile_checkpoint_backup.sno`.

**Critical: how to force a fresh run**
The checkpoint lives in two places — both must be cleared before relaunching:
1. `<site>/input/initial_profile.sno`
2. `/dev/shm/snowpack_<sid>/input/initial_profile.sno` (ramdisk — setup only copies if not already present, so it persists across runs if not explicitly cleared)

**Critical: always use `adaptive` for these sites, not `--water-transport RICHARDSEQUATION`**
Pure RE from step 0 crashes immediately on melt-season spinup (SOLVER DUMP flood, 3.5M lines, Python timeout after 900s on step 1). The original runs all used adaptive mode (BUCKET for first ~2 weeks, then RE). Rate with adaptive: ~6.5 steps/s, total ~5 min for 1819 steps.

### Comparison scripts
- `compare_RE_theta_r.py` — 4-panel: temperature and LWC for both runs
- `compare_LWC.py` — 3-panel: fixed LWC, dynamic LWC, ΔLWC difference (log scale + RdBu_r)
- `check_density_vs_diff.py` — 3-panel: density, ice fraction, ΔLWC

### LWC difference analysis results
Correlation of ΔLWC (dynamic − fixed) with density over 365,619 valid pixels:
- Pearson r = +0.42, Spearman ρ = +0.24

**Unexpected finding:** Blue regions (dynamic has *less* LWC) are concentrated in **low-density firn** (300–750 kg/m³), not high-density ice. Mean ΔLWC by bin:
- 300–450 kg/m³: −0.148 (61% blue)
- 450–650 kg/m³: −0.07 to −0.08 (50–56% blue)
- 750–917 kg/m³: +0.07 to +0.11 (44–46% blue)

**Working hypothesis:** Higher θᵣ in low-density firn reclassifies water as immobile residual, altering percolation routing — water that pooled in low-density layers in the fixed run is redistributed or drains faster in the dynamic run. Mechanism not fully resolved.
