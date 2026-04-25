# SNOWPACK Autorun — Project Documentation

> **For Claude Code sessions**: Read this file at the start of every session. Working directory: `/home/yeti/Documents/autorun_snowpack/`. Primary files: `autorun_snowpack.py`, `app.py`, `settings.toml`. After completing any meaningful change: update this file, then `git add`, `git commit`, `git push` — always in that order, always in the same turn as the code change. Never end a session with uncommitted work.

---

## 1. Project Purpose

Automated end-to-end pipeline for running the SNOWPACK snow/firn model on borehole temperature datasets from glaciological field sites. The pipeline:

1. Reads hourly temperature observations from boreholes (CSV).
2. Downloads ERA5 meteorological forcing (via CDS API).
3. Builds SNOWPACK input files (`.sno` initial profile, `.ini` configuration).
4. Runs SNOWPACK hour-by-hour with temperature assimilation — nudging the modelled firn temperature profile toward observations.
5. Outputs a `.pro` timeseries file of modelled snow/firn properties.
6. Visualises results interactively via a Streamlit web app.

**SNOWPACK binary**: `/home/yeti/snowmodel/snowpack-master/bin/snowpack`
(Patched and recompiled — see §12. Do not replace with stock binaries.)

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
│
├── AllCoreDataCommonFormat/
│   ├── Concatenated_Temperature_files/    # Borehole temperature CSVs, all sites
│   └── Depth_change_estimate/PROMICE/     # Cumulative surface height change
│
├── plot_obs_vs_model.py                           # Observed vs modelled figures (all sites)
├── plot_T4_obs_vs_model.py                        # T4-specific legacy script (kept for app button)
├── plot_UP18_obs_vs_model.py
├── plot_observed_temps.py                         # Full 2022–2025 multi-site temperature heatmap
├── plot_observed_temps_2023.py                    # 2023 summer, no depth correction
├── plot_observed_temps_2023_depth_corrected.py    # 2023 summer, PROMICE depth correction
│
├── 2007_T2_10m/             # Site directory — canonical adaptive-RE run
├── 2007_T2_10m_bucket/      # Same site, BUCKET-only comparison run (--run-tag bucket)
├── 2019_T2minus_32m/
├── 2022_T3_25m/
├── 2022_T4_25m/
├── 2023_CP_25m/
├── 2023_UP18_25m/
└── 2022_2022_T3_25m_25m/    # Spurious duplicate directory — safe to delete
```

Each site directory mirrors `{year}_{site}_{depth}m`. Output `.pro` files are named `{year}-{site}-{depth}m_TEMP_ASSIM_RUN.pro`.

Each site run directory contains:
```
{site_dir}/
├── input/          # initial_profile.sno, site_forcing.smet
├── cfgfiles/       # site_run.ini (rewritten each step)
├── current_snow/   # SNOWPACK checkpoint .sno files (rolling, keep_last_n_sno=3)
├── output/         # .pro, .ini, .png, run_status.json, water_transport_*.csv
│   └── chunks/     # Intermediate .pro chunks (merged at run end)
├── era_tmp/        # ERA5 download cache
└── cache/          # Geopotential .nc cache
```

---

## 3. Running the Model

### 3.1 CLI Invocation

**CRITICAL**: The `site_id` is constructed as `f"{year}_{site}_{depth}m"`. Always pass components separately:

```bash
# Correct
cd /home/yeti/Documents/autorun_snowpack
python autorun_snowpack.py --site T3 --year 2022 --depth 25

# WRONG — creates doubled site_id "2022_2022_T3_25m_25m"
python autorun_snowpack.py --site 2022_T3_25m
```

**Key CLI flags**:

| Flag | Default | Purpose |
|------|---------|---------|
| `--site SITE` | required | e.g. `T3`, `T2`, `T2minus` |
| `--year YEAR` | required | e.g. `2022` |
| `--depth DEPTH` | required | Sensor depth in metres, e.g. `25` |
| `--run-tag TAG` | None | Suffix for project_id; output goes to `{site_id}_{TAG}/` while data files still read from canonical `{site_id}/` |
| `--water-transport` | None (uses settings.toml) | `adaptive` / `BUCKET` / `RICHARDSEQUATION` |
| `--run-until DATE` | `""` | Stop early, e.g. `"2023-12-31 00:00"` |
| `--fresh` | False | Wipe checkpoint before starting |
| `--fresh-mode` | `archive` | `archive` (compress previous output) or `delete` |

### 3.2 Starting a New Run vs Resuming

**Resume** (default): If `input/initial_profile.sno` exists and its `ProfileDate` is later than the forcing start date, the loop fast-forwards to that timestamp. No flags needed — just re-run the same command.

**Fresh start**: Use `--fresh` to wipe the checkpoint and restart from the beginning. The app's "Fresh start" checkbox does the same. The checkpoint lives in **two places** — both are cleared on fresh start:
1. `{site_dir}/input/initial_profile.sno`
2. `/dev/shm/snowpack_{sid}/input/initial_profile.sno` (ramdisk — persists across runs unless explicitly cleared)

Fresh start can **archive** (recommended) or **delete** previous output. Archive writes `output/archives/run_YYYYMMDD_HHMMSS.tar.gz` containing the previous `.pro`, `.ini`, and `autorun.log`.

**Always use `adaptive` water transport for these sites, not pure `RICHARDSEQUATION` from step 0.** Pure RE on a cold initial profile crashes immediately (SOLVER DUMP flood, ~3.5 M log lines, Python timeout after 900 s on step 1). The adaptive scheme runs BUCKET for stabilisation first.

---

## 4. Simulation Pipeline (`autorun_snowpack.py`)

### 4.1 Startup Sequence

1. Load `settings.toml`; CLI args override.
2. Build `site_id` / `project_id`.
3. Read borehole temperature CSV (`AllCoreDataCommonFormat/`).
4. **Checkpoint check**: read `ProfileDate` from existing `.sno`. If it is later than the first forcing timestamp, skip SNO build (resuming).
5. If not resuming: build `.sno` from observed density + temperature profiles; download ERA5; write `.ini`.
6. Call `cycle_hourly_snowpack_with_moving_profile`.
7. After loop completes: call `plot_obs_vs_model.py {sid}` to regenerate the static PNG.

### 4.2 Hourly Loop (`cycle_hourly_snowpack_with_moving_profile`)

For each timestep `t0 → t1`:
1. Check for scheme switch / chunk rotation (see §5, §9).
2. Run SNOWPACK (daemon or subprocess).
3. Check for RE convergence failure (must come **before** `if not ok: raise`).
4. Write per-step diagnostics: `run_status.json`, water-transport event check.
5. Validate SNO geometry.
6. Assimilate temperature (see §6).
7. Sync checkpoint SNO to disk (if using ramdisk and SNO was written this step).

---

## 5. Water Transport Scheme

### 5.1 Background

SNOWPACK supports two water transport schemes:
- **BUCKET**: simple bucket-fill. Always stable, fast, less physically accurate.
- **RICHARDSEQUATION (RE)**: solves Richards equation. More physically correct for liquid water percolation, but can fail to converge on saturated or near-ice firn.

### 5.2 Adaptive Mode (current default)

The `water_transport = "adaptive"` setting in `settings.toml` implements:

1. **Stabilisation phase** (first `stabilization_hours`, default 15 h): always BUCKET. This lets the initial profile equilibrate before RE is asked to handle potentially difficult states from the observed density profile.
2. **Switch to RE** after stabilisation. Daemon is respawned with updated INI.
3. **Convergence fallback**: if RE reports `"Richards-Equation solver: no convergence"`:
   - Switch to BUCKET immediately.
   - If the step timed out (`ok=False`): retry the same timestep with BUCKET via daemon respawn.
   - If the step completed in SafeMode (`ok=True`): continue; future steps use BUCKET.
   - Use BUCKET for the next 24 model hours, then automatically switch back to RE.

**Critical implementation detail**: The convergence check must run *before* `if not ok: raise RuntimeError`. When SNOWPACK times out, `ok=False`; if the check came after, the exception would fire before the BUCKET retry could happen.

### 5.3 Setting History

| When | Change |
|------|--------|
| Initial | Hard-coded BUCKET throughout |
| Later | Adaptive mode added: BUCKET stabilisation → RE switch → automatic fallback |
| 2026-04-21 | `water_transport` added to `settings.toml [run]`; CLI `--water-transport` default changed to `None` (falls back to settings); selectbox added to app Settings tab |

The current setting in `settings.toml` is `water_transport = "adaptive"`.

### 5.4 Water-Transport Log

`output/water_transport_log.csv` — a dense per-step record (one row per completed hourly step) with columns `datetime,scheme`. Used for overlaying BUCKET vs RE periods on diagnostic plots.

**How it is written** (event-based, not per-step):
- During the run, only scheme *transitions* are appended to `output/water_transport_events.csv` (typically 2–5 rows per run). This file is rewritten on each transition so it survives crashes.
- `_step_scheme` is captured immediately before `using_re` can be mutated by the convergence check, so SafeMode-completed RE steps are recorded correctly as RE even though the next 24 h will use BUCKET.
- At run end or crash (in the `finally` block), events are expanded into the dense `water_transport_log.csv` by walking the event list.

`read_water_transport_log(sid)` in `app.py` loads the dense CSV as a datetime-indexed DataFrame with a `scheme` column (deduplicates by keeping the last entry per timestamp).

Both diagnostic files are deleted on fresh start.

---

## 6. Temperature Assimilation

### 6.1 Algorithm

Each hour, the modelled temperature profile is nudged toward observed borehole temperatures:

1. Sensor depths are PROMICE-corrected at each timestep: `actual_depth = install_depth + surface_height_change`.
2. Observed temperatures are clipped to `[temp_min_c, temp_max_c]` (default `[−60, 0]°C`).
3. Linear interpolation (`np.interp`) estimates target temperature at every model layer midpoint.
4. Blending: `T_new = (1 − alpha) × T_model + alpha × T_obs` for layers within the observed depth range.
5. The maximum temperature change per step is capped at `max_adjust_per_hour_c × n_hours`.
6. Wet layers are only drained when observed T < `wet_layer_drain_threshold_c` (default −1.0°C).

**Above-surface sensors are included** as interpolation anchors. Sensors still above the snowpack surface report air temperature, which is clamped to 0°C before interpolation. This improves near-surface nudging during the early deployment period. (The original code excluded sensors with `actual_depth_m < 0`.)

### 6.2 SETTEMPS Fast Path

When the daemon is active and no volume-fraction changes are needed (no wet-layer draining, within `SETTEMPS_SNO_WRITE_INTERVAL = 24` steps since last SNO write), temperature adjustments are injected directly into the daemon's in-memory state via the `SETTEMPS` command rather than writing a `.sno` file and issuing `RELOAD_SNO`. This bypasses the ~50–100 ms SNO file round-trip per step. Falls back to full SNO write + `RELOAD_SNO` when wet draining occurs or the interval expires (for crash recovery).

### 6.3 Parameter History

| Parameter | Original | Current | When changed |
|-----------|----------|---------|--------------|
| `alpha` | 0.1 (hard-coded) | 0.1 (settings.toml) | Moved to settings |
| `max_adjust_per_hour_c` | 0.1°C (hard-coded) | **1.0°C** (settings.toml) | Moved to settings 2026-04-21; user changed value via app |
| `temp_min_c` | −60°C (hard-coded) | −60°C (settings.toml) | Moved to settings |
| `temp_max_c` | 0°C (hard-coded) | 0°C (settings.toml) | Moved to settings |
| Above-surface sensors | Excluded (`depth >= 0` filter) | **Included** (clamped to 0°C) | Changed 2026-04-21 |
| `assimilation_interval_h` | 1 (hard-coded) | 1 (settings.toml) | Added 2026-04-22; scales max_adjust cap proportionally |

All assimilation parameters are editable in the app's ⚙ Settings tab under "Model & assimilation".

---

## 7. Snow/Firn Profile Management (`.sno` Files)

### 7.1 Initial Profile Build

From observed borehole density and temperature data:
- Layer thickness, temperature, ice volume fraction, and void fraction are computed.
- Grain microstructure defaults (rg, rb, dd, sp, mk) are set from `settings.toml [snow_defaults]`.
- Basal layers are appended with temperature estimated by extrapolating the gradient of the deepest 5 m of the observed profile to 15 m below the lowest thermistor (prevents unconstrained basal drift). Mode and parameters in `settings.toml [basal]`.

### 7.2 Ice Volume Fraction Cap (`max_vol_frac_ice`)

**Problem**: SNOWPACK's compaction physics can drive `θ_i → 1.0` exactly in ice lenses. When `θ_i = 1.0`, porosity = 0, which makes the RE van Genuchten retention curve and hydraulic conductivity `K(θ)` degenerate. The RE Newton solver enters an infinite retry loop emitting `"set surfacefluxrate from 0 to 0"` without advancing. This caused the 2022_T3_25m crash (9 layers with `θ_i = 1.0`, 18,000 warnings, 900 s daemon timeout).

**Fix**: `max_vol_frac_ice` (default 0.98, range 0.90–0.99, in `settings.toml [physics]`, editable in app). Applied at every SNO write:

- **Restart-file paths** (all paths through `rewrite_sno_profiledate_and_clip_timestamps`): when `θ_i > MAX_VOL_FRAC_ICE`, `Layer_Thick` is scaled by `θ_i_old / θ_i_new` so that ice mass per unit area (`ρ_i × θ_i × thick`) and sensible heat per unit area (`ρ_i × c_i × T × θ_i × thick`) are both conserved. The air pore `θ_v` absorbs the freed volume. `HS_Last` in the header is updated to reflect the slightly taller column (~1.2 cm for the T3 profile).
- **Initial profile build** (`write_sno_file`): simple cap only — measurement precision at ρ ≈ 917 kg/m³ cannot distinguish 898 from 917, so the 2% correction is within instrument error and no thickness compensation is applied.

Conservation properties (restart paths): mass ✓, sensible heat ✓, latent heat budget ✓ (ice mass preserved).

At 0.98, effective ice-layer density ≈ 898 kg/m³ vs 917 kg/m³ — a residual 2% air pore that keeps RE well-posed.

---

## 8. ERA5 Forcing and SMET File

### 8.1 Variables Used

| ERA5 field | SMET field | Conversion |
|-----------|-----------|------------|
| `t2m` (K) | `TA` (°C stored) | −273.15; SMET `units_offset` adds 273.15 back for SNOWPACK |
| `d2m` + `t2m` | `RH` (fraction) | Magnus formula |
| `u10`, `v10` | `VW`, `DW` | vector magnitude/direction |
| `ssrd` (J m⁻²) | `ISWR` (W m⁻²) | ÷3600 |
| `strd` (J m⁻²) | `ILWR` (W m⁻²) | ÷3600 |
| `tp` (m) | `PSUM` (mm) | incremental diff ×1000 |

### 8.2 Altitude

Derived from ERA5 surface geopotential (`z`, m²/s²) at the nearest grid point: `altitude_m = z / 9.80665`. The geopotential file is downloaded once per site and cached in `{site_dir}/cache/`. Falls back to `meta.elevation` from the temperature CSV header if the cache is unavailable.

Previously: always used the elevation from the CSV header. Changed 2026-04-21.

### 8.3 Forcing Adjustments

Per-variable multiplicative and additive adjustments are stored in `settings.toml [forcing_adjustments]` and applied via the SMET header (`units_multiplier`, `units_offset`):

```
physical = stored_ERA5_value × multiplier + offset
```

For `TA`, the base offset of 273.15 (°C → K) is combined with the user additive: `smet_offset = 273.15 + ta_offset`. All other fields have base offset = 0.

**History**: Precipitation was originally multiplied by 1.5 as a hard-coded factor in Python before writing to SMET. This was removed in 2026-04-21 and replaced with `psum_multiplier = 1.5` in `[forcing_adjustments]` (currently set to 1.5 to preserve prior behaviour). All forcing adjustments are now editable from the app Settings tab under "ERA5 forcing adjustments".

---

## 9. Performance: Daemon, RAM Disk, Chunking

### 9.1 SNOWPACK Daemon

**Problem** (original): Each assimilation step spawned a new SNOWPACK process. The dominant startup cost was MeteoIO re-reading and pre-buffering the SMET forcing file — O(record_length) per step.

**Solution**: Modified `snowpack/Main.cc` to support `--daemon` mode. In daemon mode, SNOWPACK initialises once (reads SNO, pre-buffers SMET) then enters a command loop on stdin:

| Command | Action |
|---------|--------|
| `RUN YYYY-MM-DDTHH:MM` | Advance simulation to that date, write SNO checkpoint, print `CHECKPOINT <date>`, wait |
| `RELOAD_SNO` | Re-read the Python-modified `.sno` from disk into memory; IOManager buffer stays warm |
| `SETTEMPS i0:T0,i1:T1,...` | Apply temperature adjustments directly to in-memory `Edata[i].Te`; print `READY` |
| `QUIT` | Write final SNO and exit |

The Hazard object is reinitialised per chunk; IOManager, vecXdata, and SunObject persist.

`SnowpackDaemon` Python class manages the subprocess (spawn, wait_for_checkpoint, reload_and_run, settemps_and_run, respawn, quit). Daemon stdout is the pipe channel; stderr is drained by a background thread and printed to console.

Daemon respawn (one cold start per scheme switch, ~1 s) is used for: BUCKET → RE switch, RE → BUCKET fallback, .pro chunk rotation. Enabled by default: `use_daemon = true` in `settings.toml`.

### 9.2 RAM Disk

`use_ramdisk = true` in `settings.toml`. `_setup_ramdisk()` creates `/dev/shm/snowpack_{sid}/` and copies hot working files (SNO, cfgfiles, current_snow) there. Large sequential-write outputs (output/, era_tmp/, cache/) stay on disk via symlinks. After each assimilation step where the SNO was written, it is synced back to disk (not done when SETTEMPS was used, since SNO was not written that step).

### 9.3 .pro File Chunking

`pro_chunk_hours = 720` in `settings.toml [run]` (default 720 model hours = 30 days; 0 = disabled).

**Why**: For multi-year runs the `.pro` file grows to hundreds of MB. Chunking keeps individual files manageable and allows earlier chunks to be analysed while the run continues.

**Mechanism**: Every `pro_chunk_hours` model steps, the current `.pro` is moved to `output/chunks/chunk_NNNN.pro` and the daemon is respawned so SNOWPACK writes to a fresh `.pro`. At run end (or crash), `concatenate_pro_chunks()` merges all chunks + current `.pro` into the final `.pro` by taking the header from chunk 0 and appending data records from all sources. Non-daemon mode rotates the file before calling `run_snowpack_one_step`. Chunks and diagnostic files are cleared on fresh start.

### 9.4 ETA Progress Bar

`output/run_status.json` is written at each completed step with `run_start_model`, `run_start_wall`, `step_model`, `step_wall`. The app reads this, computes the current model-time-per-wall-second rate, and appends `· ETA X.X h / N min / <2 min` to the progress bar label. Only shown while running; rate resets on each resume (reflects the current session's speed, not cumulative since the original run start).

---

## 10. Output Files

### 10.1 PRO File Format

SNOWPACK `.pro` files are the main output. Header contains field definitions; data records follow with one block per timestep.

Key record codes:

| Code | Quantity | Unit |
|------|----------|------|
| `0500` | Timestamp | `DD.MM.YYYY HH:MM:SS` |
| `0501` | Layer heights (top of each element from bottom) | cm |
| `0502` | Density | kg/m³ |
| `0503` | Temperature | °C |
| `0506` | Liquid water content (LWC) | % |
| `0513` | Grain type | integer code |
| `0515` | Ice volume fraction | — |
| `0516` | Air volume fraction | — |

Layers are in Lagrangian coordinates (heights change as snow settles/melts). The app and `compare_runs.py` regrid to a fixed Eulerian depth grid (`_to_grid`) using nearest-layer assignment.

**Glacial ice artifact filter**: Dense basal ice layers (density ≥ 900 kg/m³ and `h > 0`) are excluded from depth-profile plots to prevent them dominating the colour scale.

### 10.2 Per-Run Diagnostic Files

All written to `output/`; cleared on fresh start.

| File | Written | Content |
|------|---------|---------|
| `run_status.json` | Each step | `run_start_model`, `run_start_wall`, `step_model`, `step_wall` — used by app to compute ETA |
| `water_transport_events.csv` | On scheme transitions only | Sparse: `datetime,scheme` rows at transition points; survives crash |
| `water_transport_log.csv` | At run end / crash | Dense: one `datetime,scheme` row per completed hourly step; expanded from events |

---

## 11. Streamlit App (`app.py`)

```bash
cd /home/yeti/Documents/autorun_snowpack
streamlit run app.py
```

### 11.1 Tabs

| Tab | Contents |
|-----|----------|
| ▶ Run | Site selector, launch/stop controls, progress bar with ETA, status |
| ⚙ Settings | Edit and save `settings.toml` |
| 📊 Results | Per-site interactive plots (grain type, temperature, LWC, density) |
| 🌡 Obs vs Model | Static observed vs modelled temperature figures |

### 11.2 Run Tab

- **Site selector**: four radio buttons (Not yet run / Completed / Incomplete / In progress), each with count. Selecting a group shows a dropdown of matching sites.
- **Progress bar**: `{current_date} / {end_date}  ({N}%)  · ETA X.X h`. Stale `.pro` data is suppressed for fresh launches until the subprocess has written new output.
- **Crash log**: shown when status is "Crashed / stopped", but suppressed immediately after a new launch is triggered (prevents the old crash log from flashing during startup lag). Uses `fresh_launch_times` session-state key.
- **`is_running`**: checks `.autorun.pid` file, then `/proc/*/cwd` fallback (detects terminal-launched runs without a PID file).

### 11.3 Settings Tab

Five accordion sections: Paths, Run, Model & assimilation, Basal boundary, ERA5 forcing adjustments. All settings in `settings.toml` are editable here. Notable controls:

- **Water transport** selectbox: `adaptive` / `BUCKET` / `RICHARDSEQUATION`
- **RE stabilisation (hours)**: BUCKET phase duration in adaptive mode
- **.pro chunk size**: model hours per chunk (0 = off)
- **Max ice vol. fraction**: 0.90–0.99, controls `max_vol_frac_ice` RE convergence fix
- **ERA5 forcing adjustments**: per-variable multiplier and offset grid (applied via SMET header)

### 11.4 Results Tab

Four checkboxes (none checked by default):

| Checkbox | Plots |
|----------|-------|
| Grain type | Grain type hover heatmap (HTML iframe, JS-rendered) |
| Temperature | Modelled temperature heatmap + optional observed T overlay |
| LWC & Refreezing | LWC heatmap (log scale) + cumulative refreezing line |
| Density | Density heatmap (HTML iframe, JS-rendered) |

**Temperature heatmap**: Turbo colourmap, capped at 0°C. White isotherm at **−0.2°C** via Plotly Contour trace. Observed T overlaid as line series per sensor depth. Both modelled and observed share the same `coloraxis`.

**LWC plot**: Data pre-transformed with `np.log10(np.clip(..., _LWC_FLOOR, None))`; `customdata` carries raw values for hover. Explicit tick labels at 0.01, 0.1, 1, 5, ≥10 kg m⁻².

**Cumulative refreezing** (from `load_pro`):
```python
d_ice = np.diff(ice_col)          # ice volume fraction change
d_lwc = np.diff(lwc_col)          # LWC change
refreezing = np.minimum(
    np.where(d_ice > 0, d_ice, 0.0),    # ice increasing
    np.where(d_lwc < 0, -d_lwc, 0.0),   # LWC decreasing
)
cumul = np.concatenate([[0.0], np.cumsum(refreezing)])
```
No credit system — refreezing at depth is permanent in these firn columns, so simple cumulative sum is correct.

**Caching**: `@st.cache_data` with `_mtime` parameter (file modification timestamp) busts the cache when the `.pro` file changes during a live run.

**X-axis sync**: `make_subplots(shared_xaxes=True)` — zoom/pan on any panel syncs all others.

### 11.5 Obs vs Model Tab

Sites: T2 (2007), T2 bucket (2007), T2− (2019), T3, T4, CP, UP18, plus any additional completed runs auto-discovered by `sites_with_results()`.

- "Regenerate this site" expander: start date, end date, max depth (default 10 m) → runs `plot_obs_vs_model.py` subprocess.
- "Regenerate all sites" button runs `plot_obs_vs_model.py` (no args).
- Auto-regeneration runs at end of each `main()` call.

---

## 12. SNOWPACK Source Patches

The stock SNOWPACK binary is patched in two files. Both are tracked in `snowpack/` in this repo. Rebuild with `snowpack/build_snowpack.sh` (copies patched files into the source tree before `cmake`).

### 12.1 `snowpack/Main.cc` — Daemon Mode + SETTEMPS

Added `--daemon` flag support (see §9.1). Key changes:
- `--daemon` argument parsing.
- Command loop replacing the normal run-once execution.
- `SETTEMPS` branch: parses `i:T` pairs, bounds-checks against `Xdata.getNumberOfElements()`, sets `Edata[i].Te`.
- Hazard reinitialised per `RUN` command (duration can change); IOManager and vecXdata persist.

### 12.2 `snowpack/vanGenuchten.cc` — Density-Dependent Residual Saturation (θᵣ)

**History**: The stock SNOWPACK RE implementation hard-coded `θᵣ = 0.02` (2% residual water content) for all snow/firn regardless of density. SNOWPACK's BUCKET scheme already used the full Coléou & Lesaffre (1998) formula `θᵣ = f(θ_ice, ρ)`. This inconsistency meant RE and BUCKET made different physical assumptions about how much water could remain immobile.

**Change**: Modified `SetVGParamsSnow()` to replace the hard-coded `0.02` cap with `ElementData::snowResidualWaterContent(EMS->theta[ICE])` — the same formula BUCKET uses. Added `#include <snowpack/DataClasses.h>`.

**Effect by density (from T2minus comparison run)**:
- Low-density firn (300–400 kg/m³): θᵣ ≈ 0.05–0.08 (was 0.02 — 3–4× increase)
- Typical firn (most of the column): θᵣ ≈ 0.028–0.042 (~1.5× the old fixed value)
- Near-ice (>850 kg/m³): θᵣ approaches 0 (pore-space constraint dominates)

**LWC impact**: Comparison of fixed vs dynamic θᵣ on T2minus (365,619 layer-timesteps, Pearson r = +0.42 with density):
- Low-density firn (300–750 kg/m³): dynamic run has *less* LWC (−0.07 to −0.15). Higher θᵣ reclassifies mobile water as immobile residual, reducing percolation.
- Dense firn/ice (750–917 kg/m³): dynamic run has *more* LWC (+0.07 to +0.11).

---

## 13. Plot Conventions

- **Depth axis**: always depth from surface (0 at top, increasing downward). PRO code 0501 gives element-top heights in cm above the base — convert: `depth_m = (surface_height_cm − element_height_cm) / 100.0`. Never plot height-above-base as y-axis.
- **Isotherm**: all static figures draw a white contour at **−0.05°C**; colorbar gray cutoff also at −0.05°C.
- **Colorscale**: discrete Turbo, 2°C bins from −20°C to −0.05°C, gray above.
- **Sensor depth lines**: black, α=0.4, linewidth=0.5, on observed panels; track PROMICE-corrected depth over time.

---

## 14. Configuration Reference (`settings.toml`)

```toml
[paths]
snowpack_exe   # Path to SNOWPACK binary
data_root      # Root for AllCoreDataCommonFormat/ (blank = auto-detect)

[run]
run_until              # Stop early ("YYYY-MM-DD HH:MM" or "")
keep_hourly_archives   # Save per-step raw/adjusted .sno pairs
keep_last_n_sno        # Rolling SNO checkpoint count (default 3)
water_transport        # "adaptive" | "BUCKET" | "RICHARDSEQUATION"
stabilization_hours    # BUCKET phase length before RE switch (default 15)
assimilation_interval_h  # Hours per simulation cycle (default 1)
use_ramdisk            # Run hot files from /dev/shm (default true)
use_daemon             # Keep SNOWPACK alive between steps (default true)
pro_chunk_hours        # .pro chunk size in model hours; 0 = off (default 720)

[model]
calculation_step_length_min  # SNOWPACK internal timestep (default 15 min)
height_of_meteo_values       # Sensor height (default 1 m)
height_of_wind_value         # Wind sensor height (default 1 m)
alpha                        # Assimilation blend coefficient (default 0.1)
default_elevation_m          # Fallback if geopotential unavailable

[physics]
ice_density              # kg/m³ (default 917)
water_density            # kg/m³ (default 1000)
max_vol_frac_ice         # RE convergence cap: 0.90–0.99 (default 0.98)
default_water_frac_at_zero  # Initial liquid fraction at 0°C
zero_temp_tol            # Numerical tolerance for melting-point check

[assimilation]
temp_min_c               # Lower clamp for observed T (default −60°C)
temp_max_c               # Upper clamp for observed T (default 0°C)
max_adjust_per_hour_c    # Max nudge per hour (default 1.0°C; was 0.1°C)
min_obs_for_adjust       # Minimum sensors needed to nudge (default 3)
wet_layer_drain_threshold_c  # Only drain wet layers when obs T < this (default −1°C)

[snow_defaults]          # Microstructure values for new SNO layers (rg, rb, dd, sp, etc.)

[basal]
tsg_mode                 # "profile_gradient" | "zero" | "soil_temp"
tsg_lookback_m           # Depth range for gradient fit (default 5 m)
tsg_target_offset_m      # Extrapolation depth below deepest sensor (default 15 m)
basal_temp_min_c         # Floor for basal T estimate (default −40°C)

[era5]                   # CDS API dataset names

[forcing_adjustments]    # Per-variable multiplier and offset for TA, RH, VW, ISWR, ILWR, PSUM
                         # Applied via SMET header; psum_multiplier currently 1.5
```

---

## 15. Comparison Tool (`compare_runs.py`)

Produces a 2×3 matplotlib PNG comparing two `.pro` files:

```
[T_A]   [T_B]   [ΔT = B−A]
[LWC_A] [LWC_B] [ΔLWC = B−A]
```

```bash
python compare_runs.py \
    --site-a 2007_T2_10m   --label-a "Adaptive (RE+BUCKET)" \
    --site-b 2007_T2_10m_bucket --label-b "BUCKET only"
```

**Grid alignment**: both runs are interpolated onto the intersection of their time ranges (hourly, nearest-neighbour) and the shallower of their two depth maxima (linear depth interpolation).

---

## 16. Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| Doubled site ID `2022_2022_T3_25m_25m` | `--site 2022_T3_25m` instead of `--site T3 --year 2022 --depth 25` | Always pass components separately |
| App shows "Not started" for a terminal run | No `.autorun.pid` file | `echo <PID> > {site_dir}/.autorun.pid` |
| RE never triggers fallback (crash instead) | Convergence check placed after `if not ok: raise RuntimeError` | Check must come before the ok-guard |
| Resume point overwritten on restart | Old code ran `write_sno_file` before reading ProfileDate | Now: checkpoint detection runs before any SNO build |
| RE warning spam in log | RE SafeMode is very verbose | `_LOG_FILTER` regex in `app.py` suppresses it in the UI |
| CP obs vs model parse error | Trailing comma in `.pro` produces empty string token | `float(v) for v in parts if v.strip()` guard in parse_pro |
| run_status.json missing / ETA not showing | Model process started before ETA code was committed | Restart the run; ETA only works with code from 2026-04-24 onward |
| SNOWPACK exits immediately with zero steps on new site (ramdisk) | `_setup_ramdisk` creates the SMET symlink only if the disk file already exists; on a first run the file is downloaded after ramdisk setup, leaving ramdisk `input/` without a link — SNOWPACK finds zero stations | Fixed: after `build_smet_from_downloaded_era`, a symlink is created in ramdisk `input/` if it is missing |

---

## 17. Active Research: T2minus Comparison Runs

Two adaptive RE runs exist side-by-side for 2019_T2minus_32m:

| Run | .pro file | θᵣ |
|-----|-----------|-----|
| Fixed θᵣ (baseline) | `output/2019-T2minus-32m_TEMP_ASSIM_RUN_RE_fixed_theta_r.pro` | 0.02 hard-coded |
| Dynamic θᵣ (current) | `output/2019-T2minus-32m_TEMP_ASSIM_RUN.pro` | Coléou formula |

Checkpoint backups: `current_snow_backup_fixed_theta_r/`, `input/initial_profile_checkpoint_backup.sno`.

**θᵣ range in dynamic run** (from ice volume fraction field 515):
- Overall range: 0.000–0.080 (0.08 cap never reached; zero occurs where pore-space cap dominates, θᵢ > 0.971)
- Median: 0.029, mean: 0.029
- 92.5% of all layer-timesteps have θᵣ > 0.02 — the old fixed cap was binding nearly everywhere

**LWC difference analysis** (365,619 valid pixels, Pearson r = +0.42 with density):
- Blue (dynamic < fixed) concentrated in low-density firn (300–750 kg/m³)
- Red (dynamic > fixed) in near-ice layers (750–917 kg/m³)

**Working hypothesis**: Higher θᵣ in low-density firn reclassifies mobile water as immobile residual, reducing percolation; water that pooled in low-density layers in the fixed run is redistributed or drains faster in the dynamic run. Mechanism not fully resolved.

### Comparison Scripts

| Script | Output |
|--------|--------|
| `compare_RE_theta_r.py` | 4-panel: temperature and LWC for both runs |
| `compare_LWC.py` | 3-panel: fixed LWC, dynamic LWC, ΔLWC (log scale, RdBu_r) |
| `check_density_vs_diff.py` | 3-panel: density, ice fraction, ΔLWC |

### Water-transport scheme overlay on interactive plots

A "Scheme overlay" checkbox added to the Results tab plot selector. When checked alongside any of the Plotly-based plots (Temperature, LWC & Refreezing, Residual saturation), BUCKET periods are shaded amber (`rgba(255,160,0,0.20)`) across all active subplots using Plotly `add_vrect` with `yref="paper"`. A square legend marker labelled "BUCKET" is injected so the shading is self-explanatory. RE periods are unshaded (RE is the normal state; highlighting deviations is more informative). If `water_transport_log.csv` is absent a caption explains the requirement. Grain-type and density HTML iframes are not affected.

### `reconstruct_water_transport_log.py` — backfill scheme logs for completed runs

Parses an existing `autorun.log` to produce `output/water_transport_log.csv` and `output/water_transport_events.csv` for runs that predate the 2026-04-24 per-step logging. Handles all transition message formats in the log:

| Log message | Meaning | Event written |
|---|---|---|
| `Stabilization complete at T0` | t0 = step start; first RE step ends at t0+interval | `(t0+h, RE)` |
| `T1: RE SafeMode convergence failure` | t1 = step end (ran as RE); next step is BUCKET | `(t1+h, BUCKET)` |
| `→T1: RE timed out ... retrying with BUCKET` | t1 = step end (retried as BUCKET) | `(t1, BUCKET)` |
| `T0: RE fallback period over` | t0 = step start; this step runs as RE | `(t0+h, RE)` |
| `Resume: INI set to X` | scheme at resume point | initial scheme |

Splits on `"Fresh start:"` lines to handle logs with multiple runs. Step interval inferred from median of first 50 consecutive step gaps.

Run for all sites: `python reconstruct_water_transport_log.py`
Run for one site: `python reconstruct_water_transport_log.py 2022_T3_25m`
