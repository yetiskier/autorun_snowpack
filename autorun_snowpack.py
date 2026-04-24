#!/usr/bin/env python3
"""
autorun_snowpack.py

Workflow:
1) Read Tempconcatenated metadata and temperatures
2) Read PROMICE cumulative surface change
3) Read firn-core density profile
4) Build initial .sno file
5) Reuse input/site_forcing.smet if it already exists
6) Otherwise download ERA5-Land hourly gridded data for all time-varying forcing variables
7) Download ERA5 geopotential once per site and cache it locally
8) Write forcing .smet
9) Write .ini
10) Run SNOWPACK hour by hour
11) Save raw restart .sno files in current_snow/
12) Adjust temperatures using moving corrected sensor depths and continue
"""

from __future__ import annotations

import argparse
import calendar
import json
import re
import shutil
import subprocess
import sys
import tomllib
import zipfile
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import xarray as xr

try:
    import cdsapi
except Exception:
    cdsapi = None


# =============================================================================
# PATHS AND USER SETTINGS
# Populated at runtime by configure() — do not edit values here.
# Edit settings.toml in the parent directory instead.
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# All other globals are set by configure() at the start of main().
PROJECT_DIR = SCRIPT_DIR          # placeholder
DATA_DIR    = SCRIPT_DIR.parent   # placeholder

TEMP_FILE     : Path
PROMICE_FILE  : Path
DENSITY_FILE  : Path

KEEP_HOURLY_ARCHIVES : bool
KEEP_LAST_N_SNO      : int

INPUT_DIR        : Path
CFG_DIR          : Path
OUTPUT_DIR       : Path
CURRENT_SNOW_DIR : Path
ERA_TMP_DIR      : Path
CACHE_DIR        : Path

INITIAL_SNO_FILE  : Path
FORCING_SMET_FILE : Path
CFG_INI_FILE      : Path

SNOWPACK_EXE            : str
DEFAULT_ELEVATION_M     : float
DEFAULT_WATER_FRAC_AT_ZERO : float
ZERO_TEMP_TOL           : float
ICE_DENSITY             : float
WATER_DENSITY           : float
MAX_VOL_FRAC_ICE        : float
PRO_CHUNK_HOURS         : int

CALCULATION_STEP_LENGTH_MIN : float
HEIGHT_OF_METEO_VALUES      : float
HEIGHT_OF_WIND_VALUE        : float
ALPHA                       : float

ERA5_DATASET    : str
ERA5LAND_DATASET: str

SMET_NODATA = -999
SMET_TZ     = 0
SNO_TZ      = 0
SNO_SOURCE  = "Generated from Tempconcatenated, PROMICE, firn density, and ERA5-Land/ERA5 reanalysis"

SLOPE_ANGLE : float
SLOPE_AZI   : float
SOIL_ALBEDO : float
BARE_SOIL_Z0: float
CANOPY_HEIGHT             : float
CANOPY_LAI                : float
CANOPY_DIRECT_THROUGHFALL : float
WIND_SCALING_FACTOR       : float
TIMECOUNTDELTAHS          : float

DEFAULT_CONDUC_S   : float
DEFAULT_HEATCAPAC_S: float
DEFAULT_RG         : float
DEFAULT_RB         : float
DEFAULT_DD         : float
DEFAULT_SP         : float
DEFAULT_MK         : float
DEFAULT_MASS_HOAR  : float
DEFAULT_NE         : float
DEFAULT_CDOT       : float
DEFAULT_METAMO     : float

TSG_MODE            : str
TSG_LOOKBACK_M      : float
TSG_TARGET_OFFSET_M : float
ADD_BASAL_LAYERS          : bool
BASAL_LAYER_THICKNESSES_M : list
BASAL_TREND_LOOKBACK_M    : float
BASAL_TEMP_MIN_C          : float

TEMP_MIN_C              : float
TEMP_MAX_C              : float
MAX_ADJUST_PER_HOUR_C   : float
MIN_OBS_FOR_ADJUST      : int
WET_LAYER_DRAIN_THRESHOLD_C : float

WATER_TRANSPORT          : str
STABILIZATION_HOURS      : int
ASSIMILATION_INTERVAL_H  : int
USE_RAMDISK              : bool
USE_DAEMON               : bool
DISK_INPUT_DIR           : Path   # real disk path used for checkpoint sync

# Write a full SNO every this many daemon steps; SETTEMPS used in between.
SETTEMPS_SNO_WRITE_INTERVAL = 24

FORCING_TA_MULT   : float
FORCING_TA_ADD    : float
FORCING_RH_MULT   : float
FORCING_RH_ADD    : float
FORCING_VW_MULT   : float
FORCING_VW_ADD    : float
FORCING_ISWR_MULT : float
FORCING_ISWR_ADD  : float
FORCING_ILWR_MULT : float
FORCING_ILWR_ADD  : float
FORCING_PSUM_MULT : float
FORCING_PSUM_ADD  : float


# =============================================================================
# CLI + SETTINGS LOADER
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Autorun SNOWPACK for a firn-core temperature string site.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--site",  default="T3",
                   help="Site/traverse name, e.g. T3 or UP18")
    p.add_argument("--year",  type=int, default=2022,
                   help="Year the core was drilled, e.g. 2022")
    p.add_argument("--depth", type=int, default=25,
                   help="Temperature string depth in metres, e.g. 25")
    p.add_argument("--run-until", default=None, metavar="YYYY-MM-DD HH:MM",
                   help="Stop simulation at this timestamp (overrides settings.toml)."
                        " Omit for full record.")
    p.add_argument("--settings", type=Path, default=None, metavar="PATH",
                   help="Path to settings.toml (default: autorun_snowpack/settings.toml)")
    p.add_argument("--run-tag", default=None, metavar="TAG",
                   help="Suffix appended to the project directory (e.g. 'bucket') so the "
                        "run writes to <site_id>_<tag>/ without touching the original output. "
                        "Data input files are still read from the untagged site directory.")
    p.add_argument("--water-transport", default=None,
                   choices=["adaptive", "BUCKET", "RICHARDSEQUATION"],
                   help="Water transport scheme: 'adaptive' = BUCKET stabilisation then RE "
                        "(default); 'BUCKET' = BUCKET throughout; 'RICHARDSEQUATION' = RE "
                        "throughout (no stabilisation phase).")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore any existing checkpoint and restart from the beginning of "
                        "the temperature record. Clears initial_profile.sno from both disk "
                        "and ramdisk, clears current_snow/, and archives or removes the "
                        "existing .pro output so the progress bar starts from zero.")
    p.add_argument("--fresh-mode", default="archive", choices=["archive", "delete"],
                   help="What to do with the previous .pro/.ini/log output when --fresh is "
                        "set. 'archive' (default) creates a timestamped .tar.gz in "
                        "output/archives/. 'delete' removes the files.")
    return p.parse_args()


def load_settings(args: argparse.Namespace) -> dict:
    if args.settings:
        settings_path = args.settings
    else:
        settings_path = SCRIPT_DIR / "settings.toml"
    if not settings_path.exists():
        raise FileNotFoundError(
            f"Settings file not found: {settings_path}\n"
            f"Create one from the template in the autorun_snowpack/ directory."
        )
    with open(settings_path, "rb") as fh:
        return tomllib.load(fh)


def _setup_ramdisk(site_id: str, disk_project_dir: Path) -> Path:
    """Mirror hot working files to /dev/shm; symlink large static files to disk."""
    ram_dir = Path(f"/dev/shm/snowpack_{site_id}")
    ram_dir.mkdir(exist_ok=True)

    for sub in ("input", "cfgfiles", "current_snow"):
        (ram_dir / sub).mkdir(exist_ok=True)

    # Large sequential-write output stays on disk
    for sub in ("output", "era_tmp", "cache"):
        link = ram_dir / sub
        if not link.exists():
            link.symlink_to(disk_project_dir / sub)

    # SMET is large and read-only — symlink to disk
    ram_smet = ram_dir / "input" / "site_forcing.smet"
    disk_smet = disk_project_dir / "input" / "site_forcing.smet"
    if not ram_smet.exists() and disk_smet.exists():
        ram_smet.symlink_to(disk_smet)

    # Copy checkpoint SNO to RAM if it exists
    disk_sno = disk_project_dir / "input" / "initial_profile.sno"
    ram_sno  = ram_dir / "input" / "initial_profile.sno"
    if disk_sno.exists() and not ram_sno.exists():
        shutil.copy2(disk_sno, ram_sno)

    # Copy current_snow files to RAM
    disk_cs = disk_project_dir / "current_snow"
    if disk_cs.exists():
        for f in disk_cs.iterdir():
            dst = ram_dir / "current_snow" / f.name
            if not dst.exists():
                shutil.copy2(f, dst)

    return ram_dir


def configure(args: argparse.Namespace, cfg: dict) -> None:
    """Populate all module-level globals from CLI args + settings.toml."""
    global PROJECT_DIR, DATA_DIR
    global TEMP_FILE, PROMICE_FILE, DENSITY_FILE
    global KEEP_HOURLY_ARCHIVES, KEEP_LAST_N_SNO
    global INPUT_DIR, CFG_DIR, OUTPUT_DIR, CURRENT_SNOW_DIR, ERA_TMP_DIR, CACHE_DIR
    global INITIAL_SNO_FILE, FORCING_SMET_FILE, CFG_INI_FILE
    global SNOWPACK_EXE, DEFAULT_ELEVATION_M, DEFAULT_WATER_FRAC_AT_ZERO
    global ZERO_TEMP_TOL, ICE_DENSITY, WATER_DENSITY, MAX_VOL_FRAC_ICE, PRO_CHUNK_HOURS
    global CALCULATION_STEP_LENGTH_MIN, HEIGHT_OF_METEO_VALUES, HEIGHT_OF_WIND_VALUE, ALPHA
    global ERA5_DATASET, ERA5LAND_DATASET
    global SLOPE_ANGLE, SLOPE_AZI, SOIL_ALBEDO, BARE_SOIL_Z0
    global CANOPY_HEIGHT, CANOPY_LAI, CANOPY_DIRECT_THROUGHFALL
    global WIND_SCALING_FACTOR, TIMECOUNTDELTAHS
    global DEFAULT_CONDUC_S, DEFAULT_HEATCAPAC_S, DEFAULT_RG, DEFAULT_RB
    global DEFAULT_DD, DEFAULT_SP, DEFAULT_MK, DEFAULT_MASS_HOAR
    global DEFAULT_NE, DEFAULT_CDOT, DEFAULT_METAMO
    global TSG_MODE, TSG_LOOKBACK_M, TSG_TARGET_OFFSET_M
    global ADD_BASAL_LAYERS, BASAL_LAYER_THICKNESSES_M, BASAL_TREND_LOOKBACK_M, BASAL_TEMP_MIN_C
    global TEMP_MIN_C, TEMP_MAX_C, MAX_ADJUST_PER_HOUR_C, MIN_OBS_FOR_ADJUST, ASSIMILATION_INTERVAL_H
    global WET_LAYER_DRAIN_THRESHOLD_C
    global WATER_TRANSPORT, STABILIZATION_HOURS, USE_RAMDISK, USE_DAEMON, DISK_INPUT_DIR
    global FORCING_TA_MULT, FORCING_TA_ADD, FORCING_RH_MULT, FORCING_RH_ADD
    global FORCING_VW_MULT, FORCING_VW_ADD, FORCING_ISWR_MULT, FORCING_ISWR_ADD
    global FORCING_ILWR_MULT, FORCING_ILWR_ADD, FORCING_PSUM_MULT, FORCING_PSUM_ADD

    site_id  = f"{args.year}_{args.site}_{args.depth}m"
    base_dir = SCRIPT_DIR                  # autorun_snowpack/

    data_root_str = cfg["paths"].get("data_root", "").strip()
    DATA_DIR    = Path(data_root_str) if data_root_str else base_dir

    # --run-tag lets a comparison run write to a separate directory while
    # still reading data from the canonical (untagged) site files.
    project_id  = f"{site_id}_{args.run_tag}" if args.run_tag else site_id
    PROJECT_DIR = base_dir / project_id

    TEMP_FILE    = DATA_DIR / f"AllCoreDataCommonFormat/Concatenated_Temperature_files/{site_id}_Tempconcatenated.csv"
    PROMICE_FILE = DATA_DIR / f"AllCoreDataCommonFormat/Depth_change_estimate/PROMICE/{site_id}_daily_PROMICE_snowfall.csv"
    DENSITY_FILE = DATA_DIR / f"AllCoreDataCommonFormat/CoreDataEGIG/{site_id}_den.csv"

    INPUT_DIR        = PROJECT_DIR / "input"
    CFG_DIR          = PROJECT_DIR / "cfgfiles"
    OUTPUT_DIR       = PROJECT_DIR / "output"
    CURRENT_SNOW_DIR = PROJECT_DIR / "current_snow"
    ERA_TMP_DIR      = PROJECT_DIR / "era_tmp"
    CACHE_DIR        = PROJECT_DIR / "cache"

    for d in [INPUT_DIR, CFG_DIR, OUTPUT_DIR, CURRENT_SNOW_DIR, ERA_TMP_DIR, CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    INITIAL_SNO_FILE  = INPUT_DIR / "initial_profile.sno"
    FORCING_SMET_FILE = INPUT_DIR / "site_forcing.smet"
    CFG_INI_FILE      = CFG_DIR   / "site_run.ini"

    SNOWPACK_EXE = cfg["paths"]["snowpack_exe"]

    m = cfg["model"]
    CALCULATION_STEP_LENGTH_MIN = float(m["calculation_step_length_min"])
    HEIGHT_OF_METEO_VALUES      = float(m["height_of_meteo_values"])
    HEIGHT_OF_WIND_VALUE        = float(m["height_of_wind_value"])
    ALPHA                       = float(m["alpha"])
    DEFAULT_ELEVATION_M         = float(m["default_elevation_m"])

    ph = cfg["physics"]
    DEFAULT_WATER_FRAC_AT_ZERO = float(ph["default_water_frac_at_zero"])
    ZERO_TEMP_TOL              = float(ph["zero_temp_tol"])
    ICE_DENSITY                = float(ph["ice_density"])
    WATER_DENSITY              = float(ph["water_density"])
    _mvfi = float(ph.get("max_vol_frac_ice", 0.98))
    MAX_VOL_FRAC_ICE = max(0.90, min(0.99, _mvfi))

    r = cfg["run"]
    PRO_CHUNK_HOURS = int(r.get("pro_chunk_hours", 720))

    sd = cfg["snow_defaults"]
    DEFAULT_CONDUC_S    = float(sd["conduc_s"])
    DEFAULT_HEATCAPAC_S = float(sd["heatcapac_s"])
    DEFAULT_RG          = float(sd["rg"])
    DEFAULT_RB          = float(sd["rb"])
    DEFAULT_DD          = float(sd["dd"])
    DEFAULT_SP          = float(sd["sp"])
    DEFAULT_MK          = float(sd["mk"])
    DEFAULT_MASS_HOAR   = float(sd["mass_hoar"])
    DEFAULT_NE          = float(sd["ne"])
    DEFAULT_CDOT        = float(sd["cdot"])
    DEFAULT_METAMO      = float(sd["metamo"])

    si = cfg["site_defaults"]
    SLOPE_ANGLE               = float(si["slope_angle"])
    SLOPE_AZI                 = float(si["slope_azi"])
    SOIL_ALBEDO               = float(si["soil_albedo"])
    BARE_SOIL_Z0              = float(si["bare_soil_z0"])
    CANOPY_HEIGHT             = float(si["canopy_height"])
    CANOPY_LAI                = float(si["canopy_lai"])
    CANOPY_DIRECT_THROUGHFALL = float(si["canopy_direct_throughfall"])
    WIND_SCALING_FACTOR       = float(si["wind_scaling_factor"])
    TIMECOUNTDELTAHS          = float(si["timecountdeltahs"])

    b = cfg["basal"]
    TSG_MODE                  = str(b["tsg_mode"])
    TSG_LOOKBACK_M            = float(b["tsg_lookback_m"])
    TSG_TARGET_OFFSET_M       = float(b["tsg_target_offset_m"])
    ADD_BASAL_LAYERS          = bool(b["add_basal_layers"])
    BASAL_LAYER_THICKNESSES_M = list(b["basal_layer_thicknesses_m"])
    BASAL_TREND_LOOKBACK_M    = float(b["basal_trend_lookback_m"])
    BASAL_TEMP_MIN_C          = float(b["basal_temp_min_c"])

    a = cfg["assimilation"]
    TEMP_MIN_C                  = float(a["temp_min_c"])
    TEMP_MAX_C                  = float(a["temp_max_c"])
    MAX_ADJUST_PER_HOUR_C       = float(a["max_adjust_per_hour_c"])
    MIN_OBS_FOR_ADJUST          = int(a["min_obs_for_adjust"])
    WET_LAYER_DRAIN_THRESHOLD_C = float(a["wet_layer_drain_threshold_c"])

    e = cfg["era5"]
    ERA5_DATASET    = str(e["dataset"])
    ERA5LAND_DATASET = str(e["land_dataset"])

    fa = cfg.get("forcing_adjustments", {})
    FORCING_TA_MULT   = float(fa.get("ta_multiplier",   1.0))
    FORCING_TA_ADD    = float(fa.get("ta_offset",        0.0))
    FORCING_RH_MULT   = float(fa.get("rh_multiplier",   1.0))
    FORCING_RH_ADD    = float(fa.get("rh_offset",        0.0))
    FORCING_VW_MULT   = float(fa.get("vw_multiplier",   1.0))
    FORCING_VW_ADD    = float(fa.get("vw_offset",        0.0))
    FORCING_ISWR_MULT = float(fa.get("iswr_multiplier", 1.0))
    FORCING_ISWR_ADD  = float(fa.get("iswr_offset",      0.0))
    FORCING_ILWR_MULT = float(fa.get("ilwr_multiplier", 1.0))
    FORCING_ILWR_ADD  = float(fa.get("ilwr_offset",      0.0))
    FORCING_PSUM_MULT = float(fa.get("psum_multiplier", 1.0))
    FORCING_PSUM_ADD  = float(fa.get("psum_offset",      0.0))

    KEEP_HOURLY_ARCHIVES    = bool(cfg["run"]["keep_hourly_archives"])
    KEEP_LAST_N_SNO         = int(cfg["run"]["keep_last_n_sno"])
    WATER_TRANSPORT         = str(cfg["run"].get("water_transport", "adaptive"))
    STABILIZATION_HOURS     = int(cfg["run"].get("stabilization_hours", 15))
    ASSIMILATION_INTERVAL_H = int(cfg["run"].get("assimilation_interval_h", 1))
    USE_RAMDISK             = bool(cfg["run"].get("use_ramdisk", False))
    USE_DAEMON              = bool(cfg["run"].get("use_daemon", False))

    DISK_INPUT_DIR = INPUT_DIR  # save real disk path before any ramdisk redirect

    if USE_RAMDISK:
        ram_dir = _setup_ramdisk(project_id, PROJECT_DIR)
        PROJECT_DIR      = ram_dir
        INPUT_DIR        = ram_dir / "input"
        CFG_DIR          = ram_dir / "cfgfiles"
        CURRENT_SNOW_DIR = ram_dir / "current_snow"
        INITIAL_SNO_FILE = INPUT_DIR / "initial_profile.sno"
        CFG_INI_FILE     = CFG_DIR  / "site_run.ini"
        print(f"RAM disk active: {ram_dir}")

    print(f"Site:        {site_id}")
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Data root:   {DATA_DIR}")


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SiteMetadata:
    station_name: str
    station_id: str
    latitude: float
    longitude: float
    elevation: float
    drilled: Optional[pd.Timestamp]


# =============================================================================
# READ INPUT FILES
# =============================================================================

def read_tempconcatenated(path: Path, default_elevation_m: float = 2000.0) -> Tuple[pd.DataFrame, SiteMetadata]:
    lines = path.read_text(encoding="utf-8").splitlines()

    hole = None
    lat = None
    lon = None
    elev = None
    drilled = None

    for line in lines[:10]:
        s = line.strip()

        if s.startswith("Hole:"):
            parts = [p.strip() for p in s.split(",")]
            if len(parts) > 1:
                hole = parts[1]

        elif s.startswith("Location:"):
            parts = [p.strip() for p in s.split(",")]
            if len(parts) >= 3:
                lat = float(parts[1])
                lon = float(parts[2])

            m = re.search(r"elev(?:ation)?\s*=?\s*([0-9.+-]+)", s, re.IGNORECASE)
            if m:
                elev = float(m.group(1))

        elif s.startswith("Elevation:"):
            parts = [p.strip() for p in s.split(",")]
            if len(parts) > 1:
                elev = float(parts[1])

        elif s.startswith("Drilled:"):
            parts = [p.strip() for p in s.split(",")]
            if len(parts) > 1:
                drilled = pd.to_datetime(parts[1], errors="coerce")

    if hole is None or lat is None or lon is None:
        raise ValueError(f"Could not parse Hole / Location metadata from {path}")

    if elev is None:
        elev = float(default_elevation_m)

    df = pd.read_csv(path, skiprows=4)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    rename_map = {}
    for c in df.columns:
        try:
            rename_map[c] = float(c)
        except Exception:
            pass
    df = df.rename(columns=rename_map)

    numeric_cols = [c for c in df.columns if isinstance(c, (int, float))]
    df = df[numeric_cols].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    meta = SiteMetadata(
        station_name=hole,
        station_id=hole,
        latitude=float(lat),
        longitude=float(lon),
        elevation=float(elev),
        drilled=drilled if pd.notna(drilled) else None,
    )
    return df, meta


def read_promice(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").set_index("timestamp")

    if "cumulative_surface_total_change_cm" not in df.columns:
        raise ValueError("PROMICE file must contain cumulative_surface_total_change_cm")

    df["cumulative_surface_total_change_m"] = (
        pd.to_numeric(df["cumulative_surface_total_change_cm"], errors="coerce") / 100.0
    )
    return df


def read_density_profile(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, skiprows=5)

    required = ["From (cm)", "To (cm)", "Density (kg/m^3)"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Density file missing required columns: {missing}")

    out = pd.DataFrame({
        "top_m": pd.to_numeric(df["From (cm)"], errors="coerce") / 100.0,
        "bottom_m": pd.to_numeric(df["To (cm)"], errors="coerce") / 100.0,
        "density_kg_m3": pd.to_numeric(df["Density (kg/m^3)"], errors="coerce"),
    }).dropna()

    out["mid_m"] = 0.5 * (out["top_m"] + out["bottom_m"])
    out["thickness_m"] = out["bottom_m"] - out["top_m"]
    out = out[(out["thickness_m"] > 0) & np.isfinite(out["density_kg_m3"])].copy()
    return out.sort_values("mid_m").reset_index(drop=True)


# =============================================================================
# GENERIC HELPERS
# =============================================================================

def prune_sno_files(directory: Path, keep_last_n: int = 2) -> None:
    files = sorted(
        [
            p for p in directory.glob("*.sno*")
            if p.is_file()
            and not p.name.startswith("_")
            and not p.name.startswith("raw_")
            and not p.name.startswith("adj_")
        ],
        key=lambda p: p.stat().st_mtime
    )
    for p in files[:-keep_last_n]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def prune_haz_files(directory: Path, keep_last_n: int = 0) -> None:
    files = sorted(
        [p for p in directory.glob("*.haz*") if p.is_file()],
        key=lambda p: p.stat().st_mtime
    )
    for p in files[:-keep_last_n] if keep_last_n > 0 else files:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def estimate_tsg_from_temp_profile(
    temp_df: pd.DataFrame,
    lookback_m: float = 5.0,
    target_offset_m: float = 1.0,
) -> pd.Series:
    tsg = pd.Series(index=temp_df.index, dtype=float)

    for ts, row in temp_df.iterrows():
        valid = []
        for depth, temp in row.items():
            if np.isfinite(temp):
                valid.append((float(depth), float(temp)))

        if len(valid) < 2:
            tsg.loc[ts] = np.nan
            continue

        prof = pd.DataFrame(valid, columns=["depth_m", "T_C"]).sort_values("depth_m")

        deepest = prof["depth_m"].max()
        use = prof[prof["depth_m"] >= deepest - lookback_m].copy()

        if len(use) < 2:
            use = prof.tail(min(3, len(prof))).copy()

        x = use["depth_m"].to_numpy(dtype=float)
        y = use["T_C"].to_numpy(dtype=float)

        a, b = np.polyfit(x, y, 1)

        target_depth = deepest + target_offset_m
        t_est = a * target_depth + b
        t_est = float(np.clip(t_est, TEMP_MIN_C, TEMP_MAX_C))
        tsg.loc[ts] = t_est

    tsg = tsg.interpolate(method="time", limit_direction="both")
    return tsg


def fit_deep_profile_gradient(
    corrected_temp_profile: pd.DataFrame,
    lookback_m: float = 5.0,
) -> tuple[float, float]:
    prof = corrected_temp_profile.copy()
    prof = prof[
        np.isfinite(prof["actual_depth_m"]) &
        np.isfinite(prof["temperature_C"]) &
        (prof["actual_depth_m"] >= 0.0)
    ].sort_values("actual_depth_m")

    if len(prof) < 2:
        raise ValueError("Need at least two valid temperatures to fit deep profile gradient.")

    deepest = float(prof["actual_depth_m"].max())
    use = prof[prof["actual_depth_m"] >= deepest - lookback_m].copy()

    if len(use) < 2:
        use = prof.tail(min(3, len(prof))).copy()

    x = use["actual_depth_m"].to_numpy(dtype=float)
    y = use["temperature_C"].to_numpy(dtype=float)

    a, b = np.polyfit(x, y, 1)
    return float(a), float(b)


def compute_dynamic_basal_temperature_bounds(
    corrected_temp_profile: pd.DataFrame,
    lookback_m: float = 5.0,
    max_warming_above_deepest_sensor_c: float = 0.0,
    basal_temp_min_c: float = -40.0,
) -> tuple[float, float]:
    """
    Build site-specific lower temperature bounds from the deepest measured temperatures.

    Generic rule:
      - lower bound is a broad physical floor
      - upper bound is at most the deepest measured temperature plus a small allowance
      - for firn stability, use max_warming_above_deepest_sensor_c = 0.0
    """
    prof = corrected_temp_profile.copy()
    prof = prof[
        np.isfinite(prof["actual_depth_m"]) &
        np.isfinite(prof["temperature_C"]) &
        (prof["actual_depth_m"] >= 0.0)
    ].sort_values("actual_depth_m")

    if len(prof) < 2:
        raise ValueError("Need at least two valid temperatures to compute basal bounds.")

    deepest = float(prof["actual_depth_m"].max())
    use = prof[prof["actual_depth_m"] >= deepest - lookback_m].copy()
    if len(use) < 2:
        use = prof.tail(min(3, len(prof))).copy()

    deepest_measured_temp = float(use["temperature_C"].iloc[-1])
    basal_max_c = deepest_measured_temp + max_warming_above_deepest_sensor_c

    return float(basal_temp_min_c), float(basal_max_c)


def add_basal_layers_to_density_profile(
    density_df: pd.DataFrame,
    corrected_temp_profile: pd.DataFrame,
    layer_thicknesses_m: list[float],
    lookback_m: float = 5.0,
    temp_min_c: float = -40.0,
    temp_max_c: float = -0.1,
) -> pd.DataFrame:
    out = density_df.copy().sort_values("mid_m").reset_index(drop=True)

    a, b = fit_deep_profile_gradient(
        corrected_temp_profile=corrected_temp_profile,
        lookback_m=lookback_m,
    )

    bottom = float(out["bottom_m"].max())
    rho_bottom = float(out.iloc[-1]["density_kg_m3"])

    new_rows = []
    current_top = bottom

    for thick in layer_thicknesses_m:
        thick = float(thick)
        top_m = current_top
        bottom_m = current_top + thick
        mid_m = 0.5 * (top_m + bottom_m)

        t_c = a * mid_m + b
        t_c = float(np.clip(t_c, temp_min_c, temp_max_c))

        new_rows.append({
            "top_m": top_m,
            "bottom_m": bottom_m,
            "mid_m": mid_m,
            "thickness_m": thick,
            "density_kg_m3": rho_bottom,
            "T_C": t_c,
        })

        current_top = bottom_m

    basal_df = pd.DataFrame(new_rows)
    out = pd.concat([out, basal_df], ignore_index=True)
    return out.sort_values("mid_m").reset_index(drop=True)


def estimate_tsg_from_corrected_profile(
    corrected_temp_profile: pd.DataFrame,
    density_df: pd.DataFrame,
    lookback_m: float = 5.0,
    target_offset_m: float = 1.0,
) -> float:
    """
    Estimate lower boundary temperature from the deep profile trend, but do not
    allow it to be warmer than the deepest observed temperature.
    """
    prof = corrected_temp_profile.copy()
    prof = prof[
        np.isfinite(prof["actual_depth_m"]) &
        np.isfinite(prof["temperature_C"]) &
        (prof["actual_depth_m"] >= 0.0)
    ].sort_values("actual_depth_m")

    if len(prof) < 2:
        raise ValueError("Need at least two valid temperatures to estimate TSG.")

    deepest_obs_temp = float(prof["temperature_C"].iloc[-1])

    a, b = fit_deep_profile_gradient(
        corrected_temp_profile=corrected_temp_profile,
        lookback_m=lookback_m,
    )

    # Do not allow warming with depth below deepest observation
    a = min(a, 0.0)

    modeled_bottom = float(density_df["bottom_m"].max())
    if ADD_BASAL_LAYERS:
        modeled_bottom += float(sum(BASAL_LAYER_THICKNESSES_M))

    target_depth = modeled_bottom + target_offset_m

    tsg_c = a * target_depth + b
    tsg_c = min(tsg_c, deepest_obs_temp)
    tsg_c = float(np.clip(tsg_c, BASAL_TEMP_MIN_C, deepest_obs_temp))
    return tsg_c


def build_corrected_temp_profile(temp_row: pd.Series, surface_change_m: float) -> pd.DataFrame:
    rows = []
    for install_depth, T in temp_row.items():
        if not np.isfinite(T):
            continue

        rows.append({
            "install_depth_m": float(install_depth),
            "actual_depth_m": float(install_depth) + float(surface_change_m),
            "temperature_C": float(T),
        })

    if len(rows) == 0:
        return pd.DataFrame(
            columns=["install_depth_m", "actual_depth_m", "temperature_C"]
        )

    out = pd.DataFrame(rows)
    return out.sort_values("actual_depth_m").reset_index(drop=True)


def build_hourly_corrected_temp_profiles(
    temp_df: pd.DataFrame,
    promice_df: pd.DataFrame,
    max_interp_gap_hours: int = 24,
) -> dict[pd.Timestamp, pd.DataFrame]:
    temp_df = keep_hourly(temp_df)
    hourly_index = make_hourly_index(temp_df.index.min(), temp_df.index.max())

    temp_hourly_orig = temp_df.reindex(hourly_index)
    temp_hourly_fill = interpolate_small_temp_gaps(
        temp_hourly_orig,
        limit_hours=max_interp_gap_hours,
    )
    support_mask = build_temp_support_mask(temp_hourly_orig, temp_hourly_fill)

    cumulative_change = build_hourly_surface_change(promice_df, hourly_index)

    profiles = {}
    empty_count = 0

    for ts in hourly_index:
        row = temp_hourly_fill.loc[ts].copy()
        supported = support_mask.loc[ts]

        row.loc[~supported] = np.nan

        surf_change = (
            float(cumulative_change.loc[ts])
            if pd.notna(cumulative_change.loc[ts])
            else 0.0
        )

        prof = build_corrected_temp_profile(row, surf_change)

        if prof.empty:
            empty_count += 1

        profiles[ts] = prof

    if empty_count > 0:
        print(
            f"Warning: {empty_count} hourly corrected temperature profiles are empty "
            f"after limiting interpolation to gaps <= {max_interp_gap_hours} hours."
        )

    return profiles


def read_sno_layer_table(sno_path: Path) -> pd.DataFrame:
    lines = sno_path.read_text(encoding="utf-8").splitlines(keepends=True)
    _, _, _, df = parse_sno_data_table(lines)
    return df


def validate_sno_geometry(
    sno_path: Path,
    max_layer_thickness_m: float = 2.0,
    max_total_thickness_m: float = 40.0,
) -> None:
    df = read_sno_layer_table(sno_path)

    if "Layer_Thick" not in df.columns:
        raise RuntimeError(f"{sno_path} does not contain Layer_Thick")

    thick = pd.to_numeric(df["Layer_Thick"], errors="coerce").to_numpy(dtype=float)

    if not np.all(np.isfinite(thick)):
        raise RuntimeError(f"{sno_path} contains non-finite layer thickness values")

    total_thickness = float(np.sum(thick))
    max_thickness = float(np.max(thick))


    if max_thickness > max_layer_thickness_m:
        raise RuntimeError(
            f"Corrupted restart file: max layer thickness = {max_thickness:.3f} m "
            f"(limit {max_layer_thickness_m:.3f} m)"
        )

    if total_thickness > max_total_thickness_m:
        raise RuntimeError(
            f"Corrupted restart file: total thickness = {total_thickness:.3f} m "
            f"(limit {max_total_thickness_m:.3f} m)"
        )


def apply_initial_temperature_adjustment(
    input_sno_file: Path,
    long_df: pd.DataFrame,
    t0: pd.Timestamp,
    alpha: float,
) -> None:
    obs_profile = get_profile_observations_for_time(long_df, t0)

    tmp0 = input_sno_file.parent / "_initial_adjusted_tmp.sno"

    update_sno_temperatures_from_moving_profile(
        sno_in=input_sno_file,
        sno_out=tmp0,
        obs_profile=obs_profile,
        alpha=alpha,
        new_profile_date=t0,
    )

    shutil.copy2(tmp0, input_sno_file)

    try:
        tmp0.unlink()
    except FileNotFoundError:
        pass

    print(f"{t0}: applied initial temperature adjustment to {input_sno_file.name}")


def evaluate_stable_basal_extrapolation(
    depths_m: np.ndarray,
    corrected_temp_profile: pd.DataFrame,
    lookback_m: float,
    basal_temp_min_c: float,
    max_warming_above_deepest_sensor_c: float,
) -> np.ndarray:
    """
    Extrapolate deep temperatures below the deepest observed sensor, but do not allow
    the extrapolated profile to become warmer with depth than the deepest observed sensor.
    """
    a, b = fit_deep_profile_gradient(
        corrected_temp_profile=corrected_temp_profile,
        lookback_m=lookback_m,
    )

    basal_min_c, basal_max_c = compute_dynamic_basal_temperature_bounds(
        corrected_temp_profile=corrected_temp_profile,
        lookback_m=lookback_m,
        max_warming_above_deepest_sensor_c=max_warming_above_deepest_sensor_c,
        basal_temp_min_c=basal_temp_min_c,
    )

    prof = corrected_temp_profile.copy()
    prof = prof[
        np.isfinite(prof["actual_depth_m"]) &
        np.isfinite(prof["temperature_C"]) &
        (prof["actual_depth_m"] >= 0.0)
    ].sort_values("actual_depth_m")

    deepest_measured_temp = float(prof["temperature_C"].iloc[-1])

    t = a * depths_m + b

    # Prevent warming below the deepest observed temperature
    t = np.minimum(t, deepest_measured_temp + max_warming_above_deepest_sensor_c)

    # Apply broad cold floor
    t = np.clip(t, basal_min_c, None)

    return t


def format_cds_date_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> str:
    start_date = pd.to_datetime(start_ts).strftime("%Y-%m-%d")
    end_date = pd.to_datetime(end_ts).strftime("%Y-%m-%d")
    return f"{start_date}/{end_date}"


def enforce_fraction_closure(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for i in range(len(df)):
        vi = float(df.at[i, "Vol_Frac_I"])
        vw = float(df.at[i, "Vol_Frac_W"])
        vv = float(df.at[i, "Vol_Frac_V"])
        vs = float(df.at[i, "Vol_Frac_S"]) if "Vol_Frac_S" in df.columns else 0.0

        total = vi + vw + vv + vs

        if not np.isfinite(total) or total <= 0:
            raise RuntimeError(f"Invalid fraction total at row {i}: {total}")

        # Only correct if needed
        if abs(total - 1.0) > 1e-12:
            scale = 1.0 / total
            df.at[i, "Vol_Frac_I"] = vi * scale
            df.at[i, "Vol_Frac_W"] = vw * scale
            df.at[i, "Vol_Frac_V"] = vv * scale
            if "Vol_Frac_S" in df.columns:
                df.at[i, "Vol_Frac_S"] = vs * scale

    return df


# =============================================================================
# BUILD INITIAL SNO
# =============================================================================

def first_valid_temp_profile(temp_df: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Series]:
    for ts, row in temp_df.iterrows():
        if np.isfinite(row.to_numpy(dtype=float)).any():
            return ts, row
    raise ValueError("No valid temperature profile found in Tempconcatenated.")


def promice_surface_change_at(promice_df: pd.DataFrame, ts: pd.Timestamp) -> float:
    s = promice_df["cumulative_surface_total_change_m"].sort_index()
    prior = s.loc[s.index <= ts]
    if len(prior) == 0:
        return 0.0
    return float(prior.iloc[-1])


def interpolate_temperature_to_density_layers(
    corrected_temp_profile: pd.DataFrame,
    density_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build initial layer temperatures for the .sno file.

    Rules:
      - interpolate directly within observed temperature range
      - below deepest observed temperature, extrapolate with deep gradient
      - but never allow temperatures below the deepest observed sensor to be warmer
        than that deepest observed temperature
      - add optional basal layers using the same rule
    """
    prof = corrected_temp_profile.copy()
    prof = prof[
        np.isfinite(prof["actual_depth_m"]) &
        np.isfinite(prof["temperature_C"]) &
        (prof["actual_depth_m"] >= 0.0)
    ].sort_values("actual_depth_m")

    if len(prof) < 2:
        raise ValueError("Need at least two valid subsurface temperatures.")

    x = prof["actual_depth_m"].to_numpy(dtype=float)
    y = prof["temperature_C"].to_numpy(dtype=float)

    deepest_obs_depth = float(x[-1])
    deepest_obs_temp = float(y[-1])

    # Fit the deepest trend
    a, b = fit_deep_profile_gradient(
        corrected_temp_profile=corrected_temp_profile,
        lookback_m=BASAL_TREND_LOOKBACK_M,
    )

    # Important: do not allow warming with depth below deepest observation
    a = min(a, 0.0)

    out = density_df.copy().sort_values("mid_m").reset_index(drop=True)
    mids = out["mid_m"].to_numpy(dtype=float)

    t_vals = np.empty_like(mids)

    # Within observed range: direct interpolation
    in_obs = mids <= deepest_obs_depth
    t_vals[in_obs] = np.interp(mids[in_obs], x, y, left=y[0], right=y[-1])

    # Below deepest observed sensor: extrapolate, but never warmer than deepest observed temp
    below_obs = mids > deepest_obs_depth
    if np.any(below_obs):
        t_extrap = a * mids[below_obs] + b
        t_extrap = np.minimum(t_extrap, deepest_obs_temp)
        t_extrap = np.clip(t_extrap, BASAL_TEMP_MIN_C, deepest_obs_temp)
        t_vals[below_obs] = t_extrap

    out["T_C"] = t_vals

    # Add optional extra basal layers
    if ADD_BASAL_LAYERS:
        rho_bottom = float(out.iloc[-1]["density_kg_m3"])
        current_top = float(out["bottom_m"].max())

        extra_rows = []
        for thick in BASAL_LAYER_THICKNESSES_M:
            thick = float(thick)
            top_m = current_top
            bottom_m = current_top + thick
            mid_m = 0.5 * (top_m + bottom_m)

            t_c = a * mid_m + b
            t_c = min(t_c, deepest_obs_temp)
            t_c = float(np.clip(t_c, BASAL_TEMP_MIN_C, deepest_obs_temp))

            extra_rows.append({
                "top_m": top_m,
                "bottom_m": bottom_m,
                "mid_m": mid_m,
                "thickness_m": thick,
                "density_kg_m3": rho_bottom,
                "T_C": t_c,
            })

            current_top = bottom_m

        out = pd.concat([out, pd.DataFrame(extra_rows)], ignore_index=True)

    return out.sort_values("mid_m").reset_index(drop=True)


def compute_volume_fractions(
    T_C: float,
    density_kg_m3: float,
    ice_density: Optional[float] = None,
    water_density: Optional[float] = None,
    default_water_frac_at_zero: Optional[float] = None,
    zero_temp_tol: Optional[float] = None,
) -> Tuple[float, float, float]:
    if ice_density is None:
        ice_density = ICE_DENSITY
    if water_density is None:
        water_density = WATER_DENSITY
    if default_water_frac_at_zero is None:
        default_water_frac_at_zero = DEFAULT_WATER_FRAC_AT_ZERO
    if zero_temp_tol is None:
        zero_temp_tol = ZERO_TEMP_TOL
    """
    Convert bulk density and temperature into SNOWPACK volume fractions.

    Rules:
      - For subfreezing layers, assume no liquid water.
      - For layers at/above 0 C, assign a small default liquid-water fraction.
      - Cap impossible densities to maintain physically valid fractions.
    """
    T_C = float(T_C)
    rho_in = float(density_kg_m3)

    if not np.isfinite(T_C) or not np.isfinite(rho_in):
        raise ValueError(f"Non-finite input to compute_volume_fractions: T={T_C}, rho={rho_in}")

    if rho_in < 0.0:
        raise ValueError(f"Negative density is not allowed: {rho_in:.3f}")

    rho = rho_in
    capped = False

    if T_C < -zero_temp_tol:
        # Cold layer: dry firn/ice only
        if rho > ice_density:
            rho = ice_density
            capped = True

        vol_frac_w = 0.0
        vol_frac_i = rho / ice_density
        vol_frac_v = 1.0 - vol_frac_i

    else:
        # Temperate or slightly above-zero layer: allow small liquid-water fraction
        vol_frac_w = float(default_water_frac_at_zero)

        max_bulk_rho = ice_density * (1.0 - vol_frac_w) + water_density * vol_frac_w
        if rho > max_bulk_rho:
            rho = max_bulk_rho
            capped = True

        vol_frac_i = (rho - water_density * vol_frac_w) / ice_density
        vol_frac_v = 1.0 - vol_frac_i - vol_frac_w

    # Small floating-point cleanup
    if abs(vol_frac_i) < 1e-12:
        vol_frac_i = 0.0
    if abs(vol_frac_w) < 1e-12:
        vol_frac_w = 0.0
    if abs(vol_frac_v) < 1e-12:
        vol_frac_v = 0.0

    # Guard against numerical drift
    if vol_frac_i < -1e-10 or vol_frac_w < -1e-10 or vol_frac_v < -1e-10:
        raise ValueError(
            f"Invalid fractions for T={T_C:.4f}, density={rho_in:.3f} "
            f"(after capping rho={rho:.3f}): "
            f"Vi={vol_frac_i:.6f}, Vw={vol_frac_w:.6f}, Vv={vol_frac_v:.6f}"
        )

    vol_frac_i = max(0.0, vol_frac_i)
    vol_frac_w = max(0.0, vol_frac_w)
    vol_frac_v = max(0.0, vol_frac_v)

    total = vol_frac_i + vol_frac_w + vol_frac_v
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(
            f"Non-physical total fraction for T={T_C:.4f}, density={rho_in:.3f}: total={total}"
        )

    vol_frac_i /= total
    vol_frac_w /= total
    vol_frac_v /= total

    if capped:
        print(
            f"Warning: capped density from {rho_in:.3f} to {rho:.3f} kg m^-3 "
            f"for T={T_C:.4f} C"
        )

    return vol_frac_i, vol_frac_w, vol_frac_v


def build_layer_timestamps(first_temp_ts: pd.Timestamp, n_layers: int) -> List[pd.Timestamp]:
    end_ts = first_temp_ts.floor("min") - pd.Timedelta(minutes=1)
    start_year = end_ts.year - n_layers + 1
    timestamps = [
        pd.Timestamp(year=start_year + i, month=1, day=1, hour=0, minute=0)
        for i in range(n_layers)
    ]
    if timestamps[-1] >= first_temp_ts:
        raise RuntimeError("Generated layer timestamps are not all before first temperature timestamp.")
    return timestamps


def validate_layer_df_for_sno(layer_df: pd.DataFrame) -> None:
    required = ["T_C", "density_kg_m3", "thickness_m", "mid_m"]
    missing = [c for c in required if c not in layer_df.columns]
    if missing:
        raise ValueError(f"layer_df missing required columns: {missing}")

    bad = []

    for idx, r in layer_df.iterrows():
        T = float(r["T_C"])
        rho = float(r["density_kg_m3"])
        thick = float(r["thickness_m"])
        mid = float(r["mid_m"])

        if not np.isfinite(T) or not np.isfinite(rho) or not np.isfinite(thick) or not np.isfinite(mid):
            bad.append((idx, mid, thick, T, rho, "non-finite value"))
            continue

        if thick <= 0.0:
            bad.append((idx, mid, thick, T, rho, "non-positive thickness"))

        if rho < 0.0:
            bad.append((idx, mid, thick, T, rho, "negative density"))

        if T < -ZERO_TEMP_TOL and rho > ICE_DENSITY:
            print(
                f"Warning: cold layer above pure-ice density at row={idx}, "
                f"mid={mid:.3f} m, T={T:.4f} C, rho={rho:.3f} kg m^-3; "
                f"will cap to {ICE_DENSITY:.3f}"
            )

        if T >= -ZERO_TEMP_TOL:
            max_bulk_rho = ICE_DENSITY * (1.0 - DEFAULT_WATER_FRAC_AT_ZERO) + WATER_DENSITY * DEFAULT_WATER_FRAC_AT_ZERO
            if rho > max_bulk_rho:
                print(
                    f"Warning: temperate layer above max allowed bulk density at row={idx}, "
                    f"mid={mid:.3f} m, T={T:.4f} C, rho={rho:.3f} kg m^-3; "
                    f"will cap to {max_bulk_rho:.3f}"
                )

    if bad:
        msg = "\n".join(
            f"row={idx}, mid_m={mid}, thickness_m={thick}, T_C={T}, rho={rho}: {why}"
            for idx, mid, thick, T, rho, why in bad
        )
        raise ValueError(f"Invalid layer_df entries before SNO build:\n{msg}")


def build_sno_dataframe(layer_df: pd.DataFrame, first_temp_ts: pd.Timestamp) -> pd.DataFrame:
    validate_layer_df_for_sno(layer_df)

    df = layer_df.sort_values("mid_m", ascending=False).reset_index(drop=True)
    timestamps = build_layer_timestamps(first_temp_ts, len(df))

    rows = []
    for i, (_, r) in enumerate(df.iterrows()):
        T = float(r["T_C"])
        thick = float(r["thickness_m"])
        rho = float(r["density_kg_m3"])

        try:
            vol_i, vol_w, vol_v = compute_volume_fractions(T, rho)
        except Exception as e:
            raise ValueError(
                f"Failed volume-fraction calculation at layer index={i}, "
                f"mid_m={float(r['mid_m']):.3f}, top_m={float(r['top_m']):.3f}, "
                f"bottom_m={float(r['bottom_m']):.3f}, T_C={T:.4f}, "
                f"density={rho:.3f}"
            ) from e

        rows.append({
            "timestamp": timestamps[i].strftime("%Y-%m-%dT%H:%M:%S"),
            "Layer_Thick": thick,
            "T": T,
            "Vol_Frac_I": vol_i,
            "Vol_Frac_W": vol_w,
            "Vol_Frac_V": vol_v,
            "Vol_Frac_S": 0.0,
            "Rho_S": rho,
            "Conduc_S": DEFAULT_CONDUC_S,
            "HeatCapac_S": DEFAULT_HEATCAPAC_S,
            "rg": DEFAULT_RG,
            "rb": DEFAULT_RB,
            "dd": DEFAULT_DD,
            "sp": DEFAULT_SP,
            "mk": DEFAULT_MK,
            "mass_hoar": DEFAULT_MASS_HOAR,
            "ne": DEFAULT_NE,
            "CDot": DEFAULT_CDOT,
            "metamo": DEFAULT_METAMO,
        })

    out = pd.DataFrame(rows)

    totals = out["Vol_Frac_I"] + out["Vol_Frac_W"] + out["Vol_Frac_V"]
    if not np.allclose(totals.to_numpy(), 1.0, atol=1e-8):
        raise RuntimeError("Volume fractions do not sum to 1 after build_sno_dataframe().")

    return out


def write_sno_file(
    out_path: Path,
    meta: SiteMetadata,
    sno_df: pd.DataFrame,
    hs_last: float,
    erosion_level: int,
    profile_date: pd.Timestamp,
) -> None:
    fields = (
        "timestamp Layer_Thick T Vol_Frac_I Vol_Frac_W Vol_Frac_V "
        "Vol_Frac_S Rho_S Conduc_S HeatCapac_S rg rb dd sp mk "
        "mass_hoar ne CDot metamo"
    )
    units_offset = "0 0 273.15 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("SMET 1.1 ASCII\n")
        f.write("[HEADER]\n")
        f.write(f"station_id       = {meta.station_id}\n")
        f.write(f"station_name     = {meta.station_name}\n")
        f.write(f"latitude         = {meta.latitude:.6f}\n")
        f.write(f"longitude        = {meta.longitude:.6f}\n")
        f.write(f"altitude         = {meta.elevation:.3f}\n")
        f.write("nodata           = -999\n")
        f.write(f"tz               = {SNO_TZ}\n")
        f.write(f"source           = {SNO_SOURCE}\n")
        f.write(f"ProfileDate      = {profile_date.strftime('%Y-%m-%dT%H:%M:%S')}\n")
        f.write(f"HS_Last          = {hs_last:.6f}\n")
        f.write(f"SlopeAngle       = {SLOPE_ANGLE:.2f}\n")
        f.write(f"SlopeAzi         = {SLOPE_AZI:.2f}\n")
        f.write("nSoilLayerData   = 0\n")
        f.write(f"nSnowLayerData   = {len(sno_df)}\n")
        f.write(f"SoilAlbedo       = {SOIL_ALBEDO:.2f}\n")
        f.write(f"BareSoil_z0      = {BARE_SOIL_Z0:.3f}\n")
        f.write(f"CanopyHeight     = {CANOPY_HEIGHT:.2f}\n")
        f.write(f"CanopyLeafAreaIndex = {CANOPY_LAI:.6f}\n")
        f.write(f"CanopyDirectThroughfall = {CANOPY_DIRECT_THROUGHFALL:.2f}\n")
        f.write(f"WindScalingFactor = {WIND_SCALING_FACTOR:.2f}\n")
        f.write(f"ErosionLevel     = {erosion_level}\n")
        f.write(f"TimeCountDeltaHS = {TIMECOUNTDELTAHS:.1f}\n")
        f.write(f"fields           = {fields}\n")
        f.write(f"units_offset     = {units_offset}\n")
        f.write("[DATA]\n")

        for _, r in sno_df.iterrows():
            vi = min(round(float(r['Vol_Frac_I']), 10), MAX_VOL_FRAC_ICE)
            vw = round(float(r['Vol_Frac_W']), 10)
            vs = round(float(r['Vol_Frac_S']), 10)
            vv = round(1.0 - vi - vw - vs, 10)
            f.write(
                f"{r['timestamp']} "
                f"{r['Layer_Thick']:.5f} "
                f"{r['T']:.6f} "
                f"{vi:.10f} "
                f"{vw:.10f} "
                f"{vv:.10f} "
                f"{vs:.10f} "
                f"{r['Rho_S']:.6f} "
                f"{r['Conduc_S']:.3f} "
                f"{r['HeatCapac_S']:.3f} "
                f"{r['rg']:.3f} "
                f"{r['rb']:.3f} "
                f"{r['dd']:.3f} "
                f"{r['sp']:.3f} "
                f"{r['mk']:.3f} "
                f"{r['mass_hoar']:.3f} "
                f"{r['ne']:.3f} "
                f"{r['CDot']:.3f} "
                f"{r['metamo']:.3f}\n"
            )


# =============================================================================
# ERA HELPERS
# =============================================================================

def inspect_file_type(path: Path) -> str:
    with open(path, "rb") as f:
        sig = f.read(16)

    if sig.startswith(b"CDF"):
        return "netcdf3"
    if sig.startswith(b"\x89HDF"):
        return "netcdf4"
    if sig.startswith(b"PK"):
        return "zip"
    if sig.startswith(b"GRIB"):
        return "grib"

    txt = sig.decode("utf-8", errors="ignore").lower()
    if "<!doctype" in txt or "<html" in txt or "<?xml" in txt:
        return "html_or_xml"

    return "unknown"


def report_download(path: Path) -> None:
    size_mb = path.stat().st_size / (1024 * 1024)
    ftype = inspect_file_type(path)
    print(f"Downloaded {path} | {size_mb:.2f} MB | detected type: {ftype}")


def unzip_all_datasets(zip_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    nc_files = sorted(extract_dir.rglob("*.nc"))
    if nc_files:
        return nc_files

    extracted_files = [p for p in extract_dir.rglob("*") if p.is_file()]
    nc_files = []
    for p in extracted_files:
        ftype = inspect_file_type(p)
        if ftype in ("netcdf3", "netcdf4"):
            nc_files.append(p)

    if not nc_files:
        raise RuntimeError(
            f"No NetCDF files found after unzipping {zip_path}. "
            f"Extracted files: {[str(p) for p in extracted_files]}"
        )

    return sorted(nc_files)


def open_one_dataset(path: Path) -> tuple[xr.Dataset, list[Path]]:
    ftype = inspect_file_type(path)

    if ftype in ("netcdf3", "netcdf4"):
        return xr.open_dataset(path, engine="netcdf4"), []

    if ftype == "zip":
        extract_dir = path.parent / f"{path.stem}_unzipped"
        nc_paths = unzip_all_datasets(path, extract_dir)

        datasets = [xr.open_dataset(p, engine="netcdf4") for p in nc_paths]
        try:
            ds = xr.merge(datasets, compat="override", join="exact")
        except Exception:
            for d in datasets:
                try:
                    d.close()
                except Exception:
                    pass
            raise

        return ds, nc_paths + [extract_dir]

    if ftype == "grib":
        raise RuntimeError(f"{path} is GRIB, not NetCDF.")

    if ftype == "html_or_xml":
        raise RuntimeError(f"{path} looks like an HTML/XML error response from CDS, not a data file.")

    raise RuntimeError(f"Could not determine file type for {path}")


def get_time_name(ds: xr.Dataset) -> str:
    for cand in ("time", "valid_time"):
        if cand in ds.coords or cand in ds.dims:
            return cand
    raise ValueError(f"Could not identify time coordinate. Found coords={list(ds.coords)} dims={list(ds.dims)}")


def build_area(lat: float, lon: float, pad_deg: float = 0.1) -> list[float]:
    north = lat + pad_deg
    west = lon - pad_deg
    south = lat - pad_deg
    east = lon + pad_deg
    return [north, west, south, east]


def month_range(start_ts: pd.Timestamp, end_ts: pd.Timestamp) -> list[tuple[int, int]]:
    months = []
    y, m = start_ts.year, start_ts.month
    while (y < end_ts.year) or (y == end_ts.year and m <= end_ts.month):
        months.append((y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def nearest_point_da(da: xr.DataArray, lat: float, lon: float) -> xr.DataArray:
    """
    Return a point series from either:
      1) a gridded field with latitude/longitude dimensions, or
      2) an ERA5-Land timeseries product that already represents a single point.
    """
    coord_names = set(da.coords)
    dim_names = set(da.dims)

    lat_name = None
    lon_name = None

    if "latitude" in coord_names:
        lat_name = "latitude"
    elif "lat" in coord_names:
        lat_name = "lat"

    if "longitude" in coord_names:
        lon_name = "longitude"
    elif "lon" in coord_names:
        lon_name = "lon"

    # Case 1: true gridded data, where lat/lon are dimensions or indexed coordinates
    if lat_name in dim_names and lon_name in dim_names:
        return da.sel({lat_name: lat, lon_name: lon}, method="nearest")

    # Case 2: point timeseries product, where latitude/longitude are scalar coords
    if lat_name in coord_names and lon_name in coord_names:
        return da

    # Case 3: no lat/lon coordinates at all
    return da


def open_many_datasets(paths: list[Path]) -> tuple[xr.Dataset, list[Path]]:
    all_cleanup = []
    dsets = []

    for p in paths:
        ds, tmp = open_one_dataset(p)
        dsets.append(ds)
        all_cleanup.extend(tmp)

    if len(dsets) == 1:
        return dsets[0], all_cleanup

    try:
        ds = xr.concat(dsets, dim=get_time_name(dsets[0]))
    except Exception:
        for d in dsets:
            try:
                d.close()
            except Exception:
                pass
        raise

    return ds, all_cleanup


# =============================================================================
# ERA DOWNLOADS
# =============================================================================

def site_cache_name(meta: SiteMetadata) -> str:
    lat_s = f"{meta.latitude:.4f}".replace(".", "p").replace("-", "m")
    lon_s = f"{meta.longitude:.4f}".replace(".", "p").replace("-", "m")
    return f"{meta.station_id}_{lat_s}_{lon_s}"


def download_era5land_timeseries(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    lat: float,
    lon: float,
    out_path: Path,
) -> Path:
    if cdsapi is None:
        raise RuntimeError("cdsapi is not installed or importable.")

    c = cdsapi.Client()

    request = {
        "variable": [
            "2m_dewpoint_temperature",
            "2m_temperature",
            "total_precipitation",
            "surface_solar_radiation_downwards",
            "surface_thermal_radiation_downwards",
            "10m_u_component_of_wind",
            "10m_v_component_of_wind",
            "soil_temperature_level_1",
        ],
        "location": {
            "longitude": float(lon),
            "latitude": float(lat),
        },
        "date": [format_cds_date_range(start_ts, end_ts)],
        "data_format": "netcdf",
    }

    c.retrieve(ERA5LAND_DATASET, request).download(str(out_path))
    report_download(out_path)
    return out_path


def download_era5_static_geopotential_once(
    lat: float,
    lon: float,
    out_path: Path,
) -> Path:
    if out_path.exists():
        print(f"Using cached ERA5 geopotential file: {out_path}")
        return out_path

    if cdsapi is None:
        raise RuntimeError("cdsapi is not installed or importable.")

    c = cdsapi.Client()
    # ERA5 native grid is 0.25°, so use ≥0.5° padding to guarantee at least
    # one grid point falls inside the bounding box regardless of site position.
    area = build_area(lat, lon, pad_deg=0.5)

    c.retrieve(
        ERA5_DATASET,
        {
            "product_type": "reanalysis",
            "variable": ["geopotential"],
            "year": "2023",
            "month": "01",
            "day": ["01"],
            "time": ["00:00"],
            "area": area,
            "data_format": "netcdf",
            "download_format": "unarchived",
        },
        str(out_path),
    )
    report_download(out_path)
    return out_path


def download_era5land_and_static(
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
    meta: SiteMetadata,
) -> tuple[Path, Path]:
    era5land_out = ERA_TMP_DIR / f"{site_cache_name(meta)}_era5land_timeseries.nc"
    static_file = CACHE_DIR / f"{site_cache_name(meta)}_era5_geopotential.nc"

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_land = ex.submit(
            download_era5land_timeseries,
            start_ts,
            end_ts,
            meta.latitude,
            meta.longitude,
            era5land_out,
        )
        fut_static = ex.submit(
            download_era5_static_geopotential_once,
            meta.latitude,
            meta.longitude,
            static_file,
        )

        era5land_file = fut_land.result()
        static_file = fut_static.result()

    return era5land_file, static_file


# =============================================================================
# BUILD FORCING
# =============================================================================

def calc_rh_from_t_and_td(air_temp_C: np.ndarray, dew_temp_C: np.ndarray) -> np.ndarray:
    alpha1 = 17.625
    beta1 = 243.04
    es_td = np.exp((alpha1 * dew_temp_C) / (beta1 + dew_temp_C))
    es_t = np.exp((alpha1 * air_temp_C) / (beta1 + air_temp_C))
    rh = es_td / es_t
    return np.clip(rh, 0.0, 1.0)


def calc_wind_speed_dir(u10: np.ndarray, v10: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    vw = np.sqrt(u10**2 + v10**2)
    dw = np.degrees(np.arctan2(v10, u10) - (np.pi / 2.0))
    dw[dw < 0.0] += 360.0
    return vw, dw


def calc_incremental_precip_from_cumulative(tp_m: np.ndarray) -> np.ndarray:
    tp_m = np.asarray(tp_m, dtype=float)
    if tp_m.size == 0:
        return tp_m
    d = np.diff(tp_m)
    d = np.insert(d, 0, tp_m[0])
    psum_mm = d * 1000.0
    neg = psum_mm < 0.0
    psum_mm[neg] = tp_m[neg] * 1000.0
    psum_mm[psum_mm < 0.0] = 0.0
    return psum_mm


def build_forcing_from_era5land(
    era5land_files: list[Path],
    temp_df: pd.DataFrame,
    promice_df: pd.DataFrame,
    density_df: pd.DataFrame,
    meta: SiteMetadata,
    tsg_mode: str = "zero",
) -> tuple[pd.DataFrame, list[Path]]:
    dsL, tmpL = open_many_datasets(era5land_files)
    cleanup_paths = tmpL

    timeL = get_time_name(dsL)

    required = ["t2m", "ssrd", "tp", "stl1", "d2m", "u10", "v10", "strd"]
    available = sorted(dsL.data_vars)
    print("ERA5-Land data variables found:", available)

    missing = [v for v in required if v not in dsL.data_vars]
    if missing:
        raise ValueError(
            f"ERA5-Land file missing variables: {missing}. "
            f"Available data variables: {available}"
        )

    print("t2m dims:", dsL["t2m"].dims)
    print("t2m coords:", list(dsL["t2m"].coords))

    t2m = nearest_point_da(dsL["t2m"], meta.latitude, meta.longitude)
    ssrd = nearest_point_da(dsL["ssrd"], meta.latitude, meta.longitude)
    tp = nearest_point_da(dsL["tp"], meta.latitude, meta.longitude)
    stl1 = nearest_point_da(dsL["stl1"], meta.latitude, meta.longitude)
    d2m = nearest_point_da(dsL["d2m"], meta.latitude, meta.longitude)
    u10 = nearest_point_da(dsL["u10"], meta.latitude, meta.longitude)
    v10 = nearest_point_da(dsL["v10"], meta.latitude, meta.longitude)
    strd = nearest_point_da(dsL["strd"], meta.latitude, meta.longitude)

    df = pd.DataFrame({
        "timestamp": pd.to_datetime(dsL[timeL].values),
        "t2m": np.asarray(t2m.values, dtype=float),
        "ssrd": np.asarray(ssrd.values, dtype=float),
        "tp": np.asarray(tp.values, dtype=float),
        "stl1": np.asarray(stl1.values, dtype=float),
        "d2m": np.asarray(d2m.values, dtype=float),
        "u10": np.asarray(u10.values, dtype=float),
        "v10": np.asarray(v10.values, dtype=float),
        "strd": np.asarray(strd.values, dtype=float),
    })

    dsL.close()

    tmin = pd.to_datetime(temp_df.index.min()).floor("h") - pd.Timedelta(days=2)
    tmax = pd.to_datetime(temp_df.index.max()).ceil("h") + pd.Timedelta(days=2)
    df = df[(df["timestamp"] >= tmin) & (df["timestamp"] <= tmax)].copy()
    df = df.sort_values("timestamp").drop_duplicates("timestamp")

    ta_C = df["t2m"].to_numpy(dtype=float) - 273.15
    td_C = df["d2m"].to_numpy(dtype=float) - 273.15
    rh = calc_rh_from_t_and_td(ta_C, td_C)

    vw, dw = calc_wind_speed_dir(
        df["u10"].to_numpy(dtype=float),
        df["v10"].to_numpy(dtype=float),
    )

    # ERA5-Land radiation fields are hourly accumulations in J m^-2.
    # Convert to average flux over the preceding hour in W m^-2.
    iswr = df["ssrd"].to_numpy(dtype=float) / 3600.0
    ilwr = df["strd"].to_numpy(dtype=float) / 3600.0

    # Incremental precipitation. calc_incremental_precip_from_cumulative()
    # already returns mm from cumulative meters, so do not multiply again.
    psum = calc_incremental_precip_from_cumulative(
        df["tp"].to_numpy(dtype=float)
    )



    if tsg_mode == "soil_temp":
        tsg_C = df["stl1"].to_numpy(dtype=float) - 273.15

    elif tsg_mode == "profile_gradient":
        corrected_profiles = build_hourly_corrected_temp_profiles(
            temp_df=temp_df,
            promice_df=promice_df,
            max_interp_gap_hours=24,
        )

        tsg_vals = []
        for ts in pd.to_datetime(df["timestamp"]):
            prof = corrected_profiles.get(ts)
            if prof is None or prof.empty:
                tsg_vals.append(np.nan)
            else:
                try:
                    tsg_vals.append(
                        estimate_tsg_from_corrected_profile(
                            corrected_temp_profile=prof,
                            density_df=density_df,
                            lookback_m=TSG_LOOKBACK_M,
                            target_offset_m=TSG_TARGET_OFFSET_M,
                        )
                    )
                except Exception:
                    tsg_vals.append(np.nan)

        tsg_series = pd.Series(tsg_vals, index=pd.to_datetime(df["timestamp"]), dtype=float)
        tsg_series = tsg_series.interpolate(method="time", limit_direction="both")
        tsg_C = tsg_series.to_numpy(dtype=float)

    else:
        tsg_C = np.zeros_like(ta_C)

    forcing_df = pd.DataFrame({
        "timestamp": pd.to_datetime(df["timestamp"]),
        "TA": ta_C,
        "RH": rh,
        "TSG": tsg_C,
        "VW": vw,
        "DW": dw,
        "ISWR": iswr,
        "ILWR": ilwr,
        "PSUM": psum,
    })

    return forcing_df, cleanup_paths


def read_era5_geopotential_altitude(static_file: Path, lat: float, lon: float) -> Optional[float]:
    """Return ERA5 surface geopotential height (m) at the nearest grid point."""
    try:
        ds = xr.open_dataset(static_file)
        if "z" not in ds:
            print(f"Warning: 'z' not found in {static_file}; using site elevation instead.")
            ds.close()
            return None
        z_val = float(np.asarray(nearest_point_da(ds["z"], lat, lon).values).ravel()[0])
        ds.close()
        return z_val / 9.80665   # m²/s² → geopotential height in m
    except Exception as exc:
        print(f"Warning: could not read ERA5 geopotential altitude: {exc}")
        return None


def write_smet_file(out_path: Path, meta: SiteMetadata, forcing_df: pd.DataFrame,
                    era5_altitude_m: Optional[float] = None) -> None:
    altitude = era5_altitude_m if era5_altitude_m is not None else meta.elevation

    # SMET: physical = stored * multiplier + offset
    # Base offsets convert stored °C → K for TA and TSG; user adjustments are additive in °C.
    # fields: timestamp TA   RH   TSG    VW   DW  ISWR  ILWR  PSUM
    offsets     = [0, 273.15 + FORCING_TA_ADD,   FORCING_RH_ADD,   273.15,
                      FORCING_VW_ADD,   0, FORCING_ISWR_ADD, FORCING_ILWR_ADD, FORCING_PSUM_ADD]
    multipliers = [1, FORCING_TA_MULT,  FORCING_RH_MULT,  1,
                      FORCING_VW_MULT,  1, FORCING_ISWR_MULT, FORCING_ILWR_MULT, FORCING_PSUM_MULT]
    off_str  = " ".join(f"{v:g}" for v in offsets)
    mult_str = " ".join(f"{v:g}" for v in multipliers)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("SMET 1.1 ASCII\n")
        f.write("[HEADER]\n")
        f.write(f"station_id       = {meta.station_id}\n")
        f.write(f"station_name     = {meta.station_name}\n")
        f.write(f"latitude         = {meta.latitude:.6f}\n")
        f.write(f"longitude        = {meta.longitude:.6f}\n")
        f.write(f"altitude         = {altitude:.3f}\n")
        f.write("nodata           = -999\n")
        f.write(f"tz               = {SMET_TZ}\n")
        f.write("fields           = timestamp TA RH TSG VW DW ISWR ILWR PSUM\n")
        f.write(f"units_offset     = {off_str}\n")
        f.write(f"units_multiplier = {mult_str}\n")
        f.write("[DATA]\n")

        for _, r in forcing_df.iterrows():
            ts = pd.to_datetime(r["timestamp"]).strftime("%Y-%m-%dT%H:%M:%S")
            f.write(
                f"{ts} "
                f"{r['TA']:8.2f} "
                f"{r['RH']:10.3f} "
                f"{r['TSG']:8.2f} "
                f"{r['VW']:6.2f} "
                f"{r['DW']:6.1f} "
                f"{r['ISWR']:10.3f} "
                f"{r['ILWR']:10.3f} "
                f"{r['PSUM']:10.6f}\n"
            )


def remove_path(p: Path) -> None:
    try:
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    except FileNotFoundError:
        pass


def cleanup_temp_era_files(*paths: Path) -> None:
    for p in paths:
        remove_path(Path(p))


def build_smet_from_downloaded_era(
    temp_df: pd.DataFrame,
    promice_df: pd.DataFrame,
    density_df: pd.DataFrame,
    meta: SiteMetadata
) -> Optional[pd.DataFrame]:
    if FORCING_SMET_FILE.exists():
        print(f"Using existing forcing SMET: {FORCING_SMET_FILE}")
        return None

    start_ts = pd.to_datetime(temp_df.index.min()).floor("h") - pd.Timedelta(days=2)
    end_ts = pd.to_datetime(temp_df.index.max()).ceil("h") + pd.Timedelta(days=2)

    era5land_file, static_file = download_era5land_and_static(
        start_ts=start_ts,
        end_ts=end_ts,
        meta=meta,
    )

    temp_paths_to_delete = []
    try:
        forcing_df, extracted_temp_paths = build_forcing_from_era5land(
            era5land_files=[era5land_file],
            temp_df=temp_df,
            promice_df=promice_df,
            density_df=density_df,
            meta=meta,
            tsg_mode=TSG_MODE,
        )
        temp_paths_to_delete.extend(extracted_temp_paths)

        era5_alt = read_era5_geopotential_altitude(static_file, meta.latitude, meta.longitude)
        if era5_alt is not None:
            print(f"ERA5 geopotential altitude: {era5_alt:.1f} m (site elevation: {meta.elevation:.1f} m)")
        write_smet_file(FORCING_SMET_FILE, meta, forcing_df, era5_altitude_m=era5_alt)
        print(f"Wrote forcing SMET: {FORCING_SMET_FILE}")

    except Exception:
        print("SMET creation failed; keeping downloaded ERA files and extracted contents for debugging.")
        raise

    cleanup_temp_era_files(era5land_file, *temp_paths_to_delete)
    print("Deleted temporary ERA5-Land files after successful SMET creation")
    print(f"Retained cached ERA5 geopotential file: {static_file}")

    return forcing_df


# =============================================================================
# WRITE INI
# =============================================================================

def write_ini_file(out_path: Path, smet_name: str, sno_name: str,
                   water_transport: str = "BUCKET") -> None:
    text = f"""[General]
BUFFER_SIZE = 370
BUFF_BEFORE = 1.5
BUFF_GRIDS = 10

[Input]
PSUM_PH::create = PRECSPLITTING
PSUM_PH::PRECSPLITTING::type = THRESH
PSUM_PH::PRECSPLITTING::snow = 274.35
COORDSYS = UPS
COORDPARAM = N
TIME_ZONE = 0.00

METEO = SMET
METEOPATH = ./input
METEOFILE1 = {smet_name}
SNOW = SMET
SNOWPATH = ./input
SNOWFILE1 = {sno_name}

[Output]
COORDSYS = UPS
COORDPARAM = N
TIME_ZONE = 0.00

METEOPATH = ./output
WRITE_PROCESSED_METEO = FALSE
EXPERIMENT = TEMP_ASSIM_RUN

SNOW_WRITE = TRUE
SNOW = SMET
SNOWPATH = ./current_snow
SNOW_DAYS_BETWEEN = 0.0416667
FIRST_BACKUP = 0.0

PROF_WRITE = TRUE
PROF_FORMAT = PRO
AGGREGATE_PRO = FALSE
PROF_START = 0.0
PROF_DAYS_BETWEEN = 0.0416667
HARDNESS_IN_NEWTON = FALSE

TS_WRITE = FALSE
TS_FORMAT = SMET
TS_START = 0.0
TS_DAYS_BETWEEN = 0.0416667
AVGSUM_TIME_SERIES = TRUE
CUMSUM_MASS = TRUE
PRECIP_RATES = TRUE
OUT_CANOPY = FALSE
OUT_HAZ = FALSE
OUT_SOILEB = FALSE
OUT_HEAT = TRUE
OUT_T = TRUE
OUT_LW = TRUE
OUT_SW = TRUE
OUT_MASS = TRUE
OUT_METEO = TRUE
OUT_STAB = TRUE

[Snowpack]
CALCULATION_STEP_LENGTH = {CALCULATION_STEP_LENGTH_MIN:.3f}
ROUGHNESS_LENGTH = 0.002
HEIGHT_OF_METEO_VALUES = {HEIGHT_OF_METEO_VALUES:g}
HEIGHT_OF_WIND_VALUE = {HEIGHT_OF_WIND_VALUE:g}
ENFORCE_MEASURED_SNOW_HEIGHTS = FALSE
SW_MODE = INCOMING
ATMOSPHERIC_STABILITY = MO_MICHLMAYR
CANOPY = FALSE
MEAS_TSS = FALSE
CHANGE_BC = TRUE
THRESH_CHANGE_BC = 0.5
SNP_SOIL = FALSE

[SnowpackAdvanced]
VARIANT = POLAR
RESEARCH = TRUE
NUMBER_SLOPES = 1
FORCE_RH_WATER = FALSE
HOAR_THRESH_RH = 0.70
HOAR_DENSITY_BURIED = 200.0
MIN_DEPTH_SUBSURF = 0.
T_CRAZY_MIN = 200.
T_CRAZY_MAX = 280.
METAMORPHISM_MODEL = DEFAULT
NEW_SNOW_GRAIN_SIZE = 0.2
STRENGTH_MODEL = DEFAULT
VISCOSITY_MODEL = DEFAULT
ENABLE_VAPOUR_TRANSPORT = TRUE
WATERTRANSPORTMODEL_SNOW = {water_transport}
LB_COND_WATERFLUX = FREEDRAINAGE
COUPLEDPHASECHANGES = FALSE
SOIL_EVAP_MODEL = EVAP_RESISTANCE
HN_DENSITY = PARAMETERIZED
HN_DENSITY_PARAMETERIZATION = LEHNING_NEW
HEIGHT_NEW_ELEM = 0.020
MINIMUM_L_ELEMENT = 0.002500
COMBINE_ELEMENTS = TRUE

[Filters]
ENABLE_METEO_FILTERS = TRUE
TA::FILTER1 = MIN_MAX
TA::ARG1::MIN = 190
TA::ARG1::MAX = 280

RH::FILTER1 = MIN_MAX
RH::ARG1::SOFT = FALSE
RH::ARG1::MIN = 0.01
RH::ARG1::MAX = 5.2
RH::ARG1::MIN_RESET = 0.01
RH::ARG1::MAX_RESET = 1.2
RH::FILTER2 = MIN_MAX
RH::ARG2::SOFT = true
RH::ARG2::MIN = 0.01
RH::ARG2::MAX = 1.0

ISWR::FILTER1 = MIN_MAX
ISWR::ARG1::MIN = -10
ISWR::ARG1::MAX = 1500
ISWR::FILTER2 = MIN_MAX
ISWR::ARG2::SOFT = true
ISWR::ARG2::MIN = 0
ISWR::ARG2::MAX = 1500

ILWR::FILTER1 = MIN_MAX
ILWR::ARG1::MIN = 30
ILWR::ARG1::MAX = 355
ILWR::FILTER2 = MIN_MAX
ILWR::ARG2::SOFT = true
ILWR::ARG2::MIN = 35
ILWR::ARG2::MAX = 350

TSG::FILTER1 = MIN_MAX
TSG::ARG1::MIN = 200
TSG::ARG1::MAX = 275

VW::FILTER1 = MIN_MAX
VW::ARG1::MIN = -2
VW::ARG1::MAX = 70
VW::FILTER2 = MIN_MAX
VW::ARG2::SOFT = true
VW::ARG2::MIN = 0
VW::ARG2::MAX = 50

ENABLE_TIME_FILTERS = TRUE

[Interpolations1D]
MAX_GAP_SIZE = 439200

PSUM::RESAMPLE1 = ACCUMULATE
PSUM::ARG1::PERIOD = 900

RHO_HN::RESAMPLE1 = NONE

TSG::RESAMPLE1 = LINEAR
TSG::ARG1::MAX_GAP_SIZE = 219600

VW::RESAMPLE1 = NEAREST
VW::ARG1::EXTRAPOLATE = true

DW::RESAMPLE1 = NEAREST
DW::ARG1::EXTRAPOLATE = true

[InputEditing]
ENABLE_TIMESERIES_EDITING = TRUE

[SnowpackSeaice]
CHECK_INITIAL_CONDITIONS = FALSE

[TechSnow]
SNOW_GROOMING = FALSE
"""
    out_path.write_text(text, encoding="utf-8")


# =============================================================================
# MOVING SENSOR TABLES FOR ASSIMILATION
# =============================================================================

def keep_hourly(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df.index.minute == 0) & (df.index.second == 0)].copy()


def make_hourly_index(t0: pd.Timestamp, t1: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.date_range(t0.floor("h"), t1.ceil("h"), freq="1h")


def build_temp_support_mask(
    original_hourly: pd.DataFrame,
    filled_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    True where a temperature value is considered valid for assimilation:
      - originally observed, or
      - filled by interpolation across a short gap (<= interpolation limit)

    False where the value is still unsupported after interpolation, which means
    the gap was too long and assimilation should not use that sensor at that time.
    """
    support = pd.DataFrame(index=filled_hourly.index, columns=filled_hourly.columns, dtype=bool)

    for c in filled_hourly.columns:
        orig_valid = original_hourly[c].notna()
        filled_valid = filled_hourly[c].notna()
        support[c] = orig_valid | filled_valid

    return support


def interpolate_small_temp_gaps(df: pd.DataFrame, limit_hours: int = 24) -> pd.DataFrame:
    """
    Interpolate along time only through short gaps.

    pandas 'limit' here is the maximum number of consecutive NaNs to fill, so
    limit_hours=24 means gaps of 24 hourly samples or less can be bridged.
    Longer gaps remain NaN.
    """
    out = df.copy()
    for c in out.columns:
        out[c] = out[c].interpolate(
            method="time",
            limit=limit_hours,
            limit_direction="both",
        )
    return out


def build_hourly_surface_change(promice_df: pd.DataFrame, hourly_index: pd.DatetimeIndex) -> pd.Series:
    s = promice_df["cumulative_surface_total_change_m"].copy().sort_index()
    s = s.reindex(s.index.union(hourly_index)).sort_index().interpolate(method="time")
    s = s.reindex(hourly_index)
    return s


def build_corrected_depth_tables(
    temp_df: pd.DataFrame,
    promice_df: pd.DataFrame,
    max_interp_gap_hours: int = 24,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    temp_df = keep_hourly(temp_df)
    hourly_index = make_hourly_index(temp_df.index.min(), temp_df.index.max())

    temp_hourly_orig = temp_df.reindex(hourly_index)
    temp_hourly = interpolate_small_temp_gaps(temp_hourly_orig, limit_hours=max_interp_gap_hours)

    cumulative_change = build_hourly_surface_change(promice_df, hourly_index)

    corrected_depth = pd.DataFrame(index=hourly_index)
    for install_depth in temp_hourly.columns:
        corrected_depth[install_depth] = float(install_depth) + cumulative_change

    corrected_depth["cumulative_surface_total_change_m"] = cumulative_change
    return corrected_depth, temp_hourly


def build_long_observation_table(
    temp_hourly: pd.DataFrame,
    corrected_depth_wide: pd.DataFrame,
    temp_hourly_original: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    rows = []
    install_depths = [c for c in temp_hourly.columns]

    if temp_hourly_original is None:
        temp_hourly_original = temp_hourly.copy()

    support_mask = build_temp_support_mask(temp_hourly_original, temp_hourly)

    for ts in temp_hourly.index:
        for d0 in install_depths:
            temp_val = temp_hourly.at[ts, d0]
            is_supported = bool(support_mask.at[ts, d0]) if pd.notna(support_mask.at[ts, d0]) else False

            if not is_supported or pd.isna(temp_val):
                temp_val = np.nan

            rows.append({
                "timestamp": ts,
                "sensor_name": f"sensor_{d0:g}",
                "install_depth_m": float(d0),
                "actual_depth_m": float(corrected_depth_wide.at[ts, d0]) if pd.notna(corrected_depth_wide.at[ts, d0]) else np.nan,
                "temperature_C": float(temp_val) if pd.notna(temp_val) else np.nan,
            })

    long_df = pd.DataFrame(rows).sort_values(["timestamp", "actual_depth_m"])
    return long_df


# =============================================================================
# SNO PARSING / UPDATE
# =============================================================================

def parse_sno_header_fields(lines: list[str]) -> list[str]:
    for line in lines:
        s = line.strip()
        if s.lower().startswith("fields"):
            _, rhs = s.split("=", 1)
            return rhs.strip().split()
    raise RuntimeError("Could not find 'fields =' in .sno header.")


def find_data_start(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if line.strip().upper() == "[DATA]":
            return i + 1
    raise RuntimeError("Could not find [DATA] section in .sno file.")


def parse_sno_data_table(lines: list[str]) -> tuple[int, int, list[str], pd.DataFrame]:
    fields = parse_sno_header_fields(lines)
    data_start = find_data_start(lines)

    rows = []
    end_idx = data_start

    for j in range(data_start, len(lines)):
        raw = lines[j].strip()
        if not raw:
            end_idx = j
            break

        parts = raw.split()
        if len(parts) != len(fields):
            end_idx = j
            break

        row = {}
        parse_failed = False
        for k, field in enumerate(fields):
            val = parts[k]
            if field.lower() == "timestamp":
                row[field] = val
            else:
                try:
                    row[field] = float(val)
                except ValueError:
                    parse_failed = True
                    break

        if parse_failed:
            end_idx = j
            break

        rows.append(row)
        end_idx = j + 1

    df = pd.DataFrame(rows, columns=fields)
    if df.empty:
        raise RuntimeError("Parsed [DATA] section is empty in .sno file.")

    return data_start, end_idx, fields, df


def get_sno_temp_units_offset(lines: list[str]) -> float:
    """Return the units_offset value for the T column from the .sno header.

    Script-generated files declare ``units_offset = 0 0 273.15 ...``, meaning T
    is stored in Celsius (SNOWPACK adds 273.15 when reading).  SNOWPACK-written
    restart files omit units_offset entirely and store T in Kelvin directly.
    Returns 273.15 if T is stored in Celsius, 0.0 if stored in Kelvin.
    """
    fields_line = None
    offset_line = None
    for line in lines:
        s = line.strip()
        if s.startswith("fields"):
            fields_line = s
        elif s.startswith("units_offset"):
            offset_line = s
        if s == "[DATA]":
            break

    if fields_line is None or offset_line is None:
        return 0.0

    fields = fields_line.split("=", 1)[1].strip().split()
    offsets = offset_line.split("=", 1)[1].strip().split()

    try:
        temp_idx = fields.index("T")
        return float(offsets[temp_idx])
    except (ValueError, IndexError):
        return 0.0


def guess_depth_and_temperature_columns(df: pd.DataFrame) -> tuple[str, str]:
    temp_candidates = ["T", "temp", "Temp", "temperature", "Temperature", "Te"]
    depth_candidates = ["Layer_Thick", "layer_thick", "thickness", "Thickness", "L", "z", "depth"]

    temp_col = None
    depth_col = None

    for c in df.columns:
        if c in temp_candidates or c.lower() in [x.lower() for x in temp_candidates]:
            temp_col = c
            break

    for c in df.columns:
        if c in depth_candidates or c.lower() in [x.lower() for x in depth_candidates]:
            depth_col = c
            break

    if temp_col is None:
        raise RuntimeError(f"Could not identify temperature column in .sno file. Columns: {list(df.columns)}")
    if depth_col is None:
        raise RuntimeError(f"Could not identify layer thickness column in .sno file. Columns: {list(df.columns)}")

    return depth_col, temp_col


def get_profile_observations_for_time(long_df: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    g = long_df[long_df["timestamp"] == ts].copy()
    g = g[np.isfinite(g["actual_depth_m"]) & np.isfinite(g["temperature_C"])]
    return g.sort_values("actual_depth_m")


def interpolate_observed_profile_to_model_depths(
    obs_depths: np.ndarray,
    obs_temps: np.ndarray,
    model_depths: np.ndarray,
) -> np.ndarray:
    valid = np.isfinite(obs_depths) & np.isfinite(obs_temps)
    if valid.sum() < 2:
        raise ValueError("Need at least two valid observations to interpolate.")

    x = obs_depths[valid]
    y = obs_temps[valid]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    interp = np.interp(model_depths, x, y, left=y[0], right=y[-1])
    return interp


def rewrite_sno_profiledate_and_clip_timestamps(
    lines: list[str],
    new_profile_date: pd.Timestamp,
) -> list[str]:
    out = lines.copy()
    profile_date_str = pd.to_datetime(new_profile_date).strftime("%Y-%m-%dT%H:%M:%S")

    for i, line in enumerate(out):
        s = line.strip()
        if s.startswith("ProfileDate"):
            left, _ = line.split("=", 1)
            out[i] = f"{left}= {profile_date_str}\n"
            break

    start, end, fields, df = parse_sno_data_table(out)
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        ts = ts.where(ts <= new_profile_date, new_profile_date)
        df["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")

    # Fix volume-fraction sums (SNOWPACK restart files use 6-dp precision which
    # can leave sums at 0.999999) and enforce the ice-fraction cap.
    # For the cap: ice mass and sensible heat are conserved by scaling Layer_Thick
    # proportionally — ρ_i × θ_i × thick is invariant, and so is θ_i × c_i × T × thick.
    # The "created" air pore (Δθ_v = Δθ_i / original_θ_i × θ_i_new) is physically real:
    # SNOWPACK's compaction model drove θ_i to 1.0 as a numerical artefact; the cap
    # restores the small residual porosity that RE needs for a finite K(θ).
    frac_i_col = next((f for f in fields if f == "Vol_Frac_I"), None)
    frac_w_col = next((f for f in fields if f == "Vol_Frac_W"), None)
    frac_v_col = next((f for f in fields if f == "Vol_Frac_V"), None)
    frac_s_col = next((f for f in fields if f == "Vol_Frac_S"), None)
    thick_col  = next((f for f in fields if f == "Layer_Thick"), None)

    hs_changed = False
    if frac_i_col and thick_col:
        for idx in df.index:
            vi_raw = float(df.at[idx, frac_i_col])
            if vi_raw > MAX_VOL_FRAC_ICE:
                scale = vi_raw / MAX_VOL_FRAC_ICE
                df.at[idx, frac_i_col] = MAX_VOL_FRAC_ICE
                df.at[idx, thick_col]  = float(df.at[idx, thick_col]) * scale
                hs_changed = True
    elif frac_i_col:
        # No thickness column — cap without mass conservation (fallback only)
        for idx in df.index:
            vi_raw = float(df.at[idx, frac_i_col])
            if vi_raw > MAX_VOL_FRAC_ICE:
                df.at[idx, frac_i_col] = MAX_VOL_FRAC_ICE

    if hs_changed:
        new_hs_last = df[thick_col].astype(float).sum()
        for i, line in enumerate(out[:start]):
            if line.strip().startswith("HS_Last"):
                left, _ = line.split("=", 1)
                out[i] = f"{left}= {new_hs_last:.6f}\n"
                break

    new_rows = []
    for _, row in df.iterrows():
        if frac_i_col and frac_v_col:
            _vi = round(float(row[frac_i_col]), 10)
            _vw = round(float(row[frac_w_col]), 10) if frac_w_col else 0.0
            _vs = round(float(row[frac_s_col]), 10) if frac_s_col else 0.0
            _vv = round(1.0 - _vi - _vw - _vs, 10)
        vals = []
        for field in fields:
            val = row[field]
            if field.lower() == "timestamp":
                vals.append(str(val))
            elif frac_i_col and field == frac_i_col:
                vals.append(f"{_vi:.10f}")
            elif frac_w_col and field == frac_w_col:
                vals.append(f"{_vw:.10f}")
            elif frac_v_col and field == frac_v_col:
                vals.append(f"{_vv:.10f}")
            elif frac_s_col and field == frac_s_col:
                vals.append(f"{_vs:.10f}")
            else:
                vals.append(f"{float(val):.10f}")
        new_rows.append(" ".join(vals) + "\n")

    out[start:end] = new_rows

    return out



def get_sno_liquid_water_columns(df: pd.DataFrame) -> list[str]:
    candidates = [
        "Vol_Frac_W",
        "Vol_Frac_W_Pref",
        "Vol_Frac_Pref_W",
        "water_pref",
    ]
    return [c for c in candidates if c in df.columns]


def validate_sno_fraction_sums(
    df: pd.DataFrame,
    tol: float = 5e-6,
) -> None:
    required = ["Vol_Frac_I", "Vol_Frac_W", "Vol_Frac_V"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required SNO fraction columns: {missing}")

    for i in range(len(df)):
        vi = float(df.at[i, "Vol_Frac_I"])
        vw = float(df.at[i, "Vol_Frac_W"])
        vv = float(df.at[i, "Vol_Frac_V"])
        vs = float(df.at[i, "Vol_Frac_S"]) if "Vol_Frac_S" in df.columns else 0.0

        total = vi + vw + vv + vs
        if not np.isfinite(total):
            raise RuntimeError(
                f"Non-finite volume fraction total at row {i}: "
                f"I={vi:.12f}, W={vw:.12f}, V={vv:.12f}, S={vs:.12f}, total={total}"
            )

        # Allow tiny roundoff drift from SNOWPACK-written restart files.
        # Values like 0.999999 or 1.000001 are usually just 6-decimal output rounding,
        # not a physical inconsistency.
        if abs(total - 1.0) > tol:
            raise RuntimeError(
                f"Volume fractions do not sum to 1 at row {i}: "
                f"I={vi:.12f}, W={vw:.12f}, V={vv:.12f}, S={vs:.12f}, total={total:.12f}"
            )


def enforce_enthalpy_safe_restart_state(df: pd.DataFrame, temp_col: str) -> pd.DataFrame:
    df = df.copy()
    water_cols = get_sno_liquid_water_columns(df)

    for i in range(len(df)):
        total_water = 0.0
        for c in water_cols:
            total_water += float(df.at[i, c])

        # Wet layer must remain exactly at 0 C
        if total_water > 0.0:
            df.at[i, temp_col] = 0.0

        T_C = float(df.at[i, temp_col])

        # Frozen layer must be exactly dry
        if T_C < 0.0:
            for c in water_cols:
                df.at[i, c] = 0.0

    return df



def update_sno_temperatures_from_moving_profile(
    sno_in: Path,
    sno_out: Path,
    obs_profile: pd.DataFrame,
    alpha: float,
    new_profile_date: pd.Timestamp,
    n_hours: int = 1,
    return_adjustments: bool = False,
) -> "None | tuple[list[tuple[int, float]], bool]":
    """Adjust layer temperatures in-place.

    When return_adjustments=True: does NOT write sno_out.  Returns
    (adjustments, needs_reload) where adjustments = [(layer_idx_from_bottom, T_K), ...]
    and needs_reload=True means wet-layer draining or no-obs case occurred and
    the caller must fall back to a full SNO write + RELOAD_SNO.
    When return_adjustments=False (default): writes sno_out and returns None.
    """
    profile = obs_profile.copy()
    profile = profile[
        np.isfinite(profile["actual_depth_m"]) &
        np.isfinite(profile["temperature_C"])
    ].sort_values("actual_depth_m")

    lines = sno_in.read_text(encoding="utf-8").splitlines(keepends=True)

    # Detect whether T is stored in Celsius (units_offset=273.15, script-generated)
    # or Kelvin (units_offset absent/0, SNOWPACK restart). Convert to Celsius for
    # blending/clamping, then convert back before writing.
    temp_units_offset = get_sno_temp_units_offset(lines)
    # kelvin_shift: subtract from stored value to get Celsius
    # e.g. offset=0 → stored is Kelvin → shift=273.15; offset=273.15 → stored is C → shift=0
    kelvin_shift = 273.15 - temp_units_offset

    if len(profile) < MIN_OBS_FOR_ADJUST:
        # Even when skipping temperature assimilation, we must still fix enthalpy
        # (clamp wet layers to 0 °C) and volume-fraction sums, because SNOWPACK
        # restart files use 6-decimal precision which can leave sums at 0.999999
        # and may contain above-zero wet layers at exactly T_CRAZY_MAX (276 K).
        _ds, _de, _fields_early, _df_early = parse_sno_data_table(lines)
        _, _tc_early = guess_depth_and_temperature_columns(_df_early)
        _df_early[_tc_early] = _df_early[_tc_early].astype(float) - kelvin_shift
        _df_early = enforce_enthalpy_safe_restart_state(_df_early, temp_col=_tc_early)
        _df_early = enforce_fraction_closure(_df_early)
        _df_early[_tc_early] = _df_early[_tc_early].astype(float) + kelvin_shift
        # Rebuild lines with corrected values using the residual-aware writer
        _frac_i = next((f for f in _fields_early if f == "Vol_Frac_I"), None)
        _frac_w = next((f for f in _fields_early if f == "Vol_Frac_W"), None)
        _frac_v = next((f for f in _fields_early if f == "Vol_Frac_V"), None)
        _frac_s = next((f for f in _fields_early if f == "Vol_Frac_S"), None)
        _new_rows = []
        for _, _row in _df_early.iterrows():
            if _frac_i and _frac_v:
                _vi = round(float(_row[_frac_i]), 10)
                _vw = round(float(_row[_frac_w]), 10) if _frac_w else 0.0
                _vs = round(float(_row[_frac_s]), 10) if _frac_s else 0.0
                _vv = round(1.0 - _vi - _vw - _vs, 10)
            _vals = []
            for _f in _fields_early:
                _v = _row[_f]
                if _f.lower() == "timestamp":
                    _vals.append(str(_v))
                elif _frac_i and _f == _frac_i:
                    _vals.append(f"{_vi:.10f}")
                elif _frac_w and _f == _frac_w:
                    _vals.append(f"{_vw:.10f}")
                elif _frac_v and _f == _frac_v:
                    _vals.append(f"{_vv:.10f}")
                elif _frac_s and _f == _frac_s:
                    _vals.append(f"{_vs:.10f}")
                else:
                    _vals.append(f"{float(_v):.10f}")
            _new_rows.append(" ".join(_vals) + "\n")
        lines[_ds:_de] = _new_rows
        lines = rewrite_sno_profiledate_and_clip_timestamps(lines, new_profile_date)
        sno_out.write_text("".join(lines), encoding="utf-8")
        if return_adjustments:
            # No obs — fraction corrections applied; caller must RELOAD_SNO
            return ([], True)
        return

    obs_depths = profile["actual_depth_m"].to_numpy(dtype=float)
    obs_temps_C = profile["temperature_C"].to_numpy(dtype=float)
    obs_temps_C = np.clip(obs_temps_C, TEMP_MIN_C, TEMP_MAX_C)

    start, end, fields, df = parse_sno_data_table(lines)
    thick_col, temp_col = guess_depth_and_temperature_columns(df)

    thickness = df[thick_col].to_numpy(dtype=float)
    model_temps_C = df[temp_col].to_numpy(dtype=float) - kelvin_shift

    thickness_surface_order = thickness[::-1]
    tops = np.concatenate(([0.0], np.cumsum(thickness_surface_order[:-1])))
    centers_surface_order = tops + 0.5 * thickness_surface_order
    model_depths = centers_surface_order[::-1]

    obs_interp_C = interpolate_observed_profile_to_model_depths(
        obs_depths=obs_depths,
        obs_temps=obs_temps_C,
        model_depths=model_depths,
    )

    obs_min_depth = np.nanmin(obs_depths)
    obs_max_depth = np.nanmax(obs_depths)
    margin = 0.10
    in_obs_range = (model_depths >= obs_min_depth + margin) & (model_depths <= obs_max_depth - margin)

    water_cols = get_sno_liquid_water_columns(df)
    model_water = np.zeros(len(df), dtype=float)
    for c in water_cols:
        model_water += pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    new_temps_C = model_temps_C.copy()

    proposed_C = model_temps_C + alpha * (obs_interp_C - model_temps_C)
    delta_C = proposed_C - model_temps_C
    max_delta = MAX_ADJUST_PER_HOUR_C * n_hours
    delta_C = np.clip(delta_C, -max_delta, max_delta)

    # Dry layers: assimilate normally when within observed range and safely frozen.
    safe_for_assim = (
        in_obs_range &
        (model_water <= 0.0) &
        (model_temps_C <= -0.2)
    )

    # Wet layers: assimilate if observations clearly show sub-zero temperatures.
    # This breaks the thermal lock where RICHARDSEQUATION retains liquid water
    # and enforce_enthalpy_safe_restart_state keeps resetting those layers to 0 °C,
    # preventing them from ever refreezing.
    wet_drain_eligible = (
        in_obs_range &
        (model_water > 0.0) &
        np.isfinite(obs_interp_C) &
        (obs_interp_C < WET_LAYER_DRAIN_THRESHOLD_C)
    )

    new_temps_C[safe_for_assim] = model_temps_C[safe_for_assim] + delta_C[safe_for_assim]
    new_temps_C[wet_drain_eligible] = model_temps_C[wet_drain_eligible] + delta_C[wet_drain_eligible]
    new_temps_C = np.clip(new_temps_C, TEMP_MIN_C, TEMP_MAX_C)

    if not np.all(np.isfinite(new_temps_C)):
        raise RuntimeError("Non-finite temperatures generated while adjusting .sno")

    # Drain liquid water from any layer that has been nudged below freezing.
    # Physically: the observed sub-zero temperature means the layer's heat content
    # is insufficient to sustain liquid water; SNOWPACK will re-equilibrate from
    # the cold, dry state on the next timestep.
    # Volume balance: water volume moves to air (Vol_Frac_V); enforce_fraction_closure
    # will recompute Vv as the residual.
    newly_frozen = new_temps_C < 0.0
    if newly_frozen.any():
        for c in water_cols:
            arr = pd.to_numeric(df[c], errors="coerce").fillna(0.0).to_numpy()
            arr[newly_frozen] = 0.0
            df[c] = arr

    # Store as Celsius so enforce_enthalpy_safe_restart_state can compare against 0 °C
    df[temp_col] = new_temps_C

    # Make the restart enthalpy-safe without renormalizing phase fractions.
    # After the drain above, wet layers the observations say are cold are already
    # dry; enforce_enthalpy_safe_restart_state only needs to handle remaining
    # genuinely wet (T ≈ 0 °C) layers.
    df = enforce_enthalpy_safe_restart_state(df, temp_col=temp_col)
    df = enforce_fraction_closure(df)

    # Convert back to the file's native units (Kelvin for SNOWPACK restarts)
    df[temp_col] = df[temp_col].astype(float) + kelvin_shift

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        ts = ts.where(ts <= new_profile_date, new_profile_date)
        df["timestamp"] = ts.dt.strftime("%Y-%m-%dT%H:%M:%S")

    frac_i_field = next((f for f in fields if f == "Vol_Frac_I"), None)
    frac_w_field = next((f for f in fields if f == "Vol_Frac_W"), None)
    frac_v_field = next((f for f in fields if f == "Vol_Frac_V"), None)
    frac_s_field = next((f for f in fields if f == "Vol_Frac_S"), None)

    new_rows = []
    for _, row in df.iterrows():
        if frac_i_field and frac_v_field:
            vi = round(float(row[frac_i_field]), 10)
            vw = round(float(row[frac_w_field]), 10) if frac_w_field else 0.0
            vs = round(float(row[frac_s_field]), 10) if frac_s_field else 0.0
            vv = round(1.0 - vi - vw - vs, 10)
        vals = []
        for field in fields:
            if field.lower() == "timestamp":
                vals.append(str(row[field]))
            elif field == frac_i_field:
                vals.append(f"{vi:.10f}")
            elif field == frac_w_field:
                vals.append(f"{vw:.10f}")
            elif field == frac_v_field:
                vals.append(f"{vv:.10f}")
            elif field == frac_s_field:
                vals.append(f"{vs:.10f}")
            else:
                vals.append(f"{float(row[field]):.10f}")
        new_rows.append(" ".join(vals) + "\n")

    if return_adjustments:
        # Return (layer_idx_from_bottom, T_K) pairs; caller sends these via SETTEMPS.
        # When wet-layer draining occurred (needs_reload=True) we also write the SNO
        # so the caller can fall back to RELOAD_SNO with correct volume fractions.
        final_temps_K = df[temp_col].to_numpy(dtype=float)
        adjustments = [(idx, float(T)) for idx, T in enumerate(final_temps_K)]
        needs_reload = bool(newly_frozen.any())
        if needs_reload:
            lines[start:end] = new_rows
            lines = rewrite_sno_profiledate_and_clip_timestamps(lines, new_profile_date)
            sno_out.write_text("".join(lines), encoding="utf-8")
        return (adjustments, needs_reload)

    lines[start:end] = new_rows
    lines = rewrite_sno_profiledate_and_clip_timestamps(lines, new_profile_date)
    sno_out.write_text("".join(lines), encoding="utf-8")


# =============================================================================
# RUN SNOWPACK
# =============================================================================

def get_latest_sno_file(directory: Path, fallback_root: Path | None = None) -> Path:
    candidates = []
    for pattern in ("*.sno", "*.SNO"):
        candidates.extend(directory.glob(pattern))

    if not candidates and fallback_root is not None:
        for pattern in ("**/*.sno", "**/*.SNO"):
            candidates.extend(fallback_root.glob(pattern))

    candidates = [
        p for p in candidates
        if p.is_file()
        and not p.name.startswith("_")
        and not p.name.startswith("raw_")
        and not p.name.startswith("adj_")
    ]
    candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)

    if not candidates:
        print(f"No .sno files found in {directory}")
        if fallback_root is not None:
            print(f"Also searched recursively under {fallback_root}")
        raise FileNotFoundError(f"No .sno files found in {directory}")

    return candidates[-1]


def run_snowpack_one_step(
    snowpack_exe: str,
    ini_file: Path,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    work_dir: Path,
    timeout_s: int = 900,
) -> tuple[bool, str]:
    import time

    ini_rel = ini_file.relative_to(work_dir)

    cmd = [
        snowpack_exe,
        "-c", str(ini_rel),
        "-b", start_time.strftime("%Y-%m-%dT%H:%M"),
        "-e", end_time.strftime("%Y-%m-%dT%H:%M"),
    ]

    print("Running:", " ".join(cmd))
    print("Working directory:", work_dir)

    proc = subprocess.Popen(
        cmd,
        cwd=work_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines = []
    t0 = time.time()

    try:
        while True:
            line = proc.stdout.readline()
            if line:
                print(line, end="")
                lines.append(line)

            ret = proc.poll()
            if ret is not None:
                for line in proc.stdout:
                    print(line, end="")
                    lines.append(line)
                ok = (ret == 0)
                return ok, "".join(lines)

            if time.time() - t0 > timeout_s:
                proc.kill()
                return False, f"SNOWPACK timed out after {timeout_s} s\n" + "".join(lines)

    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass


class SnowpackDaemon:
    """Persistent SNOWPACK process communicating via stdin/stdout pipes.

    The daemon binary accepts --daemon mode: it loads the SNO + SMET once,
    runs a chunk, writes the SNO checkpoint, signals CHECKPOINT <date> on
    stdout, then waits for RELOAD_SNO / RUN / QUIT commands on stdin.
    """

    def __init__(
        self,
        snowpack_exe: str,
        ini_file: Path,
        work_dir: Path,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> None:
        import threading
        ini_rel = ini_file.relative_to(work_dir)
        cmd = [
            snowpack_exe, "--daemon",
            "-c", str(ini_rel),
            "-b", start_time.strftime("%Y-%m-%dT%H:%M"),
            "-e", end_time.strftime("%Y-%m-%dT%H:%M"),
        ]
        print("Spawning SNOWPACK daemon:", " ".join(cmd))
        self.proc = subprocess.Popen(
            cmd, cwd=str(work_dir),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        self.log_lines: list[str] = []
        t = threading.Thread(target=self._drain_stderr, daemon=True)
        t.start()

    def _drain_stderr(self) -> None:
        for line in self.proc.stderr:
            print(line, end="", flush=True)
            self.log_lines.append(line)

    def wait_for_checkpoint(self, timeout_s: int = 900) -> str:
        """Read stdout until CHECKPOINT; print other lines. Returns ISO date string."""
        import time
        t0 = time.time()
        while True:
            if time.time() - t0 > timeout_s:
                self.proc.kill()
                raise RuntimeError(f"SNOWPACK daemon timed out after {timeout_s} s")
            line = self.proc.stdout.readline()
            if not line:
                rc = self.proc.poll()
                raise RuntimeError(
                    f"SNOWPACK daemon exited unexpectedly (rc={rc})\n"
                    + "".join(self.log_lines[-50:])
                )
            stripped = line.rstrip("\n")
            if stripped.startswith("CHECKPOINT "):
                ts = stripped.split(None, 1)[1]
                print(f"[daemon] CHECKPOINT {ts}")
                return ts
            print(stripped)

    def reload_and_run(self, new_end: pd.Timestamp, timeout_s: int = 900) -> tuple[str, str]:
        """Send RELOAD_SNO + RUN, wait for CHECKPOINT.

        Returns (checkpoint_iso_str, log_text_for_this_chunk).
        """
        self.log_lines.clear()
        self.proc.stdin.write("RELOAD_SNO\n")
        self.proc.stdin.write(f"RUN {new_end.strftime('%Y-%m-%dT%H:%M')}\n")
        self.proc.stdin.flush()
        cp = self.wait_for_checkpoint(timeout_s=timeout_s)
        return cp, "".join(self.log_lines)

    def settemps_and_run(
        self,
        adjustments: list[tuple[int, float]],
        new_end: pd.Timestamp,
        timeout_s: int = 900,
    ) -> tuple[str, str]:
        """Send SETTEMPS (layer temps in Kelvin) + RUN, wait for CHECKPOINT.

        adjustments: list of (layer_index_from_bottom, T_Kelvin) pairs.
        Returns (checkpoint_iso_str, log_text_for_this_chunk).
        """
        self.log_lines.clear()
        payload = ",".join(f"{idx}:{T_K:.6f}" for idx, T_K in adjustments)
        self.proc.stdin.write(f"SETTEMPS {payload}\n")
        self.proc.stdin.flush()
        # Wait for READY acknowledgement before sending RUN
        while True:
            line = self.proc.stdout.readline()
            if not line:
                rc = self.proc.poll()
                raise RuntimeError(
                    f"SNOWPACK daemon exited during SETTEMPS (rc={rc})\n"
                    + "".join(self.log_lines[-50:])
                )
            if line.rstrip("\n") == "READY":
                break
            print(line.rstrip("\n"))
        self.proc.stdin.write(f"RUN {new_end.strftime('%Y-%m-%dT%H:%M')}\n")
        self.proc.stdin.flush()
        cp = self.wait_for_checkpoint(timeout_s=timeout_s)
        return cp, "".join(self.log_lines)

    def respawn(
        self,
        snowpack_exe: str,
        ini_file: Path,
        work_dir: Path,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
    ) -> None:
        """Quit current daemon and start a fresh one (e.g. after water-transport switch)."""
        self.quit()
        self.__init__(snowpack_exe, ini_file, work_dir, start_time, end_time)

    def quit(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            self.proc.stdin.write("QUIT\n")
            self.proc.stdin.flush()
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:
            self.proc.kill()
            self.proc.wait()


def rotate_pro_chunk(pro_path: Path, chunk_dir: Path, chunk_idx: int) -> None:
    """Move the current .pro to chunks/chunk_NNNN.pro for later concatenation."""
    chunk_dir.mkdir(exist_ok=True)
    chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.pro"
    shutil.move(str(pro_path), str(chunk_path))
    print(f"Pro chunk {chunk_idx}: rotated to {chunk_path.name} "
          f"({chunk_path.stat().st_size / 1e6:.1f} MB)")


def _pro_is_data_line(line: str) -> bool:
    """True if a 0500, line is an actual timestep record (DD.MM.YYYY format)."""
    return bool(re.match(r"^0500,\d{2}\.\d{2}\.\d{4}", line))


def concatenate_pro_chunks(pro_path: Path) -> None:
    """Merge all chunk files + current .pro into the final .pro, then delete chunks.

    Takes the header (everything before the first timestep) from chunk 0 and
    appends only the data records from each subsequent source in order.
    """
    chunk_dir = pro_path.parent / "chunks"
    if not chunk_dir.exists():
        return
    chunks = sorted(chunk_dir.glob("chunk_*.pro"))
    if not chunks:
        return

    print(f"Concatenating {len(chunks)} .pro chunk(s) → {pro_path.name} …")
    tmp_path = pro_path.with_name(pro_path.name + ".concat_tmp")

    header_lines: list[str] = []
    with open(chunks[0], encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if _pro_is_data_line(line):
                break
            header_lines.append(line)

    with open(tmp_path, "w", encoding="utf-8") as out:
        out.writelines(header_lines)
        for src in list(chunks) + ([pro_path] if pro_path.exists() else []):
            with open(src, encoding="utf-8", errors="replace") as fh:
                in_data = False
                for line in fh:
                    if not in_data and _pro_is_data_line(line):
                        in_data = True
                    if in_data:
                        out.write(line)

    tmp_path.replace(pro_path)
    for chunk in chunks:
        try:
            chunk.unlink()
        except OSError:
            pass
    try:
        chunk_dir.rmdir()
    except OSError:
        pass
    print(f"Concatenation complete: {pro_path.stat().st_size / 1e6:.1f} MB")


def cycle_hourly_snowpack_with_moving_profile(
    temp_hourly: pd.DataFrame,
    long_df: pd.DataFrame,
    ini_file: Path,
    input_sno_file: Path,
    alpha: float,
    stabilization_hours: int = 15,
    water_transport: str = "adaptive",
    assimilation_interval_h: int = 1,
    use_daemon: bool = False,
    pro_chunk_hours: int = 0,
) -> None:
    """Run SNOWPACK hour-by-hour with temperature assimilation.

    water_transport:
      'adaptive'        – BUCKET for stabilization_hours then RICHARDSEQUATION
                          with automatic BUCKET fallback on convergence failure.
      'BUCKET'          – BUCKET throughout; no RE switch, no fallback.
      'RICHARDSEQUATION'– RE throughout from step 0; no stabilisation phase.

    Uses BUCKET water transport for the first ``stabilization_hours`` hours so
    the highly-icy initial profile can equilibrate, then rewrites the INI to
    RICHARDSEQUATION + FREEDRAINAGE for the remainder of the run.
    """
    times = temp_hourly.index
    if len(times) < 2:
        raise ValueError("Need at least two hourly timestamps.")

    # ── Resume detection ──────────────────────────────────────────────── #
    # If the .sno file's ProfileDate is already ahead of times[0], we are
    # resuming a previously interrupted run.  Skip the initial adjustment
    # and start the loop from the matching index.
    i_start = 0
    try:
        sno_text = input_sno_file.read_text(errors="replace")
        for line in sno_text.splitlines():
            if line.strip().startswith("ProfileDate"):
                sno_date = pd.Timestamp(line.split("=", 1)[1].strip())
                matches = np.where(times == sno_date)[0]
                if len(matches) and matches[0] > 0:
                    i_start = int(matches[0])
                    print(f"Resuming from {sno_date} (step {i_start})")
                break
    except Exception:
        pass

    # Determine initial INI scheme based on water_transport mode
    if water_transport == "BUCKET":
        ini_scheme = "BUCKET"
        switched = True     # never switch away from BUCKET
        using_re = False
    elif water_transport == "RICHARDSEQUATION":
        ini_scheme = "RICHARDSEQUATION"
        switched = True     # already in RE, no stabilisation phase
        using_re = True
    else:  # adaptive
        ini_scheme = "BUCKET"
        switched = i_start >= stabilization_hours
        using_re = switched

    bucket_fallback_remaining = 0

    # Per-step tracking: .pro chunking, ETA status
    _chunk_idx: int = 0
    _steps_since_chunk: int = 0
    _run_start_model: "pd.Timestamp | None" = None
    _run_start_wall:  "datetime | None" = None

    # Water-transport event log: records only scheme transitions.
    # Expanded to a dense per-step CSV at run end / crash (in the finally block).
    _wt_initial_scheme = "RICHARDSEQUATION" if using_re else "BUCKET"
    _wt_events: list[tuple[pd.Timestamp, str]] = [(times[i_start], _wt_initial_scheme)]
    _wt_last_scheme: str = _wt_initial_scheme
    _wt_completed: list[pd.Timestamp] = []
    _wt_events_path = OUTPUT_DIR / "water_transport_events.csv"

    if i_start == 0:
        apply_initial_temperature_adjustment(
            input_sno_file=input_sno_file,
            long_df=long_df,
            t0=times[0],
            alpha=alpha,
        )
        write_ini_file(
            out_path=ini_file,
            smet_name=FORCING_SMET_FILE.name,
            sno_name=INITIAL_SNO_FILE.name,
            water_transport=ini_scheme,
        )
    else:
        # Resuming — set INI to match current expected scheme
        resume_scheme = ("RICHARDSEQUATION" if using_re else "BUCKET")
        write_ini_file(
            out_path=ini_file,
            smet_name=FORCING_SMET_FILE.name,
            sno_name=INITIAL_SNO_FILE.name,
            water_transport=resume_scheme,
        )
        print(f"Resume: INI set to {resume_scheme}")

    # ── Spawn persistent daemon (if requested and steps remain) ───────── #
    daemon: SnowpackDaemon | None = None
    if use_daemon and i_start < len(times) - 1:
        i_first_end = min(i_start + assimilation_interval_h, len(times) - 1)
        daemon = SnowpackDaemon(
            snowpack_exe=SNOWPACK_EXE,
            ini_file=ini_file,
            work_dir=PROJECT_DIR,
            start_time=times[i_start],
            end_time=times[i_first_end],
        )
        # First chunk runs automatically; wait for its CHECKPOINT
        daemon.wait_for_checkpoint(timeout_s=900)

    # SETTEMPS state: pending adjustments for next RUN; steps since last SNO write.
    _pending_adjustments: "list[tuple[int,float]] | None" = None
    _settemps_steps_since_sno_write: int = 0

    try:
     for i in range(i_start, len(times) - 1, assimilation_interval_h):
        t0 = times[i]
        i_end = min(i + assimilation_interval_h, len(times) - 1)
        t1 = times[i_end]
        n_hours = i_end - i

        # ── Stabilization → RE switch (adaptive mode only) ────────────── #
        if water_transport == "adaptive" and not switched and i >= stabilization_hours:
            print(f"Stabilization complete at {t0} — switching to RICHARDSEQUATION")
            write_ini_file(
                out_path=ini_file,
                smet_name=FORCING_SMET_FILE.name,
                sno_name=INITIAL_SNO_FILE.name,
                water_transport="RICHARDSEQUATION",
            )
            switched = True
            using_re = True
            if daemon:
                # Daemon has old INI — respawn with new RICHARDSEQUATION INI
                print(f"[daemon] respawning with RICHARDSEQUATION")
                daemon.respawn(SNOWPACK_EXE, ini_file, PROJECT_DIR, t0, t1)
                daemon.wait_for_checkpoint(timeout_s=900)
                ok, msg = True, ""
                _pending_adjustments = None
                _settemps_steps_since_sno_write = 0
            else:
                ok, msg = run_snowpack_one_step(
                    snowpack_exe=SNOWPACK_EXE, ini_file=ini_file,
                    start_time=t0, end_time=t1, work_dir=PROJECT_DIR,
                )

        # ── BUCKET fallback countdown → restore RE when done ──────────── #
        elif bucket_fallback_remaining > 0:
            bucket_fallback_remaining = max(0, bucket_fallback_remaining - n_hours)
            if bucket_fallback_remaining == 0:
                print(f"{t0}: RE fallback period over — switching back to RICHARDSEQUATION")
                write_ini_file(
                    out_path=ini_file,
                    smet_name=FORCING_SMET_FILE.name,
                    sno_name=INITIAL_SNO_FILE.name,
                    water_transport="RICHARDSEQUATION",
                )
                using_re = True
                if daemon:
                    daemon.respawn(SNOWPACK_EXE, ini_file, PROJECT_DIR, t0, t1)
                    daemon.wait_for_checkpoint(timeout_s=900)
                    ok, msg = True, ""
                    _pending_adjustments = None
                    _settemps_steps_since_sno_write = 0
                else:
                    ok, msg = run_snowpack_one_step(
                        snowpack_exe=SNOWPACK_EXE, ini_file=ini_file,
                        start_time=t0, end_time=t1, work_dir=PROJECT_DIR,
                    )
            else:
                if daemon and i > i_start:
                    if _pending_adjustments is not None:
                        cp, msg = daemon.settemps_and_run(_pending_adjustments, t1, timeout_s=900)
                    else:
                        cp, msg = daemon.reload_and_run(t1, timeout_s=900)
                    ok = True
                elif daemon and i == i_start:
                    ok, msg = True, "".join(daemon.log_lines)
                else:
                    ok, msg = run_snowpack_one_step(
                        snowpack_exe=SNOWPACK_EXE, ini_file=ini_file,
                        start_time=t0, end_time=t1, work_dir=PROJECT_DIR,
                    )

        # ── .pro chunk rotation (daemon mode) ────────────────────────── #
        # Rotate before the step so the new daemon writes to a fresh .pro.
        elif (pro_chunk_hours > 0
              and _steps_since_chunk >= pro_chunk_hours
              and daemon is not None):
            _pro_now = next(iter(OUTPUT_DIR.glob("*_TEMP_ASSIM_RUN.pro")), None)
            if _pro_now is not None and _pro_now.exists():
                rotate_pro_chunk(_pro_now, OUTPUT_DIR / "chunks", _chunk_idx)
                _chunk_idx += 1
            _steps_since_chunk = 0
            daemon.respawn(SNOWPACK_EXE, ini_file, PROJECT_DIR, t0, t1)
            daemon.wait_for_checkpoint(timeout_s=900)
            ok, msg = True, ""
            _pending_adjustments = None
            _settemps_steps_since_sno_write = 0

        # ── Normal step ───────────────────────────────────────────────── #
        else:
            # Non-daemon chunk rotation: move .pro before this step so SNOWPACK
            # creates a fresh one (daemon-less mode only; daemon mode uses respawn above).
            if (pro_chunk_hours > 0
                    and _steps_since_chunk >= pro_chunk_hours
                    and daemon is None):
                _pro_now = next(iter(OUTPUT_DIR.glob("*_TEMP_ASSIM_RUN.pro")), None)
                if _pro_now is not None and _pro_now.exists():
                    rotate_pro_chunk(_pro_now, OUTPUT_DIR / "chunks", _chunk_idx)
                    _chunk_idx += 1
                _steps_since_chunk = 0

            if daemon and i > i_start:
                if _pending_adjustments is not None:
                    cp, msg = daemon.settemps_and_run(_pending_adjustments, t1, timeout_s=900)
                else:
                    cp, msg = daemon.reload_and_run(t1, timeout_s=900)
                ok = True
            elif daemon and i == i_start:
                # First chunk already ran at daemon spawn
                ok, msg = True, "".join(daemon.log_lines)
            else:
                ok, msg = run_snowpack_one_step(
                    snowpack_exe=SNOWPACK_EXE,
                    ini_file=ini_file,
                    start_time=t0,
                    end_time=t1,
                    work_dir=PROJECT_DIR,
                )

        # ── RE convergence failure → fall back to BUCKET (adaptive only) ─ #
        # Must be checked BEFORE the ok-guard so a RE timeout triggers a
        # BUCKET retry of the same step rather than an immediate crash.
        # Capture the scheme that ran THIS step before using_re can change.
        _step_scheme = "RICHARDSEQUATION" if using_re else "BUCKET"
        if water_transport == "adaptive" and using_re and "Richards-Equation solver: no convergence" in msg:
            write_ini_file(
                out_path=ini_file,
                smet_name=FORCING_SMET_FILE.name,
                sno_name=INITIAL_SNO_FILE.name,
                water_transport="BUCKET",
            )
            using_re = False
            bucket_fallback_remaining = 24

            if not ok:
                # RE timed out — retry this step with BUCKET
                _step_scheme = "BUCKET"
                print(f"{t0}→{t1}: RE timed out with convergence failures; "
                      f"retrying with BUCKET (fallback for 24 h)")
                if daemon:
                    daemon.respawn(SNOWPACK_EXE, ini_file, PROJECT_DIR, t0, t1)
                    daemon.wait_for_checkpoint(timeout_s=900)
                    ok, msg = True, ""
                    _pending_adjustments = None
                    _settemps_steps_since_sno_write = 0
                else:
                    ok, msg = run_snowpack_one_step(
                        snowpack_exe=SNOWPACK_EXE,
                        ini_file=ini_file,
                        start_time=t0,
                        end_time=t1,
                        work_dir=PROJECT_DIR,
                    )
            else:
                print(f"{t1}: RE SafeMode convergence failure; "
                      f"switching to BUCKET for next 24 model hours")

        if not ok:
            raise RuntimeError("SNOWPACK failed\n" + msg)

        # ── Per-step diagnostics: ETA status + water-transport events ─── #
        _step_wall = datetime.now(timezone.utc)
        if _run_start_wall is None:
            _run_start_wall  = _step_wall
            _run_start_model = t1
        _run_status = {
            "run_start_model": _run_start_model.isoformat(),
            "run_start_wall":  _run_start_wall.isoformat(),
            "step_model":      t1.isoformat(),
            "step_wall":       _step_wall.isoformat(),
        }
        (OUTPUT_DIR / "run_status.json").write_text(
            json.dumps(_run_status), encoding="utf-8"
        )
        # Record completed step; write events file only on scheme transition
        _wt_completed.append(t1)
        if _step_scheme != _wt_last_scheme:
            _wt_events.append((t1, _step_scheme))
            _wt_last_scheme = _step_scheme
            with open(_wt_events_path, "w", encoding="utf-8") as _wf:
                _wf.write("datetime,scheme\n")
                for _ev_t, _ev_s in _wt_events:
                    _wf.write(f"{_ev_t.isoformat()},{_ev_s}\n")
        _steps_since_chunk += 1

        latest_sno = get_latest_sno_file(CURRENT_SNOW_DIR, fallback_root=PROJECT_DIR)

        try:
            validate_sno_geometry(
                latest_sno,
                max_layer_thickness_m=100.0,
                max_total_thickness_m=100.0,
            )
        except Exception:
            backup = CURRENT_SNOW_DIR / "_latest_unadjusted_backup.sno"
            if backup.exists():
                shutil.copy2(backup, input_sno_file)
            raise

        obs_profile = get_profile_observations_for_time(long_df, t1)

        # Use SETTEMPS fast path when daemon is active and not due for a periodic
        # SNO write (SETTEMPS_SNO_WRITE_INTERVAL).  Falls back to full SNO write
        # whenever wet-layer draining occurs or we're in KEEP_HOURLY_ARCHIVES mode.
        _use_settemps = (
            daemon is not None
            and not KEEP_HOURLY_ARCHIVES
            and _settemps_steps_since_sno_write < SETTEMPS_SNO_WRITE_INTERVAL
        )

        if KEEP_HOURLY_ARCHIVES:
            raw_archive = CURRENT_SNOW_DIR / f"raw_{t1:%Y%m%dT%H%M%S}.sno"
            adjusted_archive = CURRENT_SNOW_DIR / f"adj_{t1:%Y%m%dT%H%M%S}.sno"
            shutil.copy2(latest_sno, raw_archive)

            update_sno_temperatures_from_moving_profile(
                sno_in=raw_archive,
                sno_out=adjusted_archive,
                obs_profile=obs_profile,
                alpha=alpha,
                new_profile_date=t1,
                n_hours=n_hours,
            )

            shutil.copy2(adjusted_archive, input_sno_file)
            _pending_adjustments = None
            _settemps_steps_since_sno_write = 0
            print(f"{t1}: wrote raw={raw_archive.name}, adjusted={adjusted_archive.name}")
        elif _use_settemps:
            adj_result = update_sno_temperatures_from_moving_profile(
                sno_in=latest_sno,
                sno_out=input_sno_file,
                obs_profile=obs_profile,
                alpha=alpha,
                new_profile_date=t1,
                n_hours=n_hours,
                return_adjustments=True,
            )
            adjustments, needs_reload = adj_result
            if needs_reload:
                # Wet drain or no-obs: SNO was written; daemon must RELOAD_SNO next step
                _pending_adjustments = None
                _settemps_steps_since_sno_write = 0
                print(f"{t1}: adjustments (wet-drain/no-obs) — SNO written, will RELOAD_SNO")
            else:
                _pending_adjustments = adjustments
                _settemps_steps_since_sno_write += 1
                print(f"{t1}: {len(adjustments)} layer temps → SETTEMPS")
        else:
            tmp_adjusted = CURRENT_SNOW_DIR / "_adjusted_tmp.sno"

            update_sno_temperatures_from_moving_profile(
                sno_in=latest_sno,
                sno_out=tmp_adjusted,
                obs_profile=obs_profile,
                alpha=alpha,
                new_profile_date=t1,
                n_hours=n_hours,
            )

            shutil.copy2(tmp_adjusted, input_sno_file)

            try:
                tmp_adjusted.unlink()
            except FileNotFoundError:
                pass

            prune_sno_files(CURRENT_SNOW_DIR, keep_last_n=KEEP_LAST_N_SNO)
            prune_haz_files(OUTPUT_DIR, keep_last_n=0)
            prune_haz_files(CURRENT_SNOW_DIR, keep_last_n=0)
            _pending_adjustments = None
            _settemps_steps_since_sno_write = 0
            print(f"{t1}: updated {input_sno_file.name}")

        latest_sno_backup = CURRENT_SNOW_DIR / "_latest_unadjusted_backup.sno"

        try:
            same_file = latest_sno.exists() and latest_sno_backup.exists() and latest_sno.samefile(latest_sno_backup)
        except FileNotFoundError:
            same_file = False

        if not same_file:
            shutil.copy2(latest_sno, latest_sno_backup)

        # Sync checkpoint SNO back to disk so a crash doesn't lose progress.
        # Skip when SETTEMPS was used (SNO not written this step).
        if (USE_RAMDISK and input_sno_file != DISK_INPUT_DIR / input_sno_file.name
                and _pending_adjustments is None):
            shutil.copy2(input_sno_file, DISK_INPUT_DIR / input_sno_file.name)

    finally:
        if daemon is not None:
            daemon.quit()

        # Merge any .pro chunks into the final .pro (runs on both clean finish and crash)
        if pro_chunk_hours > 0:
            _pro_final = next(iter(OUTPUT_DIR.glob("*_TEMP_ASSIM_RUN.pro")), None)
            if _pro_final is not None:
                try:
                    concatenate_pro_chunks(_pro_final)
                except Exception as _e:
                    print(f"Warning: .pro chunk concatenation failed: {_e}")

        # Expand sparse water-transport events to a dense per-step CSV
        if _wt_completed:
            try:
                _dense_rows = ["datetime,scheme\n"]
                _ev_idx = 0
                _cur_scheme = _wt_events[0][1]
                for _step_t in _wt_completed:
                    while (_ev_idx + 1 < len(_wt_events)
                           and _wt_events[_ev_idx + 1][0] <= _step_t):
                        _ev_idx += 1
                        _cur_scheme = _wt_events[_ev_idx][1]
                    _dense_rows.append(f"{_step_t.isoformat()},{_cur_scheme}\n")
                (OUTPUT_DIR / "water_transport_log.csv").write_text(
                    "".join(_dense_rows), encoding="utf-8"
                )
                print(f"Water-transport log: {len(_wt_completed)} steps, "
                      f"{len(_wt_events)} transition(s)")
            except Exception as _e:
                print(f"Warning: water-transport log expansion failed: {_e}")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    cfg  = load_settings(args)
    configure(args, cfg)

    print("TEMP_FILE   :", TEMP_FILE, TEMP_FILE.exists())
    print("PROMICE_FILE:", PROMICE_FILE, PROMICE_FILE.exists())
    print("DENSITY_FILE:", DENSITY_FILE, DENSITY_FILE.exists())
    print("SNO_FILE    :", INITIAL_SNO_FILE, INITIAL_SNO_FILE.exists())
    print("SMET_FILE   :", FORCING_SMET_FILE, FORCING_SMET_FILE.exists())

    for p in [TEMP_FILE, PROMICE_FILE]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    temp_df, meta = read_tempconcatenated(TEMP_FILE, default_elevation_m=DEFAULT_ELEVATION_M)
    promice_df = read_promice(PROMICE_FILE)

    # ── Fresh start: wipe checkpoints so resume detection finds nothing ─── #
    if args.fresh:
        _sid_str = f"{args.year}_{args.site}_{args.depth}m"
        if args.run_tag:
            _sid_str = f"{_sid_str}_{args.run_tag}"
        _ram_dir = Path(f"/dev/shm/snowpack_{_sid_str}")

        # ── Archive or delete previous output ──────────────────────────── #
        _files_to_clear = (
            list(OUTPUT_DIR.glob("*.pro"))
            + list(OUTPUT_DIR.glob("*.ini"))
            + ([PROJECT_DIR / "autorun.log"] if (PROJECT_DIR / "autorun.log").exists() else [])
        )
        _files_to_clear = [f for f in _files_to_clear if f.exists()]

        # Always delete (never archive) the per-run diagnostic files so they
        # don't mix data from different runs.
        for _diag in ["run_status.json", "water_transport_log.csv", "water_transport_events.csv"]:
            _dp = OUTPUT_DIR / _diag
            if _dp.exists():
                _dp.unlink()
        _chunks_dir = OUTPUT_DIR / "chunks"
        if _chunks_dir.exists():
            for _cf in _chunks_dir.glob("chunk_*.pro"):
                _cf.unlink()
            try:
                _chunks_dir.rmdir()
            except OSError:
                pass

        if _files_to_clear:
            if args.fresh_mode == "archive":
                import tarfile as _tarfile
                from datetime import datetime as _dt
                _archive_dir = OUTPUT_DIR / "archives"
                _archive_dir.mkdir(exist_ok=True)
                _stamp = _dt.now().strftime("%Y%m%d_%H%M%S")
                _archive_path = _archive_dir / f"run_{_stamp}.tar.gz"
                with _tarfile.open(_archive_path, "w:gz") as _tar:
                    for _f in _files_to_clear:
                        _tar.add(_f, arcname=_f.name)
                print(f"Fresh start: archived previous output → {_archive_path.name}")
            else:
                for _f in _files_to_clear:
                    _f.unlink()
                print("Fresh start: previous output deleted.")

        # ── Clear checkpoints ───────────────────────────────────────────── #
        for _sno_path in [
            INITIAL_SNO_FILE,
            DISK_INPUT_DIR / "initial_profile.sno",
            _ram_dir / "input" / "initial_profile.sno",
        ]:
            if _sno_path.exists():
                _sno_path.unlink()
        for _cs_dir in [CURRENT_SNOW_DIR, _ram_dir / "current_snow"]:
            if _cs_dir.exists():
                for _f in _cs_dir.glob("*.sno*"):
                    _f.unlink()
        print("Fresh start: checkpoints cleared.")

    # ── Detect resume checkpoint before doing any expensive setup ──────── #
    _resuming = False
    if INITIAL_SNO_FILE.exists() and FORCING_SMET_FILE.exists():
        try:
            _first_ts, _ = first_valid_temp_profile(temp_df)
            _profile_date = _first_ts.floor("min")
            for _line in INITIAL_SNO_FILE.read_text(errors="replace").splitlines():
                if _line.strip().startswith("ProfileDate"):
                    _checkpoint_date = pd.Timestamp(_line.split("=", 1)[1].strip())
                    if _checkpoint_date > _profile_date:
                        _resuming = True
                        print(f"Resuming from checkpoint at {_checkpoint_date} — "
                              f"skipping SNO/SMET rebuild")
                    break
        except Exception:
            pass

    if _resuming:
        # INI must always be (re)written so water_transport and paths are current
        write_ini_file(
            out_path=CFG_INI_FILE,
            smet_name=FORCING_SMET_FILE.name,
            sno_name=INITIAL_SNO_FILE.name,
        )
        print(f"Wrote INI: {CFG_INI_FILE}")
    else:
        density_df = read_density_profile(DENSITY_FILE)

        first_ts, first_row = first_valid_temp_profile(temp_df)
        surface_change_m = promice_surface_change_at(promice_df, first_ts)

        corrected_temp_profile = build_corrected_temp_profile(first_row, surface_change_m)
        layer_df = interpolate_temperature_to_density_layers(corrected_temp_profile, density_df)
        validate_layer_df_for_sno(layer_df)
        sno_df = build_sno_dataframe(layer_df, first_ts)

        hs_last = float(layer_df["bottom_m"].max())
        erosion_level = len(sno_df) + 6
        profile_date = first_ts.floor("min")

        totals = sno_df["Vol_Frac_I"] + sno_df["Vol_Frac_W"] + sno_df["Vol_Frac_V"]
        if not np.allclose(totals.to_numpy(), 1.0, atol=1e-8):
            raise RuntimeError("Volume fractions do not sum to 1.")

        parsed_ts = pd.to_datetime(sno_df["timestamp"])
        if not parsed_ts.is_monotonic_increasing:
            raise RuntimeError("Layer timestamps are not strictly increasing.")
        if (parsed_ts >= first_ts).any():
            raise RuntimeError("At least one layer timestamp is not before the first temperature timestamp.")

        write_sno_file(
            out_path=INITIAL_SNO_FILE,
            meta=meta,
            sno_df=sno_df,
            hs_last=hs_last,
            erosion_level=erosion_level,
            profile_date=profile_date,
        )
        print(f"Wrote initial SNO: {INITIAL_SNO_FILE}")

        build_smet_from_downloaded_era(
            temp_df=temp_df,
            promice_df=promice_df,
            density_df=density_df,
            meta=meta,
        )

        write_ini_file(
            out_path=CFG_INI_FILE,
            smet_name=FORCING_SMET_FILE.name,
            sno_name=INITIAL_SNO_FILE.name,
        )
        print(f"Wrote INI: {CFG_INI_FILE}")

    temp_df_hourly = keep_hourly(temp_df)
    hourly_index = make_hourly_index(temp_df_hourly.index.min(), temp_df_hourly.index.max())
    temp_hourly_original = temp_df_hourly.reindex(hourly_index)

    corrected_depth_wide, temp_hourly = build_corrected_depth_tables(
        temp_df,
        promice_df,
        max_interp_gap_hours=24,
    )

    long_df = build_long_observation_table(
        temp_hourly=temp_hourly,
        corrected_depth_wide=corrected_depth_wide,
        temp_hourly_original=temp_hourly_original,
    )

    # --run-until CLI arg overrides settings.toml; empty string means full record.
    run_until_str = args.run_until or cfg["run"].get("run_until", "").strip() or None
    RUN_UNTIL = run_until_str
    if RUN_UNTIL is not None:
        cutoff = pd.Timestamp(RUN_UNTIL)
        temp_hourly = temp_hourly[temp_hourly.index <= cutoff]
        long_df     = long_df[pd.to_datetime(long_df["timestamp"]) <= cutoff]

    cycle_hourly_snowpack_with_moving_profile(
        temp_hourly=temp_hourly,
        long_df=long_df,
        ini_file=CFG_INI_FILE,
        input_sno_file=INITIAL_SNO_FILE,
        alpha=ALPHA,
        stabilization_hours=STABILIZATION_HOURS,
        water_transport=args.water_transport if args.water_transport is not None else WATER_TRANSPORT,
        assimilation_interval_h=ASSIMILATION_INTERVAL_H,
        use_daemon=USE_DAEMON,
        pro_chunk_hours=PRO_CHUNK_HOURS,
    )

    # Generate static PNG figures via visualize_pro.py
    viz_script = Path(__file__).resolve().parent / "visualize_pro.py"
    if viz_script.exists():
        print(f"Generating figures with {viz_script.name} …")
        subprocess.run(
            [sys.executable, str(viz_script),
             "--site", args.site, "--year", str(args.year), "--depth", str(args.depth)],
            cwd=str(Path(__file__).resolve().parent),
            check=False,
        )

    # Generate obs vs model temperature figure
    ovm_script = Path(__file__).resolve().parent / "plot_obs_vs_model.py"
    if ovm_script.exists():
        _sid = f"{args.year}_{args.site}_{args.depth}m"
        print(f"Generating obs vs model figure for {_sid} …")
        subprocess.run(
            [sys.executable, str(ovm_script), _sid],
            cwd=str(Path(__file__).resolve().parent),
            check=False,
        )

    print("Done.")


if __name__ == "__main__":
    main()
