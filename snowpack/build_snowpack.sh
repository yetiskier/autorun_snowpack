#!/usr/bin/env bash
# Build a patched SNOWPACK 3.7.0 binary with daemon + SETTEMPS support.
#
# BEFORE RUNNING: download the SNOWPACK 3.7.0 source archive from
#   https://code.wsl.ch/snow-models/snowpack/-/releases
# and, if MeteoIO is not already installed, MeteoIO 2.10.0 from
#   https://code.wsl.ch/snow-models/meteoio/-/releases
#
# Usage:
#   bash build_snowpack.sh <path-to-snowpack-source-dir-or-tarball> \
#                          [<path-to-meteoio-source-dir-or-tarball>]
#
# The second argument is only needed if MeteoIO is not already installed
# (check with: pkg-config --exists meteoio || ls /usr/local/include/meteoio).
#
# The patched binary is written to:
#   <repo_root>/bin/snowpack
# Point settings.toml [paths] snowpack_exe at that path.
#
# Example:
#   bash build_snowpack.sh ~/Downloads/snowpack-3.7.0.tar.gz \
#                          ~/Downloads/meteoio-2.10.0.tar.gz

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$SCRIPT_DIR/_build"
PREFIX="$SCRIPT_DIR/_install"

SNOWPACK_ARG="${1:-}"
METEOIO_ARG="${2:-}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info() { echo "[build_snowpack] $*"; }
die()  { echo "[build_snowpack] ERROR: $*" >&2; exit 1; }

require() {
    command -v "$1" >/dev/null 2>&1 || die "'$1' not found — please install it."
}

# Unpack a tarball to $BUILD_DIR and print the resulting directory path.
unpack() {
    local archive="$1" label="$2"
    info "Extracting $label …"
    local before
    before=$(ls "$BUILD_DIR")
    tar -xzf "$archive" -C "$BUILD_DIR"
    local after
    after=$(ls "$BUILD_DIR")
    # Find the newly created directory
    local new_dir
    new_dir=$(comm -13 <(echo "$before" | sort) <(echo "$after" | sort) | head -1)
    [ -n "$new_dir" ] || die "Could not determine extracted directory for $label"
    echo "$BUILD_DIR/$new_dir"
}

require cmake
require make
require g++
mkdir -p "$BUILD_DIR"

# ---------------------------------------------------------------------------
# Step 1 — Resolve SNOWPACK source
# ---------------------------------------------------------------------------

[ -n "$SNOWPACK_ARG" ] || die "Usage: bash build_snowpack.sh <snowpack-src-or-tarball> [<meteoio-src-or-tarball>]
Download SNOWPACK 3.7.0 from: https://code.wsl.ch/snow-models/snowpack/-/releases"

if [ -f "$SNOWPACK_ARG" ]; then
    SNOWPACK_SRC=$(unpack "$SNOWPACK_ARG" "SNOWPACK")
elif [ -d "$SNOWPACK_ARG" ]; then
    SNOWPACK_SRC="$SNOWPACK_ARG"
else
    die "Not a file or directory: $SNOWPACK_ARG"
fi

info "SNOWPACK source: $SNOWPACK_SRC"

# ---------------------------------------------------------------------------
# Step 2 — MeteoIO
# ---------------------------------------------------------------------------

METEOIO_CMAKE_HINT=""

meteoio_installed() {
    pkg-config --exists meteoio 2>/dev/null ||
    [ -f /usr/local/include/meteoio/MeteoIO.h ] ||
    [ -f /usr/include/meteoio/MeteoIO.h ]
}

if meteoio_installed; then
    info "MeteoIO found system-wide — skipping MeteoIO build."
elif [ -n "$METEOIO_ARG" ]; then
    if [ -f "$METEOIO_ARG" ]; then
        METEOIO_SRC=$(unpack "$METEOIO_ARG" "MeteoIO")
    elif [ -d "$METEOIO_ARG" ]; then
        METEOIO_SRC="$METEOIO_ARG"
    else
        die "Not a file or directory: $METEOIO_ARG"
    fi

    info "MeteoIO source: $METEOIO_SRC"
    METEOIO_BUILD="$BUILD_DIR/meteoio-build"
    mkdir -p "$METEOIO_BUILD"

    info "Configuring MeteoIO …"
    cmake -S "$METEOIO_SRC" -B "$METEOIO_BUILD" \
        -DCMAKE_INSTALL_PREFIX="$PREFIX" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_TESTING=OFF \
        -DBUILD_DOCUMENTATION=OFF \
        2>&1 | tail -3

    info "Building MeteoIO …"
    make -C "$METEOIO_BUILD" -j"$(nproc)"

    info "Installing MeteoIO to $PREFIX …"
    make -C "$METEOIO_BUILD" install 2>&1 | tail -3

    METEOIO_CMAKE_HINT="-DCMAKE_PREFIX_PATH=$PREFIX"
else
    die "MeteoIO not found. Provide the MeteoIO source as a second argument.
Download MeteoIO 2.10.0 from: https://code.wsl.ch/snow-models/meteoio/-/releases"
fi

# ---------------------------------------------------------------------------
# Step 3 — Apply patched Main.cc
# ---------------------------------------------------------------------------

info "Applying patched Main.cc …"
cp "$SCRIPT_DIR/Main.cc" "$SNOWPACK_SRC/applications/snowpack/Main.cc"

# ---------------------------------------------------------------------------
# Step 4 — Build SNOWPACK
# ---------------------------------------------------------------------------

SNOWPACK_BUILD="$BUILD_DIR/snowpack-build"
mkdir -p "$SNOWPACK_BUILD"

info "Configuring SNOWPACK …"
cmake -S "$SNOWPACK_SRC" -B "$SNOWPACK_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DBUILD_DOCUMENTATION=OFF \
    $METEOIO_CMAKE_HINT \
    2>&1 | tail -3

info "Building SNOWPACK …"
make -C "$SNOWPACK_BUILD" -j"$(nproc)"

# ---------------------------------------------------------------------------
# Step 5 — Copy binary
# ---------------------------------------------------------------------------

BIN_DIR="$REPO_ROOT/bin"
mkdir -p "$BIN_DIR"
cp "$SNOWPACK_BUILD/bin/snowpack" "$BIN_DIR/snowpack"

info ""
info "Build complete. Binary at: $BIN_DIR/snowpack"
info "Set this in settings.toml:"
info "  [paths]"
info "  snowpack_exe = \"$BIN_DIR/snowpack\""
