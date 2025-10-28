#!/usr/bin/env bash
set -euo pipefail

# Build PTX for multiple SM architectures.
# Usage: ./build_ptx.sh

DIR=$(cd "$(dirname "$0")" && pwd)
SRC="$DIR/kernels.cu"
OUTDIR="$DIR"

SMS=(70 75 80 86)

for sm in "${SMS[@]}"; do
  echo "Building PTX for sm_${sm}..."
  nvcc -ptx -arch=sm_${sm} -O3 -std=c++11 "$SRC" -o "$OUTDIR/sm${sm}.ptx"
done

# Fallback generic PTX (compute_70)
echo "Building generic PTX (compute_70)..."
nvcc -ptx -arch=compute_70 -O3 -std=c++11 "$SRC" -o "$OUTDIR/kernels.ptx"

echo "Done. Place MVS_CUPY_PTX to one of:"
ls -1 "$OUTDIR"/*.ptx || true
