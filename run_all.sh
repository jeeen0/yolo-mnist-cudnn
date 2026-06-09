#!/usr/bin/env bash
# Orin runtime: video -> pgm (03) then classify + measure (04).
# Edit the vars below or pass them in via the environment.
set -euo pipefail
cd "$(dirname "$0")"

VIDEOS="${VIDEOS:-clips/*.mp4}"          # rehearsal clips
WEIGHTS="${WEIGHTS:-best.engine}"        # TensorRT engine built on the Orin
BIN="${BIN:-./mnistCUDNN/mnistCUDNN}"    # classifier binary
SAMPLES="${SAMPLES:-mnistCUDNN/data}"    # 1/3/5 regression PGMs
OUT="${OUT:-runtime/pgm}"

echo "== 03: video -> pgm =="
python scripts/03_pipeline.py --videos "$VIDEOS" --weights "$WEIGHTS" --out "$OUT"

echo "== 04: classify + measure =="
python scripts/04_eval.py --bin "$BIN" --pgm-dir "$OUT" --samples-dir "$SAMPLES"

echo "done. see runtime/results/results.csv"
