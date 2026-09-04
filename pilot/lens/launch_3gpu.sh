#!/bin/bash
# Run all three models at once, one per GPU, on a single 3x A100 80GB pod.
# Each model is ~54-70GB in bf16 plus its lens, so one model per 80GB card with
# no sharding: three independent processes, three log files, ~3x less wall clock.
set -u
cd "$(dirname "$0")"

run() {  # gpu, hf_repo, lens_dir, tag, extra
  CUDA_VISIBLE_DEVICES=$1 python3 run_lens.py \
    --model "$2" --lens "$LENS_ROOT/$3/r-lens/lens.pt" \
    --out "$OUT/$4" --n "$N" --batch "$BATCH" --orient "$ORIENT" \
    ${5:-} > "$OUT/$4.log" 2>&1 &
  echo "  gpu$1 -> $4 (pid $!)"
}

LENS_ROOT=${LENS_ROOT:-/workspace/lenses}
OUT=${OUT:-/workspace/results}
N=${N:-200}
BATCH=${BATCH:-8}
ORIENT=${ORIENT:-left}
mkdir -p "$OUT"

echo "launching 3 models in parallel (n=$N, batch=$BATCH, orient=$ORIENT)"
run 0 Qwen/Qwen3.6-27B        qwen3.6-27b      qwen27b
run 1 Qwen/Qwen3.6-35B-A3B    qwen3.6-35b-a3b  qwen35b-moe
run 2 google/gemma-3-27b-it   gemma-3-27b-it   gemma27b
wait
echo "all done; results in $OUT"
grep -H "median rank" "$OUT"/*.log
