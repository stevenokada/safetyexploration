#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
M=google/gemma-3-27b-it
run(){ echo "### $*"; python3 pilot2.py --model "$M" --n 8 --out "S_$1_$2_k$3" \
   --task "$1" --conditions "$2" --ks "$3" "${@:4}" 2>&1 | grep -E "audit|FAIL"; }
run serial            immediate 2
run serial            immediate 8
run serial            filler    2
run serial            cot       4
run arith             immediate 2
run arith             immediate 4
run arith             cot       3
run arith_parallel_max immediate 4
run parallel          immediate 8
run parallel_count    immediate 8
echo DONE
