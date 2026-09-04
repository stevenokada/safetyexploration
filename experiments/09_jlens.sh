#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
MODELS="qwen/qwen3.6-27b qwen/qwen3.6-35b-a3b google/gemma-3-27b-it"

echo "########## depth sweep (serial: immediate / filler / cot) ##########"
for m in $MODELS; do
  echo "### $m"
  python3 ../src/harness.py --task serial --conditions immediate,filler,cot --n 40 \
    --ks 1,2,3,4,6,8 --model "$m" --out "J_serial_$(basename $m)" 2>&1 | tail -24
done

echo "########## parallel control (matched work, depth 2) ##########"
for m in $MODELS; do
  for t in parallel parallel_count; do
    echo "### $m | $t"
    python3 ../src/harness.py --task $t --conditions immediate,filler --n 40 \
      --ks 2,3,4,6,8 --model "$m" --out "J_${t}_$(basename $m)" 2>&1 | tail -13
  done
done

echo "########## width sweep to k=24 (32-node table) ##########"
for m in $MODELS; do
  for t in parallel parallel_count serial; do
    echo "### $m | $t | nodes=32"
    python3 ../src/harness.py --task $t --conditions immediate --n 40 --nodes 32 \
      --ks 2,4,8,12,16,20,24 --model "$m" --out "JW_${t}_$(basename $m)" 2>&1 | tail -10
  done
done
echo "########## ALL DONE ##########"
