#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
MODELS="anthropic/claude-opus-4.5 google/gemini-2.5-flash deepseek/deepseek-chat"

echo "########## RUN A: fencepost / wording control (serial, immediate) ##########"
for m in $MODELS; do
  for w in default explicit; do
    echo "### $m | $w"
    python3 ../src/harness.py --task serial --wording $w --conditions immediate --n 40 \
      --ks 1,2,3,4,6,8 --model "$m" --out "A_${w}_$(basename $m)" 2>&1 | tail -9
  done
done

echo "########## RUN B: parallel control (serial vs parallel vs parallel_count) ##########"
for m in $MODELS; do
  for t in serial parallel parallel_count; do
    echo "### $m | $t"
    python3 ../src/harness.py --task $t --conditions immediate,filler --n 40 \
      --ks 2,3,4,6,8 --model "$m" --out "B_${t}_$(basename $m)" 2>&1 | tail -12
  done
done
echo "########## ALL DONE ##########"
