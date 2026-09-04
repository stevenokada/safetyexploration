#!/bin/bash
export OPENROUTER_API_KEY=$(cat ~/.openrouter_key)
for m in deepseek/deepseek-chat google/gemini-2.5-flash anthropic/claude-opus-4.5 moonshotai/kimi-k2-0905 openai/gpt-5.1; do
  echo "### $m"
  python3 pilot.py --n 30 --ks 1,2,3,4,6,8 --model "$m" --out "grid2_$(basename $m)" 2>&1 | tail -25
done
echo "### qwen/qwen3-max (low concurrency)"
python3 pilot.py --n 30 --ks 1,2,3,4,6,8 --model qwen/qwen3-max --concurrency 4 --out grid2_qwen3-max 2>&1 | tail -25
echo "### ALL DONE"
