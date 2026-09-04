#!/bin/bash
# Pre-download weights and lenses in parallel. Do this once; it is network-bound,
# not GPU-bound, so it is the part a faster card does not speed up.
set -eu
export HF_HOME=${HF_HOME:-/workspace/hf}
mkdir -p /workspace/lenses

# Gemma is gated: export HF_TOKEN before running, or swap in the unsloth mirror.
python3 - <<'PY' &
from huggingface_hub import snapshot_download
for r in ["Qwen/Qwen3.6-27B", "Qwen/Qwen3.6-35B-A3B", "google/gemma-3-27b-it"]:
    print("downloading", r, flush=True)
    snapshot_download(r, allow_patterns=["*.safetensors","*.json","*.txt","*.model"])
PY
python3 - <<'PY' &
from huggingface_hub import snapshot_download
snapshot_download("camilablank/workspace-lenses", local_dir="/workspace/lenses",
                  allow_patterns=["qwen3.6-27b/r-lens/*", "qwen3.6-35b-a3b/r-lens/*",
                                  "gemma-3-27b-it/r-lens/*"])
PY
wait
echo "assets ready"
du -sh /workspace/lenses "$HF_HOME" 2>/dev/null || true
