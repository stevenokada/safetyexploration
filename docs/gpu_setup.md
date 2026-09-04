# RunPod session: R-lens readout on three models

One pod, 3× A100 80GB, one model per GPU. Everything below has been written but
**not executed** — there is no GPU on the development machine — so the order
matters: each step validates an assumption the next step depends on.

## Pod

- **GPU**: 3 × A100 80GB (Community Cloud ≈ $1.19/hr each, so ≈ $3.57/hr for the pod)
- **Template**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` or similar
- **Container disk**: 50GB
- **Volume**: 400GB mounted at `/workspace` — three bf16 27B–35B checkpoints are
  ~170GB, plus ~7GB of lenses, plus headroom
- **Ports**: expose 8888 if you want Jupyter; not required

Why one pod with 3 GPUs rather than 3 pods: the models download once to a shared
volume instead of three times, and there is one machine to tear down.

Why bf16 rather than 4-bit: a 27B model is ~54GB and fits an 80GB card
unquantized. Published work using this method had to concede that 4-bit
quantization might distort residual-stream geometry in ways that specifically
affect lens readouts. Since this measurement is about *where in the layer stack*
intermediates appear, running bf16 removes that caveat outright.

## Setup

```bash
cd /workspace && git clone <your repo> repo && cd repo/pilot/lens
pip install transformers accelerate huggingface_hub pandas pyarrow
export HF_HOME=/workspace/hf
export HF_TOKEN=...          # required: google/gemma-3-27b-it is gated
./fetch_assets.sh            # ~170GB, network-bound, run once
```

## Step 1 — inspect the lens (2 minutes, do not skip)

```bash
python3 inspect_lens.py --lens /workspace/lenses/qwen3.6-27b/r-lens/lens.pt
```

Prints the shape and dtype of `J`, and how `source_layers` indexes the model.
`J` is square, so its orientation (`J @ h` vs `h @ J`) cannot be read off the
shape — step 2 settles it.

## Step 2 — selftest (5 minutes, the load-bearing check)

```bash
python3 run_lens.py --selftest \
  --model Qwen/Qwen3.6-27B \
  --lens /workspace/lenses/qwen3.6-27b/r-lens/lens.pt
```

At a late source layer the lens should roughly reproduce the model's own
next-token distribution, since little computation remains before the target
layer. The script reports top-10 overlap for both orientations at several depths.

- One orientation clearly higher → use it as `--orient`
- **Both near zero → stop.** Normalization or `source_layers` indexing is wrong.
  Collecting data at this point yields plausible-looking numbers that mean
  nothing, which is worse than an error.

## Step 3 — full run (about 1–2 hours)

```bash
N=200 BATCH=8 ORIENT=left ./launch_3gpu.sh
```

Three processes, one per GPU, logging to `/workspace/results/*.log`. Batch 8 is
a starting point; 80GB has room for 16–32 at these sequence lengths, and larger
batches are the main throughput lever — a faster GPU tier is not.

Cost control: `--tail 48` scores only the trailing positions, where the answer
forms. Scoring all ~400 positions is roughly 10× the work and mostly reads the
prompt back to you. Raise it only for the filler-span analysis, and pair that
with `--layer-stride 2`.

## Step 4 — read the result

Each log ends with the median rank of true intermediates versus the null control
(a two-digit value that is not any intermediate). That comparison is the whole
experiment in one line:

- intermediates rank **far better** than null → the model is representing
  un-emitted hops, and the position/layer pattern says where
- the two are **similar** → the lens is not surfacing this computation, and no
  amount of downstream plotting will change that

A positive result is a hypothesis, not a finding. Confirm it causally by
transplanting the KV cache at exactly the positions the lens implicates and
checking that the answer moves. That test does not depend on the lens being
right, which is why it is the one that settles the question.

## Teardown

```bash
# terminate the pod, then delete the volume
```

The 400GB volume bills at $0.07/GB/month ≈ **$28/month, whether or not a pod is
attached**. That is more than the compute for this experiment. Re-downloading
costs about ten minutes and pennies, so delete the volume unless another session
is imminent.
