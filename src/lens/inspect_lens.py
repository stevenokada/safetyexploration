"""
STEP 1 on the pod. Run this before anything else.

The lens files ship as a pickled dict with keys like
['J', 'n_prompts', 'source_layers', 'd_model', 'provenance'], but the exact
orientation of J (whether the readout is J @ h or h @ J), its dtype, and how
source_layers indexes the model's layers cannot be confirmed without the file
in hand. Everything downstream depends on getting that right, and getting it
wrong produces plausible-looking garbage rather than an error.

This prints the structure so run_lens.py can be pointed at the right layout.

  python3 inspect_lens.py --lens /workspace/lenses/qwen3.6-27b/r-lens/lens.pt
"""
import argparse

import torch


def describe(obj, name="lens", depth=0, max_depth=3):
    pad = "  " * depth
    if isinstance(obj, dict):
        print(f"{pad}{name}: dict with {len(obj)} keys")
        for k, v in obj.items():
            describe(v, str(k), depth + 1, max_depth)
    elif isinstance(obj, torch.Tensor):
        print(f"{pad}{name}: Tensor shape={tuple(obj.shape)} dtype={obj.dtype} "
              f"device={obj.device} "
              f"min={obj.float().min():.4g} max={obj.float().max():.4g} "
              f"mean={obj.float().mean():.4g}")
    elif isinstance(obj, (list, tuple)):
        print(f"{pad}{name}: {type(obj).__name__} len={len(obj)}")
        if obj and depth < max_depth:
            describe(obj[0], f"{name}[0]", depth + 1, max_depth)
            if len(obj) > 1:
                describe(obj[-1], f"{name}[-1]", depth + 1, max_depth)
    else:
        s = repr(obj)
        print(f"{pad}{name}: {type(obj).__name__} = {s[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lens", required=True)
    args = ap.parse_args()

    lens = torch.load(args.lens, map_location="cpu", weights_only=False)
    describe(lens)

    print("\n--- what run_lens.py needs to know ---")
    J = lens["J"] if isinstance(lens, dict) and "J" in lens else None
    if J is None:
        print("no 'J' key; inspect the structure above and set --j-key accordingly")
        return
    if isinstance(J, torch.Tensor):
        print(f"J shape {tuple(J.shape)}")
        if J.dim() == 3:
            n, a, b = J.shape
            print(f"  reads as {n} source layers of a {a}x{b} map")
            if a == b:
                print("  square: orientation cannot be inferred from shape alone.")
                print("  Use run_lens.py --selftest, which checks the readout at a late")
                print("  layer against the model's real next-token distribution; the")
                print("  correct orientation agrees strongly, the wrong one does not.")
        elif J.dim() == 2:
            print("  single map, not per-layer: expect --layer to be ignored")
    sl = lens.get("source_layers") if isinstance(lens, dict) else None
    if sl is not None:
        print(f"source_layers = {sl}")
        print("  these index hidden_states[]; note hidden_states[0] is the embedding")
        print("  output, so hidden_states[i+1] is the output of layer i")
    if isinstance(lens, dict) and "d_model" in lens:
        print(f"d_model = {lens['d_model']}")
    if isinstance(lens, dict) and "provenance" in lens:
        print(f"provenance = {lens['provenance']}")


if __name__ == "__main__":
    main()
