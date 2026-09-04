"""
STEP 2 on the pod. Batched R-lens readout over the arithmetic task.

Question this answers: at each (layer, token position) of a forward pass, is the
ground-truth intermediate for hop m present in the model's residual stream?

Method: run prompts in batches with output_hidden_states=True (which returns the
residual stream at every layer, so no manual hooks are needed), apply the R-lens
readout softmax(W_U . norm(J_l @ h_l)) at the layers and positions we care about,
and record the RANK and PROBABILITY of each hop's target token. Because the task
generator forces every intermediate to have a distinct tens digit, and a two-digit
number's first token IS its tens digit, one token identifies one hop unambiguously.

Controls written alongside every real measurement:
  null      a two-digit value that is NOT any intermediate, giving the
            false-positive floor for "this hop is present"
  baseline  the same readout with no filler / on a shuffled prompt, so a hit can
            be compared against what the position produces with nothing to compute

IMPORTANT: this has never been executed against real weights or a real lens file
-- there is no GPU on the machine where it was written. Run --selftest first; it
validates the lens orientation and normalization against the model's own output
distribution, and will catch the most likely way this is wrong.

  python3 run_lens.py --selftest --model Qwen/Qwen3.6-27B --lens .../lens.pt
  python3 run_lens.py --model Qwen/Qwen3.6-27B --lens .../lens.pt --n 200 --out qwen27b
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import task_arith  # noqa: E402


# ---------------------------------------------------------------- prompts

def build_prompt(tok, rng, k, condition, filler_n=100):
    """Same generator as the behavioral runs, so lens results line up with the
    accuracy curves rather than describing a slightly different task."""
    q, gold, inter, reasoning = task_arith.generate(rng, k)
    if condition == "filler":
        q += "\n\n" + " ".join(str(i) for i in range(1, filler_n + 1))
    msgs = [
        {"role": "system", "content": "Answer immediately. Respond with ONLY the "
                                      "final number in the format 'Answer: <number>'. "
                                      "Do not write anything else, do not reason."},
        {"role": "user", "content": q},
    ]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text += "Answer:"          # prefill, matching the behavioral forced-answer setup
    return text, gold, inter


def first_token_id(tok, value):
    """The token a number's leading digit maps to. Intermediates are two digits
    with distinct tens digits, so this id identifies which hop it is."""
    ids = tok.encode(str(value), add_special_tokens=False)
    return ids[0]


# ---------------------------------------------------------------- lens

class Lens:
    def __init__(self, path, orient="left", device="cuda", dtype=torch.bfloat16):
        blob = torch.load(path, map_location="cpu", weights_only=False)
        self.meta = {k: v for k, v in blob.items() if k != "J"} if isinstance(blob, dict) else {}
        J = blob["J"] if isinstance(blob, dict) else blob
        self.J = J.to(device=device, dtype=dtype)
        self.orient = orient
        self.source_layers = list(self.meta.get("source_layers", range(self.J.shape[0])))

    def map_for(self, layer):
        if self.J.dim() == 2:
            return self.J
        try:
            idx = self.source_layers.index(layer)
        except ValueError:
            return None
        return self.J[idx]

    def apply(self, h, layer):
        """h: [N, d_model] residual stream vectors -> [N, d_model] mapped to the
        target layer. Orientation is a free parameter until --selftest fixes it."""
        M = self.map_for(layer)
        if M is None:
            return None
        return h @ M.T if self.orient == "left" else h @ M


def readout_logits(model, lens, h, layer):
    """softmax(W_U . norm(J_l h)) -- returns raw logits; caller ranks or softmaxes."""
    z = lens.apply(h, layer)
    if z is None:
        return None
    norm = model.model.norm                      # final RMSNorm before unembedding
    return model.lm_head(norm(z))


# ---------------------------------------------------------------- selftest

@torch.no_grad()
def selftest(model, tok, lens, device):
    """The single most important check. At a LATE source layer the lens should
    approximately reproduce the model's own next-token distribution, because
    little computation remains between there and the target layer. If the
    orientation or normalization is wrong, agreement collapses. This distinguishes
    'correctly wired' from 'plausible garbage', which nothing downstream can."""
    text = "The capital of France is"
    enc = tok(text, return_tensors="pt").to(device)
    out = model(**enc, output_hidden_states=True)
    real = out.logits[0, -1].float()
    real_top = torch.topk(real, 10).indices.tolist()

    n_layers = len(out.hidden_states) - 1
    print(f"model has {n_layers} layers; lens covers source_layers="
          f"{lens.source_layers[:4]}...{lens.source_layers[-4:]}")
    print(f"model top-10 next tokens: {[tok.decode([i]) for i in real_top]}\n")

    for orient in ("left", "right"):
        lens.orient = orient
        print(f"orientation={orient}")
        for frac in (0.6, 0.8, 0.95):
            layer = int(n_layers * frac)
            h = out.hidden_states[layer][0, -1:].to(lens.J.dtype)
            lg = readout_logits(model, lens, h, layer)
            if lg is None:
                print(f"  layer {layer:3d}: not covered by this lens")
                continue
            top = torch.topk(lg[0].float(), 10).indices.tolist()
            overlap = len(set(top) & set(real_top))
            print(f"  layer {layer:3d}: top-10 overlap with model = {overlap}/10  "
                  f"{[tok.decode([i]) for i in top[:5]]}")
    print("\nPick the orientation whose late-layer overlap is clearly higher and "
          "pass it as --orient. If BOTH are near zero, the normalization or the "
          "source_layers indexing is wrong -- stop and fix that before collecting.")


# ---------------------------------------------------------------- main sweep

@torch.no_grad()
def run(model, tok, lens, args, device):
    rows = []
    rng = random.Random(args.seed)
    n_layers = model.config.num_hidden_layers
    layers = [l for l in range(0, n_layers + 1, args.layer_stride)
              if lens.map_for(l) is not None]
    print(f"scoring {len(layers)} layers (stride {args.layer_stride})")

    for k in args.ks:
        for cond in args.conditions:
            batch, meta = [], []
            for i in range(args.n):
                text, gold, inter = build_prompt(tok, rng, k, cond, args.filler_n)
                batch.append(text); meta.append((i, gold, inter))
                if len(batch) < args.batch and i < args.n - 1:
                    continue

                enc = tok(batch, return_tensors="pt", padding=True,
                          padding_side="left").to(device)
                t0 = time.time()
                out = model(**enc, output_hidden_states=True)
                # positions: the tail of the sequence, where the answer forms
                T = enc["input_ids"].shape[1]
                pos = list(range(max(0, T - args.tail), T))

                for bi, (idx, gold, inter) in enumerate(meta):
                    targets = {f"hop{m+1}": first_token_id(tok, v)
                               for m, v in enumerate(inter)}
                    # null control: a two-digit value that is not any intermediate
                    nulls = [v for v in range(10, 100) if str(v) not in inter]
                    targets["null"] = first_token_id(tok, rng.choice(nulls))

                    for layer in layers:
                        h = out.hidden_states[layer][bi, pos].to(lens.J.dtype)
                        lg = readout_logits(model, lens, h, layer)
                        if lg is None:
                            continue
                        lgf = lg.float()
                        order = lgf.argsort(dim=-1, descending=True)
                        for label, tid in targets.items():
                            rank = (order == tid).nonzero()[:, 1]
                            prob = lgf.softmax(-1)[:, tid]
                            for pi, p in enumerate(pos):
                                rows.append({
                                    "k": k, "condition": cond, "prompt": idx,
                                    "layer": layer, "pos_from_end": T - p,
                                    "target": label, "rank": int(rank[pi]),
                                    "prob": float(prob[pi]),
                                })
                print(f"  k={k} {cond} batch of {len(batch)}: "
                      f"{time.time()-t0:.1f}s, {len(rows)} rows")
                batch, meta = [], []

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_parquet(f"{args.out}.parquet")
    print(f"\nwrote {args.out}.parquet  ({len(df):,} rows)")

    hop = df[df.target != "null"]
    null = df[df.target == "null"]
    print(f"median rank of true intermediates: {hop['rank'].median():.0f}")
    print(f"median rank of null control:       {null['rank'].median():.0f}")
    print("A real signal means intermediates rank far better than null. If the two "
          "are similar, the lens is not surfacing the computation on this task.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lens", required=True)
    ap.add_argument("--out", default="lens_out")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--ks", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--conditions", nargs="+", default=["immediate", "filler"])
    ap.add_argument("--filler-n", type=int, default=100, dest="filler_n")
    ap.add_argument("--tail", type=int, default=48,
                    help="how many trailing positions to score (full-sequence scoring "
                         "is ~10x the cost and mostly reads the prompt back)")
    ap.add_argument("--layer-stride", type=int, default=1, dest="layer_stride")
    ap.add_argument("--orient", default="left", choices=["left", "right"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"loading {args.model} in bfloat16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device)
    model.eval()
    lens = Lens(args.lens, orient=args.orient, device=device)
    print(f"lens meta: { {k: v for k, v in lens.meta.items() if k != 'provenance'} }")

    if args.selftest:
        selftest(model, tok, lens, device)
        return
    run(model, tok, lens, args, device)


if __name__ == "__main__":
    main()
