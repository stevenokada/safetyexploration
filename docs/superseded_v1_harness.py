"""
Pilot: latent serial-depth ceiling via k-hop chained lookup on in-context random mappings.

Conditions:
  immediate : forced instant answer (assistant prefill "Answer:", tiny max_tokens)
  filler    : counting filler "1 2 ... 100" appended to question, then forced answer
  cot       : free chain-of-thought, parse final "Answer: XX"

Design notes:
  - Mapping is a single 12-cycle over two-letter codes -> k hops (k<12) are always
    k distinct steps; mapping lines are shuffled so text order reveals nothing.
  - Prompt length is constant across k (all 12 mappings always shown).
  - 3 few-shot examples per condition (with filler in the filler condition),
    following Brauer et al. / Greenblatt setups.

Usage:
  OPENROUTER_API_KEY=... python3 pilot.py [--model deepseek/deepseek-chat] [--n 40] [--dry-run]
"""
import argparse, asyncio, csv, json, os, random, re, string, sys, time
from pathlib import Path

import httpx

API_URL = "https://openrouter.ai/api/v1/chat/completions"

KS = [1, 2, 3, 4, 6, 8]
CONDITIONS = ["immediate", "filler", "cot"]
N_NODES = 12          # mappings per prompt (single cycle)
FILLER_N = 100        # counting filler length
FEWSHOT = 3

# ---------------------------------------------------------------- task gen

def gen_codes(rng, n):
    """n distinct two-letter uppercase codes."""
    codes = set()
    while len(codes) < n:
        codes.add("".join(rng.choices(string.ascii_uppercase, k=2)))
    return list(codes)

def gen_example(rng, k):
    """One k-hop instance. Returns (mapping_lines, start, answer, full_chain)."""
    codes = gen_codes(rng, N_NODES)
    rng.shuffle(codes)                       # cycle order
    cycle = {codes[i]: codes[(i + 1) % N_NODES] for i in range(N_NODES)}
    lines = [f"{a} -> {b}" for a, b in cycle.items()]
    rng.shuffle(lines)                       # hide chain structure in text
    start = rng.choice(codes)
    node, full = start, [start]
    for _ in range(N_NODES - 1):
        node = cycle[node]
        full.append(node)                    # full 12-cycle from start
    return lines, start, full[k], full

def question_text(lines, start, k):
    hops = "once" if k == 1 else f"{k} times"
    return (
        "Mapping table:\n" + "\n".join(lines) + "\n\n"
        f"Start at {start} and follow the mapping {hops}. What code do you land on?"
    )

FILLER_TEXT = " ".join(str(i) for i in range(1, FILLER_N + 1))

def build_messages(rng, k, condition):
    """Few-shot examples + target question. Returns (messages, gold, chain)."""
    if condition == "cot":
        instr = ("Solve the problem. Think step by step, then give your final "
                 "answer on its own line in the format 'Answer: XX'.")
    else:
        instr = ("Answer immediately. Respond with ONLY the final code in the "
                 "format 'Answer: XX'. Do not write anything else, do not reason.")
    messages = [{"role": "system", "content": instr}]

    for _ in range(FEWSHOT):
        fl, fs, fa, fc = gen_example(rng, k)
        q = question_text(fl, fs, k)
        if condition == "filler":
            q += "\n\n" + FILLER_TEXT
        messages.append({"role": "user", "content": q})
        if condition == "cot":
            steps = " -> ".join(fc[:k + 1])
            messages.append({"role": "assistant",
                             "content": f"Following the chain: {steps}.\nAnswer: {fa}"})
        else:
            messages.append({"role": "assistant", "content": f"Answer: {fa}"})

    lines, start, gold, chain = gen_example(rng, k)
    q = question_text(lines, start, k)
    if condition == "filler":
        q += "\n\n" + FILLER_TEXT
    messages.append({"role": "user", "content": q})
    if condition != "cot":
        messages.append({"role": "assistant", "content": "Answer:"})  # prefill
    return messages, gold, chain

# ---------------------------------------------------------------- scoring

ANS_RE = re.compile(r"Answer:\s*([A-Z]{2})\b")
CODE_RE = re.compile(r"\b[A-Z]{2}\b")

def extract_answer(condition, completion):
    if condition == "cot":
        m = ANS_RE.findall(completion)
        return m[-1] if m else None
    m = CODE_RE.search(completion)           # completion continues "Answer:" prefill
    return m.group(0) if m else None

def leak_flag(condition, completion, chain):
    """Did a forced-immediate response sneak in reasoning? Verbose output or
    multiple codes = leaked reasoning. A single wrong code is NOT a leak."""
    if condition == "cot":
        return False
    return len(completion.split()) > 4 or len(CODE_RE.findall(completion)) > 1

# ---------------------------------------------------------------- api

async def call_api(client, sem, model, messages, max_tokens, temperature=0.0,
                   retries=6, allow_reasoning=False):
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if not allow_reasoning:  # hidden reasoning would contaminate forced-answer conditions
        payload["reasoning"] = {"enabled": False}
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.post(API_URL, json=payload, timeout=120)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                return data["choices"][0]["message"]["content"] or ""
            except Exception as e:
                if attempt == retries - 1:
                    return f"__ERROR__ {e}"
                await asyncio.sleep(3 * 2 ** attempt + random.random())

async def run_one(client, sem, model, rng_seed, k, condition, idx):
    rng = random.Random(rng_seed)
    messages, gold, chain = build_messages(rng, k, condition)
    max_tokens = 4000 if condition == "cot" else 8
    completion = await call_api(client, sem, model, messages, max_tokens,
                                allow_reasoning=(condition == "cot"))
    pred = None if completion.startswith("__ERROR__") else extract_answer(condition, completion)
    return {
        "k": k, "condition": condition, "idx": idx,
        "gold": gold, "pred": pred,
        "correct": int(pred == gold),
        "leak": int(leak_flag(condition, completion, chain)),
        "eff_hops": chain.index(pred) if pred in chain else -1,  # position on full cycle; >k = overshoot
        "chain": " ".join(chain[:k + 1]),
        "error": int(completion.startswith("__ERROR__")),
        "completion": completion[:2000],
    }

# ---------------------------------------------------------------- main

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek/deepseek-chat")
    ap.add_argument("--n", type=int, default=40, help="examples per (k, condition) cell")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--ks", default=",".join(map(str, KS)))
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true", help="print one prompt per condition and exit")
    args = ap.parse_args()

    conds = args.conditions.split(",")
    ks = [int(x) for x in args.ks.split(",")]

    if args.dry_run:
        for cond in conds:
            msgs, gold, chain = build_messages(random.Random(123), 4, cond)
            print(f"\n{'='*60}\nCONDITION {cond} (gold={gold}, chain={' -> '.join(chain)})")
            for m in msgs[-2:] if cond != "cot" else msgs[-1:]:
                print(f"--- {m['role']} ---\n{m['content'][:1500]}")
        return

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY")

    out = args.out or f"results_{args.model.split('/')[-1]}_{int(time.time())}"
    sem = asyncio.Semaphore(args.concurrency)
    headers = {"Authorization": f"Bearer {key}",
               "HTTP-Referer": "https://localhost", "X-Title": "serial-depth-pilot"}

    tasks = []
    async with httpx.AsyncClient(headers=headers) as client:
        for k in ks:
            for cond in conds:
                for i in range(args.n):
                    # same seed across conditions -> same instances, paired comparison
                    seed = hash((args.seed, k, i)) & 0xFFFFFFFF
                    tasks.append(run_one(client, sem, args.model, seed, k, cond, i))
        print(f"Running {len(tasks)} calls on {args.model} ...")
        t0 = time.time()
        rows = await asyncio.gather(*tasks)
        print(f"Done in {time.time()-t0:.0f}s")

    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # summary
    import pandas as pd
    df = pd.DataFrame(rows)
    df["on_cycle"] = df.eff_hops >= 0
    df["overshoot"] = df.eff_hops > df.k
    summary = df.groupby(["condition", "k"]).agg(
        acc=("correct", "mean"), on_cycle=("on_cycle", "mean"),
        eff_hops=("eff_hops", lambda s: s[s >= 0].mean()),
        overshoot=("overshoot", "sum"),
        leaks=("leak", "sum"), errs=("error", "sum")).reset_index()
    print("\n" + summary.to_string(index=False))
    chance = 1 / (N_NODES - 1)
    print(f"\n(chance ~= {chance:.2f}; leak = forced-answer response that verbalized reasoning)")

    # plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cond, g in df.groupby("condition"):
        s = g.groupby("k")["correct"].mean()
        ax.plot(s.index, s.values, marker="o", label=cond)
    ax.axhline(chance, ls=":", c="gray", label="chance")
    ax.set_xlabel("serial depth k (hops)"); ax.set_ylabel("accuracy")
    ax.set_title(f"Latent serial depth: {args.model}  (n={args.n}/cell)")
    ax.set_ylim(0, 1.02); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{out}.png", dpi=150)
    print(f"\nWrote {out}.csv and {out}.png")

if __name__ == "__main__":
    asyncio.run(main())
