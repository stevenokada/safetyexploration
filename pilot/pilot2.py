"""
Pilot v2: latent serial depth, with (a) fencepost/wording control and
(b) a matched PARALLELIZABLE control task.

Two tasks, both over the same in-context random 12-cycle mapping table,
both answered with a two-letter code (identical scoring, identical chance-ish):

  serial   : "Start at X and follow the mapping k times."
             -> k DEPENDENT lookups. Serial depth = k. Work = k lookups.
  parallel : "Exactly one of these k codes maps to a code starting with 'L'.
              Which one?"
             -> k INDEPENDENT lookups + one aggregation. Serial depth = 2.
                Work = k lookups.

Matched: same table, same #lookups (k), same answer format. Differs only in
whether the lookups must be chained. Theory (Merrill & Sabharwal; the bound in
Brown-Cohen/Lindner/Shah 2026) predicts filler tokens add parallel width but
NOT serial depth, so filler should help `parallel` at large k and not `serial`.

Wording control (for the k=2 overshoot finding):
  default  : "follow the mapping k times"
  explicit : spells out the fencepost ("applying it once to X gives the code
             X maps to; applying it twice gives the code THAT code maps to")

Usage:
  OPENROUTER_API_KEY=... python3 pilot2.py --task serial --wording explicit ...
  OPENROUTER_API_KEY=... python3 pilot2.py --task parallel --conditions immediate,filler ...
"""
import argparse, asyncio, csv, os, random, re, string, sys, time, zlib

import httpx

import task_arith

API_URL = "https://openrouter.ai/api/v1/chat/completions"
REGISTRY = "models.json"

def load_registry():
    import json as _json
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), REGISTRY)) as f:
        return _json.load(f)["jlens"]["models"]

def check_model(model, allow_unregistered):
    """Only open-weight models with a published J-Lens are permitted, so every
    behavioral result can be followed up mechanistically on the same checkpoint."""
    reg = load_registry()
    if model in reg:
        m = reg[model]
        if not m.get("openrouter", True):
            sys.exit(f"{model} has a J-Lens but is not served on OpenRouter "
                     f"({m['gpu']}); run it locally instead.")
        return m
    if allow_unregistered:
        print(f"WARNING: {model} is not in the J-Lens registry — results cannot be "
              f"followed up with internals. Proceeding because --allow-unregistered was set.")
        return None
    sys.exit(f"{model} is not an open-weight model with a published J-Lens.\n"
             f"Permitted: " + ", ".join(k for k, v in reg.items() if v.get("openrouter"))
             + "\nOverride with --allow-unregistered if you really want a closed model.")

N_NODES = 12
FILLER_N = 100
FEWSHOT = 3
FILLER_TEXT = " ".join(str(i) for i in range(1, FILLER_N + 1))

CODE_RE = re.compile(r"\b[A-Z]{2}\b")
ANS_RE = re.compile(r"Answer:\s*([A-Z]{2})\b")
NUM_RE = re.compile(r"\b\d+\b")
NUM_ANS_RE = re.compile(r"Answer:\s*(\d+)\b")

# ---------------------------------------------------------------- shared gen

def set_fewshot(n):
    """Few-shot count has been fixed at 3 since the first pilot and never varied.
    It is both a cost lever and a possible confound: every shot shows filler
    followed immediately by a correct answer, which may teach the model that the
    filler region is ignorable scaffolding and suppress any benefit from it."""
    global FEWSHOT
    FEWSHOT = n

def set_filler(n):
    """Filler length must be settable per run: matching it to the model's actual
    CoT length is the point of the sweep, and 100 tokens is ~5% of that."""
    global FILLER_N, FILLER_TEXT
    FILLER_N = n
    FILLER_TEXT = " ".join(str(i) for i in range(1, n + 1))

def set_nodes(n):
    global N_NODES
    N_NODES = n

def gen_codes(rng, n):
    codes = set()
    while len(codes) < n:
        codes.add("".join(rng.choices(string.ascii_uppercase, k=2)))
    return list(codes)

def gen_cycle(rng):
    """Random single 12-cycle. Returns (cycle dict, shuffled display lines, codes)."""
    codes = gen_codes(rng, N_NODES)
    rng.shuffle(codes)
    cycle = {codes[i]: codes[(i + 1) % N_NODES] for i in range(N_NODES)}
    lines = [f"{a} -> {b}" for a, b in cycle.items()]
    rng.shuffle(lines)
    return cycle, lines, codes

def table_text(lines):
    return "Mapping table:\n" + "\n".join(lines)

# ---------------------------------------------------------------- serial task

def gen_serial(rng, k):
    """Returns (lines, start, gold, full_cycle_from_start)."""
    cycle, lines, codes = gen_cycle(rng)
    start = rng.choice(codes)
    node, full = start, [start]
    for _ in range(N_NODES - 1):
        node = cycle[node]
        full.append(node)
    return lines, start, full[k], full

def serial_question(lines, start, k, wording):
    hops = "once" if k == 1 else f"{k} times"
    q = f"{table_text(lines)}\n\nStart at {start} and follow the mapping {hops}."
    if wording == "explicit":
        q += (f" (Following it once from {start} gives the code {start} maps to; "
              f"following it twice gives the code that code maps to; and so on for "
              f"exactly {k} step{'s' if k != 1 else ''}.)")
    q += " What code do you land on?"
    return q

# -------------------------------------------------------------- parallel task

def gen_parallel(rng, k, max_tries=200):
    """k independent lookups + unique-first-letter aggregation.

    Returns (lines, starts, gold_start, target_letter). gold_start is the one
    start code whose IMAGE begins with target_letter (unique by construction).
    """
    for _ in range(max_tries):
        cycle, lines, codes = gen_cycle(rng)
        starts = rng.sample(codes, k)
        images = [cycle[s] for s in starts]
        first_letters = [im[0] for im in images]
        uniq = [i for i, L in enumerate(first_letters)
                if first_letters.count(L) == 1]
        if not uniq:
            continue
        pick = rng.choice(uniq)
        return lines, starts, starts[pick], first_letters[pick]
    raise RuntimeError("could not construct parallel instance")

def gen_parallel_count(rng, k, max_tries=200):
    """k independent lookups + count aggregation. No reverse-search shortcut:
    the answer depends on ALL k images. Split letters A-M / N-Z.
    Rejection-sample so the count is not degenerate (avoids 0 or k)."""
    for _ in range(max_tries):
        cycle, lines, codes = gen_cycle(rng)
        starts = rng.sample(codes, k)
        cnt = sum(1 for s in starts if cycle[s][0] <= "M")
        if k >= 2 and cnt in (0, k):
            continue
        return lines, starts, str(cnt)
    return lines, starts, str(cnt)

def parallel_count_question(lines, starts, k):
    lst = ", ".join(starts)
    return (f"{table_text(lines)}\n\n"
            f"Consider these {k} codes: {lst}.\n"
            f"Apply the mapping ONCE to each of them. How many of the {k} results "
            f"begin with a letter in A-M? Answer with just the number.")

def parallel_question(lines, starts, letter, k):
    lst = ", ".join(starts)
    return (f"{table_text(lines)}\n\n"
            f"Consider these {k} codes: {lst}.\n"
            f"Apply the mapping ONCE to each of them. Exactly one of them maps to "
            f"a code beginning with the letter {letter}. Which of the {k} codes is it? "
            f"Answer with that starting code.")

# ---------------------------------------------------------------- prompting

def build_messages(rng, k, condition, task, wording):
    """Returns (messages, gold, meta) where meta carries task-specific extras."""
    fmt = ("depot name" if task == "arith_parallel_max" else
           "number" if task in ("parallel_count", "arith", "arith_parallel") else "code")
    if condition == "cot":
        instr = ("Solve the problem. Think step by step, then give your final "
                 f"answer on its own line in the format 'Answer: <{fmt}>'.")
    else:
        instr = (f"Answer immediately. Respond with ONLY the final {fmt} in the "
                 f"format 'Answer: <{fmt}>'. Do not write anything else, do not reason.")
    messages = [{"role": "system", "content": instr}]

    def make(kk):
        if task == "serial":
            lines, start, gold, full = gen_serial(rng, kk)
            q = serial_question(lines, start, kk, wording)
            steps = " -> ".join(full[:kk + 1])
            reasoning = f"Following the chain: {steps}."
            return q, gold, reasoning, full
        elif task == "arith":
            q, gold, inter, reasoning = task_arith.generate(rng, kk)
            return q, gold, reasoning, None
        elif task in ("arith_parallel", "arith_parallel_max"):
            agg = "max" if task.endswith("_max") else "sum"
            q, gold, inter, reasoning = task_arith.generate_parallel(rng, kk, agg=agg)
            return q, gold, reasoning, None
        elif task == "parallel_count":
            lines, starts, gold = gen_parallel_count(rng, kk)
            q = parallel_count_question(lines, starts, kk)
            reasoning = (f"Applying the mapping to each of {', '.join(starts)} and "
                         f"counting those beginning A-M gives {gold}.")
            return q, gold, reasoning, None
        else:
            lines, starts, gold, letter = gen_parallel(rng, kk)
            q = parallel_question(lines, starts, letter, kk)
            reasoning = (f"Applying the mapping to each: "
                         + ", ".join(f"{s}" for s in starts)
                         + f". The one landing on a code starting with {letter} is {gold}.")
            return q, gold, reasoning, None

    for _ in range(FEWSHOT):
        q, gold, reasoning, _ = make(k)
        if condition == "filler":
            q += "\n\n" + FILLER_TEXT
        messages.append({"role": "user", "content": q})
        messages.append({"role": "assistant",
                         "content": (f"{reasoning}\nAnswer: {gold}" if condition == "cot"
                                     else f"Answer: {gold}")})

    q, gold, _, full = make(k)
    if condition == "filler":
        q += "\n\n" + FILLER_TEXT
    messages.append({"role": "user", "content": q})
    if condition != "cot":
        messages.append({"role": "assistant", "content": "Answer:"})
    return messages, gold, full

# ---------------------------------------------------------------- scoring

def extract_answer(condition, completion, task="serial"):
    numeric = task in ("parallel_count", "arith", "arith_parallel")
    if task == "arith_parallel_max":
        m = re.findall(r"\b(Alpha|Bravo|Delta|Echo|Golf|Hotel|India|Ridge)\b", completion)
        return m[-1] if (condition == "cot" and m) else (m[0] if m else None)
    if condition == "cot":
        m = (NUM_ANS_RE if numeric else ANS_RE).findall(completion)
        return m[-1] if m else None
    m = (NUM_RE if numeric else CODE_RE).search(completion)
    return m.group(0) if m else None

def leak_flag(condition, completion, task="serial"):
    if condition == "cot":
        return False
    tokens = (NUM_RE if task in ("parallel_count", "arith", "arith_parallel") else CODE_RE).findall(completion)
    return len(completion.split()) > 4 or len(tokens) > 1

# ---------------------------------------------------------------- api

async def call_api(client, sem, model, messages, max_tokens, temperature=0.0,
                   retries=6, allow_reasoning=False):
    payload = {"model": model, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    if not allow_reasoning:
        payload["reasoning"] = {"enabled": False}
    async with sem:
        for attempt in range(retries):
            try:
                r = await client.post(API_URL, json=payload, timeout=180)
                if r.status_code == 429 or r.status_code >= 500:
                    raise httpx.HTTPStatusError("retryable", request=r.request, response=r)
                r.raise_for_status()
                d = r.json()
                ch = d["choices"][0]
                msg = ch.get("message", {})
                return {
                    "text": msg.get("content") or "",
                    "finish": ch.get("finish_reason", ""),
                    "out_tokens": (d.get("usage") or {}).get("completion_tokens", -1),
                    # nonzero here means the provider reasoned invisibly despite our request
                    "reasoning_tokens": ((d.get("usage") or {}).get(
                        "completion_tokens_details") or {}).get("reasoning_tokens", 0),
                }
            except Exception as e:
                if attempt == retries - 1:
                    return {"text": f"__ERROR__ {e}", "finish": "error",
                            "out_tokens": -1, "reasoning_tokens": 0}
                await asyncio.sleep(3 * 2 ** attempt + random.random())

async def run_one(client, sem, model, seed, k, condition, task, wording, idx):
    rng = random.Random(seed)
    messages, gold, full = build_messages(rng, k, condition, task, wording)
    max_tokens = 4000 if condition == "cot" else 8
    res = await call_api(client, sem, model, messages, max_tokens,
                         allow_reasoning=(condition == "cot"))
    completion = res["text"]
    err = completion.startswith("__ERROR__")
    pred = None if err else extract_answer(condition, completion, task)
    # landing position along the cycle (serial only); -1 = off-cycle
    eff = -1
    if task == "serial" and full and pred in full:
        eff = full.index(pred)
    return {
        "task": task, "wording": wording, "k": k, "condition": condition, "idx": idx,
        "gold": gold, "pred": pred, "correct": int(pred == gold),
        "eff_hops": eff, "leak": int(leak_flag(condition, completion, task)),
        "error": int(err), "finish": res["finish"],
        "out_tokens": res["out_tokens"], "reasoning_tokens": res["reasoning_tokens"],
        "model": model.split("/")[-1], "completion": completion[:2000],
        "prompt": messages[-2]["content"][:6000] if condition != "cot" else messages[-1]["content"][:6000],
        "seed": seed,
    }

# ---------------------------------------------------------------- main

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen/qwen3.6-27b")
    ap.add_argument("--allow-unregistered", action="store_true",
                    help="permit a model with no published J-Lens (no internals follow-up possible)")
    ap.add_argument("--task", default="serial",
                    choices=["serial", "parallel", "parallel_count",
                             "arith", "arith_parallel", "arith_parallel_max"])
    ap.add_argument("--wording", default="default", choices=["default", "explicit"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fewshot", type=int, default=3,
                    help="number of few-shot examples (was fixed at 3 in all earlier runs)")
    ap.add_argument("--filler-n", type=int, default=100, dest="filler_n",
                    help="number of counting-filler tokens appended before the answer")
    ap.add_argument("--nodes", type=int, default=12, help="mapping table size (must exceed max k)")
    ap.add_argument("--conditions", default="immediate,filler,cot")
    ap.add_argument("--ks", default="1,2,3,4,6,8")
    ap.add_argument("--out", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conds = args.conditions.split(",")
    ks = [int(x) for x in args.ks.split(",")]
    set_nodes(args.nodes)
    set_filler(args.filler_n)
    set_fewshot(args.fewshot)
    if max(ks) >= args.nodes:
        sys.exit(f"--nodes ({args.nodes}) must exceed max k ({max(ks)})")

    if args.dry_run:
        for cond in conds:
            msgs, gold, _ = build_messages(random.Random(7), 4, cond, args.task, args.wording)
            print(f"\n{'='*70}\n{args.task}/{args.wording}/{cond}  gold={gold}")
            print(msgs[-2]["content"][:1400] if cond != "cot" else msgs[-1]["content"][:1400])
        return

    info = check_model(args.model, args.allow_unregistered)
    if info:
        print(f"{args.model}: {info['params']} {info['arch']}, lens={info['lens']}")

    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        sys.exit("Set OPENROUTER_API_KEY")

    out = args.out or f"v2_{args.task}_{args.wording}_{args.model.split('/')[-1]}"
    sem = asyncio.Semaphore(args.concurrency)
    headers = {"Authorization": f"Bearer {key}",
               "HTTP-Referer": "https://localhost", "X-Title": "serial-depth-pilot"}

    async with httpx.AsyncClient(headers=headers) as client:
        tasks = []
        for k in ks:
            for cond in conds:
                for i in range(args.n):
                    # zlib.crc32, NOT hash(): Python randomizes string hashing per
                    # process, so hash()-derived seeds differ between invocations.
                    # That silently broke pairing whenever two conditions were run
                    # as separate processes, and made prompts unreproducible.
                    key = f"{args.seed}|{args.task}|{k}|{i}".encode()
                    seed = zlib.crc32(key) & 0xFFFFFFFF
                    tasks.append(run_one(client, sem, args.model, seed, k, cond,
                                         args.task, args.wording, i))
        print(f"Running {len(tasks)} calls | {args.model} | {args.task}/{args.wording}")
        t0 = time.time()
        rows = await asyncio.gather(*tasks)
        print(f"Done in {time.time()-t0:.0f}s")

    with open(f"{out}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    import pandas as pd
    df = pd.DataFrame(rows)
    agg = {"acc": ("correct", "mean"), "leaks": ("leak", "sum"), "errs": ("error", "sum")}
    if args.task == "serial":
        df["over"] = df.eff_hops > df.k
        df["under"] = (df.eff_hops >= 0) & (df.eff_hops < df.k)
        agg.update(eff_hops=("eff_hops", lambda s: s[s >= 0].mean()),
                   over=("over", "mean"), under=("under", "mean"))
    summary = df.groupby(["condition", "k"]).agg(**agg).reset_index()
    print("\n" + summary.round(3).to_string(index=False))
    print(f"\nWrote {out}.csv")

    try:
        import inspect_runs
        a = inspect_runs.audit(inspect_runs.load(f"{out}.csv"))
        bad = a[a.status == "FAIL"]
        if len(bad):
            cols = [c for c in ["cond","k","n","acc","cot_reasoned","leak","why"] if c in a.columns]
            print(f"\n!! VALIDITY AUDIT: {len(bad)}/{len(a)} cells FAILED — "
                  f"do not treat these as depth measurements")
            print(bad[cols].to_string(index=False))
        else:
            print(f"validity audit: all {len(a)} cells passed")
    except Exception as e:
        print(f"(audit skipped: {e})")

if __name__ == "__main__":
    asyncio.run(main())
