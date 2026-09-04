"""
Precondition check: are this task's intermediates readable by a single-token lens?

R-lens surfaces one vocabulary token at a time. The arithmetic task relies on a
specific property to stay lens-readable: a two-digit intermediate must tokenize
with its TENS DIGIT AS THE FIRST TOKEN, and all intermediates in a trial must
have distinct tens digits, so a single-token readout identifies which hop it is.

That property is a fact about each model's tokenizer, not something we control.
Run this before any lens experiment; a model that fails is not usable for the
hop-attribution analysis even though its behavioral numbers are still fine.

  python3 check_tokenization.py
"""
import random
import sys

from tokenizers import Tokenizer

from tasks import arithmetic as task_arith

# google/gemma-* repos are gated (401 without an HF token). The unsloth mirrors
# ship the same tokenizer; if you have HF_TOKEN set, prefer the official repo.
MODELS = {
    "qwen3.6-27b":     "Qwen/Qwen3.6-27B",
    "qwen3.6-35b-a3b": "Qwen/Qwen3.6-35B-A3B",
    "gemma-3-27b-it":  "unsloth/gemma-3-27b-it",   # mirror of google/gemma-3-27b-it
}

TRIALS = 200
K = 4


def check(name, tok):
    out = {"model": name, "vocab": tok.get_vocab_size()}

    def ids(s):
        return tok.encode(s, add_special_tokens=False).tokens

    # 1. how do bare integers tokenize?
    out["single_0_9"] = sum(1 for n in range(10) if len(ids(str(n))) == 1)
    out["single_10_99"] = sum(1 for n in range(10, 100) if len(ids(str(n))) == 1)

    # 2. the load-bearing property: first token of a 2-digit number == its tens digit
    firsts_ok = sum(1 for n in range(10, 100) if ids(str(n))[0] == str(n // 10))
    out["tens_is_first_token"] = firsts_ok

    # 3. same, but as the number appears mid-sentence (leading space)
    sp_ok = sum(1 for n in range(10, 100)
                if [t for t in ids(f" {n}") if t.strip(" Ġ▁")][0] == str(n // 10))
    out["tens_is_first_token_spaced"] = sp_ok

    # 4. against REAL generated trials, not synthetic integers
    bad = []
    for i in range(TRIALS):
        _, _, inter, _ = task_arith.generate(random.Random(9000 + i), K)
        first = [ids(x)[0] for x in inter]
        if first != [x[0] for x in inter] or len(set(first)) != len(first):
            bad.append((inter, first))
    out["trials_ok"] = TRIALS - len(bad)
    out["example_failure"] = bad[0] if bad else None

    # 5. the max-variant answer is a depot name; it should be a single token
    out["depot_names_single"] = sum(
        1 for d in task_arith.DEPOTS if len(ids(f" {d}")) == 1)
    out["depot_names_total"] = len(task_arith.DEPOTS)
    return out


def main():
    rows = []
    for name, repo in MODELS.items():
        try:
            tok = Tokenizer.from_pretrained(repo)
        except Exception as e:
            print(f"{name}: COULD NOT LOAD ({str(e)[:60]}) — cannot certify this model")
            rows.append({"model": name, "error": True})
            continue
        rows.append(check(name, tok))

    print(f"{'model':18s} {'vocab':>7s} {'0-9':>5s} {'10-99':>6s} "
          f"{'tens=1st':>9s} {'spaced':>7s} {'trials':>8s} {'depots':>7s}")
    for r in rows:
        if r.get("error"):
            print(f"{r['model']:18s}    (not loaded)")
            continue
        print(f"{r['model']:18s} {r['vocab']:7d} {r['single_0_9']:4d}/10 "
              f"{r['single_10_99']:4d}/90 {r['tens_is_first_token']:6d}/90 "
              f"{r['tens_is_first_token_spaced']:5d}/90 {r['trials_ok']:5d}/{TRIALS} "
              f"{r['depot_names_single']:3d}/{r['depot_names_total']}")

    print()
    ok = True
    for r in rows:
        if r.get("error"):
            print(f"  {r['model']}: UNVERIFIED — do not use for lens analysis")
            ok = False
            continue
        if r["trials_ok"] < TRIALS:
            print(f"  {r['model']}: FAILS hop identifiability on "
                  f"{TRIALS - r['trials_ok']}/{TRIALS} trials, e.g. {r['example_failure']}")
            ok = False
        elif r["depot_names_single"] < r["depot_names_total"]:
            print(f"  {r['model']}: intermediates fine, but "
                  f"{r['depot_names_total'] - r['depot_names_single']} depot names are "
                  f"multi-token — the max-variant ANSWER is not single-token here")
        else:
            print(f"  {r['model']}: OK — intermediates and answers are single-token readable")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
