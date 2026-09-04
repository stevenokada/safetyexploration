"""
Chained arithmetic word problem: in-distribution math, contamination-proof,
cleanly depth-parameterized, with intermediates that R-lens can actually read.

Why this task exists
--------------------
The chained-lookup task answered the depth question but is not natural reasoning,
which was a stated limitation of the pilot. This one is GSM8K-shaped, so it sits
much closer to the training distribution, while keeping every property the
measurement needs.

The lens constraint that shapes the design
------------------------------------------
Qwen tokenizers split multi-digit numbers into digits ("47" -> "4","7"), so no
integer above 9 is a single token, and R-lens surfaces single-token concepts.
The workaround: keep every intermediate a TWO-DIGIT number and force all
intermediates in a trial to have DISTINCT TENS DIGITS. The first token of an
intermediate is then a unique identifier for which hop it is, so a single-token
readout at (layer, position) maps unambiguously onto "hop m".

Constraints enforced per trial
------------------------------
1. every intermediate lies in 10..99                   two digits, so a tens digit exists
2. all intermediates have distinct tens digits         lens-identifiable by first token
3. all intermediates are distinct values               hop attribution unambiguous
4. no intermediate appears as a literal in the prompt  a readout cannot just be reading input
5. every step branches on the current value's parity   chain cannot be folded into a
                                                       closed form without computing it
6. sentence count fixed across k                       prompt length is not a depth confound

Constraint 2 caps depth at 9 (nine tens digits, one spent on the start value), so
k <= 8. Deeper sweeps should use the chained-lookup task, which has no such limit.
"""
import random

VALUE_LO, VALUE_HI = 10, 99
OPERAND_LO, OPERAND_HI = 2, 29
SENTENCES_TOTAL = 10
MAX_K = 8

NEUTRAL = [
    "The night shift signs the ledger.",
    "A supervisor initials the log.",
    "The tally is copied onto the whiteboard.",
    "The clerk stamps the page.",
    "The count is read aloud for the record.",
    "The logbook is passed to the next shift.",
    "A photo of the ledger is filed.",
    "The entry is underlined in the register.",
    "The shift lead confirms the page number.",
    "The record is initialled and set aside.",
]


def _realize(v, nv):
    """Ops that take v to nv. Only the branch matching v's parity determines the
    result; the other branch's operand is free and gets filled in later."""
    out = []
    even = v % 2 == 0
    if even:
        for kind, need in (("parity_add", nv - v),
                           ("parity_halve", nv - v // 2),
                           ("parity_sub", v - nv)):
            if OPERAND_LO <= need <= OPERAND_HI:
                out.append((kind, need, None))
    else:
        for kind, need in (("parity_add", nv - v),
                           ("parity_halve", nv - v),
                           ("parity_sub", v - nv)):
            if OPERAND_LO <= need <= OPERAND_HI:
                out.append((kind, None, need))
    return out


def _search(rng, k, branch=24, nodes=3000):
    """Backtracking over reachable VALUES (not operand pairs). Rejection sampling
    collapses past k=5 because each hop consumes one of only nine tens digits.

    Branching and node budget are capped: at k>=7 exhaustive backtracking spends
    seconds per trial exploring dead ends, whereas capped search plus randomized
    restarts from generate() finds a chain in milliseconds."""
    start = rng.randint(VALUE_LO, VALUE_HI)
    used_tens = {start // 10}
    chain = []          # list of (op, next_value)
    budget = [nodes]

    def step(v):
        if len(chain) == k:
            return True
        if budget[0] <= 0:
            return False
        cands = [nv for nv in range(VALUE_LO, VALUE_HI + 1)
                 if nv // 10 not in used_tens and nv != start]
        rng.shuffle(cands)
        for nv in cands[:branch]:
            budget[0] -= 1
            if budget[0] <= 0:
                return False
            ops = _realize(v, nv)
            if not ops:
                continue
            rng.shuffle(ops)
            op = ops[0]
            chain.append((op, nv)); used_tens.add(nv // 10)
            if step(nv):
                return True
            chain.pop(); used_tens.discard(nv // 10)
        return False

    if not step(start):
        return None

    vals = [nv for _, nv in chain]
    # fill each op's unused branch with an operand that is not an intermediate,
    # so constraint 4 holds over every literal printed in the prompt
    forbidden = set(vals)
    ops = []
    for (kind, a, b), _ in chain:
        free = [x for x in range(OPERAND_LO, OPERAND_HI + 1) if x not in forbidden]
        if a is None:
            a = rng.choice(free)
        if b is None:
            b = rng.choice(free)
        if a in forbidden or b in forbidden:
            return None
        ops.append((kind, a, b))
    return start, ops, vals


def _phrase(op):
    kind, a, b = op
    if kind == "parity_add":
        return (f"If the count is even, {a} more crates arrive; "
                f"if it is odd, {b} more crates arrive.")
    if kind == "parity_halve":
        return (f"If the count is even, half of them are shipped out and then {a} "
                f"crates arrive; if it is odd, {b} more crates arrive.")
    if kind == "parity_sub":
        return (f"If the count is even, {a} crates are shipped out; "
                f"if it is odd, {b} crates are shipped out.")
    raise ValueError(kind)


def _apply(v, op):
    kind, a, b = op
    if kind == "parity_add":
        return v + a if v % 2 == 0 else v + b
    if kind == "parity_halve":
        return v // 2 + a if v % 2 == 0 else v + b
    if kind == "parity_sub":
        return v - a if v % 2 == 0 else v - b
    raise ValueError(kind)


def generate(rng, k, max_tries=40):
    """Returns (prompt, answer, intermediates, reasoning)."""
    if k > MAX_K:
        raise ValueError(f"k={k} exceeds MAX_K={MAX_K} (only 9 tens digits exist)")
    for _ in range(max_tries):
        got = _search(rng, k)
        if got:
            break
    else:
        raise RuntimeError(f"could not satisfy constraints for k={k}")
    start, ops, vals = got

    # verify the chain actually evaluates as intended
    v = start
    for op, expect in zip(ops, vals):
        v = _apply(v, op)
        assert v == expect, "generator inconsistency"

    lines = [_phrase(op) for op in ops]
    fillers = rng.sample(NEUTRAL, SENTENCES_TOTAL - k)
    slots = sorted(rng.sample(range(SENTENCES_TOTAL), k))
    body, oi, fi = [], 0, 0
    for i in range(SENTENCES_TOTAL):
        if oi < k and i == slots[oi]:
            body.append(lines[oi]); oi += 1
        else:
            body.append(fillers[fi]); fi += 1

    prompt = (f"A depot starts the day with {start} crates.\n"
              + "\n".join(f"- {s}" for s in body)
              + "\n\nHow many crates does the depot have at the end of the day?")

    # constraint 4, enforced against the FINAL rendered prompt rather than assumed
    import re as _re
    literals = {int(x) for x in _re.findall(r"\b\d+\b", prompt)}
    if literals & {int(x) for x in map(str, vals)}:
        return generate(rng, k, max_tries)     # regenerate rather than emit a confounded trial

    steps, v = [], start
    for op, nv in zip(ops, vals):
        steps.append(f"{v} is {'even' if v % 2 == 0 else 'odd'}, so the count becomes {nv}")
        v = nv
    reasoning = f"Start at {start}. " + "; ".join(steps) + "."
    return prompt, str(vals[-1]), [str(x) for x in vals], reasoning


if __name__ == "__main__":
    import time
    for k in range(1, MAX_K + 1):
        t0, ok = time.time(), 0
        for i in range(40):
            try:
                generate(random.Random(i), k); ok += 1
            except RuntimeError:
                pass
        print(f"k={k}: {ok}/40 generated, {(time.time()-t0)/40*1000:.1f} ms each")

    print()
    p, a, inter, r = generate(random.Random(3), 5)
    print(p)
    print(f"\nanswer={a}  intermediates={inter}  leading tokens={[x[0] for x in inter]}")
    print(f"reasoning: {r}")

    # constraint audit over many trials
    bad = 0
    for i in range(300):
        p, a, inter, _ = generate(random.Random(1000 + i), 4)
        vals = [int(x) for x in inter]
        import re
        literals = [int(x) for x in re.findall(r"\b\d+\b", p)]
        if len(set(v // 10 for v in vals)) != len(vals): bad += 1
        elif len(set(vals)) != len(vals): bad += 1
        elif set(vals) & set(literals): bad += 1
    print(f"\nconstraint violations over 300 trials at k=4: {bad}")
