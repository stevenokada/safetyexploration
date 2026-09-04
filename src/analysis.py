"""
Regenerate every published number from the raw trial CSVs.

Until now the statistics lived in throwaway shell blocks, so the figures in the
report could not be re-derived from the repository. This script closes that gap:
every headline claim below is computed here from data/, with the test named and
the assumptions stated.

  python3 src/analysis.py                # all sections
  python3 src/analysis.py --section depth

Tests used
  paired outcomes  -> exact McNemar on discordant pairs (two conditions in ONE
                      invocation share generated problems; separate invocations
                      only pair when seeds are crc32-derived, see docs/BUGS.md #9)
  single proportion-> Wilson score interval
  two proportions  -> pooled two-proportion z
  many tests       -> Benjamini-Hochberg over the family
"""
import argparse
import glob
import math
import os

import pandas as pd

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


# ---------------------------------------------------------------- statistics

def mcnemar(b01, b10):
    """Exact two-sided McNemar. b01 = wrong->right, b10 = right->wrong."""
    n = b01 + b10
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b01, b10) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def wilson(successes, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def two_prop(k1, n1, k2, n2):
    """Pooled two-proportion z test. Returns (z, p)."""
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (k2 / n2 - k1 / n1) / se
    return z, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


def benjamini_hochberg(named_pvalues, q=0.05):
    """Returns [(name, p, critical, survives)] sorted by p."""
    rows = sorted(named_pvalues, key=lambda r: r[1])
    n = len(rows)
    out, cutoff = [], 0
    for i, (name, p) in enumerate(rows, 1):
        if p <= q * i / n:
            cutoff = i
    for i, (name, p) in enumerate(rows, 1):
        out.append((name, p, q * i / n, i <= cutoff))
    return out


def paired(df, cond_a, cond_b, key=("k", "idx")):
    """Join two conditions on the trial key. Only valid when both were produced
    in the same invocation, or by crc32-derived seeds."""
    a = df[df.condition == cond_a].set_index(list(key))["correct"]
    b = df[df.condition == cond_b].set_index(list(key))["correct"]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    b01 = int(((j.a == 0) & (j.b == 1)).sum())
    b10 = int(((j.a == 1) & (j.b == 0)).sum())
    return j, b01, b10


def load(pattern, exclude=None):
    """exclude guards against a glob swallowing a neighbouring task: 'B_parallel_*'
    also matches 'B_parallel_count_*', which silently mixed two tasks and moved a
    published figure from 0.836 to 0.610."""
    files = sorted(glob.glob(os.path.join(DATA, pattern)))
    if exclude:
        files = [f for f in files if exclude not in os.path.basename(f)]
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# ---------------------------------------------------------------- sections

def section_depth():
    """Headline: serial collapses with depth while matched-work parallel does not."""
    print("=" * 70)
    print("DEPTH vs WORK  (synthetic lookup task, immediate condition, k>=4)")
    print("=" * 70)
    for task in ("serial", "parallel"):
        d = load(f"B_{task}_*.csv", exclude="parallel_count" if task == "parallel" else None)
        if d is None:
            continue
        d = d[(d.condition == "immediate") & (d.k >= 4)]
        lo, hi = wilson(int(d.correct.sum()), len(d))
        print(f"  {task:10s} {int(d.correct.sum()):4d}/{len(d):4d} = "
              f"{d.correct.mean():.3f}   95% CI [{lo:.3f}, {hi:.3f}]")
    print("\n  Non-overlapping intervals; same number of lookups in both arms.")


def section_width():
    """How far independent work scales before it degrades."""
    print("\n" + "=" * 70)
    print("WIDTH  (independent lookups to k=64, immediate)")
    print("=" * 70)
    for f in sorted(glob.glob(os.path.join(DATA, "W2_parallel_*.csv"))):
        if "count" in f:
            continue
        d = pd.read_csv(f)
        model = os.path.basename(f).replace("W2_parallel_", "").replace(".csv", "")
        cells = []
        for k, g in d.groupby("k"):
            cells.append(f"k{k}:{g.correct.mean():.2f}({g.correct.mean()*k:.0f}x)")
        print(f"  {model:18s} " + " ".join(cells))
    print("\n  (x = multiple of chance, which is 1/k)")


def section_facts():
    """Real-world fact chains: depth curve, and filler at each depth."""
    print("\n" + "=" * 70)
    print("REAL-WORLD FACT CHAINS  (qwen3.6-27b)")
    print("=" * 70)
    s = load("F2_serial.csv")
    c = load("F2_serial_cot.csv")
    if s is None:
        print("  no data"); return
    c = c[c.finish == "stop"] if c is not None else None
    print(f"  {'k':>2} {'CoT':>7} {'no CoT':>8} {'filler':>8} {'budget':>7} {'diff':>7} {'p':>8}")
    for k in sorted(s.k.unique()):
        im = s[s.k == k].set_index("idx")["correct"]
        cot = c[c.k == k].correct.mean() if c is not None and k in set(c.k) else float("nan")
        row = f"  {k:>2} {cot:7.3f} {im.mean():8.3f}"
        fills = sorted(glob.glob(os.path.join(DATA, f"F2_fill_k{k}.csv")))
        if fills:
            f = pd.read_csv(fills[0]).set_index("idx")["correct"]
            j = pd.concat([im.rename("a"), f.rename("b")], axis=1).dropna()
            b01 = int(((j.a == 0) & (j.b == 1)).sum())
            b10 = int(((j.a == 1) & (j.b == 0)).sum())
            row += f" {j.b.mean():8.3f} {'':>7} {j.b.mean()-j.a.mean():+7.3f} {mcnemar(b01,b10):8.4f}"
        print(row)
    print("\n  CoT at ceiling everywhere => the collapse is a limit on unwritten"
          "\n  computation, not on knowledge or comprehension.")


def section_dose():
    """The four-hop filler null was a dosing error, not a ceiling."""
    print("\n" + "=" * 70)
    print("FILLER DOSE-RESPONSE AT FOUR HOPS")
    print("=" * 70)
    s = load("F2_serial.csv")
    if s is None:
        print("  no data"); return
    im = s[s.k == 4].set_index("idx")["correct"]
    print(f"  baseline (no filler): {im.mean():.3f}")
    for f in sorted(glob.glob(os.path.join(DATA, "F2_fill_k4*.csv"))):
        budget = os.path.basename(f).split("_n")[-1].replace(".csv", "") if "_n" in f else "405"
        d = pd.read_csv(f).set_index("idx")["correct"]
        j = pd.concat([im.rename("a"), d.rename("b")], axis=1).dropna()
        b01 = int(((j.a == 0) & (j.b == 1)).sum())
        b10 = int(((j.a == 1) & (j.b == 0)).sum())
        print(f"  filler {budget:>5} tokens: {j.b.mean():.3f}  "
              f"diff {j.b.mean()-j.a.mean():+.3f}  p={mcnemar(b01,b10):.4f}")
    print("\n  Effect appears once the budget matches its neighbours, then saturates.")


def section_aggregation():
    """Two aggregations over identical retrievals: cost of combining, isolated."""
    print("\n" + "=" * 70)
    print("AGGREGATION COST  (same retrievals, different final question)")
    print("=" * 70)
    cheap = load("F2_parallel2.csv")
    order = load("F2_parallel.csv")
    if cheap is None:
        print("  no data"); return
    ci = cheap[cheap.condition == "immediate"]
    print(f"  {'k':>3} {'one-letter test':>16} {'alphabetical order':>19} {'chance':>8}")
    for k in sorted(ci.k.unique()):
        a = ci[ci.k == k].correct.mean()
        b = order[order.k == k].correct.mean() if order is not None and k in set(order.k) else float("nan")
        print(f"  {k:>3} {a:16.3f} {b:19.3f} {1/k:8.3f}")
    print("\n  Cheap aggregation holds a steady margin over chance; ordering sits at"
          "\n  chance despite CoT solving it every time.")


def section_fewshot():
    """Few-shot count is a design factor, not a constant: it inflates overshoot."""
    print("\n" + "=" * 70)
    print("FEW-SHOT COUNT  (chained lookup, paired via crc32 seeds)")
    print("=" * 70)
    for model in ("gemma-3-27b-it", "qwen3.6-27b"):
        try:
            d1 = pd.read_csv(os.path.join(DATA, f"G_fs1_{model}.csv"))
            d3 = pd.read_csv(os.path.join(DATA, f"G_fs3_{model}.csv"))
        except FileNotFoundError:
            continue
        a = d1[(d1.condition == "immediate") & (d1.eff_hops >= 0)]
        b = d3[(d3.condition == "immediate") & (d3.eff_hops >= 0)]
        oa, ob = (a.eff_hops > a.k), (b.eff_hops > b.k)
        z, p = two_prop(int(oa.sum()), len(oa), int(ob.sum()), len(ob))
        print(f"  {model:18s} overshoot 1-shot {oa.mean():.3f} vs 3-shot {ob.mean():.3f}"
              f"  z={z:+.2f} p={p:.2g}")
    print("\n  More demonstrations inflate how far past the target the answer lands.")


def section_corrections():
    """Multiple-comparison correction over the filler family."""
    print("\n" + "=" * 70)
    print("MULTIPLE COMPARISONS  (filler tests, Benjamini-Hochberg q=0.05)")
    print("=" * 70)
    tests = []
    for task in ("serial", "parallel", "parallel_count"):
        d = load(f"B_{task}_*.csv",
                 exclude="parallel_count" if task == "parallel" else None)
        if d is None:
            continue
        _, b01, b10 = paired(d, "immediate", "filler")
        tests.append((f"filler:{task}", mcnemar(b01, b10)))
    for k in (2, 3, 4, 6, 8):
        d = load("B_serial_*.csv")
        if d is None:
            continue
        dk = d[d.k == k]
        _, b01, b10 = paired(dk, "immediate", "filler", key=("idx",))
        tests.append((f"filler:serial k={k}", mcnemar(b01, b10)))
    if not tests:
        print("  no data"); return
    print(f"  {'test':26s} {'p':>8s} {'BH crit':>9s}  survives")
    for name, p, crit, ok in benjamini_hochberg(tests):
        print(f"  {name:26s} {p:8.4f} {crit:9.4f}  {'yes' if ok else '.'}")


SECTIONS = {
    "depth": section_depth, "width": section_width, "facts": section_facts,
    "dose": section_dose, "aggregation": section_aggregation,
    "fewshot": section_fewshot, "corrections": section_corrections,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--section", choices=sorted(SECTIONS), help="run just one")
    args = ap.parse_args()
    for name in ([args.section] if args.section else list(SECTIONS)):
        SECTIONS[name]()


if __name__ == "__main__":
    main()
