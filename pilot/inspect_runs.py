"""
Transcript search, tagging, and validity auditing for pilot runs.

The GPT-5.1 bug (model silently ignored the "think step by step" instruction in
the CoT condition, so its CoT baseline was really a second immediate condition)
was invisible in the summary tables and only turned up on manual inspection.
`audit` catches that class of failure automatically.

Commands
  audit    per-cell validity checks; exits nonzero if any check FAILs
  tags     frequency of auto-tags, sliced however you like
  search   filter rows by pandas query and/or regex over completions
  sample   print full transcripts for eyeballing
  label    attach a manual label to matching rows (saved to labels.jsonl)

Examples
  python3 inspect_runs.py audit
  python3 inspect_runs.py audit --glob 'grid2_*.csv'
  python3 inspect_runs.py tags --where "condition=='cot'"
  python3 inspect_runs.py search --where "k>=4 and correct==0" --regex '->' -n 5
  python3 inspect_runs.py sample --tag cot_no_reasoning -n 6
  python3 inspect_runs.py label --where "model=='gpt-5.1' and condition=='cot'" \
      --tag instruction_noncompliance --note "answers without reasoning despite CoT prompt"
  python3 inspect_runs.py search --label instruction_noncompliance
"""
import argparse, glob as globmod, json, os, re, sys
from pathlib import Path

import pandas as pd

LABELS_PATH = "labels.jsonl"

# ------------------------------------------------------------------ loading

def parse_filename(fn):
    """Infer run metadata from our output naming conventions."""
    stem = Path(fn).stem
    meta = {"run": stem, "task": "serial", "wording": "default", "phase": ""}
    for pre, phase in [("grid2_", "depth-sweep"), ("grid_", "depth-sweep-v1"),
                       ("A_", "wording-control"), ("B_", "task-compare"),
                       ("W_", "width-sweep"), ("smoke", "smoke")]:
        if stem.startswith(pre):
            meta["phase"] = phase
            rest = stem[len(pre):]
            break
    else:
        rest = stem
    for t in ["parallel_count", "parallel", "serial"]:
        if rest.startswith(t + "_"):
            meta["task"] = t
            rest = rest[len(t) + 1:]
            break
    for w in ["default", "explicit"]:
        if rest.startswith(w + "_"):
            meta["wording"] = w
            rest = rest[len(w) + 1:]
            break
    meta["model"] = rest
    return meta

def load(glob_pat="*.csv"):
    frames = []
    for fn in sorted(globmod.glob(glob_pat)):
        if Path(fn).name in ("labels.jsonl",):
            continue
        try:
            d = pd.read_csv(fn)
        except Exception:
            continue
        if "completion" not in d.columns or "condition" not in d.columns:
            continue
        meta = parse_filename(fn)
        for k, v in meta.items():
            if k not in d.columns or k in ("model", "run", "phase"):
                d[k] = v
        d["file"] = fn
        d["row"] = range(len(d))
        frames.append(d)
    if not frames:
        sys.exit(f"no result CSVs matched {glob_pat!r}")
    df = pd.concat(frames, ignore_index=True)
    # new pandas str dtype preserves NA through astype(str); empty API responses
    # must become "" or every downstream regex breaks
    df["completion"] = df["completion"].astype(object).fillna("").astype(str)
    return add_tags(df)

# ------------------------------------------------------------------ tagging

CODE_RE = re.compile(r"\b[A-Z]{2}\b")
CHAIN_RE = re.compile(r"->|→|[Ss]tep\s*\d|\bthen\b")

def add_tags(df):
    """Attach behavioral tags to every transcript. Cheap, purely local rules."""
    c = df["completion"]
    df["out_len"] = c.str.len()
    df["n_codes"] = c.apply(lambda s: len(CODE_RE.findall(s)))
    df["has_chain_markup"] = c.str.contains(CHAIN_RE, regex=True, na=False)

    df = df.reset_index(drop=True)
    c = df["completion"]
    tags = [[] for _ in range(len(df))]
    def mark(mask, name):
        for i in df.index[mask.fillna(False).astype(bool)]:
            tags[i].append(name)

    is_cot = df["condition"] == "cot"
    forced = ~is_cot

    mark(df["error"] == 1, "api_error")
    mark((df["out_len"] == 0) & (df["error"] == 0), "empty_response")
    mark(df["pred"].isna() & (df["error"] == 0), "unparseable")
    # a CoT trial that produced no visible reasoning is not a CoT trial
    mark(is_cot & (df["out_len"] <= 15) & (df["error"] == 0), "cot_no_reasoning")
    mark(is_cot & ~df["has_chain_markup"] & (df["out_len"] > 15), "cot_prose_only")
    # forced-answer trials that leaked reasoning
    mark(forced & (df["out_len"] > 25), "forced_verbose")
    mark(forced & (df["n_codes"] > 1), "forced_multi_code")
    # near the max_tokens cap for forced conditions (8 tokens ~ 24-32 chars)
    mark(forced & df["out_len"].between(26, 40), "possible_truncation")

    if "eff_hops" in df.columns and "k" in df.columns:
        e, k = df["eff_hops"], df["k"]
        serial = df["task"] == "serial"
        mark(serial & (e < 0) & (df["error"] == 0), "off_cycle")
        mark(serial & (e > k), "overshoot")
        mark(serial & (e >= 0) & (e < k), "undershoot")
        mark(serial & (e == k + 1), "overshoot_by_one")
        mark(serial & (e == k), "exact")

    df["tags"] = [",".join(t) for t in tags]
    return attach_labels(df)

# ------------------------------------------------------------ manual labels

def attach_labels(df):
    df["labels"] = ""
    if not os.path.exists(LABELS_PATH):
        return df
    lab = {}
    with open(LABELS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            lab.setdefault((r["file"], r["row"]), []).append(r["tag"])
    if lab:
        df["labels"] = [",".join(sorted(set(lab.get((f, r), []))))
                        for f, r in zip(df["file"], df["row"])]
    return df

def write_labels(rows, tag, note):
    with open(LABELS_PATH, "a") as f:
        for _, r in rows.iterrows():
            f.write(json.dumps({"file": r["file"], "row": int(r["row"]),
                                "tag": tag, "note": note}) + "\n")

# ------------------------------------------------------------------ audit

def audit(df, min_cot_reasoning=0.80, min_cot_acc=0.90, max_leak=0.05,
          max_err=0.02, min_parse=0.98):
    """Per-(run, condition, k) validity checks. These are preconditions for the
    depth measurement to mean anything, not results."""
    out = []
    for (run, model, cond, k), g in df.groupby(["run", "model", "condition", "k"]):
        n = len(g)
        rec = {"run": run, "model": model, "cond": cond, "k": k, "n": n}
        fails = []
        err = (g["error"] == 1).mean()
        if err > max_err:
            fails.append(f"api_errors={err:.0%}")
        ok = g[g["error"] == 0]
        parse = ok["pred"].notna().mean() if len(ok) else 1.0
        if parse < min_parse:
            fails.append(f"unparseable={1-parse:.0%}")
        if cond == "cot":
            wrote = (~ok["tags"].str.contains("cot_no_reasoning")).mean() if len(ok) else 1.0
            rec["cot_reasoned"] = round(wrote, 3)
            if wrote < min_cot_reasoning:
                fails.append(f"cot_reasoning_rate={wrote:.0%}")
            acc = ok["correct"].mean() if len(ok) else 0
            rec["acc"] = round(acc, 3)
            if acc < min_cot_acc:
                fails.append(f"cot_ceiling={acc:.0%}")
        else:
            leak = ok["tags"].str.contains("forced_verbose|forced_multi_code").mean() if len(ok) else 0
            rec["leak"] = round(leak, 3)
            rec["acc"] = round(ok["correct"].mean(), 3) if len(ok) else float("nan")
            if leak > max_leak:
                fails.append(f"leak={leak:.0%}")
        rec["status"] = "FAIL" if fails else "ok"
        rec["why"] = "; ".join(fails)
        out.append(rec)
    return pd.DataFrame(out)

# ------------------------------------------------------------------ display

def show(df, n, width=220):
    cols = ["model", "task", "condition", "k", "gold", "pred", "correct"]
    cols = [c for c in cols if c in df.columns]
    for _, r in df.head(n).iterrows():
        head = "  ".join(f"{c}={r[c]}" for c in cols)
        print(f"\n\033[1m{head}\033[0m")
        if r.get("tags"):
            print(f"  tags: {r['tags']}")
        if r.get("labels"):
            print(f"  labels: {r['labels']}")
        print(f"  {r['file']}:{r['row']}")
        body = r["completion"].replace("\n", "\n  ")
        print(f"  {body[:width]}{'…' if len(r['completion']) > width else ''}")

def apply_filters(df, args, use_tag_filter=True):
    if args.where:
        df = df.query(args.where)
    if args.tag and use_tag_filter:
        df = df[df["tags"].str.contains(args.tag, regex=False)]
    if args.label:
        df = df[df["labels"].str.contains(args.label, regex=False)]
    if args.regex:
        df = df[df["completion"].str.contains(args.regex, regex=True, na=False)]
    if args.model:
        df = df[df["model"].str.contains(args.model, regex=False)]
    return df

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["audit", "tags", "search", "sample", "label"])
    ap.add_argument("--glob", default="*.csv")
    ap.add_argument("--where", help="pandas query, e.g. \"k>=4 and correct==0\"")
    ap.add_argument("--tag", help="filter by auto-tag; for `label`, the tag to write")
    ap.add_argument("--label", help="filter by manual label")
    ap.add_argument("--regex", help="regex over the completion text")
    ap.add_argument("--model", help="substring match on model name")
    ap.add_argument("--note", default="", help="note to store with a manual label")
    ap.add_argument("-n", type=int, default=10)
    ap.add_argument("--all", action="store_true", help="audit: show ok cells too")
    args = ap.parse_args()

    df = load(args.glob)

    if args.cmd == "audit":
        a = audit(df)
        bad = a[a.status == "FAIL"]
        cols = [c for c in ["run","model","cond","k","n","acc","cot_reasoned","leak","why"]
                if c in a.columns]
        if args.all:
            print(a[cols].to_string(index=False))
        elif len(bad):
            print(f"{len(bad)} of {len(a)} cells FAILED validity checks:\n")
            print(bad[cols].to_string(index=False))
        else:
            print(f"all {len(a)} cells passed validity checks")
        print(f"\nchecks: CoT must show reasoning >=80% of trials and reach >=90% accuracy; "
              f"forced-answer leak <=5%; api errors <=2%; parse rate >=98%")
        sys.exit(1 if len(bad) else 0)

    # for `label`, --tag names the label being written, not a filter
    df = apply_filters(df, args, use_tag_filter=(args.cmd != "label"))

    if args.cmd == "tags":
        rows = []
        for t in sorted({t for s in df["tags"] for t in s.split(",") if t}):
            sub = df[df["tags"].str.contains(t, regex=False)]
            rows.append({"tag": t, "n": len(sub), "pct": round(100*len(sub)/len(df), 1),
                         "acc": round(sub["correct"].mean(), 3)})
        print(f"{len(df)} transcripts matched\n")
        print(pd.DataFrame(rows).sort_values("n", ascending=False).to_string(index=False))

    elif args.cmd in ("search", "sample"):
        print(f"{len(df)} transcripts matched; showing {min(args.n, len(df))}")
        show(df, args.n, width=600 if args.cmd == "sample" else 220)

    elif args.cmd == "label":
        if not args.tag:
            sys.exit("--tag is required for label")
        if not len(df):
            sys.exit("no rows matched; refusing to write an empty label")
        write_labels(df, args.tag, args.note)
        print(f"labelled {len(df)} transcripts as '{args.tag}' -> {LABELS_PATH}")

if __name__ == "__main__":
    main()
