"""
Collect transcripts for an experiment page. Every version must ship with these:
the build refuses otherwise.

  python3 collect_transcripts.py --version 4 --task facts \
      --model google/gemma-3-27b-it --cells "immediate:2 filler:4 cot:2"

Reads back the prompt and completion that pilot2.py now stores per row, and
merges them into that version's report_data.json.
"""
import argparse, glob, json, os, pathlib, subprocess, sys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--cells", required=True,
                    help='space-separated condition:k pairs, e.g. "immediate:2 cot:3"')
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    import pandas as pd
    root = pathlib.Path(__file__).resolve().parent
    vdir = root / "reports" / f"v{args.version:02d}"
    if not vdir.exists():
        sys.exit(f"{vdir} does not exist -- stage the version first")

    rows = []
    for cell in args.cells.split():
        cond, k = cell.split(":")
        out = root / f"_tx_{args.task}_{cond}_k{k}"
        subprocess.run([sys.executable, "pilot2.py", "--task", args.task,
                        "--model", args.model, "--conditions", cond, "--ks", k,
                        "--n", str(args.n), "--out", str(out)],
                       cwd=root, check=True, capture_output=True)
        d = pd.read_csv(f"{out}.csv")
        for _, r in d.iterrows():
            p = str(r.get("prompt", ""))
            if p in ("", "nan"):
                continue
            rows.append({"task": args.task, "cond": cond, "k": int(k),
                         "model": str(r.get("model", "")), "gold": str(r["gold"]),
                         "pred": "" if pd.isna(r["pred"]) else str(r["pred"]),
                         "ok": int(r["correct"]),
                         "eff": int(r["eff_hops"]) if "eff_hops" in r
                                and r["eff_hops"] == r["eff_hops"] else None,
                         "toks": int(r.get("out_tokens", -1)),
                         "prompt": p[:3000], "completion": str(r["completion"])[:2500]})
        os.remove(f"{out}.csv")

    f = vdir / "report_data.json"
    data = json.loads(f.read_text()) if f.exists() else {}
    data["transcripts"] = rows
    f.write_text(json.dumps(data))
    print(f"wrote {len(rows)} transcripts into {f.relative_to(root)}")
    print("remember the narrative also needs the browser mounts:")
    print('  <div class="tx-controls" data-tx="controls"></div>\\n  <div data-tx="list"></div>')

if __name__ == "__main__":
    main()
