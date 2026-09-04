"""
Version and archive a report before publishing it as an Artifact.

Why: an Artifact URL always shows the LATEST publish. Without an archive, an
earlier version of a finding is unrecoverable, and a published page carries no
record of which data or which code produced it. This script makes each publish
reproducible: it snapshots the HTML and the exact data behind it, stamps the
page with a version and git commit, and records the artifact URL in a manifest.

Workflow
  1. python3 publish_report.py stage --summary "what changed in this version"
       -> archives reports/vNN/, stamps report.html, prints the next step
  2. publish report.html via the Artifact tool (same file path keeps the URL)
  3. python3 publish_report.py record --url <artifact-url>
       -> writes the URL into the manifest for vNN

Other commands
  log                 list every version with date, commit, and summary
  show --version N    print one version's metadata
  diff --version N    compare a past version's headline numbers to current data
"""
import argparse, hashlib, json, os, re, shutil, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
MANIFEST = REPORTS / "manifest.json"
REPORT_HTML = ROOT / "report.html"
REPORT_DATA = ROOT / "report_data.json"
STAMP_MARK = "<!--version-stamp-->"


def git(*args, default=""):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return default


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {"artifact_url": None, "versions": []}


def save_manifest(m):
    REPORTS.mkdir(exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n")


def data_fingerprint():
    """Which result files fed this report, and their content hashes. Lets a past
    version be checked against today's data files for silent drift."""
    files = {}
    for p in sorted(ROOT.glob("*.csv")):
        if p.name.startswith(("grid", "A_", "B_", "W_", "W2_", "J_", "JW_",
                              "F_", "cotlen", "ar_")):
            h = hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            n = sum(1 for _ in p.open()) - 1
            files[p.name] = {"sha256": h, "rows": n}
    return files


def headline_numbers():
    """A few numbers worth diffing across versions, pulled from report_data."""
    if not REPORT_DATA.exists():
        return {}
    d = json.loads(REPORT_DATA.read_text())
    out = {}
    try:
        for model, m in d.get("taskcmp", {}).items():
            out[f"{model} serial k=8"] = m["serial"]["immediate"]["acc"][-1]
            out[f"{model} parallel k=8"] = m["parallel"]["immediate"]["acc"][-1]
        for model, m in d.get("width", {}).items():
            out[f"{model} parallel k=24"] = m["parallel"]["acc"][-1]
        w = d.get("wording", {})
        if w:
            out["overshoot k=4 (default)"] = w["default"]["over"][3]
    except (KeyError, IndexError):
        pass
    return out


def _unused_stamp_html(html, ver, meta):
    """Insert a provenance line into the report footer. Idempotent: an existing
    stamp is replaced, so re-staging never accumulates stamps."""
    stamp = (f'{STAMP_MARK}<p style="font-family:\'IBM Plex Mono\',monospace;'
             f'font-size:12px;opacity:.75">'
             f'Version {ver} · {meta["date"][:10]} · commit {meta["git_commit"]} · '
             f'{meta["total_rows"]:,} logged trials across {meta["n_files"]} result files'
             f'</p>{STAMP_MARK}')
    pat = re.compile(re.escape(STAMP_MARK) + ".*?" + re.escape(STAMP_MARK), re.S)
    if pat.search(html):
        return pat.sub(stamp, html)
    return html.replace("</footer>", stamp + "\n</footer>")


def cmd_stage(args):
    if not REPORT_HTML.exists():
        sys.exit(f"missing {REPORT_HTML}")
    m = load_manifest()
    ver = len(m["versions"]) + 1
    # capture git state BEFORE staging writes anything, otherwise the stamp and
    # the archive dirty the tree and every version reports itself as dirty
    # scope to this directory: unrelated files elsewhere in the repo do not make
    # the report's provenance unreproducible
    dirty = bool(git("status", "--porcelain", "--", str(ROOT)))
    fp = data_fingerprint()
    meta = {
        "version": ver,
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "summary": args.summary,
        "tab_label": args.tab_label,
        "git_commit": git("rev-parse", "--short", "HEAD", default="uncommitted"),
        "git_dirty": dirty,
        "git_branch": git("branch", "--show-current", default="?"),
        "n_files": len(fp),
        "total_rows": sum(v["rows"] for v in fp.values()),
        "data_files": fp,
        "headline": headline_numbers(),
        "artifact_url": None,
    }

    vdir = REPORTS / f"v{ver:02d}"
    vdir.mkdir(parents=True, exist_ok=True)
    if REPORT_DATA.exists():
        shutil.copy2(REPORT_DATA, vdir / "report_data.json")
    narrative = vdir / "narrative.html"
    if not narrative.exists():
        # a new version needs its own prose; start from the previous version's
        prev = REPORTS / f"v{ver-1:02d}" / "narrative.html"
        if prev.exists():
            shutil.copy2(prev, narrative)
            print(f"  NOTE: copied v{ver-1:02d} narrative as a starting point — "
                  f"edit {narrative.relative_to(ROOT)} to describe THIS run")
        else:
            sys.exit(f"create {narrative} first (the prose for this version)")
    (vdir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")

    m["versions"].append(meta)
    save_manifest(m)

    # rebuild the multi-tab page so every archived run stays reachable
    import build_report
    build_report.main()
    shutil.copy2(REPORT_HTML, vdir / "report.html")

    print(f"staged v{ver:02d} -> {vdir.relative_to(ROOT)}")
    print(f"  {meta['total_rows']:,} trials across {meta['n_files']} files, "
          f"commit {meta['git_commit']}{' (dirty)' if meta['git_dirty'] else ''}")
    if meta["git_dirty"]:
        print("  NOTE: working tree has uncommitted changes; commit first for a "
              "reproducible stamp, then re-run stage.")
    print(f"\nnext: publish {REPORT_HTML.name} with the Artifact tool"
          + (f" using url={m['artifact_url']}" if m["artifact_url"] else " (new artifact)")
          + f"\nthen: python3 publish_report.py record --url <artifact-url>")


def cmd_record(args):
    m = load_manifest()
    if not m["versions"]:
        sys.exit("nothing staged yet")
    v = m["versions"][-1]
    v["artifact_url"] = args.url
    m["artifact_url"] = args.url
    save_manifest(m)
    vdir = REPORTS / f"v{v['version']:02d}"
    (vdir / "meta.json").write_text(json.dumps(v, indent=2) + "\n")
    print(f"recorded v{v['version']:02d} -> {args.url}")


def cmd_log(args):
    m = load_manifest()
    if not m["versions"]:
        sys.exit("no versions yet")
    print(f"artifact: {m['artifact_url'] or '(unpublished)'}\n")
    for v in m["versions"]:
        flag = " *dirty" if v.get("git_dirty") else ""
        print(f"v{v['version']:02d}  {v['date'][:10]}  {v['git_commit']}{flag}  "
              f"{v['total_rows']:,} trials")
        print(f"      {v['summary']}")


def cmd_show(args):
    m = load_manifest()
    hits = [v for v in m["versions"] if v["version"] == args.version]
    if not hits:
        sys.exit(f"no version {args.version}")
    print(json.dumps(hits[0], indent=2))


def cmd_diff(args):
    m = load_manifest()
    hits = [v for v in m["versions"] if v["version"] == args.version]
    if not hits:
        sys.exit(f"no version {args.version}")
    old, new = hits[0]["headline"], headline_numbers()
    print(f"v{args.version:02d} ({hits[0]['date'][:10]})  vs  current data\n")
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if a is None:
            print(f"  + {k}: {b} (new)")
        elif b is None:
            print(f"  - {k}: {a} (gone)")
        elif abs(a - b) > 1e-9:
            print(f"  ~ {k}: {a} -> {b}")
    # data drift: files whose contents changed since that version
    oldf, newf = hits[0]["data_files"], data_fingerprint()
    changed = [f for f in set(oldf) & set(newf)
               if oldf[f]["sha256"] != newf[f]["sha256"]]
    if changed:
        print(f"\n  {len(changed)} result file(s) changed on disk since v{args.version:02d}:")
        for f in sorted(changed):
            print(f"    {f}")
    for f in sorted(set(newf) - set(oldf)):
        print(f"    + {f} (added since)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("stage"); s.add_argument("--summary", required=True)
    s.add_argument("--tab-label", required=True, dest="tab_label",
                   help="short tab name, e.g. 'J-Lens · 3 models'")
    s.set_defaults(f=cmd_stage)
    s = sub.add_parser("record"); s.add_argument("--url", required=True); s.set_defaults(f=cmd_record)
    s = sub.add_parser("log"); s.set_defaults(f=cmd_log)
    s = sub.add_parser("show"); s.add_argument("--version", type=int, required=True); s.set_defaults(f=cmd_show)
    s = sub.add_parser("diff"); s.add_argument("--version", type=int, required=True); s.set_defaults(f=cmd_diff)
    args = ap.parse_args()
    args.f(args)


if __name__ == "__main__":
    main()
