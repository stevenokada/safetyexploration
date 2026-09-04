"""
Assemble report.html from every archived version, one tab per run.

An Artifact URL shows only the latest publish, so a re-run would otherwise bury
the experiments it replaces. This builds a single page that carries all of them:
a tab per version, each with its own narrative, its own data, and its own
provenance line. Old experiments stay reachable at the same URL.

Inputs
  assets/report.css, assets/charts.js, assets/masthead.html, assets/footer.html
  reports/manifest.json                    version list, newest last
  reports/vNN/narrative.html               that version's prose + chart mounts
  reports/vNN/report_data.json             that version's numbers

Usage
  python3 build_report.py                  rebuild report.html from the archive
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
REPORTS = ROOT / "reports"


def check_version(vid, narrative, data, legacy):
    """Every experiment page must ship with graphs AND a transcript browser.

    These are the two things that let a reader check a claim instead of trusting
    it, and both were added late and by hand before this gate existed. Versions
    predating the requirement are marked `legacy` in the manifest and only warn.
    """
    problems = []
    if not re.search(r'data-fig="[^"]+"', narrative):
        problems.append("no chart mounts (data-fig=...) in the narrative")
    if 'data-tx="controls"' not in narrative or 'data-tx="list"' not in narrative:
        problems.append('no transcript browser (needs data-tx="controls" and data-tx="list")')
    n_tx = len(data.get("transcripts") or [])
    if n_tx == 0:
        problems.append("no transcripts in report_data.json "
                        "(collect them with pilot2.py, which stores prompts)")
    elif n_tx < 10:
        problems.append(f"only {n_tx} transcripts; aim for >=10 across conditions")
    # a chart mount nothing fills renders as an empty box
    mounts = set(re.findall(r'data-(?:fig|leg|tbl)="([^"]+)"', narrative))
    renderer = (ASSETS / "charts.js").read_text()
    unfilled = [m for m in mounts if f'"{m}"' not in renderer]
    if unfilled:
        problems.append(f"mounts with no renderer: {', '.join(sorted(unfilled))}")
    if not problems:
        return True
    tag = "WARN (legacy)" if legacy else "FAIL"
    print(f"  {tag} {vid}:")
    for x in problems:
        print(f"      - {x}")
    return legacy


def main():
    manifest = json.loads((REPORTS / "manifest.json").read_text())
    versions = manifest["versions"]
    if not versions:
        raise SystemExit("no versions staged yet")

    tabs, panels, data = [], [], {}
    # newest first: the current run is what a reader should land on
    for i, v in enumerate(reversed(versions)):
        vid = f"v{v['version']:02d}"
        vdir = REPORTS / vid
        narrative = (vdir / "narrative.html")
        if not narrative.exists():
            continue
        data[vid] = json.loads((vdir / "report_data.json").read_text())
        current = i == 0
        label = v.get("tab_label") or vid
        tabs.append(
            f'<button role="tab" id="tab-{vid}" aria-controls="panel-{vid}" '
            f'aria-selected="{"true" if current else "false"}" data-tab="{vid}">'
            f'<span class="tv">{label}</span>'
            f'<span class="td">{v["date"][:10]} · {v["total_rows"]:,} trials</span></button>')
        panels.append(
            f'<div role="tabpanel" id="panel-{vid}" aria-labelledby="tab-{vid}" '
            f'data-panel="{vid}"{"" if current else " hidden"}>\n'
            f'<p class="provenance">{"Current run" if current else "Archived run"} · '
            f'{v["date"][:10]} · commit <code>{v["git_commit"]}</code> · '
            f'{v["total_rows"]:,} logged trials across {v["n_files"]} result files<br>'
            f'{v["summary"]}</p>\n'
            + narrative.read_text() + "\n</div>")

    print("checking every version ships graphs + a transcript browser ...")
    ok = True
    for v in versions:
        vid = f"v{v['version']:02d}"
        nar = REPORTS / vid / "narrative.html"
        dat = REPORTS / vid / "report_data.json"
        if not nar.exists():
            continue
        ok &= check_version(vid, nar.read_text(), json.loads(dat.read_text()),
                            bool(v.get("legacy")))
    if not ok:
        sys.exit("\nrefusing to build: an experiment page is missing graphs or "
                 "transcripts. Add them, or mark the version \"legacy\": true in "
                 "reports/manifest.json if it predates this requirement.")
    print("  all versions pass\n")

    notes = REPORTS / "notes.html"
    if notes.exists():
        tabs.append('<button role="tab" id="tab-notes" aria-controls="panel-notes" '
                    'aria-selected="false" data-tab="notes">'
                    '<span class="tv">Bugs &amp; corrections</span>'
                    '<span class="td">applies to all runs</span></button>')
        panels.append('<div role="tabpanel" id="panel-notes" aria-labelledby="tab-notes" '
                      'data-panel="notes" hidden>\n' + notes.read_text() + "\n</div>")

    html = f"""<title>Opaque Serial Depth</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,600;1,6..72,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">

<style>
{(ASSETS / "report.css").read_text()}

/* ---------- run tabs ---------- */
.tabs{{display:flex; gap:6px; flex-wrap:wrap; border-bottom:1px solid var(--rule);
  margin:0 0 30px; padding-bottom:0}}
.tabs button{{appearance:none; background:none; border:0; border-bottom:2px solid transparent;
  padding:10px 14px 11px; cursor:pointer; text-align:left; color:var(--muted);
  font-family:"IBM Plex Sans",sans-serif; display:flex; flex-direction:column; gap:2px;
  border-radius:3px 3px 0 0}}
.tabs button:hover{{background:var(--surface-2); color:var(--ink-2)}}
.tabs button[aria-selected="true"]{{color:var(--ink); border-bottom-color:var(--serial)}}
.tabs button:focus-visible{{outline:2px solid var(--parallel); outline-offset:-2px}}
.tabs .tv{{font-weight:600; font-size:14.5px}}
.tabs .td{{font-family:"IBM Plex Mono",monospace; font-size:11px; letter-spacing:.03em}}
.provenance{{font-family:"IBM Plex Mono",monospace; font-size:11.5px; line-height:1.7;
  color:var(--muted); background:var(--surface-2); border:1px solid var(--rule);
  border-radius:4px; padding:12px 16px; margin:0 0 34px; max-width:660px;
  margin-left:auto; margin-right:auto}}
.provenance code{{background:none; padding:0}}
</style>

<div class="wrap">
{(ASSETS / "masthead.html").read_text()}
<div class="tabs" role="tablist" aria-label="Experiment runs">
{chr(10).join(tabs)}
</div>

{chr(10).join(panels)}

{(ASSETS / "footer.html").read_text()}
</div>

<script id="data" type="application/json">{json.dumps(data)}</script>
<script>
const ALL = JSON.parse(document.getElementById('data').textContent);

{(ASSETS / "charts.js").read_text()}

const rendered = new Set();
function activate(vid, push){{
  document.querySelectorAll('[data-panel]').forEach(p => {{
    const on = p.dataset.panel === vid;
    p.hidden = !on;
  }});
  document.querySelectorAll('[data-tab]').forEach(b =>
    b.setAttribute('aria-selected', String(b.dataset.tab === vid)));
  if (!rendered.has(vid)) {{           // charts draw once, on first view
    const root = document.querySelector(`[data-panel="${{vid}}"]`);
    if (root && ALL[vid]) renderReport(root, ALL[vid]);
    rendered.add(vid);
  }}
  if (push) history.replaceState(null, '', '#' + vid);
}}

document.querySelectorAll('[data-tab]').forEach(b =>
  b.addEventListener('click', () => activate(b.dataset.tab, true)));

const initial = location.hash.slice(1);
const first = document.querySelector('[data-tab]').dataset.tab;
activate(ALL[initial] ? initial : first, false);
</script>
"""
    (ROOT / "report.html").write_text(html)
    kb = len(html) / 1024
    print(f"built report.html with {len(panels)} run tab(s), {kb:.0f} KB")
    for v in reversed(versions):
        print(f"  v{v['version']:02d}  {v['date'][:10]}  {v['total_rows']:,} trials")


if __name__ == "__main__":
    main()
