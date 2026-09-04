# Bug and correction log

Every defect found in this project, what it affected, how it surfaced, and whether
any published conclusion changed. Kept because a result is only as trustworthy as
the instrument, and several of these silently produced plausible-looking numbers.

Status key: **CHANGED** = a reported conclusion moved. **CAUGHT** = found before it
reached a conclusion.

| # | Bug | Effect | Found by | Status |
|---|-----|--------|----------|--------|
| 1 | Only the chain up to hop `k` was stored, so answers landing past `k` were recorded as "off-cycle" rather than overshoot | The central overshoot finding was invisible; wrong answers looked random | Inspecting individual predictions against the full cycle | **CHANGED** — overshoot became a headline result |
| 2 | Provider-side reasoning was disabled in *all* conditions, including chain-of-thought | The CoT validity baseline was not a real CoT condition | Reviewing per-condition token counts | **CHANGED** — CoT baselines re-run |
| 3 | GPT-5.1 ignored the reason-first instruction on ~50% of trials (model behaviour, not our code) | Its CoT baseline was a second immediate condition; its depth numbers were uninterpretable | Reading transcripts after an odd 50% ceiling | **CHANGED** — model excluded from depth claims |
| 4 | pandas' `str` dtype preserves NA through `astype(str)`, so empty API responses stayed null | Tagging crashed; 6 empty Kimi responses had been silently scored as wrong answers | Crash while building the transcript tagger | **CAUGHT** |
| 5 | `--tag` in `inspect_runs.py label` was used both as a filter and as the label to write | The label command matched zero rows and wrote nothing | Trying to label the GPT-5.1 trials | **CAUGHT** |
| 6 | `publish_report.py` read git state *after* staging had written files | Every version stamped itself "dirty" | First staging run | **CAUGHT** |
| 7 | Version fingerprint matched only the original filename prefixes | New result files were invisible; v02 undercounted 26,730 trials as 24,914 | Staging reported an implausible trial count | **CAUGHT** |
| 8 | `--filler-n` was passed to `pilot2.py`, which had no such flag | The entire first CoT-matched filler sweep made zero API calls; the log looked like empty results rather than an error | Empty summary rows in the log | **CAUGHT** |
| 9 | Seeds came from `hash()` on a tuple containing strings, and Python randomizes string hashing per process | Prompts were unreproducible, and any two conditions run as separate invocations were **not paired** despite the harness appearing to pair them | Checking whether prompts could be regenerated for the transcript browser | **CHANGED** — filler-sweep and few-shot McNemar tests recomputed as unpaired |
| 10 | Lens loader assumed `J` was a stacked 3-D tensor; the shipped file stores a dict keyed by layer | Would have crashed on first use | Running `inspect_lens.py` before the main run | **CAUGHT** |
| 11 | Full-vocabulary `argsort` (248k entries) at every layer × position | ~3,000 sorts per prompt; would have turned a 10-minute run into hours | Estimating runtime before launching | **CAUGHT** |
| 12 | Lens correctness used the token after the `Answer:` prefill, but Qwen emits a *space* token first | Every trial scored unsolved (0% vs the true ~40%) | "solved: 0.0%" contradicting the behavioural rate | **CAUGHT** |
| 13 | Serial-depth replica used `hidden_dim // num_heads`, and 5120/24 is not an integer | Qwen dense bound failed silently — the error was swallowed by a `grep` in the runner | Missing rows in the bounds table | **CAUGHT** |
| 14 | Depot name "Foxtrot" tokenizes as `Fo|xt|rot` | One of eight possible answers would have been unreadable by a single-token lens | Tokenization precondition check | **CAUGHT** |
| 15 | The first parallelizable control aggregated by **sum**, which is expensive to compute mentally | It collapsed like the serial task, which would have read as "the depth effect does not replicate on arithmetic" | Suspicion at a control failing in the same shape as the treatment | **CAUGHT** — added a cheap-aggregation (max) variant |

## Standing methodological caveats

- **Few-shot count was a hidden constant.** Fixed at 3 from the first run and never varied. A 2×2 found 1-shot both more accurate and half as prone to overshoot, so the published overshoot magnitude is partly a prompt artifact. Powered replication in progress.
- **Underpowered cells.** Most per-model, per-depth cells are n=30–40, adequate for a 67-point effect and useless for a 5-point one.
- **Lens used outside its fitted regime.** The R-lens was estimated at `t_max=128` over 25 prompts; our prompts run ~300 tokens.
- **Single-token lens targets are digits**, which the workspace paper flags as a case the method may not read.
| 16 | The facts answer parser matched the literal word "Answer" from the echoed prefill | Every trial scored `pred='Answer'`, so the whole task read as 0% at every depth when the model was in fact ~95% correct at one hop | Inspecting completions after an implausible all-zero column | **CAUGHT** |
| 17 | The answer-format hint said "city name" at every depth, but the chain answers with a letter at k=3, an element at k=4 and a number at k=5 | The model stalled after `Answer:` rather than answer in a format it was told not to use; 30% empty responses at k=4 | Checking `finish_reason` on the empty responses (all `stop`, not `length`) | **CAUGHT** |
| 18 | The audit counted a model that declines to answer as a parse failure | Conflated "we could not read the answer" with "there was no answer", hiding a real behaviour | Reading the raw completions behind an unparseable rate | **CAUGHT** — now tagged `model_declined`, scored wrong, reported separately |
| 19 | Lens correctness compared against the token after an `Answer:` prefill, but Qwen emits a space token first | Every trial scored unsolved (0.0% against a true ~40%), making the solved/unsolved split impossible | A solve rate that contradicted the behavioural runs | **CAUGHT** |

## Standing requirements (enforced by the build)

`build_report.py` refuses to build unless every experiment page carries **graphs**
and a **transcript browser** backed by at least ten real trials, and unless every
chart mount in a narrative has a renderer that fills it. Both were added by hand
and late on earlier versions, which is exactly why they are now a gate rather than
a habit. Use `collect_transcripts.py` to gather the transcripts. v01 is marked
`legacy` in the manifest: it predates prompt storage and deterministic seeds, so
its transcripts cannot be recovered.
| 20 | Chart dispatch keyed on `D.width2`, so the v03 data shape (no `width2`) fell through to the v01 code path and threw on `D.meta.leak_rate` | Every chart and the transcript browser on the newest tab silently failed to render; the page looked fine but was blank below each figure title | User reported graphs not rendering | **CHANGED** — v03 charts were broken from first publish until this fix |
| 21 | Few-shot examples were drawn before the target, so with a small fact pool the target question could appear verbatim in its own few-shot block | The model could copy the answer. 41% of facts trials at every depth; the synthetic tasks were unaffected because their instances are randomly generated | Reading a full k=3 prompt end-to-end while answering a question about what the prompts look like | **CHANGED** — inflates every facts accuracy number collected before this fix, including published v03 |
| 22 | Chain-of-thought capped at 4,000 tokens while deep fact chains reason for 3,500+ | 17/40 responses truncated at k=6, giving CoT accuracy 0.575 — which reads as "the task is too hard at this depth" rather than "the budget was too small" | Audit flagged `unparseable=43%`, prompting a check of `finish_reason` | **CHANGED** — k=6 is in fact 34/34 solvable; the apparent depth ceiling was an artifact |
| 23 | Filler budget matched to per-depth chain-of-thought length, which is not monotonic in depth | Four hops received the smallest budget of any depth (405 tokens) and showed no filler effect, which read as a ceiling. At a comparable budget the effect appears (+0.075, p=0.004) and saturates | Noticing k=6 benefited while k=4 did not, though k=6 has the lower baseline | **CHANGED** — the four-hop null was a dosing artifact |
| 24 | The `--nodes` guard (table size must exceed k) applied to tasks that have no lookup table | Every parallel-control cell at k>=12 exited before making a call; the runner's grep hid the error and the log showed only the cell header | A missing output file after the sweep reported success | **CAUGHT** |
| 25 | An orphaned `pilot2.py` survived `pkill` of its parent script and kept running pre-bugfix code against the same output filename | Would have overwritten clean results with answer-leaking data, undetectably | Checking process list while investigating slow progress | **CAUGHT** — runs now carry a code-fingerprint run id, stamped per row |
| 26 | `publish_report.py stage` copied the working data file over a version's, discarding transcripts already collected for it | Emptied the transcript browser on the version being staged | The build gate refused to build | **CAUGHT** — staging now preserves existing transcripts |
| 27 | Chart dispatch for v04 was reachable only from inside the v03 renderer, which v04's data never routes to | Every chart on the new tab would have been blank, as in bug 20 | The build gate's data-shape check | **CAUGHT** — all dispatch is now top-level and the gate understands nested conditions |
