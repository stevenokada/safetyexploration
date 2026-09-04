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
