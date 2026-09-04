# Opaque serial depth

How many reasoning steps can a language model chain together **without writing
anything down**? Chain-of-thought monitoring assumes that reasoning a model does
not verbalise, it cannot do. This measures that assumption behaviourally, on both
synthetic and real-world tasks, and compares the answer to what the architecture
theoretically permits.

**Report with charts and searchable transcripts:**
<https://claude.ai/code/artifact/17c0cd78-a8b1-4378-b7ea-24daa7337a1b>

---

## Headline findings

**Depth is the binding constraint, not the amount of work.** Given the same
number of lookups, chaining them collapses while performing them independently
does not. On the synthetic lookup task, serial accuracy at four or more hops is
0.169 against 0.836 for matched-work parallel (n=360 per arm). On real-world
fact composition, four chained retrievals give 0.017 against 0.450 for the same
four retrievals unchained.

**Parallel width survives far past serial depth.** Independent lookups hold up to
k=64 without breaking — one model is flat at 0.60 from k=8 to k=64 while chance
falls eightfold — whereas chained lookups are at the floor by k=4.

**Models use a small fraction of the depth their architecture allows.** Running
DeepMind's [serial_depth](https://github.com/google-deepmind/serial_depth)
calculator on each model's architecture gives bounds around 10,000 serial steps
for the dense 27B models and about 6,400 for the mixture-of-experts model — the
latter independently reproducing the published claim that MoE permits *less*
hidden serial computation. Behaviourally the models realise a handful of steps.

**Filler tokens buy reliability, not reach.** Given enough budget, filler
improves accuracy at the depth a model can nearly manage (+27 points at three
hops on the fact task) and does little beyond it. It never moves the ceiling.
The effect saturates: more filler past roughly 1,300 tokens changes nothing.

**Aggregation cost is a confound that masquerades as depth.** Two parallel
controls failed because combining the results was itself expensive — summing k
two-digit numbers, and ordering k strings. Swapping to a single-character test
over the same retrievals moved the model from chance to a steady margin above it.

---

## Start here

```bash
./reproduce.sh analysis    # re-derive every published number from the committed data
./reproduce.sh audit       # validity-check every collected cell
./reproduce.sh smoke       # collect a small fresh sample (needs an API key)
./reproduce.sh report      # rebuild the HTML report
```

`reproduce.sh analysis` needs no API key and takes seconds: it recomputes the
headline figures from the CSVs in `data/`, printing the test used for each.

## Repository layout

```
reproduce.sh              the four things you are likely to want
src/
  harness.py              runs prompts: builds them, calls the API, scores, writes a CSV
  analysis.py             turns those CSVs back into the published numbers
  audit.py                validity checks + transcript search and tagging
  tokenization.py         precondition check for the interpretability work
  models.json             permitted models (open weights with a published R-lens)
  tasks/
    arithmetic.py         chained arithmetic word problems + parallel controls
    facts.py              real-world fact chains to 6 hops + parallel control
  lens/                   reading intermediates out of the residual stream, on a GPU
experiments/              one script per sweep, numbered in the order they were run
data/                     every logged trial, with full prompts and completions
report/                   report builder, archived versions, published HTML
docs/
  BUGS.md                 every defect found, and whether it changed a conclusion
  gpu_setup.md            the GPU session recipe for the lens work
```

The chained-lookup task is defined inside `src/harness.py`; the other two task
families live in `src/tasks/`.

## Tasks

| task | shape | depth | notes |
|---|---|---|---|
| `serial` | chained lookups over a random in-context cycle | k | contamination-proof, constant prompt length |
| `parallel` / `parallel_count` | k independent lookups + aggregation | 2 | matched work, tested to k=64 |
| `arith` | chained arithmetic word problem, parity-branching | k | intermediates constrained to be lens-readable |
| `arith_parallel_max` | k independent depots + cheap comparison | 2 | |
| `facts` | real-world fact chain, constant answer type | 1–6 | memorised facts, nothing given in context |
| `facts_parallel` | k independent retrievals + one-letter test | 2 | tested to k=16 |

Every task carries a **chain-of-thought condition as a validity baseline**. If a
model cannot solve a depth when allowed to reason aloud, its no-CoT number at that
depth means nothing. This caught several would-be findings, including an apparent
six-hop ceiling that was in fact a token-budget truncation.

## Running it

```bash
echo 'sk-or-v1-...' > ~/.openrouter_key && chmod 600 ~/.openrouter_key

cd pilot
python3 pilot2.py --task facts --model qwen/qwen3.6-27b \
    --conditions immediate,filler,cot --n 120 --ks 1,2,3,4,5,6 --out my_run

python3 inspect_runs.py audit          # validity checks; exits nonzero on failure
python3 inspect_runs.py sample --tag cot_no_reasoning -n 5
```

Scoring is **entirely algorithmic** — a regex extraction and an exact string
match. No LLM judge anywhere in the pipeline. Ground truth comes from forward
construction and is cross-checked by independent solvers that parse the rendered
prompt.

## On the bug log

[`pilot/BUGS.md`](pilot/BUGS.md) records every defect found, what it affected, how
it surfaced, and whether a reported conclusion moved. It is long, and eight
entries changed a result that had already been reported.

The pattern worth noting: the costly failures were all **silent**. A filler sweep
that made zero API calls and logged empty results. A correctness flag that scored
every trial as failed. Data analysed as paired that was not. Few-shot examples
that handed the model its own answer in 41% of trials. None raised an error; each
produced plausible-looking output.

Two guardrails came out of that and now run automatically: a **validity audit**
after every sweep, and a **build gate** that refuses to publish a report page
lacking charts or transcripts.

## Limits

- Behavioural ceilings are **lower bounds under one prompting scheme**. Better
  elicitation, or training pressure to hide reasoning, could raise them. Nothing
  here is a safety guarantee.
- Most per-cell samples are n=30–120: fine for a 67-point effect, useless for a
  5-point one.
- The architectural bounds are replica-based; only Gemma 3 27B has a published
  figure to check against, and the replica lands within 11% of it.
- Reading intermediates from the residual stream succeeded on word-valued
  intermediates and found nothing on digit-valued ones, consistent with the lens
  authors' own warning about small integers. That result is correlational; a
  causal test by activation transplant has not been run.
