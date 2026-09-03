# Context-firewalled model-diffing validation

> **Status: preregistered proof-of-concept; results pending.**

This repository tests whether a fresh, rejection-only validator is less likely
to endorse a spurious behavioral difference than the agent that originally
discovered the hypothesis. Both validators receive exactly the same fresh,
hypothesis-targeted confirmation evidence. The only manipulated variable is
whether the validator can see the exploratory context.

The experiment contains eight blinded synthetic audits:

- four null worlds with independently sampled outputs from identical targets;
- four positive controls with a precisely specified conditional response
  rewrite applied to one randomly assigned target.

This is a small pipeline sanity check, not a definitive result and not a
mechanistic-interpretability experiment. Its purpose is to produce an honest,
fully inspectable research artifact quickly.

## Artifact map

- [Preregistration](PREREGISTRATION.md)
- [Pilot configuration](configs/pilot.json)
- [Experiment runner](src/run_experiment.py)
- [Analysis script](src/analyze.py)
- `data/raw/` — complete model inputs and outputs after the run
- `results/` — trial-level scores, aggregate tables, and figures after analysis
- `REPORT.md` — final report after results are available

## Reproduce

Requirements: Python 3.11+ and an authenticated `codex` CLI supporting
`gpt-5.6-luna`.

```bash
python3 src/run_experiment.py --config configs/pilot.json
python3 src/analyze.py --input data/raw/run.json --output-dir results
python3 -m unittest discover -s tests -v
```

The scripts use only the Python standard library. No credential is written to
the repository.

