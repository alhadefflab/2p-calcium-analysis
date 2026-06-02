# Testing

## Setup

Activate the conda environment before running any tests:

```
conda activate caiman
```

## Running unit tests

Unit tests require no data files and no GPU. Run them from the project root:

```
pytest tests/unit/
```

These cover pure analysis logic — z-scoring, responder classification, frame
arithmetic — using synthetic data. They must always pass.

## Running regression tests

Regression tests re-run the full analysis pipeline from existing ZH539 CNMF
files and compare the results to a committed baseline. They catch regressions
in neuron detection, motion correction, and responder classification.

They require:
1. A completed ZH539 pipeline run (CNMF .hdf5 files + analysis/params.yaml present)
2. A generated baseline file (`tests/regression/baseline_zh539.json`)

Set the data directory and run (Anaconda Prompt):

```
set ZH539_DIR=C:\path\to\your\ZH539
pytest tests/regression/ -v
```

Tests auto-skip if `ZH539_DIR` is not set or the baseline is a placeholder.

## Generating the baseline (run once)

After a verified pipeline run on ZH539, generate the baseline:

```
set ZH539_DIR=C:\path\to\your\ZH539
python tests/regression/save_baseline.py
```

This reads the existing CNMF files and analysis results, extracts key metrics
(neuron counts per z-plane, motion correction shifts, responder counts), and
saves them to `tests/regression/baseline_zh539.json`. Commit this file — it is
the frozen ground truth and should not change unless you deliberately decide to
update it.

## Updating the baseline after a CaImAn upgrade

After upgrading CaImAn the CNMF outputs will change numerically. The workflow:

1. Re-run the full pipeline on ZH539 with the new environment.
2. Visually verify the heatmaps and neuron counts look correct.
3. Run `python tests/regression/save_baseline.py` to regenerate the baseline.
4. Run `pytest tests/regression/ -v` — tests should pass within tolerances.
5. Commit the updated `baseline_zh539.json` as the new ground truth.

Only do this when you have verified the new results are correct. The unit tests
never need updating unless the analysis logic itself changes.

## What each regression test checks

| Test class | What it verifies |
|---|---|
| `TestNeuronCounts` | Neuron count per z-plane within ±15% of baseline |
| `TestMotionCorrection` | Mean shift per z-plane within ±1px of baseline |
| `TestResponderClassification` | Responder counts per category within ±20% of baseline |

Tolerances exist because CNMF uses stochastic initialisation and results vary
slightly between runs. They are defined in `baseline_zh539.json` and can be
tightened once the pipeline is stable.
