# 2P Calcium Imaging Analysis Pipeline

Analysis pipeline for two-photon calcium imaging data. Processes multi-session, multi-Z-plane recordings through motion correction, ROI segmentation, and neural signal extraction.

> **Note:** This codebase is under active development and contains known bugs. Use with caution and expect rough edges.

## Overview

The pipeline takes raw multi-channel `.tif` stacks from two-photon microscopy sessions and produces denoised, motion-corrected fluorescence traces per neuron across Z-planes.

**Pipeline stages:**
1. **Data loading** — reads multi-session, multi-channel TIFF stacks; supports optional temporal cropping per session
2. **Affine motion correction** — inter-session alignment using `pystackreg`; reference image generated from the middle 50% of frames
3. **Rigid / piecewise-rigid motion correction** — fine-grained within-session correction using [CaImAn](https://github.com/flatironinstitute/CaImAn)
4. **ROI identification** — cell segmentation via [Cellpose](https://github.com/MouseLand/cellpose) in seeded mode
5. **Source extraction (CNMF)** — constrained nonnegative matrix factorization via CaImAn; seeded by Cellpose masks; followed by `evaluate_components` + `select_components` to remove noise and neuropil components before traces are saved
6. **Analysis & visualization** — stimulus-aligned response analysis (`pipeline_funcs.py`) and interactive plots via Bokeh and matplotlib (`visualizationKB.py`)
7. **GUI** *(recently added — required)* — `gui.py` is now the recommended entry point for running the analysis. It replaces manual editing of hardcoded frame numbers: timing is entered in seconds and frame counts are computed automatically from the actual frame period. Supports single and multi-animal experiments. See **Usage** below.

Provenance is tracked automatically in a `provenance.yaml` file so each processing step can be skipped on re-runs if outputs already exist.

## File Structure

```
gui.py               # GUI entry point (NEW — start here)
pipeline.py          # Core pipeline steps (load, motion correct, source extract)
pipeline_funcs.py    # Post-extraction analysis (stimulus alignment, response calculation)
pipeline_utils.py    # Utilities (TIFF combining, YAML provenance, argument capture)
params.py            # All tunable parameters (MC, CNMF, video, ROI settings)
analysis_again.py    # Legacy run script; kept for reference, superseded by gui.py
scratch.py           # Exploratory / scratch code
visualizationKB.py   # Visualization functions (Bokeh interactive + matplotlib)
environment.yml      # Conda environment specification
```

## Setup

```bash
conda env create -f environment.yml
conda activate <env-name>
```

**Key dependencies:** CaImAn, Cellpose, pystackreg, OpenCV, Bokeh, tifffile, NumPy, SciPy, matplotlib

## Usage

Install the GUI dependency, then launch `gui.py`:

```bash
pip install customtkinter
python gui.py
```

The GUI walks through four tabs:

| Tab | What to set |
|-----|-------------|
| **Animals & Data** | Subject ID, output folder, session folders per animal (one folder per stimulus), single vs. multi-animal mode |
| **Recording** | Frame period in s/frame (read from the PrairieView `.xml` file), z-planes to analyse, channel assignments. Use **Auto-detect** to scan the session folder and fill in z-planes automatically |
| **Timing** | Pre-baseline discard, baseline window, and stimulus duration — all in **seconds**. The exact frame counts are shown live as you type |
| **Run** | Select which pipeline stages to run. Uncheck motion correction and CNMF to re-run only the analysis with different timing or threshold settings |

Data should be organized as flat directories of per-frame `.tif` files exported from the acquisition software (PrairieView XML format assumed). Each session folder maps to one stimulus condition.

## Known Bugs / Limitations

Items marked ✅ have been patched. Items marked ❌ are open.

### Analysis correctness

- ✅ **Stimulus window index offset** — `get_resp1_resp2` was using original frame numbers (103) as indices into `stims1`, which starts 52 frames into the recording. Classification was silently starting ~30 s late. Fixed by correcting the index to 51 (= 103 − 52) in `pipeline_funcs.py` and the matching mean z-score calculation in `analysis_again.py`.
- ✅ **CNMF component quality control missing** — `evaluate_components` + `select_components` were removed at some point during refactoring. Without them, noise components and neuropil patches passed straight through to classification. Both calls have been restored after `cnm.fit()` in `pipeline.py`.
- ✅ **`get_resp1_resp2` called with wrong number of arguments** — the per-animal loop in `analysis_again.py` was passing 2 arguments to a function that requires 3, crashing immediately on any multi-animal run. Fixed.
- ✅ **Timing defined in frames instead of seconds** — frame indices were hardcoded as rounded integers, making it impossible to set stimulus timing without editing source code, and causing off-by-one errors when the frame period changed. All timing is now computed from seconds via `round(seconds / frame_period)` and exposed in the GUI.
- ❌ **No false discovery rate correction** — responders are classified with a fixed z-score threshold of 1.64 (p < 0.05 one-tailed), applied independently to every neuron. With hundreds of neurons tested simultaneously, expected false positives accumulate. Reported counts should be treated as upper bounds; neurons just above the threshold are the most likely to be noise. Standard fix is Benjamini–Hochberg FDR correction.

### Code quality / environment

- ✅ **Video output filename typo** — source extraction video was saved as `conat_…` instead of `concat_…`. Fixed in `pipeline.py`.
- ❌ **`multi_crop_len` defined but never used** — `params.py` defines `multi_crop_len` but `_load_data` only uses `multi_crop_start`, cropping with no upper bound. If sessions differ in total frame count the output arrays will have inconsistent lengths.
- ❌ **`_load_data` channel count not validated** — no check that channel counts or frame dimensions match across sessions before concatenation.
- ❌ **`environment.yml` encoding and prefix** — the file is saved as UTF-16 LE, which `conda env create` cannot parse. It also contains a hardcoded `prefix:` path pointing to the original developer's machine that must be removed before install.
- ❌ **`analysis_again.py` contains hardcoded subject/path references** — subject ID and data paths are hardcoded and must be edited manually for each experiment. The GUI replaces this workflow but the underlying script still has the hardcoded values.
- ❌ **`scratch.py` has dead code** — references the old class-based `Pipeline` API which no longer exists, and contains duplicate variable assignments. The file does not run as-is.
- ❌ **No unit tests** — correctness depends entirely on visual inspection of outputs.

## Data

Raw data and processed outputs are excluded from this repository (too large; ~55 GB+). Data follows the naming convention:

```
data/<date>_<subject>_<stimulus>/   # raw TIFFs per session
```

Processed outputs (motion-corrected stacks, CNMF results, ROI masks) are written to a user-specified output directory.
