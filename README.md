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
5. **Source extraction (CNMF)** — constrained nonnegative matrix factorization via CaImAn; seeded by Cellpose masks
6. **Analysis & visualization** — stimulus-aligned response analysis (`pipeline_funcs.py`) and interactive plots via Bokeh and matplotlib (`visualizationKB.py`)

Provenance is tracked automatically in a `provenance.yaml` file so each processing step can be skipped on re-runs if outputs already exist.

## File Structure

```
pipeline.py          # Core pipeline steps (load, motion correct, source extract)
pipeline_funcs.py    # Post-extraction analysis (stimulus alignment, response calculation)
pipeline_utils.py    # Utilities (TIFF combining, YAML provenance, argument capture)
params.py            # All tunable parameters (MC, CNMF, video, ROI settings)
analysis_again.py    # Top-level run script; calls pipeline stages in sequence
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

Edit `params.py` to point `LOAD_PARAMS["multi_path"]` at your data directories and set the Z-plane(s) of interest. Then run:

```bash
python analysis_again.py
```

Data should be organized as flat directories of `.tif` files exported from the acquisition software (PrairieView XML format assumed).

## Known Bugs / Limitations

- Temporal cropping (`multi_crop`) indexing is inconsistent across sessions — may silently drop or misalign frames
- The `_load_data` function does not validate that channel counts match across sessions
- `analysis_again.py` contains hardcoded subject/path references that need manual updating per experiment
- `scratch.py` has dead code and unfinished branches
- Video output filenames contain a typo ("conat" instead of "concat") that propagates to saved files
- Python 3.10 and 3.11 bytecode coexists; the environment has not been fully pinned and may behave differently across versions
- No unit tests exist; correctness depends entirely on visual inspection of outputs

## Data

Raw data and processed outputs are excluded from this repository (too large; ~55 GB+). Data follows the naming convention:

```
data/<date>_<subject>_<stimulus>/   # raw TIFFs per session
```

Processed outputs (motion-corrected stacks, CNMF results, ROI masks) are written to a user-specified output directory.
