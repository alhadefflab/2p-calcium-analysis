# 2P Calcium Imaging Analysis Pipeline

Analysis pipeline for two-photon calcium imaging data. Processes multi-session, multi-Z-plane recordings through motion correction, ROI segmentation, and neural signal extraction.

> **Note:** This codebase is under active development and contains known bugs. Use with caution and expect rough edges.

## Overview

The pipeline takes raw multi-channel `.tif` stacks from two-photon microscopy sessions and produces denoised, motion-corrected fluorescence traces per neuron across Z-planes.

**Pipeline stages:**
1. **Data loading**: reads multi-session, multi-channel TIFF stacks; supports optional temporal cropping per session
2. **Affine motion correction**: inter-session alignment using `pystackreg`; reference image generated from the middle 50% of frames
3. **Rigid / piecewise-rigid motion correction**: fine-grained within-session correction using [CaImAn](https://github.com/flatironinstitute/CaImAn)
4. **ROI identification**: cell segmentation via [Cellpose](https://github.com/MouseLand/cellpose) in seeded mode
5. **Source extraction (CNMF)**: constrained nonnegative matrix factorization via CaImAn; seeded by Cellpose masks; followed by `evaluate_components` + `select_components` to remove noise and neuropil components before traces are saved
6. **Neuron curation**: post-CNMF interactive browser (`gui/neuron_viewer.py`); inspect individual fluorescence traces and calcium mini-video; accept or reject components; saves `is_cell` mask per z-plane without re-running CNMF
7. **Multi-plane duplicate review**: cross-z-plane duplicate detection using Jaccard IoU of spatial masks and trace correlation 3-D neuron reconstruction via plotly; interactive resolution per pair; see `docs/multiplane_duplicate_review.md`
8. **Analysis & visualization**: stimulus-aligned response analysis (`pipeline_funcs.py`) and plots via matplotlib (`visualization/response_plots.py`); `is_cell` masks from curation and duplicate review are applied automatically
9. **GUI**: `gui.py` is the recommended entry point for running the analysis. It replaces manual editing of hardcoded frame numbers: timing is entered in seconds and frame counts are computed automatically from the actual frame period. Supports single and multi-animal experiments, and 1-4 stimulus conditions. See **Usage** below.

Provenance is tracked automatically in a `provenance.yaml` file so each processing step can be skipped on re-runs if outputs already exist.

## File Structure

```
gui.py                         # Entry point, launches the GUI
gui/app.py                     # Main GUI window (tabs, pipeline runner)
gui/roi_editor.py              # Interactive ROI editor (Tkinter canvas)
gui/neuron.py                  # Neuron dataclass wrapping one CNMF component
gui/neuron_viewer.py           # Post-CNMF neuron browser (trace display, accept/reject)
gui/zplane_viewer.py           # Multi-plane duplicate review (3-D map, IoU detection)
pipeline.py                    # Core pipeline steps (load, motion correct, source extract)
pipeline_funcs.py              # Post-extraction analysis (z-scoring, responder classification)
pipeline_utils.py              # Utilities (TIFF combining, YAML provenance, argument capture)
params.py                      # Default parameters for MC, CNMF, Cellpose; USE_GPU auto-detected
visualization/response_plots.py  # Response heatmaps, bar charts, region plots
visualization/roi_legacy.py    # Legacy Bokeh ROI visualization helpers
docs/multiplane_duplicate_review.md  # Full guide to the multi-plane duplicate review feature
environment-cpu.yml            # Conda environment (CPU-only)
environment-gpu.yml            # Conda environment (CUDA/GPU)
```

## Setup

Choose the environment that matches your hardware (Anaconda Prompt):

---

### CPU (no GPU)

```
conda env create -f environment-cpu.yml
conda activate caiman-cpu
```

---

### GPU (CUDA)

**Step 1: Install CUDA Toolkit 12.4 (once, system-wide)**

Download the Windows installer from NVIDIA:
https://developer.nvidia.com/cuda-12-4-0-download-archive

Select: Windows → x86_64 → 11 → exe (local). Run the installer and reboot when prompted.

Verify the install:
```
nvcc --version
```

**Step 2: Create the conda environment**

```
conda env create -f environment-gpu.yml
conda activate caiman-gpu
```

**Step 3: Install GPU-enabled PyTorch**

Conda installs a CPU-only torch as a CaImAn dependency. Replace it with the CUDA build:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
```

**Step 4: Verify**

```
python -c "import torch; print(torch.cuda.is_available())"
python -c "import caiman; print(caiman.__version__)"
```

Both should succeed without errors.

---

After activating either environment, GPU usage is **automatic**: `params.py` calls `torch.cuda.is_available()` at import time and sets `USE_GPU` accordingly. No manual editing required. If you need to force CPU mode (e.g. to test without a GPU), open `params.py` and hardcode `USE_GPU = False`.

**Key dependencies:** CaImAn >= 1.12, Cellpose >= 4, PyTorch >= 2.5, pystackreg, OpenCV, Bokeh, tifffile, NumPy, SciPy

**Optional packages** — all included in the conda environment files and installed automatically on fresh `conda env create`. If you already have an existing environment, install them manually with:

```
pip install plotly ripser pyvista
```

| Package | Used by | Behaviour if absent |
|---------|---------|---------------------|
| `plotly` | Multi-plane 3-D neuron reconstruction; Manifold activation animation | Button disabled, message shown |
| `ripser` | Population Analysis → Topology tab → persistent homology (TDA) | TDA checkbox disabled |
| `pyvista` | Neural Manifold Analysis → Neuron Explorer → 3-D spatial map | Button disabled, message shown |

The core analyses (clustering, state-space trajectories, coding dimensions, subspaces) run without any of these packages.

## Usage

Launch the GUI (Anaconda Prompt):

```
python gui.py
```

The GUI walks through four tabs:

| Tab | What to set |
|-----|-------------|
| **Animals & Data** | Subject ID, output folder, **number of stimuli (1–4)**, session folders per animal (one folder per stimulus condition), single vs. multi-animal mode |
| **Recording** | Frame period in s/frame (read from the PrairieView `.xml` file), z-planes to analyse, channel assignments, Cellpose ROI detection parameters (diameter, flow threshold, cell probability threshold) |
| **Timing** | Pre-baseline discard, baseline window, and stimulus duration, all in **seconds**. The exact frame counts are shown live as you type. **Responder threshold** (z-score, default 1.64) is editable here |
| **Run** | Select which pipeline stages to run. Uncheck motion correction and CNMF to re-run only the analysis with different timing or threshold settings |

Data should be organized as flat directories of per-frame `.tif` files exported from the acquisition software (PrairieView XML format assumed). Each session folder maps to one stimulus condition.

### Number of stimuli

The **Number of stimuli** selector in the Animals & Data tab controls how many session folders are shown per animal (1 through 4). The analysis adapts automatically:

| N | Behaviour |
|---|-----------|
| 1 | Single-stimulus experiment: neurons are classified as responders or non-responders; one heatmap panel |
| 2 | Two-stimulus comparison (default): neurons split into stim-1-only / both / stim-2-only groups; two heatmap panels |
| 3–4 | Multi-stimulus: neurons grouped by their primary responding stimulus; N heatmap panels |

### Tunable analysis parameters

The following parameters are set in the GUI **Timing** tab and saved to `params.yaml` alongside each result:

| Parameter | Default | Where to change |
|-----------|---------|-----------------|
| Responder z-score threshold | 1.64 (one-tailed p < 0.05) | Timing tab |
| Pre-baseline discard | 30 s | Timing tab |
| Baseline window | 30 s | Timing tab |
| Stimulus duration | 180 s | Timing tab |
| Frame period | 0.585 s/frame | Recording tab |
| Cell diameter (Cellpose) | 15 px (or Auto) | Recording tab (Cellpose section) |
| Flow threshold (Cellpose) | 2.0 | Recording tab (Cellpose section) |
| Cell probability threshold (Cellpose) | −1.0 | Recording tab (Cellpose section) |

Parameters **not** in the GUI; edit `params.py` directly:

| Parameter | Default | Why you might change it |
|-----------|---------|------------------------|
| `CNMF_PARAMS["min_SNR"]` | 2.0 | Minimum SNR for a CNMF component to be accepted; raise to reject more noise |
| `CNMF_PARAMS["decay_time"]` | 1.8 s | Calcium indicator decay constant (1.8 s for GCaMP6s, ~0.4 s for faster indicators) |
| `CNMF_PARAMS["p"]` | 2 | AR model order (2 for GCaMP6s, 1 for faster indicators) |

`min_SNR` is the CNMF parameter most likely to need tuning per experiment.

## Known Bugs / Limitations

Items marked ✅ have been patched. Items marked ❌ are open.

### Unsupported experiment types

- ❌ **No cell-type classification from a structural channel**: the pipeline loads both imaging channels and applies motion correction to both, but the structural/reference channel is discarded after motion correction and is never used analytically. Experiments that require identifying which neurons are marker-positive cannot currently be fully analysed. The calcium traces are extracted correctly, but the per-neuron classification step, which requires projecting the motion-corrected GFP stack onto each CNMF spatial footprint and thresholding, has not been implemented. Bleed-through correction is also absent.

### Analysis correctness

- ⚠️ **Responder threshold and multiple comparisons (under review)**: responders are classified with a fixed z-score threshold (default 1.64, one-tailed p < 0.05), applied independently to every neuron. With hundreds of neurons tested simultaneously, expected false positives accumulate: at p < 0.05 and 500 neurons, ~25 false positives are expected even if nothing responds. The standard textbook fix is Benjamini-Hochberg (BH) FDR correction, which progressively raises the effective threshold as neuron count grows. However, BH assumes the tests are independent, which does not hold here: neurons from the same recording are spatially and temporally correlated (shared neuropil, network activity, hemodynamics). When tests are positively correlated, BH tends to over-correct, removing genuine responders. Both approaches (no correction and BH) are therefore potentially wrong in opposite directions, and neither is biologically grounded. A more principled alternative under consideration is permutation testing: shuffle trial labels to build an empirical null distribution from each recording's own noise structure, then derive a session-specific threshold. This approach automatically accounts for neuron count, noise distribution, and correlation structure without distributional assumptions.

### Code quality / environment

- ❌ **`multi_crop_len` defined but never used**: `params.py` defines `multi_crop_len` but `_load_data` only uses `multi_crop_start`, cropping with no upper bound. If sessions differ in total frame count the output arrays will have inconsistent lengths.
- ❌ **`_load_data` channel count not validated**: no check that channel counts or frame dimensions match across sessions before concatenation.

## Data

Raw data and processed outputs are excluded from this repository (too large; ~55 GB+). Data follows the naming convention:

```
data/<date>_<subject>_<stimulus>/   # raw TIFFs per session
```

Processed outputs (motion-corrected stacks, CNMF results, ROI masks) are written to a user-specified output directory.
