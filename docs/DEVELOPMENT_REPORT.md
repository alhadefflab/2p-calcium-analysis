# Two-Photon Calcium Imaging Analysis Pipeline — Development Report

*A complete, commit-by-commit account of the project from its first commit to the
current state, with the scientific rationale, mathematical formulations, and the
new functions, tools, and GUI components introduced at each step.*

Repository: `2p-calcium-analysis` (working name **Luceo**, Latin *luceo* — "I shine")
Period covered: 2026-05-12 → 2026-06-08 (32 commits)
Author: Juarez Culau

---

## 0. Executive summary

The project is an end-to-end analysis pipeline for **two-photon (2P) calcium imaging**
of neural activity. It ingests raw multi-session, multi-channel, multi-Z-plane TIFF
stacks exported by PrairieView and produces motion-corrected, denoised, per-neuron
fluorescence traces, then classifies and visualizes stimulus-evoked responses.

The work evolved through four clear phases:

1. **Core scientific pipeline** (initial commit): the numerical engine — loading,
   two-stage motion correction, Cellpose-seeded CNMF source extraction, and
   stimulus-aligned response analysis. Originally driven by hardcoded scripts.
2. **GUI and interactivity** (May): a Tkinter/CustomTkinter front-end replacing
   hardcoded frame numbers with second-based timing; an interactive ROI curation
   editor; per-stage resume via provenance tracking.
3. **Robustness, generalization, and reproducibility** (late May → early June):
   unit + regression tests, environment migration (CaImAn 1.13 / PyTorch GPU /
   Cellpose 4), generalization from 2 stimuli to N stimuli, session-split bug fixes,
   and a repository reorganization into proper packages.
4. **Advanced neuron- and population-level analysis** (June): a post-CNMF neuron
   viewer, multi-plane duplicate detection, and two complementary population-analysis
   frameworks — a *Sherringtonian* cell-type view and a *Hopfieldian* neural-manifold
   view.

The remainder of this report is organized as:

- **§1** Scientific & mathematical foundations (the methods, with formulae).
- **§2** Chronological commit-by-commit narrative (what changed and why).
- **§3** Current architecture and file map.
- **§4** Consolidated mathematical appendix.
- **§5** Known limitations and open problems.

---

## 1. Scientific & mathematical foundations

This section collects the methods used across the pipeline so the commit narrative
in §2 can refer back to them. The pipeline implements a standard 2P analysis chain
plus two more advanced population-analysis layers.

### 1.1 The imaging model

A 2P recording of one Z-plane is a movie `Y ∈ ℝ^{T × d₁ × d₂}` (T frames,
`d₁ × d₂` pixels). Two optical channels are acquired:

- a **functional channel** (`func_ch`, e.g. GCaMP6s) carrying the calcium signal, and
- a **structural / motion-correction channel** (`mc_ch`) used as a stable anatomical
  reference.

The goal is to recover, for each neuron *k*, a spatial **footprint**
`aₖ ∈ ℝ^{d₁d₂}` and a temporal **activity trace** `cₖ ∈ ℝ^T`.

### 1.2 Two-stage motion correction

Motion is corrected in two passes because between-session drift and within-session
jitter have different spatial structure.

**Stage 1 — Affine inter-session alignment (`pystackreg`).**
For each session a reference image is built from the temporal mean of the
**middle 50 %** of frames (most stable, least bleached/drifted):

```
ref = mean over the central 50% of frames of the mc_ch stack
```

An affine transform `T_affine` (6 DOF: translation, rotation, scale, shear) is
estimated on the `mc_ch` and then **applied identically** to the `func_ch`, so the
two channels stay co-registered. (Implementation trick: the reference frame is
prepended as frame 0 so StackReg accepts it, then stripped after.)

**Stage 2 — Rigid / piecewise-rigid intra-session correction (CaImAn).**
Fine motion is corrected with CaImAn's `MotionCorrect`. In **piecewise-rigid**
mode the field of view is tiled into overlapping patches (`strides = 64×64`,
`overlaps = 32×32`); each patch *p* gets its own shift `s_p`, constrained so it
deviates from the global rigid shift `s_rig` by at most `max_deviation_rigid`:

```
‖s_p − s_rig‖∞ ≤ max_deviation_rigid
```

**Quality metric.** Correction quality is judged by each frame's Pearson
correlation with the temporal-mean image. For frame matrix `X` (frames × pixels)
with mean frame `x̄`:

```
corr(frame_t) = ⟨x_t − mean(x_t),  x̄ − mean(x̄)⟩
                ────────────────────────────────────────
                ‖x_t − mean(x_t)‖ · ‖x̄ − mean(x̄)‖
```

A successful correction pushes every frame's correlation up and toward the y = x
identity line on a raw-vs-corrected scatter (both plotted in
`_visualize_rigcorr_results`).

### 1.3 ROI seeding (Cellpose)

CNMF is *seeded* rather than initialized blindly. A summary image is built from the
functional movie (default: the 99th percentile projection over time, which
highlights pixels that were ever bright = active somata):

```
summary(x,y) = percentile_99 over t of func(t, x, y)
```

This is optionally median-filtered, then segmented by **Cellpose** (`cyto3`
model). Each label becomes one binary seed mask. Seeding makes the factorization
well-posed and anatomically grounded. Key Cellpose knobs exposed in the GUI:
`diameter`, `flow_threshold`, `cellprob_threshold`.

### 1.4 Source extraction — CNMF

The functional movie is factorized by **Constrained Non-negative Matrix
Factorization** (CaImAn `cnmf`). The generative model is:

```
Y  ≈  A C  +  b f  +  ε
```

- `A ∈ ℝ^{d₁d₂ × K}` — spatial footprints (one column per neuron, seeded by Cellpose).
- `C ∈ ℝ^{K × T}` — denoised calcium traces (non-negative).
- `b f` — low-rank background / neuropil (`nb = 2` components).
- `ε` — noise.
- Each `cₖ` obeys an **autoregressive** calcium model of order `p` (`p = 2` for the
  slow indicator GCaMP6s), capturing the fast rise / slow exponential decay of a
  calcium transient (`decay_time = 1.8 s`).

After fitting, components are quality-controlled with
`evaluate_components` + `select_components` (SNR floor `min_SNR = 2.0`, spatial
footprint consistency, optional CNN classifier) to drop noise and neuropil before
traces are saved.

### 1.5 ΔF/F and z-scoring (`custom_df_f`, `custom_df_f_startend`)

The raw extracted fluorescence per neuron is reconstructed and **scaled by the
footprint norm** so traces are in comparable physical units:

```
F = (C [+ YrA])  ·  sqrt( diag(Aᵀ A) )
```

(`YrA` is the residual added back when `use_residuals=True`, giving the raw rather
than the purely denoised trace.) The background contribution per neuron is
`f₀ = F + Aᵀ(b f)`.

Two normalizations are supported, both using a user-defined **baseline window**
(rather than the whole recording, which would be contaminated by the response):

- **Normalize-to-median** (`method='norm_to_median'`):
  ```
  ΔF/F = (F − median_baseline(F)) / median_baseline(f₀)
  ```
- **Z-score** (`method='zscore'`, the default for response analysis):
  ```
  z = (F − mean_baseline(F)) / std_baseline(f₀)
  ```

Z-scoring expresses every neuron's activity in units of its own baseline noise, so
neurons of different brightness become directly comparable.

### 1.6 Session windows and the session-split correction

Sessions (one per stimulus condition) are concatenated along time into a single
stack so CNMF sees the same neurons across conditions. Analysis windows are then
defined **per session** in seconds and converted to frames using the true frame
period `Δt`:

```
pre_f  = round(pre_discard_s / Δt)     # discarded settling frames
base_f = round(baseline_s   / Δt)      # baseline window
stim_f = round(stim_s       / Δt)      # stimulus window
```

A key bug fixed mid-project: session 2 must begin at the **actual frame count of
session 1** (read back from the affine-corrected TIFFs via `_session_lengths`), not
at a value inferred from `stim_s`. `_session_window_indices_n` accumulates the true
offset across N sessions, and each session is z-scored against **its own** baseline.

### 1.7 Responder classification

For each neuron and each stimulus *j*, the **median z-score over the stimulus
window** is the response statistic (median is robust to transient artifacts):

```
m_{k,j} = median over stim window of z_{k,j}(t)
responds(k, j)  ⟺  m_{k,j} > θ
```

The default threshold `θ = 1.64` corresponds to a one-tailed *p* < 0.05 under a
standard normal. Neurons are then sorted into display groups (`get_resp_n`),
generalized over N stimuli:

- **N = 1:** responders vs non-responders.
- **N = 2:** stim-1-only / both / stim-2-only.
- **N = 3–4:** grouped by *primary* stimulus = `argmax_j m_{k,j}`.

Within each group, neurons are sorted by descending median z-score for a clean
heatmap.

### 1.8 Spatial response maps and sub-regions

`get_spatial_response_data` places each neuron at its CNMF **center of mass** on the
anatomical max-projection and colors it by its stimulus selectivity, letting the
user see whether responders are anatomically clustered. Users can hand-draw
**sub-region masks** (e.g. two anatomical territories); `get_region_labels` assigns
each neuron to region A (0), B (1), or unclassified (−1) by looking up its centroid
in the mask, enabling region-by-region comparative statistics.

### 1.9 Multi-plane duplicate detection

Because a soma is taller than the Z-step (24 µm) and the 2P axial PSF spreads a few
µm, one physical neuron can appear in adjacent planes and be double-counted. Each
candidate cross-plane pair is tested in three stages:

1. **Centroid pre-filter:** skip pairs > 50 px apart (a 25 µm soma at 1.21 µm/px ≈
   20 px diameter, so true copies cannot be farther).
2. **Jaccard IoU of binary footprints** (primary spatial criterion):
   ```
   IoU(m₁, m₂) = |m₁ ∩ m₂| / |m₁ ∪ m₂|          threshold 0.60
   ```
   Stronger than centroid distance: adjacent-but-distinct cells have IoU ≈ 0.
3. **Trace correlation:** Pearson *r* between the two denoised traces over the whole
   recording (threshold 0.85). This works because in quiet inter-stimulus periods a
   true duplicate tracks perfectly (same intracellular Ca²⁺ signal → *r* > 0.95),
   while distinct neurons that merely share stimulus tuning sit at *r* ≈ 0.6–0.8.

Pairs are sorted by IoU descending. The user resolves each pair (keep brighter
plane — the 2P signal is quadratic in excitation, so the brightest copy is closest
to focus) and the decision is written to the `is_cell` mask.

### 1.10 The `is_cell` curation mask

Both the post-CNMF neuron viewer and the duplicate review write a boolean array
`concat_{z}_is_cell.npy` per plane. Every downstream analysis (`custom_df_f_startend`,
`get_stims_n`, region labels, spatial maps) loads it via `_load_is_cell` and silently
restricts `A`, `C`, `YrA` to accepted neurons — **without re-running CNMF**. This
decouples expensive extraction from cheap curation/re-analysis.

### 1.11 Population analysis I — the Sherringtonian (cell-type) view (`analysis/population.py`)

Here the **neuron** is the unit of explanation. Neurons are grouped into functional
cell-types and characterized individually.

**Feature construction & clustering.** Per-neuron stimulus-window activity is
concatenated across conditions, z-scored row-wise for scale invariance, and
PCA-reduced to ≤ 20 dims for denoising. Neurons are then clustered (KMeans / GMM /
Ward hierarchical) into co-activity cell-types.

**Single-neuron selectivity index.** For N stimuli with rectified responses
`rᵢ = max(responseᵢ, 0)`:

```
SI = ( N − Σᵢ (rᵢ / r_max) ) / (N − 1)
```

`SI = 1` → responds to a single stimulus; `SI = 0` → responds equally to all.

**Cluster quality (silhouette).** For each neuron with mean intra-cluster distance
*a* and nearest-other-cluster mean distance *b*:

```
s = (b − a) / max(a, b)   ∈ [−1, 1]
```

Sweeping *k* and taking the silhouette peak justifies the number of cell-types.

**Functional connectivity & modularity.** The neuron×neuron Pearson correlation
matrix is reordered by cluster. Block structure is quantified by:

```
modularity_ratio = mean|corr| within clusters / mean|corr| between clusters    (>1 ⇒ blocks)
```

**Spatial organization (permutation test).** For each cell-type, the observed mean
pairwise 3-D centroid distance is compared to a label-permutation null
(`n_permutations = 1000`); a one-sided p-value (small ⇒ the cell-type is
anatomically compact):

```
p = (#{ null ≤ observed } + 1) / (n_permutations + 1)
```

The Z (plane) axis is rescaled so plane separation spans the same range as the in-plane
extent, preventing one axis from dominating the distance.

**Spatial topology (TDA).** Optional persistent homology (`ripser`) of the 3-D
centroid cloud (H0 components, H1 loops, H2 voids) characterizes the shape of the
anatomical layout.

### 1.12 Population analysis II — the Hopfieldian (manifold) view (`analysis/manifold.py`)

Here the **population state** is the unit. The codebase implements the five core
concepts of Ebitz & Hayden (2021), *The population doctrine in cognitive
neuroscience*, Neuron 109:3055–3068.

**Coding dimension (targeted dimensionality reduction).** The direction in neural
state space that maximally separates conditions is found by least-squares regression
of population activity `X` (time × neurons) onto condition labels `y`:

```
w = argmin_w ‖X w − y‖²,    then  w ← w / ‖w‖
```

Each neuron's entry in `w` is its **loading**; the projection `X w` over time and the
between-condition separability are reported, plus the fraction of between-condition
variance the coding dimension captures.

**Trajectory geometry.** Stim-window states are PCA-compressed (≤ 20 dims) into
per-condition trajectories. Two scalars over time:

```
speed_t   = ‖ traj_{t+1} − traj_t ‖₂            (state transition rate)
cond_dist_t = mean over condition pairs of ‖ traj_i(t) − traj_j(t) ‖₂
```

The **discrimination latency** is the first frame where `cond_dist` exceeds
*baseline mean + 2 SD*.

**Subspace decomposition.** LDA discriminant axes span the **coding subspace**
(dimension N−1); its orthogonal complement is the **nullspace**. The
coding-to-total variance ratio measures how task-informative the population activity
is.

**Manifold dimensionality (participation ratio).** In a sliding window, from the
covariance eigenvalues `λᵢ`:

```
PR = ( Σ λᵢ )² / Σ λᵢ²
```

PR ↓ during stimulus ⇒ the population becomes more coordinated / lower-dimensional.

**Neural states / attractors.** A Gaussian mixture (component count chosen by
**BIC**) clusters pooled trajectory points into discrete population states; per-state
**occupancy**, **mean dwell time**, and a row-normalized **transition matrix**
`P(i→j)` are computed. **Slow points** — local minima of trajectory speed below the
25th percentile — are flagged as candidate fixed points / attractors (|dstate/dt| → 0
near a stable state).

**Manifold topology.** Persistent homology of the trajectory point cloud (H0
clusters, H1 loops), pooled and per condition. H1 loops with persistence above 10 %
of the H0 scale are counted as significant (a signature of rotational / ring-attractor
dynamics).

---

## 2. Commit-by-commit narrative

### Phase 1 — The core scientific pipeline

**`4eaa07f` — initial commit (2026-05-12).**
The numerical engine, as a set of scripts. Key files:
- `pipeline.py` (699 lines): `load_data` → `affine_motion_correction` →
  `rigid_motion_correction` → `source_extraction`, plus the provenance machinery
  (`init`, `_get_provenance`, `_save_provenance`) and visualization helpers
  (`_visualize_rigcorr_results`, MC video export).
- `pipeline_funcs.py`: post-extraction analysis — `custom_df_f`,
  `get_stims1_stims2`, `get_resp1_resp2`, plus Bokeh ROI-picking helpers
  (`nb_pick_dots`, `select_subregions`).
- `pipeline_utils.py`: provenance dictionaries (`JSONDict`/`YAMLDict` with
  auto-save on mutation), `combine_tiffs` (merges per-frame PrairieView TIFFs into
  multipage stacks by channel/Z, parsing Cycle/Ch/Z from filenames), `capture_args`
  decorator (records each step's call arguments into provenance), and OpenCV ROI
  add/remove tools (`draw_masks`, `remove_neurons`).
- `params.py`: default parameter blocks for loading, MC, Cellpose, and CNMF.
- `analysis_again.py`, `scratch.py`: exploratory driver scripts.
- `visualizationKB.py` (1457 lines): legacy Bokeh ROI visualization.
- `environment.yml`, `README.md`, `.gitignore`.

*Why this way:* provenance-first design means every expensive step records its
inputs and output filenames to `provenance.yaml`, so re-runs can skip completed
stages — essential for a multi-hour pipeline on ~55 GB of data.

**`e5207dc` — Fix stimulus classification offset and missing CNMF QC.**
Corrected an off-by-window error in stimulus classification and ensured the CNMF
quality-control step (`evaluate_components` / `select_components`) actually runs, so
noise components are removed before traces are saved.

### Phase 2 — GUI and interactivity

**`28c4e5f` — Add GUI entry point with second-based timing (`gui.py`, 620 lines).**
The pivotal usability change: instead of editing hardcoded frame numbers, the user
enters timing in **seconds**; frame counts are derived from the actual frame period
`Δt`. Pipeline function signatures were fixed to match. This is the origin of the
seconds→frames conversion described in §1.6.

**`17c7678` — Per-stage resume.** The GUI gained checkboxes to run individual
pipeline stages, leaning on the provenance file so motion correction / CNMF can be
skipped and only the analysis re-run with new timing.

**`9e35b6b` — Matplotlib on the main thread.** Fixed an editor crash by ensuring
all matplotlib calls run on the main thread (a recurring theme — GUI toolkits and
matplotlib are not thread-safe).

**`280d0ca`, `87ca5f9`, `95c7259` — Interactive neuron add/remove + bug fixes.**
Added in-GUI ROI editing (adding/removing neurons), then fixed an OpenCV display bug
when adding neurons and a Tkinter cursor-redirect bug.

**`696d93a` — Bug fix: source extraction processed only z5.** Every call was
accidentally pinned to a single Z-plane; corrected to iterate all planes.

**`4cabef4` — Single integrated ROI-curation window + exclude-region option (+279
lines).** Consolidated ROI curation into one window and added the ability to mark
regions to exclude.

**`8f8c80f`, `b17f685` — ROI-curation visualization improvements.** Better contrast
images for curation and a display-settings control.

**`5b2332f` — Optional output folder.** Lets the user choose where results are
written (otherwise a timestamped `run-…` directory).

### Phase 3 — Robustness, reproducibility, generalization

**`e3dcf9c` — Add unit and regression tests (+910 lines).** Introduced
`tests/unit/` (analysis functions, frame layout, provenance remapping, ROI-editor
coordinate transforms) and `tests/regression/` (full-pipeline snapshot tests), plus
`pytest.ini`, `conftest.py`, and `docs/testing.md`. Also fixed a scale-calculation
bug when adding neurons during ROI curation. *Why:* the codebase had reached a size
where silent numerical regressions were a real risk.

**`54e095e`, `2b7c0ee`, `667172f` — Environment file fixes.** Several iterations to
get a reproducible Conda environment (format, numpy via pip, etc.).

**`43e8585` — Thread-safe `visualize_rigcorr_results`.** Reworked plotting to use
the Agg backend with explicit `Figure`/`FigureCanvasAgg` objects rather than the
global `pyplot` state, eliminating thread-safety crashes during MC visualization.

**`2b68208` — Session-split bug fix (important correctness fix).** Per-session
**independent** z-scoring using stored frame counts; corrected heatmap axis and
timing provenance. This is the fix described in §1.6: session boundaries now come
from the true frame counts (`_session_lengths`), and each session is normalized
against its own baseline. Tests were added to lock the behavior.

**`3e8160f` — Sub-region definition + comparative analysis + neuron count (+515
lines in `gui.py`).** Added the hand-drawn sub-region workflow (`get_region_labels`,
§1.8), comparative region analysis, and a live neuron count.

**`d0b3ef0` — Spatial response map (`get_spatial_response_data`).** Anatomy
max-projection overlaid with per-neuron stimulus-selectivity scatter (§1.8).

**`d731aa0` — Bug fix: channel mmap now saves for all Zs.**

**`625ecf1` — Regression baseline (ZH539).** Committed a ground-truth baseline
(`baseline_zh539.json`, `save_baseline.py`) and a real end-to-end regression test
(`test_zh539_pipeline.py`), replacing the earlier placeholder snapshot scaffolding.

**`f8a7693` — Migrate to CaImAn 1.13 + PyTorch GPU + Cellpose 4 (major
modernization).** Updated deprecated APIs (e.g. `cnm.fit` now mutates in place;
Cellpose `CellposeModel`/`cyto3`), split the environment into
`environment-cpu.yml` / `environment-gpu.yml`, and wired **automatic GPU detection**:
`params.py` calls `torch.cuda.is_available()` at import and sets `USE_GPU`, which
flows into MC (`use_cuda`), Cellpose (`gpu`), and CNMF. Added
`tests/unit/test_environment.py`.

**`511e024` — Repository reorganization.** Broke the 1800-line `gui.py` into a
package and deleted dead scripts:
- `gui/app.py` (main window), `gui/roi_editor.py` (Tkinter canvas ROI editor).
- `visualization/response_plots.py` (heatmaps, bar charts, region plots) and
  `visualization/roi_legacy.py` (renamed from `visualizationKB.py`).
- Removed `analysis_again.py`, `scratch.py`.
*Why:* a flat pile of scripts had become unmaintainable; clean package boundaries
were a prerequisite for the feature work that followed.

**`7654456` — Generalize pipeline to N stimuli (1–4).** This is where
`get_stims1_stims2` / `get_resp1_resp2` became thin backward-compatible wrappers
around the new `get_stims_n` / `get_resp_n` (§1.7), `_session_window_indices_n`
replaced the 2-session-only version, the GUI gained a **number-of-stimuli** selector,
the Cellpose API was fixed, and ROI parameters were exposed in the GUI.

### Phase 4 — Advanced neuron- and population-level analysis

**`139bdbb` — Neuron viewer post-CNMF (`gui/neuron_viewer.py`, `gui/neuron.py`).**
A Suite2p-style browser to inspect each component after CNMF: per-neuron
fluorescence trace (raw `C + YrA` and denoised `C`), a reconstructed calcium
mini-video, and accept/reject controls. The `Neuron` dataclass (§ `gui/neuron.py`)
wraps one CNMF column: it reshapes the sparse footprint to `(h, w)` in Fortran
order, stores raw/denoised traces, and computes a footprint-weighted centroid
(thresholded at 10 % of peak). Curation writes `concat_{z}_is_cell.npy` (§1.10)
without re-running CNMF; the pipeline was wired to load and apply it everywhere.

**`9919611` — Neuron Frequency expanded pop-up.** Richer per-neuron detail view
(expanded frequency / activity visualization) in the viewer.

**`19c95fb` — Multi-plane duplicate review (`gui/zplane_viewer.py`, 775 lines).**
The cross-plane duplicate detector of §1.9: centroid pre-filter → Jaccard IoU →
trace correlation, with a Tkinter 2-D overlay, threshold sliders, a neuron search
box, per-pair resolution cards, and a Plotly 3-D reconstruction in the browser.
Decisions feed the `is_cell` mask. Fully documented in
`docs/multiplane_duplicate_review.md`. Added `plotly` to both environments.

**`a32a31e` — README update.** Documentation refresh.

**`9e5874b` — Two-view population analysis (work in progress).** The largest single
feature commit (+3907 lines). Added the two deliberately-separated frameworks:
- `analysis/population.py` + `visualization/population_plots.py` +
  `gui/population_viewer.py` — the **Sherringtonian** cell-type view (§1.11):
  co-activity clustering, selectivity index, silhouette quality, functional
  connectivity / modularity, spatial-aggregation permutation test, and optional
  spatial TDA.
- `analysis/manifold.py` + `visualization/manifold_plots.py` +
  `gui/manifold_viewer.py` — the **Hopfieldian** manifold view (§1.12): coding
  dimensions, trajectory geometry, subspace decomposition, participation-ratio
  dimensionality, GMM neural states with BIC, and manifold topology.
- Optional dependencies `ripser` (TDA) and `pyvista` (3-D spatial map) added, each
  degrading gracefully if absent.

*Design note (recorded in project memory):* the two frameworks are kept strictly
separate on purpose — `population.py` explains structure by grouping **neurons**;
`manifold.py` explains it by the geometry of **population states**. They should not
be re-mixed.

---

## 3. Current architecture and file map

```
gui.py                              Entry point — launches the GUI
gui/app.py                          Main window: Animals / Recording / Timing / Run tabs + pipeline runner
gui/roi_editor.py                   Interactive ROI editor (Tkinter canvas)
gui/neuron.py                       Neuron dataclass (one CNMF component)
gui/neuron_viewer.py               Post-CNMF neuron browser (trace + calcium video, accept/reject)
gui/zplane_viewer.py               Multi-plane duplicate review (IoU + corr, 3-D Plotly)
gui/population_viewer.py           Sherringtonian cell-type analysis UI
gui/manifold_viewer.py             Hopfieldian manifold analysis UI
pipeline.py                         load → affine MC → rigid MC → CNMF source extraction
pipeline_funcs.py                  ΔF/F, z-scoring, N-stimulus responder classification, spatial/region data
pipeline_utils.py                  combine_tiffs, provenance dicts, capture_args, OpenCV ROI tools
params.py                          Default MC/CNMF/Cellpose params; USE_GPU auto-detected
analysis/population.py             Cell-type computation (clustering, selectivity, connectivity, spatial)
analysis/manifold.py               Manifold computation (coding dims, trajectories, states, topology)
visualization/response_plots.py    Heatmaps, bar charts, region plots
visualization/population_plots.py  Cell-type figures
visualization/manifold_plots.py    Manifold figures
visualization/roi_legacy.py        Legacy Bokeh ROI helpers
tests/unit/, tests/regression/     Unit + ZH539 end-to-end regression tests
docs/                              testing.md, multiplane_duplicate_review.md
environment-cpu.yml / -gpu.yml      Conda environments
```

**GUI flow.** `PipelineGUI` (CustomTkinter) builds four tabs —
`_build_animals_tab` (subjects, output folder, number of stimuli, session folders),
`_build_recording_tab` (frame period, Z-planes, channel map, Cellpose params),
`_build_timing_tab` (pre-discard / baseline / stim seconds with live frame counts,
responder threshold), and `_build_run_tab` (which stages to run) — then
`_run_pipeline` executes the selected stages with stdout captured into the GUI.

**Provenance.** `provenance.yaml` records, per stage and Z-plane, the captured
arguments and output filenames. `init` also **remaps absolute paths** if the project
is moved/copied to another machine, so resumes survive relocation.

---

## 4. Mathematical appendix (consolidated)

| Quantity | Formula | Where |
|---|---|---|
| MC reference image | mean of central 50 % of frames | §1.2 |
| Frame–mean correlation | `⟨x_t−x̄_t, x̄−⟨x̄⟩⟩ / (‖x_t−x̄_t‖·‖x̄−⟨x̄⟩‖)` | §1.2 |
| Summary image | `percentile₉₉ₜ func(t,x,y)` | §1.3 |
| CNMF model | `Y ≈ A C + b f + ε`, `cₖ` ~ AR(p) | §1.4 |
| Scaled fluorescence | `F = (C[+YrA])·sqrt(diag(AᵀA))` | §1.5 |
| ΔF/F (median) | `(F − med_base F)/med_base f₀` | §1.5 |
| Z-score | `(F − mean_base F)/std_base f₀` | §1.5 |
| Seconds→frames | `n_f = round(t_s / Δt)` | §1.6 |
| Response statistic | `m_{k,j} = median_stim z_{k,j}(t)` | §1.7 |
| Responder rule | `m_{k,j} > θ` (θ = 1.64) | §1.7 |
| Jaccard IoU | `|m₁∩m₂|/|m₁∪m₂|` (≥ 0.60) | §1.9 |
| Duplicate trace corr | Pearson r (≥ 0.85) | §1.9 |
| Selectivity index | `(N − Σ rᵢ/r_max)/(N−1)` | §1.11 |
| Silhouette | `(b−a)/max(a,b)` | §1.11 |
| Modularity ratio | within / between mean\|corr\| | §1.11 |
| Spatial permutation p | `(#{null≤obs}+1)/(n_perm+1)` | §1.11 |
| Coding dimension | `w = argmin ‖Xw−y‖²`, normalized | §1.12 |
| Trajectory speed | `‖traj_{t+1}−traj_t‖₂` | §1.12 |
| Condition distance | mean pairwise `‖traj_i(t)−traj_j(t)‖₂` | §1.12 |
| Participation ratio | `(Σλᵢ)²/Σλᵢ²` | §1.12 |
| State model | GMM, k by min BIC | §1.12 |
| Persistent homology | H0/H1(/H2) via Vietoris–Rips (ripser) | §1.11–1.12 |

---

## 5. Known limitations and open problems

These are carried in the README and are worth stating explicitly in any write-up:

1. **No cell-type classification from the structural channel.** The `mc_ch` is
   motion-corrected but discarded analytically. Marker-positive identification
   (projecting the structural stack onto each CNMF footprint and thresholding) and
   bleed-through correction are not implemented.

2. **Responder threshold & multiple comparisons (under review).** A fixed per-neuron
   threshold (θ = 1.64) accumulates false positives at scale (~25 expected per 500
   neurons at p < 0.05). Benjamini–Hochberg FDR assumes independence, which fails for
   spatially/temporally correlated neurons (shared neuropil, network, hemodynamics),
   tending to over-correct. The principled alternative under consideration is
   **permutation testing** — shuffle trial labels to build a session-specific
   empirical null that automatically accounts for neuron count, noise distribution,
   and correlation structure.

3. **`multi_crop_len` defined but unused.** `_load_data` only uses
   `multi_crop_start` (crops with no upper bound); sessions of differing length can
   yield inconsistent array lengths.

4. **No validation in `_load_data`.** Channel counts and frame dimensions are not
   checked for consistency across sessions before concatenation.

5. **`9e5874b` population analysis is marked work-in-progress.**

---

## 6. Suggested narrative arc for your own report

If you write this up, a clean story is:

1. **Problem** — extract and compare stimulus-evoked activity of many neurons across
   multiple conditions and focal planes from raw 2P TIFFs.
2. **Engine** — provenance-driven, resumable pipeline: two-stage motion correction →
   Cellpose-seeded CNMF → per-session z-scored responses (§1.2–1.7).
3. **From scripts to a tool** — second-based timing, interactive curation, and a
   four-tab GUI removed the need to edit code, making the pipeline usable by
   non-programmers (Phase 2).
4. **Trustworthiness** — tests, a committed regression baseline, environment
   reproducibility, GPU support, and the session-split correctness fix (Phase 3).
5. **Generality** — 1–4 stimuli, N-session windows, exposed parameters (`7654456`).
6. **Beyond single traces** — post-CNMF curation, multi-plane de-duplication, and
   the two complementary population frameworks (cell-type vs manifold), each with
   explicit, citable mathematics (Phase 4).
7. **Honesty** — the documented open problems (especially the multiple-comparisons
   question) show methodological awareness and set up future work.

*Generated as a working basis for the author's own report; verify any figure or
formula against the current source before publication.*
```