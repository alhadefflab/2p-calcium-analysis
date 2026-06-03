# Multi-plane Duplicate Review

## What it is

When you image multiple z-planes in a single 2-photon session, the same physical neuron can appear in more than one focal plane. This happens because:

1. The neuron soma is physically larger than the z-step between planes.
2. The 2-photon axial point-spread function (PSF) spreads the signal a few micrometers above and below the focal plane.

The result is that CNMF extracts what appear to be separate neurons in z1 and z2, each with its own trace, when in reality they represent one cell recorded twice. If this is not corrected, your neuron count is inflated and the same activity is double-counted in all downstream analyses.

The Multi-plane Duplicate Review stage detects these cases using spatial mask overlap and trace correlation, lets you resolve each pair interactively, and saves the result to the `is_cell` filter that the analysis stage reads automatically.

## How it fits into the pipeline

Multi-plane Duplicate Review runs after CNMF and after per-plane Neuron Curation (which removes noise components). The correct order is:

1. Motion correction
2. Source extraction (CNMF)
3. Neuron Curation (per plane) -- removes noise; reduces false duplicate matches
4. Multi-plane Duplicate Review -- cross-plane matching on the clean set
5. Stimulus response analysis -- uses the filtered neuron set

You do not need to re-run CNMF or motion correction after making changes here. The changes are saved to `concat_z_is_cell.npy` for each z-plane and the analysis stage picks them up automatically.

## Running it

In the Luceo GUI Run tab, check "Multi-plane duplicate review" and click Run. You need at least 2 z-planes with saved CNMF results. Motion correction and CNMF can be unchecked if already done.

The window loads all z-planes, computes duplicates, and opens two views:

- The tkinter window shows a 2-D spatial overlay and the duplicate resolution panel.
- A plotly 3-D reconstruction opens automatically in your default browser.

## Dependency: plotly

The 3-D browser view requires plotly. If it is not installed, the button still appears but shows a warning. Install it with:

```
pip install plotly
```

### Step 1: Centroid distance pre-filter

Pairs whose centroids are more than 50 pixels apart are skipped immediately. A neuron with a 25 micrometer soma at 1.21 micrometers per pixel spans about 20 pixels in diameter, so two copies of the same cell cannot be more than 50 pixels apart in x-y.

### Step 2: Jaccard IoU of spatial masks

The primary spatial criterion is the Jaccard index (intersection over union) of the two binary spatial footprints:

```
IoU = (pixels in both masks) / (pixels in either mask)
```

A value of 0 means no shared pixels. A value of 1 means identical footprints. The default threshold is 0.60.

Two adjacent but distinct neurons can have centroids 8 pixels apart and still have IoU near zero because their pixel masks do not overlap. This is why IoU is a stronger criterion than centroid distance alone.

### Step 3: Trace correlation

Pairs that pass the spatial test are then checked for temporal correlation. The Pearson correlation coefficient r is computed between the two denoised traces (C) over the full recording length. The default threshold is 0.85.

The key reason this works well: during baseline periods and spontaneous activity between stimuli, two genuinely different neurons will have independent fluctuations. A real duplicate will track perfectly in those quiet periods because it is the same intracellular calcium signal. This pushes genuine duplicates to r above 0.90-0.97, while nearby neurons that share stimulus selectivity tend to sit at r between 0.60 and 0.80.

### Sorting

Detected pairs are sorted by IoU descending so the most certain duplicates appear first.

## The interface

### 2-D overlay (left panel)

Shows all z-plane neuron outlines overlaid on the combined max-projection mean image. Each z-plane has a distinct colour (blue, orange, green, purple). Accepted neurons are shown at full brightness; rejected neurons are dim grey. Duplicate candidates have a white ring at their centroid and a small ID label (for example, z3 #5).

### 3-D reconstruction (browser)

Opens automatically in your default browser using plotly. Each neuron is rendered as its actual spatial footprint: boundary pixels as solid coloured dots and a sparse interior fill for a volume effect. Rejected neurons appear as dim grey outlines. Duplicate pairs are connected by white dashed lines between centroids.

Every accepted neuron has its ID displayed at its centroid in the browser view. Diamond markers indicate duplicate candidates. You can rotate, zoom, and hover over any neuron to see its ID, centroid coordinates, and peak fluorescence.

### Threshold sliders

Two sliders adjust the detection thresholds:

- Min IoU: minimum Jaccard overlap to flag a pair (default 0.60). Increase to see fewer, more certain candidates. Decrease to cast a wider net.
- Min corr: minimum trace correlation (default 0.85). Increase to require more synchronous activity.

After adjusting, press Re-detect to rerun the detection and refresh both the overlay and the duplicate list.

### Search box

Type a neuron ID to filter the duplicate list. Supported formats:

- `55` -- shows all pairs containing neuron number 55 (any plane)
- `z3` -- shows all pairs containing any neuron from z-plane 3
- `z3 #5` -- shows exactly the pair(s) involving z3 neuron 5

The search does not cross-match the z-plane number with the neuron number, so typing `55` will not incorrectly match z5 neuron 5.

### Duplicate cards

Each card shows:

- The two neuron IDs in their plane colours
- IoU, r, and centroid distance
- Centroid coordinates and peak fluorescence for each copy
- Four action buttons: Keep first plane, Keep second plane, Keep both, Reject both

Resolved cards turn green-tinted. Unresolved cards stay dark.

## Deciding which plane to keep

Use the peak fluorescence value. The plane with the higher peak is the one where the soma sits closest to the focal plane. The 2-photon signal is a quadratic process and falls steeply with distance from focus, so the brightest copy is always the most accurate.

Your acquisition parameters: 25x 0.95 NA objective, z-step 24 micrometers, 1.21 micrometers per pixel lateral.

At 24 micrometers per z-step:

- Typical cortical neuron (10 to 20 micrometers soma): appears in 1 to 2 planes.
- Brainstem neurons in the DVC, NTS, and DMV (25 to 50 micrometers soma): commonly appear in 2 to 3 consecutive planes. This is expected and correct, not an artifact.

Rules of thumb:

- 2 planes: keep the higher peak, reject the lower.
- 3 planes: keep the middle plane. It almost always has the highest peak and the most complete circular footprint. The flanking planes are imaging the top and bottom caps of the soma.
- Peaks within 10 percent of each other: the soma centre sits near the plane boundary. Either copy is acceptable. Prefer the one with the tighter circular footprint in the 3-D browser.

## How results feed into analysis

When you click Save and Close, the window writes updated `concat_z_is_cell.npy` files for each z-plane. Rejected neurons in those files have their flag set to False.

When you then run Stimulus response analysis (with motion correction and CNMF unchecked), `get_stims_n` loads the is_cell file for each z-plane and silently excludes the rejected neurons. The heatmap, responder counts, and spatial maps all reflect the filtered set automatically.

Nothing else needs to change. The CNMF result is untouched.

## Effect on neuron counts

If you find a neuron in z1, z2, and z3 and keep only z1:

- Before: 1 real cell counted 3 times in the analysis
- After: 1 real cell counted once

The total neuron count decreases by 2 for that cell. This is correct. Those 2 entries were not independent neurons; they were the same intracellular signal recorded at different focal depths. Removing them makes your responder percentages and heatmap row counts accurate.
