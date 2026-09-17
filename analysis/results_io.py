"""Per-neuron analysis results: written for each mouse, read by combine_mice.py.

Files in <mouse>/analysis/:
    neurons.csv          one row per accepted neuron (all neurons, not only responders)
    traces_stim<j>.npy   (K, T) float32 z-score trace per neuron for stimulus j,
                         same row order as neurons.csv

neurons.csv columns:
    mouse                animal label (subject, or <subject>_animal<i> in multi-animal runs)
    z_plane              e.g. z1
    plane_index          position among the accepted neurons of that plane
                         (the neuron viewer's order)
    region               AP, NTS, or unclassified (no AP outline on that plane)
    median_z_stim<j>     median z-score over the stimulus-j window
    responds_stim<j>     1 if median_z_stim<j> > threshold, else 0
    responder            1 if the neuron responds to any stimulus
    group                display group (analysis.responders.group_names), or none

Numpy / pandas only, no CaImAn import.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis.responders import stim_medians, classify_from_medians, group_names

NEURONS_CSV = "neurons.csv"
REGIONS = ("AP", "NTS")
_REGION_CODES = {0: "AP", 1: "NTS"}

def trace_file(j):
    return f"traces_stim{j + 1}.npy"

def neuron_table(stims_n, z_ids, region_labels, mouse_labels,
                 stim_onset_idx, threshold):
    """Build the neurons.csv table.

    stims_n       list of N (K, T) z-score arrays
    z_ids         (K,) int z-plane number per row
    region_labels (K,) int, 0 = AP, 1 = NTS, -1 = unclassified; or None
    mouse_labels  (K,) str per row, or one str for every row
    """
    N = len(stims_n)
    K = stims_n[0].shape[0]
    z_ids = np.asarray(z_ids, dtype=int)
    if region_labels is None:
        region_labels = np.full(K, -1, dtype=int)
    if isinstance(mouse_labels, str):
        mouse_labels = [mouse_labels] * K
    if not (len(z_ids) == len(region_labels) == len(mouse_labels) == K):
        raise ValueError(
            f"row counts differ: traces {K}, z_ids {len(z_ids)}, "
            f"regions {len(region_labels)}, mice {len(mouse_labels)}")

    medians = stim_medians(stims_n, stim_onset_idx).reshape(K, N)
    responds, group = classify_from_medians(medians, threshold)
    names = group_names(N)

    df = pd.DataFrame({
        "mouse":   [str(m) for m in mouse_labels],
        "z_plane": [f"z{z}" for z in z_ids],
    })
    df["plane_index"] = df.groupby(["mouse", "z_plane"], sort=False).cumcount()
    df["region"] = [_REGION_CODES.get(int(r), "unclassified") for r in region_labels]
    for j in range(N):
        df[f"median_z_stim{j + 1}"] = medians[:, j]
    for j in range(N):
        df[f"responds_stim{j + 1}"] = responds[:, j].astype(int)
    df["responder"] = (group >= 0).astype(int)
    df["group"] = [names[g] if g >= 0 else "none" for g in group]
    return df

def save_neuron_results(results_dir, stims_n, z_ids, region_labels, mouse_labels,
                        stim_onset_idx, threshold):
    """Write neurons.csv and traces_stim<j>.npy; return the table."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df = neuron_table(stims_n, z_ids, region_labels, mouse_labels,
                      stim_onset_idx, threshold)
    df.to_csv(results_dir / NEURONS_CSV, index=False, float_format="%.6f")
    for j, s in enumerate(stims_n):
        np.save(results_dir / trace_file(j), np.asarray(s, dtype=np.float32))
    return df

def find_analysis_dir(path):
    """The folder holding neurons.csv: `path` itself or `path`/analysis."""
    path = Path(path)
    for cand in (path, path / "analysis"):
        if (cand / NEURONS_CSV).exists():
            return cand
    raise FileNotFoundError(
        f"No {NEURONS_CSV} in {path} or {path / 'analysis'}.\n"
        "Re-run the Analysis stage for this mouse (motion correction and CNMF "
        "unchecked) to write it.")

def load_neuron_results(path):
    """Read one analysis folder.

    Returns dict(dir, table, traces, params); traces is a list of N (K, T) arrays.
    """
    d = find_analysis_dir(path)
    # keep_default_na=False: a mouse called "NA" must stay a string
    df = pd.read_csv(d / NEURONS_CSV, keep_default_na=False, dtype={"mouse": str})
    N = sum(c.startswith("median_z_stim") for c in df.columns)
    traces = [np.load(d / trace_file(j)) for j in range(N)]
    for j, t in enumerate(traces):
        if t.ndim != 2 or t.shape[0] != len(df):
            raise ValueError(
                f"{d / trace_file(j)} has shape {t.shape} but {NEURONS_CSV} has "
                f"{len(df)} rows; the files come from different runs. "
                "Re-run the Analysis stage for this mouse.")
    params = {}
    if (d / "params.yaml").exists():
        with open(d / "params.yaml") as f:
            params = yaml.safe_load(f) or {}
    return dict(dir=d, table=df, traces=traces, params=params)