"""Responder classification shared by the per-mouse analysis and combine_mice.py.

Numpy only, so scripts that pool saved results can use it without importing CaImAn.

A neuron responds to a stimulus when its median z-score over the stimulus
window (stim onset to the end of the window) is above the threshold.
Display groups:
    N=1   0 responder
    N=2   0 stim-1-only, 1 both, 2 stim-2-only
    N≥3   j = the stimulus with the highest median, among responders
    -1    non-responder (any N)
"""
import numpy as np

def stim_medians(stims_n, stim_onset_idx):
    """(K, N) median z-score of each neuron over each stimulus window."""
    return np.array([np.median(s[:, stim_onset_idx:], axis=1) for s in stims_n]).T

def group_names(N):
    """Name of each display group code 0, 1, …; code -1 is 'none'."""
    if N == 1:
        return ["responder"]
    if N == 2:
        return ["stim1_only", "both", "stim2_only"]
    return [f"stim{j + 1}_primary" for j in range(N)]

def group_labels(N, stim_names):
    """Human-readable version of group_names for figure legends."""
    if N == 1:
        return ["Responder"]
    if N == 2:
        return [f"{stim_names[0]} only", "Both", f"{stim_names[1]} only"]
    return [f"{name} primary" for name in stim_names]

def classify_from_medians(medians, threshold):
    """Responder flags and display group from per-stimulus medians.

    medians : (K, N)
    Returns responds (K, N) bool and group (K,) int (see module docstring).
    """
    medians = np.asarray(medians, dtype=float)
    K, N = medians.shape
    responds = medians > threshold
    group = np.full(K, -1, dtype=int)
    if N == 1:
        group[responds[:, 0]] = 0
    elif N == 2:
        r1, r2 = responds[:, 0], responds[:, 1]
        group[r1 & ~r2] = 0
        group[r1 & r2] = 1
        group[~r1 & r2] = 2
    else:
        any_resp = responds.any(axis=1)
        group[any_resp] = np.argmax(medians[any_resp], axis=1)
    return responds, group

def sort_key(medians, group):
    """Median that orders each neuron within its display group (sorted descending).

    stim-2-only neurons sort by stimulus 2, N≥3 groups by their primary stimulus,
    everything else by stimulus 1; non-responders by their largest median.
    """
    medians = np.asarray(medians, dtype=float)
    K, N = medians.shape
    if N == 2:
        col = np.where(group == 2, 1, 0)
    elif N >= 3:
        col = np.where(group >= 0, group, 0)
    else:
        col = np.zeros(K, dtype=int)
    key = medians[np.arange(K), col]
    if N and K:
        key = np.where(group < 0, medians.max(axis=1), key)
    return key

def responder_order(medians, group, N):
    """Responder row indices, by display group then descending sort_key.

    Returns (sorted_idx, group_sizes).
    """
    key = sort_key(medians, group)
    groups = []
    for g in range(len(group_names(N))):
        idx = np.where(group == g)[0]
        groups.append(idx[np.argsort(-key[idx])])
    sorted_idx = np.concatenate(groups).astype(int)
    return sorted_idx, [len(g) for g in groups]