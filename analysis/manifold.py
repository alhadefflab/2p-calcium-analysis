"""Neural manifold analysis for 2-photon calcium imaging.

Implements the five core concepts from:
    Ebitz R.B. & Hayden B.Y. (2021).
    The population doctrine in cognitive neuroscience.
    Neuron 109, 3055-3068.  https://doi.org/10.1016/j.neuron.2021.07.011

The population is treated as the fundamental unit — not individual neurons.
Structure is revealed by the geometry of collective activity in state space:

    1. Coding dimensions  (Section 3) — directions in state space encoding
       task variables; each neuron receives a scalar loading.
    2. State space / dynamics (Sections 1, 5) — PCA trajectories, trajectory
       speed, and condition separability over time.
    3. Subspaces (Section 4) — coding subspace vs nullspace decomposition.
    4. Manifold dimensionality (Section 2) — participation ratio over time.

No GUI imports — pure computation, safe to use in notebooks or tests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import yaml

try:
    from ripser import ripser as _ripser
    _HAS_RIPSER = True
except ImportError:
    _HAS_RIPSER = False


# ── Coding dimensions ─────────────────────────────────────────────────────────

def compute_coding_dimensions(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
) -> dict:
    """Find the coding dimension via targeted dimensionality reduction.

    The coding dimension is the direction in neural state space that
    maximally encodes the distinction between stimulus conditions, found
    by regressing population activity onto condition labels (Ebitz &
    Hayden 2021, Section 3).

    Each neuron's regression weight = its **loading** on the coding
    dimension.  Positive loaders push the population toward condition-1,
    negative toward condition-2; near-zero = nullspace participation.

    Parameters
    ----------
    stims_n       : list of N arrays (K, T) — z-scored ΔF/F
    stim_onset_idx: index of first stimulus frame

    Returns
    -------
    dict with:
        coding_vector      : (K,) normalised neuron loadings
        projection         : list of N arrays (stim_f,) pop projection over time
        separability       : (stim_f,) distance between condition projections
        explained_variance : float — fraction of between-cond variance captured
        has_coding_dim     : bool — False if N < 2
    """
    N = len(stims_n)
    K = stims_n[0].shape[0]
    stim_windows = [s[:, stim_onset_idx:] for s in stims_n]
    stim_f = stim_windows[0].shape[1]

    if N < 2:
        return {
            "coding_vector": np.zeros(K),
            "projection": [s[:, stim_onset_idx:].mean(axis=0) for s in stims_n],
            "separability": np.zeros(stim_f),
            "explained_variance": 0.0,
            "has_coding_dim": False,
        }

    # (N*stim_f, K) — each row = population state at one time point
    X = np.vstack([w.T for w in stim_windows])
    y = np.repeat(np.arange(N), stim_f).astype(float)

    coding_vector, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    norm = np.linalg.norm(coding_vector)
    if norm > 1e-10:
        coding_vector /= norm

    # Per-condition projection over time
    projection = [w.T @ coding_vector for w in stim_windows]

    # Condition separability: mean pairwise distance of projections
    if N == 2:
        separability = np.abs(projection[0] - projection[1])
    else:
        pairs = [np.abs(projection[i] - projection[j])
                 for i in range(N) for j in range(i + 1, N)]
        separability = np.mean(pairs, axis=0)

    # Explained variance: fraction of between-condition variance on coding dim
    cond_means = np.array([w.T.mean(axis=0) for w in stim_windows])  # (N, K)
    grand_mean = cond_means.mean(axis=0)
    between_var = float(np.sum((cond_means - grand_mean) ** 2))
    proj_means = cond_means @ coding_vector
    proj_var = float(np.sum((proj_means - proj_means.mean()) ** 2))
    explained = proj_var / between_var if between_var > 1e-10 else 0.0

    return {
        "coding_vector": coding_vector,
        "projection": projection,
        "separability": separability,
        "explained_variance": float(explained),
        "has_coding_dim": True,
    }


# ── Trajectory geometry ───────────────────────────────────────────────────────

def compute_trajectory_geometry(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
    n_pca_dims: int = 20,
) -> dict:
    """Analyse state-space trajectory geometry (Ebitz & Hayden 2021, Sections 1 + 5).

    Trajectory speed (|Δstate/Δt|): frame-to-frame Euclidean distance in
    PC-compressed state space.  A spike after stimulus onset indicates a
    rapid state transition driven by the stimulus.

    Condition distance: pairwise Euclidean distance between the N condition
    trajectories at each time point.  Rises from near-zero (shared baseline
    state) when the population begins encoding stimulus identity — its onset
    marks the population-level discrimination latency.

    Parameters
    ----------
    stims_n       : list of N arrays (K, T)
    stim_onset_idx: stimulus onset frame
    n_pca_dims    : PCA dimensions for distance computation (noise reduction)

    Returns
    -------
    dict with:
        speed              : list of N arrays (stim_f - 1,)
        condition_distance : (stim_f,) — pairwise mean distance
        pca_trajectories   : list of N arrays (stim_f, n_pca)
        onset_frame        : int — frame where distance exceeds 2 SD above baseline
        pca                : fitted sklearn PCA object
    """
    from sklearn.decomposition import PCA

    K = stims_n[0].shape[0]
    N = len(stims_n)
    stim_windows = [s[:, stim_onset_idx:] for s in stims_n]
    stim_f = stim_windows[0].shape[1]

    n_pca = min(n_pca_dims, K, stim_f)
    pca = PCA(n_components=n_pca)
    pca.fit(np.vstack([w.T for w in stim_windows]))
    trajectories = [pca.transform(w.T) for w in stim_windows]

    speed = [np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1))
             for traj in trajectories]

    if N >= 2:
        pairs = [np.sqrt(np.sum((trajectories[i] - trajectories[j]) ** 2, axis=1))
                 for i in range(N) for j in range(i + 1, N)]
        condition_distance = np.mean(pairs, axis=0)
    else:
        condition_distance = np.zeros(stim_f)

    # Onset: first frame where distance > baseline mean + 2 SD
    n_baseline = min(10, stim_f // 4)
    bd = condition_distance[:n_baseline]
    threshold = bd.mean() + 2.0 * bd.std()
    above = np.where(condition_distance > threshold)[0]
    onset_frame = int(above[0]) if len(above) > 0 else stim_f

    return {
        "speed": speed,
        "condition_distance": condition_distance,
        "pca_trajectories": trajectories,
        "onset_frame": onset_frame,
        "pca": pca,
    }


# ── Subspace geometry ─────────────────────────────────────────────────────────

def compute_subspace_geometry(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
) -> dict:
    """Decompose activity into coding subspace and nullspace (Section 4).

    The coding subspace is spanned by the LDA discriminant axes.  The
    nullspace is its orthogonal complement.  A high coding-to-null variance
    ratio means most population activity is task-informative.

    Parameters
    ----------
    stims_n       : list of N arrays (K, T)
    stim_onset_idx: stimulus onset frame

    Returns
    -------
    dict with:
        coding_variance_ratio          : float — fraction in coding subspace
        null_variance_ratio            : float — 1 - coding_variance_ratio
        coding_subspace_dim            : int — N - 1 for N conditions
        projection_variance_over_time  : (stim_f,) float
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    N = len(stims_n)
    stim_windows = [s[:, stim_onset_idx:] for s in stims_n]
    stim_f = stim_windows[0].shape[1]

    if N < 2:
        return {
            "coding_variance_ratio": 0.0,
            "null_variance_ratio": 1.0,
            "coding_subspace_dim": 0,
            "projection_variance_over_time": np.zeros(stim_f),
            "lda_scalings": None,
        }

    X = np.vstack([w.T for w in stim_windows])
    y = np.repeat(np.arange(N), stim_f)

    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y)
        W = lda.scalings_  # (K, n_components)
        W_n = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-10)

        X_proj = X @ W_n
        coding_var = float(np.var(X_proj) * W_n.shape[1])
        total_var = float(np.var(X) * X.shape[1])
        ratio = float(np.clip(coding_var / total_var, 0.0, 1.0)) if total_var > 1e-10 else 0.0

        proj_var = np.array([
            float(np.var(np.array([w.T[t] for w in stim_windows]) @ W_n))
            for t in range(stim_f)
        ])
    except Exception:
        ratio = 0.0
        proj_var = np.zeros(stim_f)
        W_n = None

    return {
        "coding_variance_ratio": ratio,
        "null_variance_ratio": 1.0 - ratio,
        "coding_subspace_dim": N - 1,
        "projection_variance_over_time": proj_var,
        "lda_scalings": W_n,
    }


# ── Dimensionality over time ──────────────────────────────────────────────────

def compute_dimensionality_over_time(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
    window_frames: int = 10,
) -> dict:
    """Sliding-window participation ratio across the full trace (Section 2).

    Reveals whether the manifold becomes more or less dimensional during the
    stimulus response.  A decrease = the population becomes more coordinated
    (lower-dimensional, more structured); an increase = more independent
    activity.

    Parameters
    ----------
    stims_n       : list of N arrays (K, T)
    stim_onset_idx: stimulus onset frame
    window_frames : width of the sliding window

    Returns
    -------
    dict with:
        pr_over_time : (T - window_frames,) — participation ratio at each frame
        pr_baseline  : float — mean PR during pre-stimulus window
        pr_stimulus  : float — mean PR during post-onset window
        window_frames: int
    """
    K = stims_n[0].shape[0]
    T = stims_n[0].shape[1]

    # Average across conditions
    mean_trace = np.mean(np.stack(stims_n, axis=0), axis=0)  # (K, T)

    pr_list = []
    for t in range(T - window_frames):
        w = mean_trace[:, t:t + window_frames]
        try:
            eigs = np.linalg.eigvalsh(np.cov(w))
            eigs = np.maximum(eigs, 0.0)
            s, s2 = eigs.sum(), (eigs ** 2).sum()
            pr_list.append(float((s ** 2) / s2) if s2 > 1e-10 else 1.0)
        except Exception:
            pr_list.append(1.0)

    pr_over_time = np.array(pr_list)
    n_pr = len(pr_over_time)

    pr_baseline = float(pr_over_time[:stim_onset_idx].mean()) \
        if stim_onset_idx > 0 and stim_onset_idx <= n_pr else float(pr_over_time[:n_pr // 2].mean())
    pr_stimulus = float(pr_over_time[stim_onset_idx:].mean()) \
        if stim_onset_idx < n_pr else float(pr_over_time.mean())

    return {
        "pr_over_time": pr_over_time,
        "pr_baseline": pr_baseline,
        "pr_stimulus": pr_stimulus,
        "window_frames": window_frames,
    }


# ── Neural states / attractors ────────────────────────────────────────────────

def _pca_trajectories(stims_n, stim_onset_idx, n_pca_dims):
    """Shared helper: fit PCA on pooled stim-window states, return (pca, trajs)."""
    from sklearn.decomposition import PCA

    K = stims_n[0].shape[0]
    stim_windows = [s[:, stim_onset_idx:] for s in stims_n]
    stim_f = stim_windows[0].shape[1]
    n_pca = max(1, min(n_pca_dims, K, stim_f))
    pca = PCA(n_components=n_pca)
    pca.fit(np.vstack([w.T for w in stim_windows]))
    trajectories = [pca.transform(w.T) for w in stim_windows]  # list of (stim_f, n_pca)
    return pca, trajectories


def _run_lengths(labels: np.ndarray) -> list[tuple[int, int]]:
    """Return [(state, run_length), ...] for consecutive runs in `labels`."""
    if len(labels) == 0:
        return []
    runs = []
    cur, length = int(labels[0]), 1
    for v in labels[1:]:
        if int(v) == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur, length = int(v), 1
    runs.append((cur, length))
    return runs


def compute_neural_states(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
    n_pca_dims: int = 10,
    n_states_range: tuple[int, int] = (2, 6),
    random_state: int = 42,
) -> dict:
    """Discover discrete population states (attractor basins) in state space.

    This is the Hopfieldian analog of 'populations': rather than grouping
    neurons, it groups recurring *population states* — regions the trajectory
    visits.  A Gaussian mixture is fit to the pooled state-space trajectory
    points, with the number of states chosen by Bayesian Information Criterion
    (BIC) over `n_states_range`.

    Slow points — local minima of trajectory speed below the 25th percentile —
    are flagged as candidate fixed points / attractors, since |dstate/dt| → 0
    near a stable state (Ebitz & Hayden 2021, Section 5; attractor dynamics).

    Parameters
    ----------
    stims_n        : list of N arrays (K, T) — z-scored ΔF/F
    stim_onset_idx : index of first stimulus frame
    n_pca_dims     : PCA dimensions for the state space
    n_states_range : (min, max) candidate state counts for the BIC sweep
    random_state   : RNG seed

    Returns
    -------
    dict with:
        pca               : fitted sklearn PCA
        trajectories      : list of N arrays (stim_f, n_pca)
        n_states          : int — selected number of states
        state_labels      : list of N arrays (stim_f,) int — state per time point
        state_centroids   : (n_states, n_pca) — GMM component means (PC space)
        occupancy         : (N, n_states) — fraction of time per state per cond
        dwell_frames      : (n_states,) — mean consecutive-frame dwell per state
        transition_matrix : (n_states, n_states) — row-normalised P(i→j)
        slow_point_frames : list of N arrays — frame indices of slow points
        speed             : list of N arrays (stim_f-1,)
        bic_scores        : dict {k: bic}
    """
    from sklearn.mixture import GaussianMixture

    pca, trajectories = _pca_trajectories(stims_n, stim_onset_idx, n_pca_dims)
    N = len(trajectories)
    stim_f = trajectories[0].shape[0]
    pooled = np.vstack(trajectories)  # (N*stim_f, n_pca)

    # BIC sweep to choose the number of states
    lo, hi = n_states_range
    hi = max(lo, min(hi, pooled.shape[0] - 1))
    bic_scores: dict[int, float] = {}
    best_k, best_bic, best_gmm = lo, np.inf, None
    for k in range(lo, hi + 1):
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="full",
                                  random_state=random_state, reg_covar=1e-5)
            gmm.fit(pooled)
            bic = float(gmm.bic(pooled))
        except Exception:
            continue
        bic_scores[k] = bic
        if bic < best_bic:
            best_k, best_bic, best_gmm = k, bic, gmm

    if best_gmm is None:  # degenerate fallback — single state
        best_k = 1
        state_labels = [np.zeros(stim_f, dtype=int) for _ in trajectories]
        centroids = pooled.mean(axis=0, keepdims=True)
    else:
        state_labels = [best_gmm.predict(traj).astype(int) for traj in trajectories]
        centroids = best_gmm.means_

    # Per-condition occupancy
    occupancy = np.zeros((N, best_k))
    for j, lab in enumerate(state_labels):
        for s in range(best_k):
            occupancy[j, s] = np.mean(lab == s)

    # Mean dwell time (consecutive frames) per state, pooled across conditions
    dwell_accum = {s: [] for s in range(best_k)}
    transition = np.zeros((best_k, best_k))
    for lab in state_labels:
        for s, length in _run_lengths(lab):
            dwell_accum[s].append(length)
        for a, b in zip(lab[:-1], lab[1:]):
            transition[int(a), int(b)] += 1
    dwell_frames = np.array([
        float(np.mean(dwell_accum[s])) if dwell_accum[s] else 0.0
        for s in range(best_k)])
    row_sums = transition.sum(axis=1, keepdims=True)
    transition_matrix = np.divide(transition, row_sums,
                                  out=np.zeros_like(transition),
                                  where=row_sums > 0)

    # Trajectory speed + slow points (candidate attractors)
    speed = [np.sqrt(np.sum(np.diff(traj, axis=0) ** 2, axis=1))
             for traj in trajectories]
    slow_point_frames = []
    for spd in speed:
        if len(spd) < 3:
            slow_point_frames.append(np.array([], dtype=int))
            continue
        thresh = np.percentile(spd, 25)
        local_min = (spd[1:-1] <= spd[:-2]) & (spd[1:-1] <= spd[2:])
        below = spd[1:-1] <= thresh
        idx = np.where(local_min & below)[0] + 1  # +1 → index into speed array
        slow_point_frames.append(idx.astype(int))

    return {
        "pca": pca,
        "trajectories": trajectories,
        "n_states": best_k,
        "state_labels": state_labels,
        "state_centroids": centroids,
        "occupancy": occupancy,
        "dwell_frames": dwell_frames,
        "transition_matrix": transition_matrix,
        "slow_point_frames": slow_point_frames,
        "speed": speed,
        "bic_scores": bic_scores,
    }


# ── Manifold topology (persistent homology) ───────────────────────────────────

def _significant_loops(h1: np.ndarray, min_persistence: float) -> int:
    """Count H1 features whose persistence (death − birth) exceeds threshold."""
    if h1 is None or len(h1) == 0:
        return 0
    pers = h1[:, 1] - h1[:, 0]
    return int(np.sum(pers > min_persistence))


def compute_manifold_topology(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
    n_pca_dims: int = 10,
    max_dim: int = 1,
    random_state: int = 42,
) -> dict:
    """Persistent homology of the population trajectory manifold.

    Characterises the *shape* of the neural manifold, not just its
    dimensionality.  Persistent homology is computed on the trajectory point
    cloud (time points in PC space):

        H0 — connected components (discrete state clusters)
        H1 — loops (cyclic / rotational dynamics, e.g. ring-attractor signatures)

    Computed both pooled across conditions (overall manifold shape) and per
    condition (lets you compare manifold geometry between stimuli).

    Parameters
    ----------
    stims_n        : list of N arrays (K, T)
    stim_onset_idx : stimulus onset frame
    n_pca_dims     : PCA dimensions for the point cloud
    max_dim        : maximum homology dimension (1 = up to loops)
    random_state   : RNG seed (unused directly; kept for API symmetry)

    Returns
    -------
    dict with:
        tda_available        : bool — ripser installed and run
        pooled_diagrams      : list of persistence diagrams [H0, H1] or None
        per_condition_diagrams: list of N diagram-lists or None
        n_loops_pooled       : int — significant H1 features (pooled)
        max_h1_persistence   : float — longest-lived loop (pooled)
        n_pca_dims           : int
    """
    if not _HAS_RIPSER:
        return {
            "tda_available": False,
            "pooled_diagrams": None,
            "per_condition_diagrams": None,
            "n_loops_pooled": 0,
            "max_h1_persistence": 0.0,
            "n_pca_dims": n_pca_dims,
        }

    _, trajectories = _pca_trajectories(stims_n, stim_onset_idx, n_pca_dims)
    pooled = np.vstack(trajectories)

    pooled_diagrams = _ripser(pooled, maxdim=max_dim)["dgms"]
    per_condition_diagrams = [
        _ripser(traj, maxdim=max_dim)["dgms"] for traj in trajectories]

    # Significance threshold: 10 % of the H0 spread (largest finite death)
    h0 = pooled_diagrams[0]
    finite_h0 = h0[np.isfinite(h0[:, 1])]
    h0_scale = float(finite_h0[:, 1].max()) if len(finite_h0) else 1.0
    min_pers = 0.10 * h0_scale

    h1 = pooled_diagrams[1] if len(pooled_diagrams) > 1 else None
    n_loops = _significant_loops(h1, min_pers)
    max_h1 = float((h1[:, 1] - h1[:, 0]).max()) if (h1 is not None and len(h1)) else 0.0

    return {
        "tda_available": True,
        "pooled_diagrams": pooled_diagrams,
        "per_condition_diagrams": per_condition_diagrams,
        "n_loops_pooled": n_loops,
        "max_h1_persistence": max_h1,
        "n_pca_dims": n_pca_dims,
    }


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_manifold_analysis(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
    results_dir: str,
    fp: float = 0.585,
    stim_names: Optional[list[str]] = None,
    centroids: Optional[np.ndarray] = None,
) -> dict:
    """Run all manifold analyses, save figures + summary, return results dict.

    Parameters
    ----------
    stims_n        : list of N arrays (K, T) — z-scored ΔF/F
    stim_onset_idx : index of first stimulus frame
    results_dir    : parent results directory; 'manifold/' subdir is created
    fp             : seconds per frame
    stim_names     : human-readable stimulus labels
    centroids      : (K, 2) array of neuron centroid (row, col) or None

    Returns
    -------
    Merged dict from all four sub-analyses plus stims_n, fp, stim_names,
    centroids, manifold_dir, results_dir.
    """
    from visualization.manifold_plots import (
        plot_coding_dimensions,
        plot_trajectory_geometry,
        plot_subspace,
        plot_dimensionality_over_time,
        plot_neural_states,
        plot_manifold_topology,
    )

    N = len(stims_n)
    K = stims_n[0].shape[0]
    if stim_names is None:
        stim_names = [f"Stimulus {j + 1}" for j in range(N)]

    manifold_dir = Path(results_dir) / "manifold"
    manifold_dir.mkdir(parents=True, exist_ok=True)

    coding     = compute_coding_dimensions(stims_n, stim_onset_idx)
    geometry   = compute_trajectory_geometry(stims_n, stim_onset_idx)
    subspace   = compute_subspace_geometry(stims_n, stim_onset_idx)
    dim_time   = compute_dimensionality_over_time(stims_n, stim_onset_idx)
    states     = compute_neural_states(stims_n, stim_onset_idx)
    topology   = compute_manifold_topology(stims_n, stim_onset_idx)

    # Figures
    plot_coding_dimensions(
        coding["coding_vector"], coding["projection"], coding["separability"],
        stim_names, fp, stim_onset_idx, str(manifold_dir))
    plot_trajectory_geometry(
        geometry["pca_trajectories"], geometry["speed"],
        geometry["condition_distance"], geometry["onset_frame"],
        stim_names, fp, str(manifold_dir))
    plot_subspace(
        subspace["coding_variance_ratio"],
        subspace["projection_variance_over_time"],
        stim_onset_idx, fp, str(manifold_dir))
    plot_dimensionality_over_time(
        dim_time["pr_over_time"], dim_time["pr_baseline"],
        dim_time["pr_stimulus"], stim_onset_idx, fp, str(manifold_dir))
    plot_neural_states(states, stim_names, fp, str(manifold_dir))
    if topology["tda_available"]:
        plot_manifold_topology(topology, stim_names, str(manifold_dir))

    # Persist numeric artefacts
    np.save(str(manifold_dir / "coding_vector.npy"), coding["coding_vector"])
    np.save(str(manifold_dir / "state_centroids.npy"), states["state_centroids"])

    summary = {
        "n_neurons": K,
        "n_stimuli": N,
        "stim_names": stim_names,
        "coding_explained_variance": round(coding["explained_variance"], 4),
        "coding_subspace_dim": subspace["coding_subspace_dim"],
        "coding_variance_ratio": round(subspace["coding_variance_ratio"], 4),
        "null_variance_ratio": round(subspace["null_variance_ratio"], 4),
        "pr_baseline": round(dim_time["pr_baseline"], 2),
        "pr_stimulus": round(dim_time["pr_stimulus"], 2),
        "onset_frame": geometry["onset_frame"],
        "onset_time_s": round(geometry["onset_frame"] * fp, 3),
        "n_states": states["n_states"],
        "state_dwell_frames": [round(float(d), 2) for d in states["dwell_frames"]],
        "tda_available": topology["tda_available"],
        "n_loops_pooled": topology["n_loops_pooled"],
        "max_h1_persistence": round(topology["max_h1_persistence"], 4),
    }
    with open(str(manifold_dir / "summary.yaml"), "w") as fh:
        yaml.dump(summary, fh, default_flow_style=False)

    return {
        **coding,
        **geometry,
        **subspace,
        **dim_time,
        "states": states,
        "topology": topology,
        "stims_n": stims_n,
        "stim_onset_idx": stim_onset_idx,
        "fp": fp,
        "stim_names": stim_names,
        "centroids": centroids,
        "manifold_dir": str(manifold_dir),
        "results_dir": results_dir,
    }
