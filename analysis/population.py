"""Single-neuron / cell-type analysis for 2-photon calcium imaging.

The Sherringtonian view: the neuron (and the circuit it forms) is the unit
of explanation.  Neurons are grouped into functional cell-types by
co-activity clustering, then characterised by single-neuron tuning /
selectivity, cluster quality, pairwise functional connectivity, and the
spatial (anatomical) organisation of the cell-types — including the
topology of their 3-D arrangement.

This is deliberately NOT the population-doctrine / manifold framework
(state space, coding dimensions, attractor geometry); for that Hopfieldian
view see analysis/manifold.py.

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


# ── Clustering ────────────────────────────────────────────────────────────────

def _build_cluster_features(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (features_pca, pca_variance_explained) used for clustering.

    Stim windows are concatenated across conditions, z-scored neuron-wise for
    scale invariance, then PCA-reduced (≤20 dims) for noise suppression.
    """
    from sklearn.decomposition import PCA

    K = stims_n[0].shape[0]
    stim_windows = [s[:, stim_onset_idx:] for s in stims_n]
    features = np.concatenate(stim_windows, axis=1)  # (K, N*stim_f)

    f_mean = features.mean(axis=1, keepdims=True)
    f_std  = features.std(axis=1, keepdims=True)
    f_std[f_std < 1e-10] = 1.0
    features_z = (features - f_mean) / f_std

    n_pca = min(20, K, features_z.shape[1])
    pca = PCA(n_components=n_pca, random_state=random_state)
    features_pca = pca.fit_transform(features_z)  # (K, n_pca)
    return features_pca, pca.explained_variance_ratio_


def compute_coactivity_clusters(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
    n_clusters: int = 6,
    method: str = "kmeans",
    random_state: int = 42,
) -> dict:
    """Cluster neurons into functional cell-types by co-activity patterns.

    Parameters
    ----------
    stims_n       : list of N arrays (K, T) — z-scored ΔF/F, full window
    stim_onset_idx: index of stimulus onset frame (baseline precedes this)
    n_clusters    : number of desired clusters
    method        : "kmeans" | "gmm" | "hierarchical"
    random_state  : RNG seed for reproducibility

    Returns
    -------
    dict with keys
        labels               : (K,) int  — cluster id per neuron [0, n_clusters)
        n_clusters           : int
        cluster_means        : list of N arrays (n_clusters, T) — mean trace per cluster
        cluster_sizes        : (n_clusters,) int
        method               : str
        pca_variance_explained: (n_pca,) float — scree for the pre-clustering PCA
        features_pca         : (K, n_pca) float — feature matrix used to cluster
    """
    K = stims_n[0].shape[0]
    n_clusters = min(n_clusters, K)

    features_pca, pca_var = _build_cluster_features(
        stims_n, stim_onset_idx, random_state)

    labels = _cluster(features_pca, n_clusters, method, random_state)

    # Per-cluster mean traces (on the original z-scored arrays)
    cluster_means = []
    for s in stims_n:
        means = np.zeros((n_clusters, s.shape[1]))
        for c in range(n_clusters):
            mask = labels == c
            if mask.any():
                means[c] = s[mask].mean(axis=0)
        cluster_means.append(means)

    cluster_sizes = np.array([(labels == c).sum() for c in range(n_clusters)])

    return {
        "labels": labels,
        "n_clusters": n_clusters,
        "cluster_means": cluster_means,
        "cluster_sizes": cluster_sizes,
        "method": method,
        "pca_variance_explained": pca_var,
        "features_pca": features_pca,
    }


def _cluster(features: np.ndarray, n_clusters: int, method: str, random_state: int) -> np.ndarray:
    if method == "kmeans":
        from sklearn.cluster import KMeans
        return KMeans(n_clusters=n_clusters, random_state=random_state,
                      n_init=10).fit_predict(features)
    if method == "gmm":
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(n_components=n_clusters,
                               random_state=random_state).fit_predict(features)
    if method == "hierarchical":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=n_clusters,
                                       linkage="ward").fit_predict(features)
    raise ValueError(f"Unknown clustering method: {method!r}")


# ── Single-neuron selectivity / tuning ────────────────────────────────────────

def compute_neuron_selectivity(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
) -> dict:
    """Per-neuron stimulus tuning and selectivity (single-neuron doctrine).

    Each neuron's mean post-onset response to every stimulus gives a tuning
    profile.  The selectivity index (SI) summarises how stimulus-specific a
    neuron is:

        SI = (N − Σ(rᵢ / r_max)) / (N − 1)     with rᵢ = max(response, 0)

    SI = 1 → responds to a single stimulus; SI = 0 → responds equally to all.

    Parameters
    ----------
    stims_n        : list of N arrays (K, T) — z-scored ΔF/F
    stim_onset_idx : stimulus onset frame index

    Returns
    -------
    dict with keys
        response_matrix : (K, N) float — mean post-onset z per neuron × stimulus
        preferred_stim  : (K,) int — argmax response stimulus per neuron
        selectivity     : (K,) float — selectivity index in [0, 1]
        n_stim          : int
    """
    N = len(stims_n)
    K = stims_n[0].shape[0]

    response = np.zeros((K, N))
    for j, s in enumerate(stims_n):
        response[:, j] = s[:, stim_onset_idx:].mean(axis=1)

    preferred_stim = np.argmax(response, axis=1).astype(int)

    if N >= 2:
        r = np.clip(response, 0.0, None)
        r_max = r.max(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_sum = np.where(r_max > 1e-10,
                                 r.sum(axis=1) / r_max, 1.0)
        selectivity = (N - ratio_sum) / (N - 1)
        selectivity = np.clip(selectivity, 0.0, 1.0)
    else:
        selectivity = np.zeros(K)

    return {
        "response_matrix": response,
        "preferred_stim": preferred_stim,
        "selectivity": selectivity,
        "n_stim": N,
    }


# ── Cluster quality ───────────────────────────────────────────────────────────

def compute_cluster_quality(
    features_pca: np.ndarray,
    labels: np.ndarray,
    method: str = "kmeans",
    k_range: tuple[int, int] = (2, 12),
    random_state: int = 42,
) -> dict:
    """Silhouette analysis to judge / justify the number of cell-types.

    The silhouette score (∈ [−1, 1]) measures how well each neuron sits in its
    own cluster vs the nearest other cluster.  Sweeping k highlights the
    cluster count the data actually supports (a peak in the curve).

    Parameters
    ----------
    features_pca : (K, n_pca) — the matrix neurons were clustered on
    labels       : (K,) — current cluster assignment (for current_silhouette)
    method       : clustering method to use for the sweep
    k_range      : (min_k, max_k) inclusive sweep range
    random_state : RNG seed

    Returns
    -------
    dict with keys
        silhouette_by_k    : dict {k: mean silhouette}
        best_k             : int — k with the highest silhouette
        current_silhouette : float — silhouette of the supplied labels
        sample_silhouette  : (K,) float — per-neuron silhouette for current labels
    """
    from sklearn.metrics import silhouette_score, silhouette_samples

    K = features_pca.shape[0]
    lo, hi = k_range
    hi = max(lo, min(hi, K - 1))

    silhouette_by_k: dict[int, float] = {}
    for k in range(lo, hi + 1):
        try:
            lab_k = _cluster(features_pca, k, method, random_state)
            if len(np.unique(lab_k)) < 2:
                continue
            silhouette_by_k[k] = float(silhouette_score(features_pca, lab_k))
        except Exception:
            continue

    best_k = max(silhouette_by_k, key=silhouette_by_k.get) if silhouette_by_k else lo

    current_silhouette = 0.0
    sample_silhouette = np.zeros(K)
    if len(np.unique(labels)) >= 2:
        try:
            current_silhouette = float(silhouette_score(features_pca, labels))
            sample_silhouette = silhouette_samples(features_pca, labels)
        except Exception:
            pass

    return {
        "silhouette_by_k": silhouette_by_k,
        "best_k": best_k,
        "current_silhouette": current_silhouette,
        "sample_silhouette": sample_silhouette,
    }


# ── Functional connectivity ───────────────────────────────────────────────────

def compute_functional_connectivity(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
    labels: np.ndarray,
) -> dict:
    """Pairwise neuron-neuron correlation, ordered by cell-type (circuit view).

    Correlations are computed over the concatenated stim-window activity.
    Sorting the matrix by cluster reveals whether co-active groups form
    coherent functional blocks (the closest Sherringtonian analog of circuits).

    Parameters
    ----------
    stims_n        : list of N arrays (K, T)
    stim_onset_idx : stimulus onset frame index
    labels         : (K,) cluster assignment

    Returns
    -------
    dict with keys
        corr            : (K, K) float — Pearson correlation matrix
        order           : (K,) int — neuron order sorting by cluster
        labels_sorted   : (K,) int — labels under `order`
        within_corr     : float — mean |corr| within clusters
        between_corr     : float — mean |corr| between clusters
        modularity_ratio: float — within / between (>1 ⇒ block structure)
    """
    stim_windows = [s[:, stim_onset_idx:] for s in stims_n]
    X = np.concatenate(stim_windows, axis=1)  # (K, N*stim_f)
    corr = np.corrcoef(X)
    corr = np.nan_to_num(corr, nan=0.0)

    order = np.argsort(labels, kind="stable")
    labels_sorted = labels[order]

    K = len(labels)
    same = labels[:, None] == labels[None, :]
    off_diag = ~np.eye(K, dtype=bool)
    abs_corr = np.abs(corr)
    within_mask = same & off_diag
    between_mask = (~same) & off_diag
    within_corr = float(abs_corr[within_mask].mean()) if within_mask.any() else 0.0
    between_corr = float(abs_corr[between_mask].mean()) if between_mask.any() else 0.0
    modularity_ratio = within_corr / between_corr if between_corr > 1e-10 else 0.0

    return {
        "corr": corr,
        "order": order,
        "labels_sorted": labels_sorted,
        "within_corr": within_corr,
        "between_corr": between_corr,
        "modularity_ratio": modularity_ratio,
    }


# ── Spatial organisation + spatial topology ───────────────────────────────────

def _centroids_3d(centroids: np.ndarray, z_ids: Optional[np.ndarray]) -> np.ndarray:
    """Build a (K, 3) coordinate array (x=col, y=row, z=plane) with balanced z.

    The z axis (imaging plane) is scaled so plane separation spans roughly the
    same range as the in-plane (x, y) extent — otherwise distances are
    dominated by whichever axis has the larger raw units.
    """
    cols = centroids[:, 1].astype(float)
    rows = centroids[:, 0].astype(float)
    if z_ids is None or len(z_ids) != len(centroids):
        z = np.zeros(len(centroids))
    else:
        z = z_ids.astype(float)
    n_planes = len(np.unique(z))
    if n_planes > 1:
        xy_extent = max(cols.max() - cols.min(), rows.max() - rows.min())
        z_scale = xy_extent / (z.max() - z.min())
        z = z * z_scale
    return np.column_stack([cols, rows, z])


def compute_spatial_organization(
    centroids: Optional[np.ndarray],
    z_ids: Optional[np.ndarray],
    labels: np.ndarray,
    run_tda: bool = False,
    tda_max_dim: int = 2,
    tda_n_points: int = 400,
    n_permutations: int = 1000,
    random_state: int = 42,
) -> dict:
    """Anatomical organisation of cell-types and the topology of their layout.

    Two questions are answered:

    1. *Do cell-types aggregate in space?*  For each cluster the mean pairwise
       3-D distance among its members is compared to a label-permutation null.
       A small observed distance (low p) ⇒ the cell-type is spatially compact,
       i.e. the functional grouping has anatomical meaning.

    2. *What is the topology of the arrangement?*  Persistent homology on the
       3-D centroid cloud (H0 components, H1 loops, H2 voids) characterises the
       shape of the population's spatial layout.

    Parameters
    ----------
    centroids      : (K, 2) row/col or None
    z_ids          : (K,) imaging-plane index or None
    labels         : (K,) cluster assignment
    run_tda        : compute spatial persistent homology (requires ripser)
    tda_max_dim    : max homology dimension (2 ⇒ include voids)
    tda_n_points   : subsample cap for TDA tractability
    n_permutations : permutations for the aggregation test
    random_state   : RNG seed

    Returns
    -------
    dict with keys
        available          : bool — centroids present
        coords3d           : (K, 3) float or None
        cluster_pvalues    : (n_clusters,) — aggregation p-value per cluster
        observed_dist      : (n_clusters,) — observed mean within-cluster dist
        null_dist          : (n_clusters,) — null mean within-cluster dist
        spatial_diagrams   : list of persistence diagrams or None
        tda_available      : bool
    """
    if centroids is None or len(centroids) == 0:
        return {
            "available": False, "coords3d": None,
            "cluster_pvalues": np.array([]), "observed_dist": np.array([]),
            "null_dist": np.array([]), "spatial_diagrams": None,
            "tda_available": False,
        }

    coords = _centroids_3d(centroids, z_ids)
    K = coords.shape[0]
    n_clusters = int(labels.max()) + 1 if len(labels) else 0
    rng = np.random.default_rng(random_state)

    def _mean_pairwise(idx: np.ndarray) -> float:
        if len(idx) < 2:
            return 0.0
        pts = coords[idx]
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
        iu = np.triu_indices(len(idx), k=1)
        return float(d[iu].mean())

    cluster_pvalues = np.ones(n_clusters)
    observed_dist = np.zeros(n_clusters)
    null_dist = np.zeros(n_clusters)
    for c in range(n_clusters):
        members = np.where(labels == c)[0]
        size = len(members)
        if size < 2:
            continue
        obs = _mean_pairwise(members)
        null = np.empty(n_permutations)
        for p in range(n_permutations):
            null[p] = _mean_pairwise(rng.choice(K, size, replace=False))
        observed_dist[c] = obs
        null_dist[c] = float(null.mean())
        # one-sided: clustered ⇒ observed smaller than random
        cluster_pvalues[c] = float((np.sum(null <= obs) + 1) / (n_permutations + 1))

    spatial_diagrams = None
    tda_available = _HAS_RIPSER and run_tda
    if tda_available:
        spatial_diagrams = compute_spatial_topology(
            coords, max_dim=tda_max_dim, n_points=tda_n_points,
            random_state=random_state)

    return {
        "available": True,
        "coords3d": coords,
        "cluster_pvalues": cluster_pvalues,
        "observed_dist": observed_dist,
        "null_dist": null_dist,
        "spatial_diagrams": spatial_diagrams,
        "tda_available": tda_available,
    }


def compute_spatial_topology(
    coords3d: np.ndarray,
    max_dim: int = 2,
    n_points: int = 400,
    random_state: int = 42,
) -> Optional[list]:
    """Persistent homology of the 3-D centroid cloud (H0/H1/H2).

    Returns the list of persistence diagrams, or None if ripser is unavailable.
    Subsamples to `n_points` for tractability.
    """
    if not _HAS_RIPSER or coords3d is None or len(coords3d) == 0:
        return None
    pts = coords3d
    if len(pts) > n_points:
        rng = np.random.default_rng(random_state)
        pts = pts[rng.choice(len(pts), n_points, replace=False)]
    return _ripser(pts, maxdim=max_dim)["dgms"]


# ── Top-level entry point ─────────────────────────────────────────────────────

def run_population_analysis(
    stims_n: list[np.ndarray],
    stim_onset_idx: int,
    results_dir: str,
    n_clusters: int = 6,
    cluster_method: str = "kmeans",
    run_tda: bool = False,
    fp: float = 0.585,
    stim_names: Optional[list[str]] = None,
    centroids: Optional[np.ndarray] = None,
    z_ids: Optional[np.ndarray] = None,
) -> dict:
    """Run all cell-type analyses, save figures + summary, return results dict.

    Sherringtonian pipeline: co-activity clustering into cell-types, then
    single-neuron selectivity, cluster quality, functional connectivity, and
    spatial organisation (+ spatial topology).  Figure generation is delegated
    to visualization.population_plots.  Never calls plt.show().

    Parameters
    ----------
    stims_n        : list of N arrays (K, T) — z-scored ΔF/F
    stim_onset_idx : index of first stimulus frame
    results_dir    : path where results are saved; a 'population/' subdir is created
    n_clusters     : initial cluster count
    cluster_method : "kmeans" | "gmm" | "hierarchical"
    run_tda        : run spatial persistent homology (requires ripser)
    fp             : seconds per frame (for time axes)
    stim_names     : optional human-readable stimulus labels
    centroids      : (K, 2) row/col neuron centroids or None
    z_ids          : (K,) imaging-plane index per neuron or None

    Returns
    -------
    Merged result dict combining all sub-analysis outputs, plus:
        stims_n, stim_onset_idx, fp, stim_names, pop_dir, centroids, z_ids
    """
    from visualization.population_plots import (
        plot_cluster_traces,
        plot_neuron_selectivity,
        plot_cluster_quality,
        plot_functional_connectivity,
        plot_spatial_organization,
        plot_spatial_topology,
    )

    N = len(stims_n)
    if stim_names is None:
        stim_names = [f"Stimulus {j + 1}" for j in range(N)]

    pop_dir = Path(results_dir) / "population"
    pop_dir.mkdir(parents=True, exist_ok=True)

    cluster_result = compute_coactivity_clusters(
        stims_n, stim_onset_idx, n_clusters=n_clusters, method=cluster_method)
    labels = cluster_result["labels"]

    selectivity = compute_neuron_selectivity(stims_n, stim_onset_idx)
    quality = compute_cluster_quality(
        cluster_result["features_pca"], labels, method=cluster_method)
    connectivity = compute_functional_connectivity(stims_n, stim_onset_idx, labels)
    spatial = compute_spatial_organization(
        centroids, z_ids, labels, run_tda=run_tda)

    # Persist cluster labels
    np.save(str(pop_dir / "clusters.npy"), labels)

    # Generate figures
    plot_cluster_traces(
        cluster_result["cluster_means"], stim_names, fp, stim_onset_idx,
        str(pop_dir))
    plot_neuron_selectivity(selectivity, stim_names, str(pop_dir))
    plot_cluster_quality(quality, cluster_result["n_clusters"], str(pop_dir))
    plot_functional_connectivity(connectivity, str(pop_dir))
    if spatial["available"]:
        plot_spatial_organization(spatial, labels, str(pop_dir))
        if spatial["tda_available"] and spatial["spatial_diagrams"] is not None:
            plot_spatial_topology(spatial["spatial_diagrams"], str(pop_dir))

    # Summary YAML for reproducibility
    summary = {
        "n_clusters": cluster_result["n_clusters"],
        "cluster_sizes": cluster_result["cluster_sizes"].tolist(),
        "cluster_method": cluster_result["method"],
        "best_k_silhouette": quality["best_k"],
        "current_silhouette": round(quality["current_silhouette"], 4),
        "mean_selectivity": round(float(selectivity["selectivity"].mean()), 4),
        "modularity_ratio": round(connectivity["modularity_ratio"], 4),
        "stim_names": stim_names,
    }
    if spatial["available"]:
        summary["spatially_clustered_celltypes"] = int(
            np.sum(spatial["cluster_pvalues"] < 0.05))
    with open(str(pop_dir / "summary.yaml"), "w") as fh:
        yaml.dump(summary, fh, default_flow_style=False)

    return {
        **cluster_result,
        "selectivity": selectivity,
        "quality": quality,
        "connectivity": connectivity,
        "spatial": spatial,
        "stims_n": stims_n,
        "stim_onset_idx": stim_onset_idx,
        "fp": fp,
        "stim_names": stim_names,
        "centroids": centroids,
        "z_ids": z_ids,
        "pop_dir": str(pop_dir),
        # n_clusters from cluster_result may differ from the argument if K < n_clusters
        "n_clusters": cluster_result["n_clusters"],
    }
