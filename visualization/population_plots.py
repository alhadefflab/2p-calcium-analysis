"""Matplotlib figure generation for the Sherringtonian cell-type tool.

Plots cluster mean traces, single-neuron selectivity/tuning, cluster-quality
(silhouette), functional connectivity, and spatial organisation + topology.
NOT derived from Ebitz & Hayden (2021) — for the Hopfieldian manifold plots
see visualization/manifold_plots.py.

All public functions use the OO Figure interface (not pyplot) and are
safe to call from a background thread.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

_CLUSTER_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#c4a35a", "#469990", "#b070e0", "#9a6324",
    "#800000", "#3cb890", "#f5b0d0", "#005f75",
]


def _save(fig: Figure, path: str) -> None:
    FigureCanvasAgg(fig).print_figure(path, dpi=150, bbox_inches="tight")


def _time_axis(n_frames: int, fp: float, stim_onset_idx: int) -> np.ndarray:
    return (np.arange(n_frames) - stim_onset_idx) * fp


# ── Cluster traces ────────────────────────────────────────────────────────────

def plot_cluster_traces(
    cluster_means: list[np.ndarray],
    stim_names: list[str],
    fp: float,
    stim_onset_idx: int,
    out_dir: str,
) -> None:
    """Mean z-score trace per cluster, one subplot column per stimulus."""
    n_clusters = cluster_means[0].shape[0]
    N = len(cluster_means)
    T = cluster_means[0].shape[1]
    t = _time_axis(T, fp, stim_onset_idx)

    fig = Figure(figsize=(4.0 * N, 1.6 * n_clusters + 0.8))
    axes = fig.subplots(n_clusters, N, squeeze=False, sharex=True)
    fig.subplots_adjust(hspace=0.5, wspace=0.3,
                        left=0.10, right=0.97, top=0.94, bottom=0.08)

    for c in range(n_clusters):
        color = _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)]
        for j in range(N):
            ax = axes[c, j]
            ax.plot(t, cluster_means[j][c], color=color, linewidth=1.5)
            ax.axvline(0, color="gray", linewidth=0.7, linestyle="--")
            ax.axhline(0, color="#cccccc", linewidth=0.5)
            if j == 0:
                ax.set_ylabel(f"Cl. {c}", fontsize=8, color=color,
                              fontweight="bold")
            if c == 0:
                ax.set_title(stim_names[j], fontsize=9)
            ax.set_xlim(t[0], t[-1])
            ax.tick_params(labelsize=7)

    for ax in axes[-1]:
        ax.set_xlabel("Time from onset (s)", fontsize=8)

    _save(fig, str(Path(out_dir) / "cluster_traces.png"))


# ── Single-neuron selectivity ─────────────────────────────────────────────────

def plot_neuron_selectivity(
    selectivity: dict,
    stim_names: list[str],
    out_dir: str,
) -> None:
    """Selectivity-index histogram, preferred-stimulus counts, tuning heatmap."""
    resp = selectivity["response_matrix"]   # (K, N)
    pref = selectivity["preferred_stim"]
    sel  = selectivity["selectivity"]
    K, N = resp.shape

    fig = Figure(figsize=(13, 4.6))
    ax_hist, ax_pref, ax_heat = fig.subplots(1, 3)
    fig.subplots_adjust(left=0.06, right=0.97, top=0.88, bottom=0.14, wspace=0.32)

    ax_hist.hist(sel, bins=20, range=(0, 1), color="#4fa1ca", alpha=0.85)
    ax_hist.axvline(float(sel.mean()), color="#d62728", ls="--", lw=1.2,
                    label=f"mean {sel.mean():.2f}")
    ax_hist.set_xlabel("Selectivity index  (0 = broad, 1 = single-stimulus)")
    ax_hist.set_ylabel("Neuron count")
    ax_hist.set_title("Selectivity distribution")
    ax_hist.legend(fontsize=9)

    counts = np.array([(pref == j).sum() for j in range(N)])
    cols = [_COLORS[j % len(_COLORS)] for j in range(N)]
    ax_pref.bar(range(N), counts, color=cols, alpha=0.85)
    ax_pref.set_xticks(range(N))
    ax_pref.set_xticklabels(
        [stim_names[j] if j < len(stim_names) else f"S{j+1}" for j in range(N)],
        rotation=30, ha="right", fontsize=8)
    ax_pref.set_ylabel("Neuron count")
    ax_pref.set_title("Preferred stimulus")

    # Tuning heatmap: neurons sorted by preferred stim then selectivity
    order = np.lexsort((-sel, pref))
    im = ax_heat.imshow(resp[order], aspect="auto", cmap="viridis",
                        interpolation="nearest")
    ax_heat.set_xticks(range(N))
    ax_heat.set_xticklabels(
        [stim_names[j] if j < len(stim_names) else f"S{j+1}" for j in range(N)],
        rotation=30, ha="right", fontsize=8)
    ax_heat.set_ylabel("Neuron (sorted by tuning)")
    ax_heat.set_title("Tuning matrix (mean z per stimulus)")
    fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="z")

    _save(fig, str(Path(out_dir) / "neuron_selectivity.png"))


# ── Cluster quality ───────────────────────────────────────────────────────────

def plot_cluster_quality(quality: dict, n_clusters: int, out_dir: str) -> None:
    """Silhouette-vs-k curve + per-neuron silhouette distribution."""
    sil_by_k = quality["silhouette_by_k"]
    best_k   = quality["best_k"]
    cur_sil  = quality["current_silhouette"]
    samples  = quality["sample_silhouette"]

    fig = Figure(figsize=(11, 4.4))
    ax_k, ax_dist = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.13, wspace=0.28)

    if sil_by_k:
        ks = sorted(sil_by_k)
        vals = [sil_by_k[k] for k in ks]
        ax_k.plot(ks, vals, "o-", color="#4fa1ca", lw=1.6)
        ax_k.axvline(best_k, color="#2ca02c", ls="--", lw=1.2,
                     label=f"best k = {best_k}")
        ax_k.axvline(n_clusters, color="#ff7f0e", ls=":", lw=1.4,
                     label=f"current k = {n_clusters}")
        ax_k.legend(fontsize=9)
    ax_k.set_xlabel("Number of clusters (k)")
    ax_k.set_ylabel("Mean silhouette")
    ax_k.set_title("Cluster-count justification")

    ax_dist.hist(samples, bins=25, color="#911eb4", alpha=0.8)
    ax_dist.axvline(cur_sil, color="#d62728", ls="--", lw=1.2,
                    label=f"mean {cur_sil:.2f}")
    ax_dist.axvline(0, color="gray", lw=0.8)
    ax_dist.set_xlabel("Per-neuron silhouette")
    ax_dist.set_ylabel("Neuron count")
    ax_dist.set_title(f"Silhouette at current k = {n_clusters}")
    ax_dist.legend(fontsize=9)

    _save(fig, str(Path(out_dir) / "cluster_quality.png"))


# ── Functional connectivity ───────────────────────────────────────────────────

def plot_functional_connectivity(connectivity: dict, out_dir: str) -> None:
    """Cluster-sorted correlation matrix + within/between summary."""
    corr   = connectivity["corr"]
    order  = connectivity["order"]
    labels_sorted = connectivity["labels_sorted"]
    within = connectivity["within_corr"]
    between = connectivity["between_corr"]

    sorted_corr = corr[np.ix_(order, order)]

    fig = Figure(figsize=(11, 5))
    ax_mat, ax_bar = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})
    fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.10, wspace=0.28)

    im = ax_mat.imshow(sorted_corr, cmap="RdBu_r", vmin=-1, vmax=1,
                       interpolation="nearest")
    # Cluster boundary lines
    boundaries = np.where(np.diff(labels_sorted) != 0)[0] + 0.5
    for b in boundaries:
        ax_mat.axhline(b, color="black", lw=0.6)
        ax_mat.axvline(b, color="black", lw=0.6)
    ax_mat.set_title("Neuron-neuron correlation (sorted by cell-type)")
    ax_mat.set_xlabel("Neuron"); ax_mat.set_ylabel("Neuron")
    fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04, label="Pearson r")

    ax_bar.bar(["Within\ntype", "Between\ntypes"], [within, between],
               color=["#2ca02c", "#7f7f7f"], alpha=0.85, width=0.5)
    ax_bar.set_ylabel("Mean |correlation|")
    ratio = within / between if between > 1e-10 else 0.0
    ax_bar.set_title(f"Block structure  (ratio {ratio:.2f})")
    for i, v in enumerate([within, between]):
        ax_bar.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)

    _save(fig, str(Path(out_dir) / "functional_connectivity.png"))


# ── Spatial organisation ──────────────────────────────────────────────────────

def plot_spatial_organization(spatial: dict, labels: np.ndarray, out_dir: str) -> None:
    """3-D anatomical cell-type map + spatial-aggregation test."""
    coords = spatial["coords3d"]
    pvals  = spatial["cluster_pvalues"]
    obs    = spatial["observed_dist"]
    null   = spatial["null_dist"]
    n_clusters = int(labels.max()) + 1 if len(labels) else 0

    fig = Figure(figsize=(13, 5.5))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_bar = fig.add_subplot(1, 2, 2)
    fig.subplots_adjust(left=0.03, right=0.96, top=0.92, bottom=0.10, wspace=0.22)

    for c in range(n_clusters):
        m = labels == c
        ax3d.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                     color=_CLUSTER_COLORS[c % len(_CLUSTER_COLORS)],
                     s=14, alpha=0.8, label=f"Cl.{c}")
    ax3d.set_xlabel("Column (px)"); ax3d.set_ylabel("Row (px)")
    ax3d.set_zlabel("Plane (scaled)")
    ax3d.set_title("Cell-types in 3-D anatomical space")
    if n_clusters <= 10:
        ax3d.legend(fontsize=7, loc="upper left")

    x = np.arange(n_clusters)
    width = 0.38
    ax_bar.bar(x - width / 2, obs, width, color="#2ca02c", alpha=0.85,
               label="Observed")
    ax_bar.bar(x + width / 2, null, width, color="#cccccc", alpha=0.85,
               label="Random null")
    for c in range(n_clusters):
        if pvals[c] < 0.05:
            ax_bar.text(c, max(obs[c], null[c]) * 1.02, "*",
                        ha="center", fontsize=14, color="#d62728")
    ax_bar.set_xticks(x); ax_bar.set_xticklabels([f"Cl.{c}" for c in x])
    ax_bar.set_ylabel("Mean within-type 3-D distance")
    ax_bar.set_title("Spatial aggregation  (* p < 0.05 ⇒ compact)")
    ax_bar.legend(fontsize=9)

    _save(fig, str(Path(out_dir) / "spatial_organization.png"))


# ── Spatial topology ──────────────────────────────────────────────────────────

def plot_spatial_topology(diagrams: list, out_dir: str) -> None:
    """Persistence diagram + barcode for the 3-D centroid arrangement."""
    dim_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    fig = Figure(figsize=(11, 4.6))
    ax_dg, ax_bc = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.13, wspace=0.28)

    finite_deaths = []
    for dgm in diagrams:
        if dgm is None or len(dgm) == 0:
            continue
        fin = dgm[np.isfinite(dgm[:, 1])]
        if len(fin):
            finite_deaths.append(fin[:, 1].max())
    lim = max(finite_deaths) * 1.1 if finite_deaths else 1.0

    ax_dg.plot([0, lim], [0, lim], color="#999999", lw=0.8, ls="--")
    for d, dgm in enumerate(diagrams):
        if dgm is None or len(dgm) == 0:
            continue
        fin = dgm[np.isfinite(dgm[:, 1])]
        if len(fin):
            ax_dg.scatter(fin[:, 0], fin[:, 1], s=26,
                          color=dim_colors[d % len(dim_colors)],
                          label=f"H{d}", alpha=0.8)
    ax_dg.set_xlim(0, lim); ax_dg.set_ylim(0, lim)
    ax_dg.set_xlabel("Birth (distance)"); ax_dg.set_ylabel("Death (distance)")
    ax_dg.set_title("Spatial persistence diagram")
    ax_dg.legend(fontsize=9)

    row = 0
    for d, dgm in enumerate(diagrams):
        if dgm is None or len(dgm) == 0:
            continue
        fin = dgm[np.isfinite(dgm[:, 1])]
        for b, death in fin:
            ax_bc.plot([b, death], [row, row],
                       color=dim_colors[d % len(dim_colors)], lw=2.0, alpha=0.8)
            row += 1
    ax_bc.set_xlabel("Filtration value (distance)")
    ax_bc.set_ylabel("Feature")
    ax_bc.set_title("Spatial barcode")
    from matplotlib.patches import Patch
    ax_bc.legend(handles=[
        Patch(color="#1f77b4", label="H0 clumps"),
        Patch(color="#d62728", label="H1 loops"),
        Patch(color="#2ca02c", label="H2 voids"),
    ], fontsize=8, loc="lower right")

    fig.suptitle("Spatial topology of cell-type arrangement", fontsize=12, y=0.98)
    _save(fig, str(Path(out_dir) / "spatial_topology.png"))
