"""Matplotlib figure generation for neural manifold analysis.

Plots derived from Ebitz & Hayden (2021) population doctrine framework:
coding dimensions, state-space trajectories, subspace decomposition,
and dimensionality over time.

All functions use Figure() + FigureCanvasAgg (no pyplot) — thread-safe.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.cm as cm
import matplotlib.colors as mcolors

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

_STATE_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
                 "#911eb4", "#42d4f4", "#f032e6", "#469990"]


def _save(fig: Figure, path: str) -> None:
    FigureCanvasAgg(fig).print_figure(path, dpi=150, bbox_inches="tight")


def _time_axis(n: int, fp: float, onset: int) -> np.ndarray:
    return (np.arange(n) - onset) * fp


# ── Coding dimensions ─────────────────────────────────────────────────────────

def plot_coding_dimensions(
    coding_vector: np.ndarray,
    projection: list[np.ndarray],
    separability: np.ndarray,
    stim_names: list[str],
    fp: float,
    stim_onset_idx: int,
    out_dir: str,
) -> None:
    """Loading bar chart + population projection over time + separability."""
    K = len(coding_vector)
    N = len(projection)
    stim_f = len(projection[0])
    t = _time_axis(stim_f, fp, 0)  # already starts at onset

    fig = Figure(figsize=(13, 5))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.35,
                          left=0.07, right=0.97, top=0.92, bottom=0.10)
    ax_bar  = fig.add_subplot(gs[:, 0])
    ax_proj = fig.add_subplot(gs[0, 1])
    ax_sep  = fig.add_subplot(gs[1, 1])

    # Loading bar chart — sorted by loading value, coloured by sign
    order = np.argsort(coding_vector)
    sorted_v = coding_vector[order]
    colours = ["#1f77b4" if v < 0 else "#ff7f0e" for v in sorted_v]
    y_pos = np.arange(K)
    ax_bar.barh(y_pos, sorted_v, color=colours, height=0.8, alpha=0.8)
    ax_bar.axvline(0, color="black", lw=0.8)
    ax_bar.set_xlabel("Loading on coding dimension")
    ax_bar.set_ylabel("Neuron (sorted)")
    ax_bar.set_yticks([])
    ax_bar.set_title("Neuron loadings")
    # Simplified legend
    from matplotlib.patches import Patch
    ax_bar.legend(handles=[
        Patch(color="#ff7f0e", label="Positive (cond-1 driver)"),
        Patch(color="#1f77b4", label="Negative (cond-2 driver)"),
    ], fontsize=8, loc="lower right")

    # Population projection over time
    for j, proj in enumerate(projection):
        color = _COLORS[j % len(_COLORS)]
        lbl = stim_names[j] if j < len(stim_names) else f"Stim {j+1}"
        ax_proj.plot(t, proj, color=color, label=lbl, linewidth=1.8)
    ax_proj.axvline(0, color="gray", lw=0.7, ls="--")
    ax_proj.axhline(0, color="#cccccc", lw=0.5)
    ax_proj.set_xlabel("Time from onset (s)")
    ax_proj.set_ylabel("Projection (a.u.)")
    ax_proj.set_title("Population projection onto coding dimension")
    ax_proj.legend(fontsize=11)

    # Separability over time
    ax_sep.plot(t, separability, color="#2ca02c", linewidth=1.8)
    ax_sep.axvline(0, color="gray", lw=0.7, ls="--")
    ax_sep.axhline(0, color="#cccccc", lw=0.5)
    ax_sep.set_xlabel("Time from onset (s)")
    ax_sep.set_ylabel("Condition distance (a.u.)")
    ax_sep.set_title("Stimulus discriminability over time")
    ax_sep.fill_between(t, 0, separability, alpha=0.15, color="#2ca02c")

    _save(fig, str(Path(out_dir) / "coding_dimensions.png"))


# ── Trajectory geometry ───────────────────────────────────────────────────────

def plot_trajectory_geometry(
    pca_trajectories: list[np.ndarray],
    speed: list[np.ndarray],
    condition_distance: np.ndarray,
    onset_frame: int,
    stim_names: list[str],
    fp: float,
    out_dir: str,
) -> None:
    """PC1/PC2 trajectories, scree, speed, condition distance."""
    N = len(pca_trajectories)
    stim_f = pca_trajectories[0].shape[0]
    t_dist = _time_axis(stim_f, fp, 0)
    t_speed = _time_axis(stim_f - 1, fp, 0)

    fig = Figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32,
                          left=0.08, right=0.97, top=0.93, bottom=0.09)
    ax_traj = fig.add_subplot(gs[0, 0])
    ax_dist = fig.add_subplot(gs[0, 1])
    ax_spd  = fig.add_subplot(gs[1, 0])
    ax_scr  = fig.add_subplot(gs[1, 1])

    # PC1/PC2 trajectories — opacity encodes time (light=early, solid=late)
    alphas = np.linspace(0.12, 1.0, stim_f)
    for j, traj in enumerate(pca_trajectories):
        color = _COLORS[j % len(_COLORS)]
        lbl = stim_names[j] if j < len(stim_names) else f"Stim {j+1}"
        for i in range(stim_f - 1):
            ax_traj.plot(traj[i:i+2, 0], traj[i:i+2, 1],
                         color=color, alpha=float(alphas[i]), lw=1.6)
        ax_traj.plot(*traj[0, :2],  "o", color=color, ms=6, label=lbl, zorder=5)
        ax_traj.plot(*traj[-1, :2], "s", color=color, ms=5, zorder=5)
    ax_traj.set_xlabel("PC 1")
    ax_traj.set_ylabel("PC 2")
    ax_traj.set_title("State-space trajectories\n(◉ start   ■ end,  opacity = time)")
    ax_traj.legend(fontsize=11)
    ax_traj.axhline(0, color="#cccccc", lw=0.5)
    ax_traj.axvline(0, color="#cccccc", lw=0.5)

    # Condition distance over time
    ax_dist.plot(t_dist, condition_distance, color="#2ca02c", lw=1.8)
    if onset_frame < stim_f:
        ax_dist.axvline(onset_frame * fp, color="#d62728", lw=1.0, ls="--",
                        label=f"Onset ≈ {onset_frame * fp:.1f} s")
        ax_dist.legend(fontsize=11)
    ax_dist.axvline(0, color="gray", lw=0.7, ls=":")
    ax_dist.set_xlabel("Time from onset (s)")
    ax_dist.set_ylabel("Euclidean distance (PC space)")
    ax_dist.set_title("Condition separability over time")
    ax_dist.fill_between(t_dist, 0, condition_distance, alpha=0.12, color="#2ca02c")

    # Per-condition speed
    for j, spd in enumerate(speed):
        color = _COLORS[j % len(_COLORS)]
        lbl = stim_names[j] if j < len(stim_names) else f"Stim {j+1}"
        ax_spd.plot(t_speed, spd, color=color, label=lbl, lw=1.4, alpha=0.85)
    ax_spd.axvline(0, color="gray", lw=0.7, ls="--")
    ax_spd.set_xlabel("Time from onset (s)")
    ax_spd.set_ylabel("|Δstate / Δt|")
    ax_spd.set_title("Trajectory speed (state-change rate)")
    ax_spd.legend(fontsize=11)

    # Scree (variance explained by each PC)
    pca_ve = None
    if hasattr(pca_trajectories[0], 'shape') and pca_trajectories[0].shape[1] > 0:
        # Compute approximate variance from trajectories
        combined = np.vstack(pca_trajectories)
        variances = combined.var(axis=0)
        total = variances.sum()
        pca_ve = variances / total if total > 0 else variances
    if pca_ve is not None:
        n_show = min(15, len(pca_ve))
        pcs = np.arange(1, n_show + 1)
        ax_scr.bar(pcs, pca_ve[:n_show] * 100, color="#4fa1ca", alpha=0.8)
        ax2 = ax_scr.twinx()
        ax2.plot(pcs, np.cumsum(pca_ve[:n_show]) * 100, "o-",
                 color="#d62728", ms=3, lw=1.3)
        ax2.axhline(85, color="#d62728", ls=":", lw=0.8)
        ax2.set_ylabel("Cumulative (%)", color="#d62728")
        ax2.tick_params(axis="y", labelcolor="#d62728")
        ax_scr.set_xticks(pcs)
    ax_scr.set_xlabel("PC")
    ax_scr.set_ylabel("Variance (%)")
    ax_scr.set_title("PC variance explained")

    _save(fig, str(Path(out_dir) / "trajectory_geometry.png"))


# ── Subspace ──────────────────────────────────────────────────────────────────

def plot_subspace(
    coding_variance_ratio: float,
    proj_var_over_time: np.ndarray,
    stim_onset_idx: int,
    fp: float,
    out_dir: str,
) -> None:
    """Coding vs null variance decomposition."""
    stim_f = len(proj_var_over_time)
    t = _time_axis(stim_f, fp, 0)

    fig = Figure(figsize=(11, 4.5))
    ax_bar, ax_time = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.08, right=0.96, top=0.90, bottom=0.13, wspace=0.32)

    null_ratio = 1.0 - coding_variance_ratio
    ax_bar.bar(["Coding\nsubspace", "Nullspace"],
               [coding_variance_ratio * 100, null_ratio * 100],
               color=["#ff7f0e", "#7f7f7f"], alpha=0.8, width=0.5)
    ax_bar.set_ylabel("% of total stim-window variance")
    ax_bar.set_title("Coding subspace vs nullspace")
    ax_bar.set_ylim(0, 105)
    for bar, val in zip(ax_bar.patches, [coding_variance_ratio, null_ratio]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, f"{val*100:.1f}%",
                    ha="center", va="bottom", fontsize=9)

    ax_time.plot(t, proj_var_over_time, color="#ff7f0e", lw=1.8)
    ax_time.axvline(0, color="gray", lw=0.7, ls="--", label="Stim onset")
    ax_time.set_xlabel("Time from onset (s)")
    ax_time.set_ylabel("Projection variance (a.u.)")
    ax_time.set_title("Task-relevant activity over time\n(variance in coding subspace)")
    ax_time.legend(fontsize=11)
    ax_time.fill_between(t, 0, proj_var_over_time, alpha=0.12, color="#ff7f0e")

    _save(fig, str(Path(out_dir) / "subspace.png"))


# ── Dimensionality over time ──────────────────────────────────────────────────

def plot_dimensionality_over_time(
    pr_over_time: np.ndarray,
    pr_baseline: float,
    pr_stimulus: float,
    stim_onset_idx: int,
    fp: float,
    out_dir: str,
) -> None:
    """Participation ratio over time — manifold dimensionality dynamics."""
    n = len(pr_over_time)
    t = _time_axis(n, fp, stim_onset_idx)

    fig = Figure(figsize=(9, 4.2))
    ax = fig.add_subplot(111)
    fig.subplots_adjust(left=0.09, right=0.97, top=0.88, bottom=0.14)

    ax.plot(t, pr_over_time, color="#9467bd", lw=1.8, label="PR (sliding window)")
    ax.axhline(pr_baseline, color="#1f77b4", lw=1.2, ls="--",
               label=f"Baseline mean  ({pr_baseline:.1f})")
    ax.axhline(pr_stimulus, color="#ff7f0e", lw=1.2, ls="--",
               label=f"Stimulus mean  ({pr_stimulus:.1f})")
    ax.axvline(0, color="gray", lw=0.8, ls=":", label="Stim onset")
    ax.set_xlabel("Time from onset (s)")
    ax.set_ylabel("Participation ratio")
    ax.set_title("Neural manifold dimensionality over time\n"
                 "(decrease = more coordinated; increase = more independent)")
    ax.legend(fontsize=11)
    ax.fill_between(t, pr_over_time.min() * 0.95, pr_over_time,
                    alpha=0.10, color="#9467bd")

    _save(fig, str(Path(out_dir) / "dimensionality_over_time.png"))


# ── Neural states / attractors ────────────────────────────────────────────────

def plot_neural_states(
    states: dict,
    stim_names: list[str],
    fp: float,
    out_dir: str,
) -> None:
    """State-coloured trajectories, occupancy raster, transitions, dwell times."""
    trajectories  = states["trajectories"]
    state_labels  = states["state_labels"]
    centroids     = states["state_centroids"]
    n_states      = states["n_states"]
    occupancy     = states["occupancy"]
    dwell_frames  = states["dwell_frames"]
    transition    = states["transition_matrix"]
    slow_frames   = states["slow_point_frames"]
    N = len(trajectories)
    stim_f = trajectories[0].shape[0]

    fig = Figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30,
                          left=0.07, right=0.96, top=0.93, bottom=0.09)
    ax_tr   = fig.add_subplot(gs[0, 0])
    ax_occ  = fig.add_subplot(gs[0, 1])
    ax_dw   = fig.add_subplot(gs[1, 0])
    ax_trn  = fig.add_subplot(gs[1, 1])

    # PC1/PC2 trajectory, each point coloured by its state; centroids as stars
    for j, traj in enumerate(trajectories):
        lab = state_labels[j]
        cols = [_STATE_COLORS[int(s) % len(_STATE_COLORS)] for s in lab]
        ax_tr.plot(traj[:, 0], traj[:, 1], color="#cccccc", lw=0.8, zorder=1)
        ax_tr.scatter(traj[:, 0], traj[:, 1], c=cols, s=18, zorder=2)
        if len(slow_frames[j]):
            sp = slow_frames[j]
            ax_tr.scatter(traj[sp, 0], traj[sp, 1], s=90, facecolors="none",
                          edgecolors="black", linewidths=1.4, zorder=3,
                          label="Slow point" if j == 0 else None)
    for s in range(n_states):
        color = _STATE_COLORS[s % len(_STATE_COLORS)]
        ax_tr.scatter(centroids[s, 0], centroids[s, 1], marker="*", s=320,
                      color=color, edgecolors="black", linewidths=1.0, zorder=4)
        ax_tr.annotate(f"S{s}", (centroids[s, 0], centroids[s, 1]),
                       fontsize=9, fontweight="bold", ha="center", va="center")
    ax_tr.set_xlabel("PC 1")
    ax_tr.set_ylabel("PC 2")
    ax_tr.set_title(f"State-space trajectories coloured by state  "
                    f"(★ = attractor centroid,  {n_states} states)")
    ax_tr.axhline(0, color="#cccccc", lw=0.5)
    ax_tr.axvline(0, color="#cccccc", lw=0.5)
    handles, labels = ax_tr.get_legend_handles_labels()
    if handles:
        ax_tr.legend(fontsize=9, loc="best")

    # Occupancy raster: state index over time, one row per condition
    cmap = mcolors.ListedColormap(
        [_STATE_COLORS[s % len(_STATE_COLORS)] for s in range(n_states)])
    raster = np.vstack(state_labels)  # (N, stim_f)
    t = _time_axis(stim_f, fp, 0)
    ax_occ.imshow(raster, aspect="auto", cmap=cmap, vmin=-0.5,
                  vmax=n_states - 0.5, interpolation="nearest",
                  extent=[t[0], t[-1], N - 0.5, -0.5])
    ax_occ.set_yticks(range(N))
    ax_occ.set_yticklabels(
        [stim_names[j] if j < len(stim_names) else f"Stim {j+1}" for j in range(N)],
        fontsize=8)
    ax_occ.set_xlabel("Time from onset (s)")
    ax_occ.set_title("State occupancy over time")

    # Mean dwell time per state (seconds)
    s_idx = np.arange(n_states)
    s_cols = [_STATE_COLORS[s % len(_STATE_COLORS)] for s in s_idx]
    ax_dw.bar(s_idx, dwell_frames * fp, color=s_cols, alpha=0.85)
    ax_dw.set_xticks(s_idx)
    ax_dw.set_xticklabels([f"S{s}" for s in s_idx])
    ax_dw.set_xlabel("State")
    ax_dw.set_ylabel("Mean dwell time (s)")
    ax_dw.set_title("Dwell time per state (stability)")

    # Transition matrix heatmap
    im = ax_trn.imshow(transition, cmap="magma", vmin=0, vmax=1,
                       interpolation="nearest")
    ax_trn.set_xticks(s_idx); ax_trn.set_yticks(s_idx)
    ax_trn.set_xticklabels([f"S{s}" for s in s_idx])
    ax_trn.set_yticklabels([f"S{s}" for s in s_idx])
    ax_trn.set_xlabel("To state")
    ax_trn.set_ylabel("From state")
    ax_trn.set_title("State transition probability")
    for i in range(n_states):
        for k in range(n_states):
            ax_trn.text(k, i, f"{transition[i, k]:.2f}", ha="center", va="center",
                        color="white" if transition[i, k] < 0.6 else "black",
                        fontsize=8)
    fig.colorbar(im, ax=ax_trn, fraction=0.046, pad=0.04)

    _save(fig, str(Path(out_dir) / "neural_states.png"))


# ── Manifold topology (persistent homology) ───────────────────────────────────

def _plot_diagram(ax, diagrams, title):
    """Birth-death persistence-diagram scatter for H0 (and H1 if present)."""
    dim_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    finite_deaths = []
    for d, dgm in enumerate(diagrams):
        if dgm is None or len(dgm) == 0:
            continue
        fin = dgm[np.isfinite(dgm[:, 1])]
        if len(fin):
            finite_deaths.append(fin[:, 1].max())
    lim = max(finite_deaths) * 1.1 if finite_deaths else 1.0
    ax.plot([0, lim], [0, lim], color="#999999", lw=0.8, ls="--", zorder=1)
    for d, dgm in enumerate(diagrams):
        if dgm is None or len(dgm) == 0:
            continue
        fin = dgm[np.isfinite(dgm[:, 1])]
        if len(fin):
            ax.scatter(fin[:, 0], fin[:, 1], s=28,
                       color=dim_colors[d % len(dim_colors)],
                       label=f"H{d}", alpha=0.8, zorder=2)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Birth"); ax.set_ylabel("Death")
    ax.set_title(title)
    ax.legend(fontsize=9)


def plot_manifold_topology(
    topology: dict,
    stim_names: list[str],
    out_dir: str,
) -> None:
    """Pooled persistence diagram + barcode, plus per-condition diagrams."""
    pooled = topology["pooled_diagrams"]
    per_cond = topology["per_condition_diagrams"]
    n_loops = topology["n_loops_pooled"]
    if pooled is None:
        return
    N = len(per_cond) if per_cond else 0

    n_cols = max(2, N)
    fig = Figure(figsize=(3.4 * n_cols, 7.5))
    gs = fig.add_gridspec(2, n_cols, hspace=0.40, wspace=0.35,
                          left=0.07, right=0.97, top=0.91, bottom=0.10)

    # Pooled persistence diagram + barcode (top row, first two columns)
    ax_dg = fig.add_subplot(gs[0, 0])
    _plot_diagram(ax_dg, pooled,
                  f"Pooled manifold  ({n_loops} significant loop"
                  f"{'s' if n_loops != 1 else ''})")

    ax_bc = fig.add_subplot(gs[0, 1])
    dim_colors = ["#1f77b4", "#d62728", "#2ca02c"]
    row = 0
    for d, dgm in enumerate(pooled):
        if dgm is None or len(dgm) == 0:
            continue
        fin = dgm[np.isfinite(dgm[:, 1])]
        for b, death in fin:
            ax_bc.plot([b, death], [row, row], color=dim_colors[d % len(dim_colors)],
                       lw=2.0, alpha=0.8)
            row += 1
    ax_bc.set_xlabel("Filtration value")
    ax_bc.set_ylabel("Feature")
    ax_bc.set_title("Pooled persistence barcode")
    from matplotlib.patches import Patch
    ax_bc.legend(handles=[Patch(color="#1f77b4", label="H0 (components)"),
                          Patch(color="#d62728", label="H1 (loops)")],
                 fontsize=8, loc="lower right")

    # Per-condition diagrams (bottom row)
    if per_cond:
        for j, diags in enumerate(per_cond):
            ax = fig.add_subplot(gs[1, j % n_cols])
            name = stim_names[j] if j < len(stim_names) else f"Stim {j+1}"
            _plot_diagram(ax, diags, name)

    fig.suptitle("Manifold topology — persistent homology", fontsize=12, y=0.97)
    _save(fig, str(Path(out_dir) / "manifold_topology.png"))
