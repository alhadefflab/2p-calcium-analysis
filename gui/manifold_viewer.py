"""Neural manifold analysis viewer.

Displays coding dimensions, state-space trajectories, subspace decomposition,
and an interactive Neuron Explorer based on:
    Ebitz & Hayden (2021). The population doctrine in cognitive neuroscience.
    Neuron 109, 3055-3068.

The Neuron Explorer (Tab 4) provides:
  - 2-D matplotlib canvas: neurons at spatial positions, coloured by loading,
    click-to-select with yellow highlight ring
  - Loading threshold auto-select sliders
  - Trace panel for selected neurons
  - "Open 3-D spatial map" button → PyVista window (spheres at col/row/z-plane,
    click to select — selections propagate back to the CTk window)
  - "Open activation animation" button → Plotly animated scatter in browser
"""
from __future__ import annotations

import os
import tempfile
import threading
import webbrowser
from pathlib import Path

import numpy as np
import tkinter as tk
import customtkinter as ctk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
           "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

_STATE_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231",
                 "#911eb4", "#42d4f4", "#f032e6", "#469990"]


class ManifoldViewerWindow(ctk.CTkToplevel):
    """Interactive viewer for neural manifold analysis results.

    Parameters
    ----------
    parent  : parent tkinter widget (PipelineGUI)
    results : dict returned by analysis.manifold.run_manifold_analysis
    """

    def __init__(self, parent, results: dict):
        super().__init__(parent)
        self.title("Neural Manifold Analysis  --  Ebitz & Hayden (2021)")
        self.geometry("1200x800")
        self.resizable(True, True)

        self._r = results
        self._stims_n       = results["stims_n"]
        self._stim_onset    = results["stim_onset_idx"]
        self._fp            = results.get("fp", 0.585)
        self._stim_names    = results.get("stim_names", [])
        self._centroids     = results.get("centroids")   # (K, 2) row/col or None
        self._coding_vector = results.get("coding_vector", np.array([]))
        self._manifold_dir  = results.get("manifold_dir", "")
        self._states        = results.get("states", {})
        self._topology      = results.get("topology", {})

        # Neuron selection state
        self._selected: set[int] = set()
        self._stim_idx = 0

        # PyVista inter-thread communication
        self._pv_picks: list[int] = []
        self._pv_lock = threading.Lock()

        self._build_ui()
        self.lift()
        self.focus_force()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.tabs = ctk.CTkTabview(self, anchor="nw")
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)
        for name in ("Coding Dimensions", "Trajectory & Dynamics",
                     "Subspace", "States / Attractors", "Topology",
                     "Neuron Explorer"):
            self.tabs.add(name)
        self._build_coding_tab()
        self._build_trajectory_tab()
        self._build_subspace_tab()
        self._build_states_tab()
        self._build_topology_tab()
        self._build_explorer_tab()

    # ── Tab 1: Coding Dimensions ──────────────────────────────────────────────

    def _build_coding_tab(self):
        tab = self.tabs.tab("Coding Dimensions")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        outer = ctk.CTkFrame(tab, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # Info panel (left)
        info = ctk.CTkFrame(outer, width=180, fg_color="transparent")
        info.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        info.columnconfigure(0, weight=1)

        ev  = self._r.get("explained_variance", 0.0)
        oc  = self._r.get("onset_frame", 0) * self._fp
        dim = self._r.get("coding_subspace_dim", 0)
        K   = self._coding_vector.shape[0] if self._coding_vector.size else 0

        ctk.CTkLabel(info, text="Coding\nDimensions",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     justify="left").pack(anchor="w", padx=8, pady=(12, 4))
        for label, val in [
            ("Neurons:", str(K)),
            ("Explained var:", f"{ev*100:.1f}%"),
            ("Subspace dim:", str(dim)),
            ("Disc. onset:", f"{oc:.2f} s"),
        ]:
            row = ctk.CTkFrame(info, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=2)
            ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=9),
                         text_color="gray").pack(side="left")
            ctk.CTkLabel(row, text=val,
                         font=ctk.CTkFont(size=10, weight="bold")).pack(side="right")

        ctk.CTkFrame(info, height=1, fg_color="gray50").pack(fill="x", padx=8, pady=8)
        ctk.CTkLabel(info,
                     text="Orange bars: neurons that\ndrive the population\ntoward condition 1.\n\n"
                          "Blue bars: drive toward\ncondition 2.\n\n"
                          "Near-zero = nullspace\n(not task-relevant).",
                     font=ctk.CTkFont(size=9), text_color="gray",
                     justify="left", wraplength=160).pack(anchor="w", padx=8)

        # Figure (right)
        fig = Figure(figsize=(10, 5))
        canvas = FigureCanvasTkAgg(fig, master=outer)
        canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        self._draw_coding_fig(fig)
        canvas.draw()

    def _draw_coding_fig(self, fig: Figure):
        cv  = self._coding_vector
        proj = self._r.get("projection", [])
        sep  = self._r.get("separability", np.array([]))
        N = len(proj)
        K = len(cv)
        stim_f = len(proj[0]) if proj else 0
        t = (np.arange(stim_f)) * self._fp  # seconds from onset

        gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.32,
                              left=0.07, right=0.97, top=0.93, bottom=0.10)
        ax_bar  = fig.add_subplot(gs[:, 0])
        ax_proj = fig.add_subplot(gs[0, 1])
        ax_sep  = fig.add_subplot(gs[1, 1])

        if K > 0:
            order  = np.argsort(cv)
            sv     = cv[order]
            cols   = ["#1f77b4" if v < 0 else "#ff7f0e" for v in sv]
            ax_bar.barh(np.arange(K), sv, color=cols, height=0.8, alpha=0.8)
            ax_bar.axvline(0, color="black", lw=0.8)
        ax_bar.set_xlabel("Loading")
        ax_bar.set_ylabel("Neuron (sorted by loading)")
        ax_bar.set_yticks([])
        ax_bar.set_title("Neuron loadings on coding dimension")

        for j in range(N):
            color = _COLORS[j % len(_COLORS)]
            lbl = self._stim_names[j] if j < len(self._stim_names) else f"Stim {j+1}"
            ax_proj.plot(t, proj[j], color=color, label=lbl, lw=1.8)
        ax_proj.axvline(0, color="gray", lw=0.7, ls="--")
        ax_proj.axhline(0, color="#cccccc", lw=0.5)
        ax_proj.set_xlabel("Time from onset (s)")
        ax_proj.set_ylabel("Projection")
        ax_proj.set_title("Population projection over time")
        if N:
            ax_proj.legend(fontsize=11)

        if sep.size:
            ax_sep.plot(t[:len(sep)], sep, color="#2ca02c", lw=1.8)
            ax_sep.fill_between(t[:len(sep)], 0, sep, alpha=0.12, color="#2ca02c")
        ax_sep.axvline(0, color="gray", lw=0.7, ls="--")
        ax_sep.set_xlabel("Time from onset (s)")
        ax_sep.set_ylabel("|proj(cond1) - proj(cond2)|")
        ax_sep.set_title("Stimulus discriminability over time")

    # ── Tab 2: Trajectory & Dynamics ─────────────────────────────────────────

    def _build_trajectory_tab(self):
        tab = self.tabs.tab("Trajectory & Dynamics")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        fig = Figure(figsize=(12, 7))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        trajs = self._r.get("pca_trajectories", [])
        speed = self._r.get("speed", [])
        cdist = self._r.get("condition_distance", np.array([]))
        onset = self._r.get("onset_frame", 0)
        N = len(trajs)
        stim_f = trajs[0].shape[0] if trajs else 0
        t = np.arange(stim_f) * self._fp

        gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.30,
                              left=0.08, right=0.97, top=0.93, bottom=0.09)
        ax_t = fig.add_subplot(gs[0, 0])
        ax_d = fig.add_subplot(gs[0, 1])
        ax_s = fig.add_subplot(gs[1, 0])
        ax_c = fig.add_subplot(gs[1, 1])

        # Trajectories
        alphas = np.linspace(0.12, 1.0, stim_f) if stim_f else []
        for j, traj in enumerate(trajs):
            color = _COLORS[j % len(_COLORS)]
            lbl = self._stim_names[j] if j < len(self._stim_names) else f"Stim {j+1}"
            for i in range(stim_f - 1):
                ax_t.plot(traj[i:i+2, 0], traj[i:i+2, 1],
                          color=color, alpha=float(alphas[i]), lw=1.6)
            ax_t.plot(*traj[0, :2],  "o", color=color, ms=6, label=lbl, zorder=5)
            ax_t.plot(*traj[-1, :2], "s", color=color, ms=5, zorder=5)
        ax_t.set_xlabel("PC 1"); ax_t.set_ylabel("PC 2")
        ax_t.set_title("State-space trajectories  (o = start, s = end)")
        ax_t.legend(fontsize=11)
        ax_t.axhline(0, color="#ccc", lw=0.5); ax_t.axvline(0, color="#ccc", lw=0.5)

        # Condition distance
        if cdist.size:
            ax_d.plot(t[:len(cdist)], cdist, color="#2ca02c", lw=1.8)
            if onset < stim_f:
                ax_d.axvline(onset * self._fp, color="#d62728", lw=1.0, ls="--",
                             label=f"Onset ~{onset * self._fp:.1f} s")
                ax_d.legend(fontsize=11)
            ax_d.fill_between(t[:len(cdist)], 0, cdist, alpha=0.10, color="#2ca02c")
        ax_d.axvline(0, color="gray", lw=0.7, ls=":")
        ax_d.set_xlabel("Time (s)"); ax_d.set_ylabel("Distance (PC space)")
        ax_d.set_title("Condition separability over time")

        # Speed per condition
        for j, spd in enumerate(speed):
            color = _COLORS[j % len(_COLORS)]
            lbl = self._stim_names[j] if j < len(self._stim_names) else f"Stim {j+1}"
            ts = np.arange(len(spd)) * self._fp
            ax_s.plot(ts, spd, color=color, label=lbl, lw=1.4, alpha=0.85)
        ax_s.axvline(0, color="gray", lw=0.7, ls="--")
        ax_s.set_xlabel("Time (s)"); ax_s.set_ylabel("|dState / dt|")
        ax_s.set_title("Trajectory speed  (rate of state change)")
        if speed: ax_s.legend(fontsize=11)

        # PR over time (compact)
        pr = self._r.get("pr_over_time", np.array([]))
        pr_base = self._r.get("pr_baseline", 0.0)
        pr_stim = self._r.get("pr_stimulus", 0.0)
        if pr.size:
            tc = _time_axis_fn(len(pr), self._fp, self._stim_onset)
            ax_c.plot(tc, pr, color="#9467bd", lw=1.5)
            ax_c.axhline(pr_base, color="#1f77b4", lw=1.0, ls="--",
                         label=f"Baseline {pr_base:.1f}")
            ax_c.axhline(pr_stim, color="#ff7f0e", lw=1.0, ls="--",
                         label=f"Stimulus {pr_stim:.1f}")
            ax_c.axvline(0, color="gray", lw=0.7, ls=":")
            ax_c.legend(fontsize=11)
        ax_c.set_xlabel("Time (s)"); ax_c.set_ylabel("PR")
        ax_c.set_title("Manifold dimensionality (PR)")

        canvas.draw()

    # ── Tab 3: Subspace ───────────────────────────────────────────────────────

    def _build_subspace_tab(self):
        tab = self.tabs.tab("Subspace")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        outer = ctk.CTkFrame(tab, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=0)
        outer.columnconfigure(1, weight=1)
        outer.rowconfigure(0, weight=1)

        # Info panel
        info = ctk.CTkFrame(outer, width=180, fg_color="transparent")
        info.grid(row=0, column=0, sticky="ns", padx=(0, 8))

        cr  = self._r.get("coding_variance_ratio", 0.0)
        nr  = self._r.get("null_variance_ratio", 1.0)
        ev  = self._r.get("explained_variance", 0.0)
        prb = self._r.get("pr_baseline", 0.0)
        prs = self._r.get("pr_stimulus", 0.0)
        delta = prs - prb

        ctk.CTkLabel(info, text="Subspace\nAnalysis",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     justify="left").pack(anchor="w", padx=8, pady=(12, 4))

        # Explained variance is the most interpretable metric — show prominently
        ctk.CTkLabel(info,
                     text=f"{ev*100:.1f}%",
                     font=ctk.CTkFont(size=26, weight="bold"),
                     text_color="#ff7f0e").pack(anchor="w", padx=8)
        ctk.CTkLabel(info,
                     text="between-condition\nvariance explained\nby coding dimension",
                     font=ctk.CTkFont(size=9), text_color="gray",
                     justify="left").pack(anchor="w", padx=8, pady=(0, 8))

        ctk.CTkFrame(info, height=1, fg_color="gray50").pack(fill="x", padx=8, pady=4)

        for label, val in [
            ("Raw coding %:", f"{cr*100:.1f}%"),
            ("Raw nullspace %:", f"{nr*100:.1f}%"),
            ("PR baseline:", f"{prb:.1f}"),
            ("PR stimulus:", f"{prs:.1f}"),
            ("dPR:", f"{delta:+.1f}"),
        ]:
            row = ctk.CTkFrame(info, fg_color="transparent")
            row.pack(fill="x", padx=8, pady=2)
            ctk.CTkLabel(row, text=label, font=ctk.CTkFont(size=9),
                         text_color="gray").pack(side="left")
            ctk.CTkLabel(row, text=val,
                         font=ctk.CTkFont(size=10, weight="bold")).pack(side="right")

        ctk.CTkFrame(info, height=1, fg_color="gray50").pack(fill="x", padx=8, pady=8)
        ctk.CTkLabel(info,
                     text="The large nullspace % is\nnormal: with 435 neurons,\nthe coding dimension is\n"
                          "1 direction out of 435.\nWhat matters is how much\nof the between-stimulus\n"
                          "variance it captures\n(shown above in orange).",
                     font=ctk.CTkFont(size=9), text_color="gray",
                     justify="left", wraplength=165).pack(anchor="w", padx=8)

        # Figure
        fig = Figure(figsize=(9, 5))
        canvas = FigureCanvasTkAgg(fig, master=outer)
        canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")

        ax_bar, ax_time = fig.subplots(1, 2)
        fig.subplots_adjust(left=0.09, right=0.97, top=0.90, bottom=0.13, wspace=0.32)

        ax_bar.bar(["Coding\nsubspace", "Nullspace"],
                   [cr * 100, nr * 100],
                   color=["#ff7f0e", "#7f7f7f"], alpha=0.8, width=0.5)
        ax_bar.set_ylabel("% of total stim-window variance")
        ax_bar.set_title("Coding subspace vs nullspace")
        ax_bar.set_ylim(0, 105)
        for patch, v in zip(ax_bar.patches, [cr, nr]):
            ax_bar.text(patch.get_x() + patch.get_width() / 2,
                        patch.get_height() + 1, f"{v*100:.1f}%",
                        ha="center", fontsize=9)

        pv = self._r.get("projection_variance_over_time", np.array([]))
        if pv.size:
            t = np.arange(len(pv)) * self._fp
            ax_time.plot(t, pv, color="#ff7f0e", lw=1.8)
            ax_time.axvline(0, color="gray", lw=0.7, ls="--", label="Stim onset")
            ax_time.fill_between(t, 0, pv, alpha=0.12, color="#ff7f0e")
            ax_time.legend(fontsize=11)
        ax_time.set_xlabel("Time from onset (s)")
        ax_time.set_ylabel("Projection variance (a.u.)")
        ax_time.set_title("Task-relevant activity over time")

        canvas.draw()

    # ── Tab 4: States / Attractors ────────────────────────────────────────────

    def _build_states_tab(self):
        tab = self.tabs.tab("States / Attractors")
        tab.columnconfigure(0, weight=0)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        st = self._states or {}
        n_states  = st.get("n_states", 0)
        dwell     = st.get("dwell_frames", np.array([]))

        # Info panel
        info = ctk.CTkFrame(tab, width=190, fg_color="transparent")
        info.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        ctk.CTkLabel(info, text="Neural States",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     justify="left").pack(anchor="w", padx=8, pady=(12, 4))
        ctk.CTkLabel(info, text=f"{n_states}",
                     font=ctk.CTkFont(size=30, weight="bold"),
                     text_color="#3cb44b").pack(anchor="w", padx=8)
        ctk.CTkLabel(info, text="discrete states\n(GMM, BIC-selected)",
                     font=ctk.CTkFont(size=9), text_color="gray",
                     justify="left").pack(anchor="w", padx=8, pady=(0, 8))
        ctk.CTkFrame(info, height=1, fg_color="gray50").pack(fill="x", padx=8, pady=6)
        ctk.CTkLabel(info,
                     text="The Hopfieldian analog\nof 'populations':\nrecurring states the\n"
                          "population visits, not\nclusters of neurons.\n\n"
                          "★ = attractor centroid\n○ = slow point\n(candidate fixed point,\n"
                          "where |dstate/dt|→0).\n\n"
                          "Dwell time = how long\nthe population stays in\na state (stability).",
                     font=ctk.CTkFont(size=9), text_color="gray",
                     justify="left", wraplength=170).pack(anchor="w", padx=8)

        fig = Figure(figsize=(11, 7))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        if not st or n_states == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No state data available",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            self._draw_states_fig(fig)
        canvas.draw()

    def _draw_states_fig(self, fig: Figure):
        st = self._states
        trajectories = st["trajectories"]
        state_labels = st["state_labels"]
        centroids    = st["state_centroids"]
        n_states     = st["n_states"]
        dwell        = st["dwell_frames"]
        transition   = st["transition_matrix"]
        slow_frames  = st["slow_point_frames"]
        N = len(trajectories)
        stim_f = trajectories[0].shape[0]

        gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.32,
                              left=0.08, right=0.95, top=0.93, bottom=0.09)
        ax_tr  = fig.add_subplot(gs[0, 0])
        ax_occ = fig.add_subplot(gs[0, 1])
        ax_dw  = fig.add_subplot(gs[1, 0])
        ax_trn = fig.add_subplot(gs[1, 1])

        for j, traj in enumerate(trajectories):
            cols = [_STATE_COLORS[int(s) % len(_STATE_COLORS)]
                    for s in state_labels[j]]
            ax_tr.plot(traj[:, 0], traj[:, 1], color="#cccccc", lw=0.8, zorder=1)
            ax_tr.scatter(traj[:, 0], traj[:, 1], c=cols, s=16, zorder=2)
            sp = slow_frames[j]
            if len(sp):
                ax_tr.scatter(traj[sp, 0], traj[sp, 1], s=85, facecolors="none",
                              edgecolors="black", linewidths=1.3, zorder=3)
        for s in range(n_states):
            color = _STATE_COLORS[s % len(_STATE_COLORS)]
            ax_tr.scatter(centroids[s, 0], centroids[s, 1], marker="*", s=300,
                          color=color, edgecolors="black", linewidths=1.0, zorder=4)
            ax_tr.annotate(f"S{s}", (centroids[s, 0], centroids[s, 1]),
                           fontsize=8, fontweight="bold", ha="center", va="center")
        ax_tr.set_xlabel("PC 1"); ax_tr.set_ylabel("PC 2")
        ax_tr.set_title("Trajectories coloured by state  (★ centroid, ○ slow point)")
        ax_tr.axhline(0, color="#cccccc", lw=0.5)
        ax_tr.axvline(0, color="#cccccc", lw=0.5)

        import matplotlib.colors as mcolors
        cmap = mcolors.ListedColormap(
            [_STATE_COLORS[s % len(_STATE_COLORS)] for s in range(n_states)])
        raster = np.vstack(state_labels)
        t = (np.arange(stim_f)) * self._fp
        ax_occ.imshow(raster, aspect="auto", cmap=cmap, vmin=-0.5,
                      vmax=n_states - 0.5, interpolation="nearest",
                      extent=[t[0], t[-1], N - 0.5, -0.5])
        ax_occ.set_yticks(range(N))
        ax_occ.set_yticklabels(
            [self._stim_names[j] if j < len(self._stim_names) else f"Stim {j+1}"
             for j in range(N)], fontsize=8)
        ax_occ.set_xlabel("Time from onset (s)")
        ax_occ.set_title("State occupancy over time")

        s_idx = np.arange(n_states)
        s_cols = [_STATE_COLORS[s % len(_STATE_COLORS)] for s in s_idx]
        ax_dw.bar(s_idx, dwell * self._fp, color=s_cols, alpha=0.85)
        ax_dw.set_xticks(s_idx); ax_dw.set_xticklabels([f"S{s}" for s in s_idx])
        ax_dw.set_xlabel("State"); ax_dw.set_ylabel("Mean dwell (s)")
        ax_dw.set_title("Dwell time per state (stability)")

        im = ax_trn.imshow(transition, cmap="magma", vmin=0, vmax=1,
                           interpolation="nearest")
        ax_trn.set_xticks(s_idx); ax_trn.set_yticks(s_idx)
        ax_trn.set_xticklabels([f"S{s}" for s in s_idx])
        ax_trn.set_yticklabels([f"S{s}" for s in s_idx])
        ax_trn.set_xlabel("To state"); ax_trn.set_ylabel("From state")
        ax_trn.set_title("Transition probability")
        for i in range(n_states):
            for k in range(n_states):
                ax_trn.text(k, i, f"{transition[i, k]:.2f}", ha="center",
                            va="center", fontsize=7,
                            color="white" if transition[i, k] < 0.6 else "black")
        fig.colorbar(im, ax=ax_trn, fraction=0.046, pad=0.04)

    # ── Tab 5: Topology ───────────────────────────────────────────────────────

    def _build_topology_tab(self):
        tab = self.tabs.tab("Topology")
        tab.columnconfigure(0, weight=0)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=1)

        tp = self._topology or {}
        available = tp.get("tda_available", False)
        n_loops   = tp.get("n_loops_pooled", 0)
        max_h1    = tp.get("max_h1_persistence", 0.0)

        info = ctk.CTkFrame(tab, width=190, fg_color="transparent")
        info.grid(row=0, column=0, sticky="ns", padx=(0, 8))
        ctk.CTkLabel(info, text="Manifold\nTopology",
                     font=ctk.CTkFont(size=13, weight="bold"),
                     justify="left").pack(anchor="w", padx=8, pady=(12, 4))

        if not available:
            ctk.CTkLabel(info,
                         text="ripser not installed.\n\npip install ripser\n\n"
                              "Persistent homology\nneeds ripser to compute\n"
                              "H0 / H1 features.",
                         font=ctk.CTkFont(size=10), text_color="#d62728",
                         justify="left", wraplength=170).pack(anchor="w", padx=8, pady=6)
            fig = Figure(figsize=(11, 7))
            canvas = FigureCanvasTkAgg(fig, master=tab)
            canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "Install ripser to compute manifold topology\n"
                              "(pip install ripser)",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            canvas.draw()
            return

        ctk.CTkLabel(info, text=f"{n_loops}",
                     font=ctk.CTkFont(size=30, weight="bold"),
                     text_color="#d62728").pack(anchor="w", padx=8)
        ctk.CTkLabel(info, text="significant loops (H1)",
                     font=ctk.CTkFont(size=9), text_color="gray").pack(
            anchor="w", padx=8, pady=(0, 6))
        ctk.CTkLabel(info, text=f"max H1 persistence: {max_h1:.3f}",
                     font=ctk.CTkFont(size=9), text_color="#ff7f0e").pack(
            anchor="w", padx=8)
        ctk.CTkFrame(info, height=1, fg_color="gray50").pack(fill="x", padx=8, pady=8)
        ctk.CTkLabel(info,
                     text="Persistent homology of\nthe population trajectory:\n\n"
                          "H0 = connected\ncomponents (separated\nstate clusters)\n\n"
                          "H1 = loops (cyclic /\nrotational dynamics,\ne.g. ring attractors)\n\n"
                          "Points far above the\ndiagonal are long-lived,\nrobust features.",
                     font=ctk.CTkFont(size=9), text_color="gray",
                     justify="left", wraplength=170).pack(anchor="w", padx=8)

        fig = Figure(figsize=(11, 7))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().grid(row=0, column=1, sticky="nsew")
        self._draw_topology_fig(fig)
        canvas.draw()

    def _draw_topology_fig(self, fig: Figure):
        tp = self._topology
        pooled   = tp["pooled_diagrams"]
        per_cond = tp.get("per_condition_diagrams") or []
        n_loops  = tp.get("n_loops_pooled", 0)
        N = len(per_cond)
        n_cols = max(2, N)
        dim_colors = ["#1f77b4", "#d62728", "#2ca02c"]

        gs = fig.add_gridspec(2, n_cols, hspace=0.42, wspace=0.38,
                              left=0.08, right=0.96, top=0.91, bottom=0.10)

        ax_dg = fig.add_subplot(gs[0, 0])
        self._draw_pers_diagram(ax_dg, pooled,
                                f"Pooled  ({n_loops} loop"
                                f"{'s' if n_loops != 1 else ''})")

        ax_bc = fig.add_subplot(gs[0, 1])
        row = 0
        for d, dgm in enumerate(pooled):
            if dgm is None or len(dgm) == 0:
                continue
            fin = dgm[np.isfinite(dgm[:, 1])]
            for b, death in fin:
                ax_bc.plot([b, death], [row, row],
                           color=dim_colors[d % len(dim_colors)], lw=2.0, alpha=0.8)
                row += 1
        ax_bc.set_xlabel("Filtration value"); ax_bc.set_ylabel("Feature")
        ax_bc.set_title("Pooled barcode")

        for j, diags in enumerate(per_cond):
            ax = fig.add_subplot(gs[1, j % n_cols])
            name = (self._stim_names[j] if j < len(self._stim_names)
                    else f"Stim {j+1}")
            self._draw_pers_diagram(ax, diags, name)

    @staticmethod
    def _draw_pers_diagram(ax, diagrams, title):
        dim_colors = ["#1f77b4", "#d62728", "#2ca02c"]
        finite_deaths = []
        for dgm in diagrams:
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
                ax.scatter(fin[:, 0], fin[:, 1], s=24,
                           color=dim_colors[d % len(dim_colors)],
                           label=f"H{d}", alpha=0.8, zorder=2)
        ax.set_xlim(0, lim); ax.set_ylim(0, lim)
        ax.set_xlabel("Birth"); ax.set_ylabel("Death")
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8)

    # ── Tab 6: Neuron Explorer ────────────────────────────────────────────────

    def _build_explorer_tab(self):
        tab = self.tabs.tab("Neuron Explorer")
        tab.columnconfigure(0, weight=2)
        tab.columnconfigure(1, weight=1)
        tab.rowconfigure(0, weight=2)
        tab.rowconfigure(1, weight=1)

        # ── spatial map (left) ────────────────────────────────────────────────
        map_frame = ctk.CTkFrame(tab, fg_color="transparent")
        map_frame.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=(0, 4))
        map_frame.rowconfigure(0, weight=1)
        map_frame.columnconfigure(0, weight=1)

        self._map_fig = Figure()
        self._map_ax  = self._map_fig.add_subplot(111)
        self._map_canvas = FigureCanvasTkAgg(self._map_fig, master=map_frame)
        self._map_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._map_canvas.mpl_connect("pick_event", self._on_map_pick)
        self._draw_spatial_map()

        # ── controls (top-right) ──────────────────────────────────────────────
        ctrl = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl.grid(row=0, column=1, sticky="nsew", pady=(0, 4))
        ctrl.columnconfigure(0, weight=1)

        ctk.CTkLabel(ctrl, text="Neuron Explorer",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=10, pady=(10, 6))

        # Threshold auto-select
        ctk.CTkLabel(ctrl, text="Auto-select by loading threshold:",
                     font=ctk.CTkFont(size=10)).pack(anchor="w", padx=10)

        thr_frame = ctk.CTkFrame(ctrl, fg_color="transparent")
        thr_frame.pack(fill="x", padx=10, pady=4)
        thr_frame.columnconfigure(1, weight=1)

        self._pos_thr_var = tk.DoubleVar(value=0.05)
        self._neg_thr_var = tk.DoubleVar(value=0.05)

        ctk.CTkLabel(thr_frame, text="+ loaders >=", width=80,
                     font=ctk.CTkFont(size=9)).grid(row=0, column=0, sticky="w")
        self._pos_lbl = ctk.CTkLabel(thr_frame, text="0.05", width=36,
                                     font=ctk.CTkFont(size=9))
        ctk.CTkSlider(thr_frame, from_=0.01, to=0.30,
                      number_of_steps=29, variable=self._pos_thr_var,
                      command=lambda v: (
                          self._pos_lbl.configure(text=f"{float(v):.2f}"),
                          self._auto_select_pos())).grid(row=0, column=1, padx=4)
        self._pos_lbl.grid(row=0, column=2)

        ctk.CTkLabel(thr_frame, text="- loaders <=", width=80,
                     font=ctk.CTkFont(size=9)).grid(row=1, column=0, sticky="w", pady=2)
        self._neg_lbl = ctk.CTkLabel(thr_frame, text="0.05", width=36,
                                     font=ctk.CTkFont(size=9))
        ctk.CTkSlider(thr_frame, from_=0.01, to=0.30,
                      number_of_steps=29, variable=self._neg_thr_var,
                      command=lambda v: (
                          self._neg_lbl.configure(text=f"{float(v):.2f}"),
                          self._auto_select_neg())).grid(row=1, column=1, padx=4)
        self._neg_lbl.grid(row=1, column=2)

        ctk.CTkButton(ctrl, text="Select nullspace",
                      command=self._select_nullspace).pack(
            fill="x", padx=10, pady=2)
        ctk.CTkButton(ctrl, text="Select all",
                      command=self._select_all).pack(fill="x", padx=10, pady=2)
        ctk.CTkButton(ctrl, text="Clear selection",
                      command=self._clear_selection).pack(
            fill="x", padx=10, pady=2)

        ctk.CTkFrame(ctrl, height=1, fg_color="gray50").pack(fill="x", padx=10, pady=6)

        self._sel_lbl = ctk.CTkLabel(ctrl, text="0 neurons selected",
                                     font=ctk.CTkFont(size=10))
        self._sel_lbl.pack(anchor="w", padx=10)

        # Stimulus selector for traces
        ctk.CTkLabel(ctrl, text="Show traces for:",
                     font=ctk.CTkFont(size=9),
                     text_color="gray").pack(anchor="w", padx=10, pady=(8, 2))
        N = len(self._stims_n)
        stim_vals = [self._stim_names[j] if j < len(self._stim_names)
                     else f"Stim {j+1}" for j in range(N)]
        if stim_vals:
            self._stim_btn = ctk.CTkSegmentedButton(
                ctrl, values=stim_vals, command=self._on_stim_changed)
            self._stim_btn.set(stim_vals[0])
            self._stim_btn.pack(fill="x", padx=10, pady=2)

        ctk.CTkFrame(ctrl, height=1, fg_color="gray50").pack(fill="x", padx=10, pady=6)

        # 3D / animation buttons
        ctk.CTkButton(
            ctrl, text="Open 3-D spatial map  (PyVista)",
            fg_color="#3b8ed0", hover_color="#1f6aa5",
            command=self._open_pyvista_3d,
        ).pack(fill="x", padx=10, pady=2)

        ctk.CTkButton(
            ctrl, text="Open activation animation  (browser)",
            fg_color="#3b8ed0", hover_color="#1f6aa5",
            command=self._open_plotly_animation,
        ).pack(fill="x", padx=10, pady=2)

        if self._centroids is None:
            ctk.CTkLabel(ctrl,
                         text="No spatial data -- 3-D buttons disabled.",
                         font=ctk.CTkFont(size=8), text_color="gray",
                         wraplength=180).pack(anchor="w", padx=10, pady=2)

        # ── trace panel (bottom-right) ────────────────────────────────────────
        trace_frame = ctk.CTkFrame(tab, fg_color="transparent")
        trace_frame.grid(row=1, column=1, sticky="nsew")
        trace_frame.rowconfigure(0, weight=1)
        trace_frame.columnconfigure(0, weight=1)

        self._trace_fig    = Figure(figsize=(5, 2.5))
        self._trace_ax     = self._trace_fig.add_subplot(111)
        self._trace_canvas = FigureCanvasTkAgg(self._trace_fig, master=trace_frame)
        self._trace_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self._draw_trace_panel()

    # ── Spatial map drawing ───────────────────────────────────────────────────

    def _draw_spatial_map(self):
        # Rebuild the whole figure each redraw. ax.clear() wipes only the axes
        # contents, not the colorbar (a separate axes), so calling fig.colorbar()
        # again on every click stacked a new colorbar each time — stealing width
        # and shrinking the map indefinitely. Clearing the figure avoids that.
        self._map_fig.clear()
        ax = self._map_fig.add_subplot(111)
        self._map_ax = ax
        cv = self._coding_vector
        K  = len(cv)

        if K == 0 or self._centroids is None:
            ax.text(0.5, 0.5, "No spatial data available",
                    ha="center", va="center", transform=ax.transAxes)
            self._map_canvas.draw()
            return

        cents = self._centroids  # (K, 2) row / col
        cols_pos = cents[:, 1]
        rows_pos = cents[:, 0]

        # Normalise loading for colourmap
        vmax = np.abs(cv).max() if np.abs(cv).max() > 1e-10 else 1.0
        norm = mpl_norm(vmin=-vmax, vmax=vmax)
        cmap = "RdBu_r"

        sc = ax.scatter(cols_pos, -rows_pos, c=cv, cmap=cmap,
                        norm=norm, s=40, alpha=0.8, picker=True, pickradius=6,
                        zorder=2)
        self._map_fig.colorbar(sc, ax=ax, label="Loading", fraction=0.04, pad=0.02)

        # Selected neurons: yellow ring
        if self._selected:
            idx = np.array(sorted(self._selected))
            ax.scatter(cols_pos[idx], -rows_pos[idx],
                       s=80, facecolors="none", edgecolors="gold",
                       linewidths=2.0, zorder=3)

        ax.set_xlabel("Column (px)")
        ax.set_ylabel("Row (px, flipped)")
        ax.set_title(f"Neurons coloured by coding dimension loading  "
                     f"({len(self._selected)} selected)")
        ax.set_aspect("equal")
        self._map_canvas.draw()

    def _on_map_pick(self, event):
        if event.ind is None or len(event.ind) == 0:
            return
        k = int(event.ind[0])
        if k in self._selected:
            self._selected.discard(k)
        else:
            self._selected.add(k)
        self._refresh_selection()

    # ── Trace panel drawing ───────────────────────────────────────────────────

    def _draw_trace_panel(self):
        ax = self._trace_ax
        ax.clear()

        if not self._selected or self._stim_idx >= len(self._stims_n):
            ax.text(0.5, 0.5,
                    "Click neurons on the map\nto see their traces here",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="gray")
            self._trace_canvas.draw()
            return

        traces = self._stims_n[self._stim_idx]  # (K, T)
        t = (np.arange(traces.shape[1]) - self._stim_onset) * self._fp
        idx = np.array(sorted(self._selected))

        for k in idx:
            ax.plot(t, traces[k], lw=0.8, alpha=0.45, color="#4fa1ca")
        mean_tr = traces[idx].mean(axis=0)
        ax.plot(t, mean_tr, lw=2.0, color="#1f77b4", label="Mean")
        ax.axvline(0, color="gray", lw=0.7, ls="--")
        ax.axhline(0, color="#cccccc", lw=0.5)
        sname = (self._stim_names[self._stim_idx]
                 if self._stim_idx < len(self._stim_names)
                 else f"Stim {self._stim_idx + 1}")
        ax.set_title(f"{len(idx)} selected neurons  --  {sname}", fontsize=9)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_ylabel("z-score", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=10)

        self._trace_canvas.draw()

    # ── Selection helpers ─────────────────────────────────────────────────────

    def _refresh_selection(self):
        self._sel_lbl.configure(text=f"{len(self._selected)} neurons selected")
        self._draw_spatial_map()
        self._draw_trace_panel()

    def _auto_select_pos(self):
        thr = float(self._pos_thr_var.get())
        cv = self._coding_vector
        self._selected = {k for k in range(len(cv)) if cv[k] >= thr}
        self._refresh_selection()

    def _auto_select_neg(self):
        thr = float(self._neg_thr_var.get())
        cv = self._coding_vector
        self._selected = {k for k in range(len(cv)) if cv[k] <= -thr}
        self._refresh_selection()

    def _select_nullspace(self):
        thr = max(float(self._pos_thr_var.get()), float(self._neg_thr_var.get()))
        cv = self._coding_vector
        self._selected = {k for k in range(len(cv)) if abs(cv[k]) < thr}
        self._refresh_selection()

    def _select_all(self):
        self._selected = set(range(len(self._coding_vector)))
        self._refresh_selection()

    def _clear_selection(self):
        self._selected.clear()
        self._refresh_selection()

    def _on_stim_changed(self, value):
        names = [self._stim_names[j] if j < len(self._stim_names)
                 else f"Stim {j+1}" for j in range(len(self._stims_n))]
        if value in names:
            self._stim_idx = names.index(value)
        self._draw_trace_panel()

    # ── PyVista 3D ────────────────────────────────────────────────────────────

    def _open_pyvista_3d(self):
        if self._centroids is None:
            return
        try:
            import pyvista as pv  # noqa: F401
        except ImportError:
            ctk.CTkToplevel(self).title("Install pyvista:  pip install pyvista")
            return

        self._pv_picks = []
        threading.Thread(target=self._pyvista_worker, daemon=True).start()
        self.after(600, self._poll_pyvista_picks)

    def _pyvista_worker(self):
        import pyvista as pv

        cents = self._centroids  # (K, 2) row / col
        cv    = self._coding_vector
        K     = len(cv)

        # Infer z-plane index per neuron from centroids if z_ids available
        # Fall back to all at z=0
        z_ids = self._r.get("z_ids_all")
        if z_ids is not None and len(z_ids) == K:
            z_vals = z_ids.astype(float)
        else:
            z_vals = np.zeros(K)
        z_scale = 50.0

        # Scale z so planes are visually separated (50 px spacing)
        z_scale = 50.0
        pts = np.column_stack([cents[:, 1], -cents[:, 0], z_vals * z_scale])
        cloud = pv.PolyData(pts)
        cloud["loading"] = cv
        cloud["abs_loading"] = np.abs(cv)

        # Point sizes: larger spheres for high-|loading| neurons so they stand out
        max_abs = float(np.abs(cv).max()) if np.abs(cv).max() > 1e-10 else 1.0
        clim = max(0.05, float(np.percentile(np.abs(cv), 95)))

        pl = pv.Plotter(title="Neural Manifold - Neuron Spatial Map (PyVista)")
        pl.add_mesh(cloud, scalars="loading", cmap="RdBu_r",
                    clim=[-clim, clim],
                    render_points_as_spheres=True, point_size=12,
                    show_scalar_bar=True,
                    scalar_bar_args={"title": "Loading\n(orange=cond1, blue=cond2)"})
        pl.enable_point_picking(
            callback=self._pyvista_pick_cb,
            use_picker=False,
            show_message=True,
            show_point=True,
        )
        pl.add_axes()
        pl.show()

    def _pyvista_pick_cb(self, point, *args):
        """Called by PyVista when a point is clicked — find closest neuron."""
        cents = self._centroids
        if cents is None or point is None:
            return
        z_ids = self._r.get("z_ids_all")
        z_scale = 50.0
        z_vals = (z_ids.astype(float) * z_scale
                  if (z_ids is not None and len(z_ids) == len(cents))
                  else np.zeros(len(cents)))

        pts = np.column_stack([cents[:, 1], -cents[:, 0], z_vals])
        try:
            pt = np.array(point, dtype=float).ravel()[:3]
            dists = np.linalg.norm(pts - pt, axis=1)
            k = int(np.argmin(dists))
        except Exception:
            return

        with self._pv_lock:
            self._pv_picks.append(k)

    def _poll_pyvista_picks(self):
        with self._pv_lock:
            picks = list(self._pv_picks)
            self._pv_picks.clear()

        for k in picks:
            if k in self._selected:
                self._selected.discard(k)
            else:
                self._selected.add(k)

        if picks:
            self._refresh_selection()

        # Keep polling as long as the tab is visible
        self.after(500, self._poll_pyvista_picks)

    # ── Plotly activation animation ───────────────────────────────────────────

    def _open_plotly_animation(self):
        if self._centroids is None:
            return
        try:
            import plotly.graph_objects as go
        except ImportError:
            ctk.CTkToplevel(self).title("Install plotly:  pip install plotly")
            return

        cents = self._centroids
        stim_idx = self._stim_idx
        traces = self._stims_n[stim_idx]  # (K, T)
        stim_f = traces.shape[1] - self._stim_onset
        if stim_f <= 0:
            return

        # If neurons are selected, show only those; else all
        if self._selected:
            nidx = np.array(sorted(self._selected))
        else:
            nidx = np.arange(len(cents))

        cols = cents[nidx, 1].tolist()
        rows = (-cents[nidx, 0]).tolist()
        activations = traces[nidx, self._stim_onset:]  # (n_sel, stim_f)
        t_axis = np.arange(stim_f) * self._fp

        frames = [
            go.Frame(
                data=[go.Scatter(
                    x=cols, y=rows, mode="markers",
                    marker=dict(
                        color=activations[:, t_i].tolist(),
                        colorscale="RdBu_r", cmin=-3, cmax=3, size=10,
                        colorbar=dict(title="z-score"),
                    ),
                    hovertext=[f"Neuron {nidx[i]}<br>z={activations[i, t_i]:.2f}"
                               for i in range(len(nidx))],
                )],
                name=f"{t_axis[t_i]:.2f}",
            )
            for t_i in range(stim_f)
        ]

        # Initial frame
        init_data = go.Scatter(
            x=cols, y=rows, mode="markers",
            marker=dict(color=activations[:, 0].tolist(),
                        colorscale="RdBu_r", cmin=-3, cmax=3, size=10,
                        colorbar=dict(title="z-score")),
        )

        sname = (self._stim_names[stim_idx] if stim_idx < len(self._stim_names)
                 else f"Stim {stim_idx + 1}")

        fig = go.Figure(
            data=[init_data],
            frames=frames,
            layout=go.Layout(
                title=f"Neuron activation over time - {sname}",
                xaxis_title="Column (px)", yaxis_title="Row (px, flipped)",
                xaxis=dict(scaleanchor="y"),
                updatemenus=[dict(
                    type="buttons", showactive=False,
                    buttons=[
                        dict(label="Play",
                             method="animate",
                             args=[None, dict(frame=dict(duration=80, redraw=True),
                                             fromcurrent=True)]),
                        dict(label="Pause",
                             method="animate",
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                               mode="immediate")]),
                    ],
                )],
                sliders=[dict(
                    steps=[dict(method="animate", args=[[f.name],
                                dict(mode="immediate",
                                     frame=dict(duration=80, redraw=True))],
                                label=f.name)
                           for f in frames],
                    x=0.05, len=0.95,
                    currentvalue=dict(prefix="t = ", suffix=" s", visible=True),
                )],
            ),
        )

        with tempfile.NamedTemporaryFile(
                suffix=".html", delete=False, mode="w", encoding="utf-8") as fh:
            fig.write_html(fh.name, include_plotlyjs="cdn")
            webbrowser.open(fh.name)


# ── small helpers ─────────────────────────────────────────────────────────────

def _time_axis_fn(n: int, fp: float, onset: int) -> np.ndarray:
    return (np.arange(n) - onset) * fp


def mpl_norm(vmin: float, vmax: float):
    import matplotlib.colors as mc
    return mc.Normalize(vmin=vmin, vmax=vmax)
