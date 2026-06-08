"""Cell-type analysis viewer (Sherringtonian view).

Displays functional cell-type clusters, single-neuron selectivity/tuning,
functional connectivity, spatial organisation, and spatial topology for the
cell-type tool (analysis/population.py).  This is NOT based on Ebitz & Hayden
(2021); for the Hopfieldian manifold viewer see gui/manifold_viewer.py.

All heavy computation is done before this window opens; the viewer renders
results, supports interactive re-clustering via the n-clusters slider, and
computes spatial topology on demand.
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import tkinter as tk
import customtkinter as ctk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3-D projection)

from analysis.population import (
    compute_coactivity_clusters,
    compute_spatial_topology,
    _HAS_RIPSER,
)

_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]
_CLUSTER_COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45",
    "#fabed4", "#469990", "#dcbeff", "#9a6324",
    "#800000", "#aaffc3", "#ffd8b1", "#000075",
]


class PopulationViewerWindow(ctk.CTkToplevel):
    """Interactive viewer for population-level neural analysis results.

    Parameters
    ----------
    parent     : parent tkinter widget (PipelineGUI)
    results    : dict returned by analysis.population.run_population_analysis
    results_dir: path to the analysis folder (used for the 'open folder' button)
    """

    def __init__(self, parent, results: dict, results_dir: str):
        super().__init__(parent)
        self.title("Population Analysis")
        self.geometry("1100x750")
        self.resizable(True, True)

        self._results    = results
        self._results_dir = results_dir
        self._pop_dir    = results.get("pop_dir", str(Path(results_dir) / "population"))

        # Data shortcuts
        self._stims_n         = results["stims_n"]
        self._stim_onset_idx  = results["stim_onset_idx"]
        self._fp              = results.get("fp", 0.585)
        self._stim_names      = results.get("stim_names", [])
        self._cluster_labels  = results["labels"].copy()
        self._cluster_method  = results.get("method", "kmeans")
        self._n_clusters      = results["n_clusters"]

        # Sherringtonian sub-analyses
        self._selectivity  = results.get("selectivity", {})
        self._quality      = results.get("quality", {})
        self._connectivity = results.get("connectivity", {})
        self._spatial      = results.get("spatial", {})

        # Threading lock
        self._recompute_lock = threading.Lock()

        self._build_ui()
        self.lift()
        self.focus_force()

    # ── UI layout ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.tabs = ctk.CTkTabview(self, anchor="nw")
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)

        for name in ("Populations", "Selectivity", "Connectivity",
                     "Spatial", "Spatial Topology", "Summary"):
            self.tabs.add(name)

        self._build_populations_tab()
        self._build_selectivity_tab()
        self._build_connectivity_tab()
        self._build_spatial_tab()
        self._build_spatial_topology_tab()
        self._build_summary_tab()

    # ── Tab 1: Populations ─────────────────────────────────────────────────────

    def _build_populations_tab(self):
        tab = self.tabs.tab("Populations")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        # Main splitter
        outer = ctk.CTkFrame(tab, fg_color="transparent")
        outer.grid(row=0, column=0, sticky="nsew")
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(0, weight=1)

        # Left: cluster trace figure
        plot_frame = ctk.CTkFrame(outer, fg_color="transparent")
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self._cluster_fig = Figure()
        self._cluster_canvas = FigureCanvasTkAgg(self._cluster_fig, master=plot_frame)
        self._cluster_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Right: controls
        ctrl = ctk.CTkFrame(outer, width=220, fg_color="transparent", corner_radius=8)
        ctrl.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        ctrl.columnconfigure(0, weight=1)

        ctk.CTkLabel(ctrl, text="Populations",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=12, pady=(12, 4))

        ctk.CTkLabel(ctrl, text="Number of clusters:",
                     font=ctk.CTkFont(size=10)).pack(anchor="w", padx=12)

        self._n_clusters_var = tk.IntVar(value=self._n_clusters)
        self._n_clust_lbl = ctk.CTkLabel(
            ctrl, text=str(self._n_clusters),
            font=ctk.CTkFont(size=12, weight="bold"), text_color="#4fa1ca")
        self._n_clust_lbl.pack(anchor="w", padx=12)

        self._clust_slider = ctk.CTkSlider(
            ctrl, from_=2, to=min(20, self._stims_n[0].shape[0]),
            number_of_steps=min(18, self._stims_n[0].shape[0] - 2),
            variable=self._n_clusters_var,
            command=self._on_n_clusters_changed,
        )
        self._clust_slider.pack(fill="x", padx=12, pady=(2, 8))

        ctk.CTkLabel(ctrl, text="Method:", font=ctk.CTkFont(size=10)).pack(
            anchor="w", padx=12)
        self._method_btn = ctk.CTkSegmentedButton(
            ctrl,
            values=["K-Means", "GMM", "Hierarchical"],
            command=self._on_method_changed,
        )
        self._method_btn.set("K-Means")
        self._method_btn.pack(fill="x", padx=12, pady=(2, 12))

        ctk.CTkFrame(ctrl, height=1, fg_color="#444444").pack(fill="x", padx=12)

        self._sizes_lbl = ctk.CTkLabel(
            ctrl, text="", font=ctk.CTkFont(family="Courier", size=9),
            text_color="gray", justify="left", anchor="w", wraplength=190)
        self._sizes_lbl.pack(anchor="w", padx=12, pady=(8, 4))

        self._status_lbl = ctk.CTkLabel(
            ctrl, text="", font=ctk.CTkFont(size=9), text_color="gray")
        self._status_lbl.pack(anchor="w", padx=12)

        ctk.CTkFrame(ctrl, height=1, fg_color="#444444").pack(
            fill="x", padx=12, pady=8)
        ctk.CTkButton(
            ctrl, text="Save clusters.npy",
            fg_color="#3b8ed0", hover_color="#1f6aa5",
            command=self._save_clusters,
        ).pack(fill="x", padx=12, pady=(0, 12))

        self._draw_cluster_traces(self._results["cluster_means"])
        self._update_sizes_label(self._results["cluster_sizes"])

    def _on_n_clusters_changed(self, value):
        n = int(float(value))
        self._n_clust_lbl.configure(text=str(n))
        self._schedule_recompute(n, self._cluster_method)

    def _on_method_changed(self, value):
        method_map = {"K-Means": "kmeans", "GMM": "gmm", "Hierarchical": "hierarchical"}
        self._cluster_method = method_map[value]
        n = int(self._n_clusters_var.get())
        self._schedule_recompute(n, self._cluster_method)

    def _schedule_recompute(self, n_clusters: int, method: str):
        self._status_lbl.configure(text="Computing…")
        t = threading.Thread(
            target=self._recompute_clusters,
            args=(n_clusters, method),
            daemon=True,
        )
        t.start()

    def _recompute_clusters(self, n_clusters: int, method: str):
        with self._recompute_lock:
            try:
                result = compute_coactivity_clusters(
                    self._stims_n, self._stim_onset_idx,
                    n_clusters=n_clusters, method=method)
                self._cluster_labels = result["labels"].copy()
                self._n_clusters = result["n_clusters"]
                self.after(0, lambda r=result: self._apply_cluster_result(r))
            except Exception as exc:
                self.after(0, lambda e=exc: self._status_lbl.configure(
                    text=f"Error: {e}"))

    def _apply_cluster_result(self, result: dict):
        self._draw_cluster_traces(result["cluster_means"])
        self._update_sizes_label(result["cluster_sizes"])
        self._status_lbl.configure(text="")

    def _draw_cluster_traces(self, cluster_means: list[np.ndarray]):
        self._cluster_fig.clear()
        n_clusters = cluster_means[0].shape[0]
        N = len(cluster_means)
        T = cluster_means[0].shape[1]
        t = (np.arange(T) - self._stim_onset_idx) * self._fp

        axes = self._cluster_fig.subplots(n_clusters, N, squeeze=False)
        self._cluster_fig.subplots_adjust(
            hspace=0.4, wspace=0.25, left=0.09, right=0.97, top=0.94, bottom=0.08)

        for c in range(n_clusters):
            color = _CLUSTER_COLORS[c % len(_CLUSTER_COLORS)]
            for j in range(N):
                ax = axes[c, j]
                ax.plot(t, cluster_means[j][c], color=color, linewidth=1.2)
                ax.axvline(0, color="gray", lw=0.5, ls="--")
                ax.axhline(0, color="#cccccc", lw=0.4)
                ax.tick_params(labelsize=6)
                if j == 0:
                    ax.set_ylabel(f"Cl.{c}", fontsize=7, color=color,
                                  fontweight="bold")
                if c == 0 and self._stim_names:
                    ax.set_title(self._stim_names[j], fontsize=8)
                ax.set_xlim(t[0], t[-1])
        for ax in axes[-1]:
            ax.set_xlabel("t (s)", fontsize=7)

        self._cluster_canvas.draw()

    def _update_sizes_label(self, cluster_sizes: np.ndarray):
        lines = [f"Cl.{c}: {int(cluster_sizes[c])} neurons"
                 for c in range(len(cluster_sizes))]
        self._sizes_lbl.configure(text="\n".join(lines))

    def _save_clusters(self):
        out = Path(self._pop_dir) / "clusters.npy"
        np.save(str(out), self._cluster_labels)
        self._status_lbl.configure(text=f"Saved → {out.name}")

    # ── Tab 2: Selectivity ─────────────────────────────────────────────────────

    def _build_selectivity_tab(self):
        tab = self.tabs.tab("Selectivity")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        fig = Figure(figsize=(12, 4.6))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        sel = self._selectivity or {}
        resp = sel.get("response_matrix")
        if resp is None or resp.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No selectivity data", ha="center", va="center",
                    transform=ax.transAxes)
            canvas.draw()
            return

        pref = sel["preferred_stim"]
        si   = sel["selectivity"]
        K, N = resp.shape
        ax_hist, ax_pref, ax_heat = fig.subplots(1, 3)
        fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.16, wspace=0.32)

        ax_hist.hist(si, bins=20, range=(0, 1), color="#4fa1ca", alpha=0.85)
        ax_hist.axvline(float(si.mean()), color="#d62728", ls="--", lw=1.2,
                        label=f"mean {si.mean():.2f}")
        ax_hist.set_xlabel("Selectivity index")
        ax_hist.set_ylabel("Neurons")
        ax_hist.set_title("Selectivity (0=broad, 1=single-stim)")
        ax_hist.legend(fontsize=9)

        counts = [int((pref == j).sum()) for j in range(N)]
        ax_pref.bar(range(N), counts,
                    color=[_COLORS[j % len(_COLORS)] for j in range(N)], alpha=0.85)
        ax_pref.set_xticks(range(N))
        ax_pref.set_xticklabels(
            [self._stim_names[j] if j < len(self._stim_names) else f"S{j+1}"
             for j in range(N)], rotation=30, ha="right", fontsize=8)
        ax_pref.set_ylabel("Neurons")
        ax_pref.set_title("Preferred stimulus")

        order = np.lexsort((-si, pref))
        im = ax_heat.imshow(resp[order], aspect="auto", cmap="viridis",
                            interpolation="nearest")
        ax_heat.set_xticks(range(N))
        ax_heat.set_xticklabels(
            [self._stim_names[j] if j < len(self._stim_names) else f"S{j+1}"
             for j in range(N)], rotation=30, ha="right", fontsize=8)
        ax_heat.set_ylabel("Neuron (sorted by tuning)")
        ax_heat.set_title("Tuning matrix")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04, label="z")
        canvas.draw()

    # ── Tab 3: Connectivity ────────────────────────────────────────────────────

    def _build_connectivity_tab(self):
        tab = self.tabs.tab("Connectivity")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        fig = Figure(figsize=(11, 5))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        cn = self._connectivity or {}
        corr = cn.get("corr")
        if corr is None or corr.size == 0:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, "No connectivity data", ha="center", va="center",
                    transform=ax.transAxes)
            canvas.draw()
            return

        order = cn["order"]
        labels_sorted = cn["labels_sorted"]
        within, between = cn["within_corr"], cn["between_corr"]
        sorted_corr = corr[np.ix_(order, order)]

        ax_mat, ax_bar = fig.subplots(1, 2, gridspec_kw={"width_ratios": [2, 1]})
        fig.subplots_adjust(left=0.06, right=0.96, top=0.90, bottom=0.10, wspace=0.28)

        im = ax_mat.imshow(sorted_corr, cmap="RdBu_r", vmin=-1, vmax=1,
                           interpolation="nearest")
        for b in np.where(np.diff(labels_sorted) != 0)[0] + 0.5:
            ax_mat.axhline(b, color="black", lw=0.6)
            ax_mat.axvline(b, color="black", lw=0.6)
        ax_mat.set_title("Correlation (sorted by cell-type)")
        ax_mat.set_xlabel("Neuron"); ax_mat.set_ylabel("Neuron")
        fig.colorbar(im, ax=ax_mat, fraction=0.046, pad=0.04, label="r")

        ax_bar.bar(["Within\ntype", "Between\ntypes"], [within, between],
                   color=["#2ca02c", "#7f7f7f"], alpha=0.85, width=0.5)
        ax_bar.set_ylabel("Mean |correlation|")
        ratio = within / between if between > 1e-10 else 0.0
        ax_bar.set_title(f"Block structure (ratio {ratio:.2f})")
        for i, v in enumerate([within, between]):
            ax_bar.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
        canvas.draw()

    # ── Tab 4: Spatial ─────────────────────────────────────────────────────────

    def _build_spatial_tab(self):
        tab = self.tabs.tab("Spatial")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        fig = Figure(figsize=(13, 5.5))
        canvas = FigureCanvasTkAgg(fig, master=tab)
        canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        sp = self._spatial or {}
        if not sp.get("available") or sp.get("coords3d") is None:
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5,
                    "No spatial data — centroids were not available for this run.",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            canvas.draw()
            return

        coords = sp["coords3d"]
        labels = self._cluster_labels
        pvals, obs, null = sp["cluster_pvalues"], sp["observed_dist"], sp["null_dist"]
        n_clusters = int(labels.max()) + 1 if len(labels) else 0

        ax3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax_bar = fig.add_subplot(1, 2, 2)
        fig.subplots_adjust(left=0.03, right=0.96, top=0.92, bottom=0.10, wspace=0.20)

        for c in range(n_clusters):
            m = labels == c
            ax3d.scatter(coords[m, 0], coords[m, 1], coords[m, 2],
                         color=_CLUSTER_COLORS[c % len(_CLUSTER_COLORS)],
                         s=14, alpha=0.8, label=f"Cl.{c}")
        ax3d.set_xlabel("Column"); ax3d.set_ylabel("Row")
        ax3d.set_zlabel("Plane")
        ax3d.set_title("Cell-types in 3-D anatomical space")
        if n_clusters <= 10:
            ax3d.legend(fontsize=7, loc="upper left")

        x = np.arange(n_clusters)
        w = 0.38
        ax_bar.bar(x - w / 2, obs, w, color="#2ca02c", alpha=0.85, label="Observed")
        ax_bar.bar(x + w / 2, null, w, color="#cccccc", alpha=0.85, label="Random")
        for c in range(n_clusters):
            if c < len(pvals) and pvals[c] < 0.05:
                ax_bar.text(c, max(obs[c], null[c]) * 1.02, "*", ha="center",
                            fontsize=14, color="#d62728")
        ax_bar.set_xticks(x); ax_bar.set_xticklabels([f"Cl.{c}" for c in x])
        ax_bar.set_ylabel("Mean within-type 3-D distance")
        ax_bar.set_title("Spatial aggregation  (* p<0.05 ⇒ compact)")
        ax_bar.legend(fontsize=9)
        canvas.draw()

    # ── Tab 5: Spatial Topology ────────────────────────────────────────────────

    def _build_spatial_topology_tab(self):
        tab = self.tabs.tab("Spatial Topology")
        tab.columnconfigure(0, weight=1)
        tab.columnconfigure(1, weight=0)
        tab.rowconfigure(0, weight=1)

        plot_frame = ctk.CTkFrame(tab, fg_color="transparent")
        plot_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 4))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)

        self._sptopo_fig = Figure()
        self._sptopo_canvas = FigureCanvasTkAgg(self._sptopo_fig, master=plot_frame)
        self._sptopo_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        info = ctk.CTkFrame(tab, width=210, fg_color="transparent", corner_radius=8)
        info.grid(row=0, column=1, sticky="ns", padx=(4, 0))

        ctk.CTkLabel(info, text="Spatial Topology",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(
            anchor="w", padx=12, pady=(12, 4))
        ctk.CTkLabel(info,
                     text="Persistent homology of the\n3-D centroid arrangement:\n\n"
                          "H0 = spatial clumps\nH1 = ring layouts\nH2 = voids (shells)\n\n"
                          "Asks whether the physical\norganisation of the\npopulation has structure.",
                     font=ctk.CTkFont(size=9), text_color="gray",
                     justify="left", wraplength=185).pack(anchor="w", padx=12)

        sp = self._spatial or {}
        have_coords = sp.get("available") and sp.get("coords3d") is not None

        self._sptopo_status = ctk.CTkLabel(
            info, text="", font=ctk.CTkFont(size=9), text_color="gray")

        if not _HAS_RIPSER:
            ctk.CTkLabel(info, text="ripser not installed\npip install ripser",
                         font=ctk.CTkFont(size=9), text_color="#d62728",
                         justify="left").pack(anchor="w", padx=12, pady=8)
        elif not have_coords:
            ctk.CTkLabel(info, text="No centroids available\nfor this run.",
                         font=ctk.CTkFont(size=9), text_color="#d62728",
                         justify="left").pack(anchor="w", padx=12, pady=8)
        else:
            ctk.CTkButton(info, text="Compute spatial topology",
                          fg_color="#3b8ed0", hover_color="#1f6aa5",
                          command=self._on_compute_spatial_topology).pack(
                fill="x", padx=12, pady=8)
        self._sptopo_status.pack(anchor="w", padx=12, pady=4)

        # If diagrams were already computed during the run, show them.
        diagrams = sp.get("spatial_diagrams")
        if diagrams is not None:
            self._draw_spatial_topology(diagrams)
        else:
            ax = self._sptopo_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Click 'Compute spatial topology'\nto run persistent homology",
                    ha="center", va="center", transform=ax.transAxes, color="gray")
            self._sptopo_canvas.draw()

    def _on_compute_spatial_topology(self):
        self._sptopo_status.configure(text="Computing… (may take a moment)")
        threading.Thread(target=self._run_spatial_topology, daemon=True).start()

    def _run_spatial_topology(self):
        try:
            coords = self._spatial["coords3d"]
            diagrams = compute_spatial_topology(coords)
            self.after(0, lambda d=diagrams: self._apply_spatial_topology(d))
        except Exception as exc:
            self.after(0, lambda e=exc: self._sptopo_status.configure(
                text=f"Error: {e}"))

    def _apply_spatial_topology(self, diagrams):
        self._sptopo_status.configure(text="")
        if diagrams is None:
            self._sptopo_status.configure(text="No output (ripser unavailable).")
            return
        self._draw_spatial_topology(diagrams)

    def _draw_spatial_topology(self, diagrams: list):
        self._sptopo_fig.clear()
        dim_colors = ["#1f77b4", "#d62728", "#2ca02c"]
        ax_dg, ax_bc = self._sptopo_fig.subplots(1, 2)
        self._sptopo_fig.subplots_adjust(
            left=0.09, right=0.96, top=0.90, bottom=0.13, wspace=0.28)

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
                ax_dg.scatter(fin[:, 0], fin[:, 1], s=24,
                              color=dim_colors[d % len(dim_colors)],
                              label=f"H{d}", alpha=0.8)
        ax_dg.set_xlim(0, lim); ax_dg.set_ylim(0, lim)
        ax_dg.set_xlabel("Birth (distance)"); ax_dg.set_ylabel("Death (distance)")
        ax_dg.set_title("Spatial persistence diagram")
        ax_dg.legend(fontsize=8)

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
        self._sptopo_canvas.draw()

    # ── Tab 6: Summary ─────────────────────────────────────────────────────────

    def _build_summary_tab(self):
        tab = self.tabs.tab("Summary")
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(0, weight=1)

        frame = ctk.CTkFrame(tab, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        # Button row
        btn_row = ctk.CTkFrame(frame, fg_color="transparent")
        btn_row.grid(row=0, column=0, sticky="ew", pady=(4, 0))
        ctk.CTkButton(
            btn_row, text="Open results folder",
            fg_color="#3b8ed0", hover_color="#1f6aa5",
            command=lambda: os.startfile(self._pop_dir),
        ).pack(side="left", padx=10, pady=6)

        # Summary text
        box = ctk.CTkTextbox(
            frame, font=ctk.CTkFont(family="Courier", size=12),
            state="normal", fg_color="transparent")
        box.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))

        box.insert("end", self._build_summary_text())
        box.configure(state="disabled")

    def _build_summary_text(self) -> str:
        r = self._results
        nc   = r.get("n_clusters", 0)
        cs   = r.get("cluster_sizes", np.array([]))
        meth = r.get("method", "")
        sn   = r.get("stim_names", [])
        sel  = self._selectivity or {}
        qual = self._quality or {}
        cn   = self._connectivity or {}
        sp   = self._spatial or {}

        mean_sel = float(sel["selectivity"].mean()) if sel.get("selectivity") is not None \
            and len(sel["selectivity"]) else 0.0

        lines = [
            "=== Cell-type Analysis Summary (Sherringtonian) ===",
            "",
            f"Stimuli: {', '.join(sn)}",
            f"Neurons analysed: {r['stims_n'][0].shape[0]}",
            "",
            "── Functional cell-types ────────────",
            f"  Method            : {meth}",
            f"  Clusters (current): {nc}",
            f"  Best k (silhouette): {qual.get('best_k', '-')}",
            f"  Silhouette @ current k: {qual.get('current_silhouette', 0.0):.3f}",
        ]
        for c, sz in enumerate(cs):
            lines.append(f"    Cluster {c}: {int(sz)} neurons")

        lines += [
            "",
            "── Single-neuron tuning ─────────────",
            f"  Mean selectivity index : {mean_sel:.3f}",
            "",
            "── Functional connectivity ──────────",
            f"  Within-type |r|  : {cn.get('within_corr', 0.0):.3f}",
            f"  Between-type |r| : {cn.get('between_corr', 0.0):.3f}",
            f"  Block ratio      : {cn.get('modularity_ratio', 0.0):.2f}",
        ]

        if sp.get("available"):
            pv = sp.get("cluster_pvalues", np.array([]))
            n_compact = int(np.sum(pv < 0.05)) if len(pv) else 0
            lines += [
                "",
                "── Spatial organisation ─────────────",
                f"  Spatially compact cell-types : {n_compact} / {len(pv)}",
                "  (* clusters whose 3-D arrangement is tighter than chance)",
            ]
        lines += ["", f"Results saved to: {self._pop_dir}"]
        return "\n".join(lines)
