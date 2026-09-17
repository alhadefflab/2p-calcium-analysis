"""
Luceo
2P Calcium Imaging Pipeline
Run:  python gui.py
Requires: pip install customtkinter
"""
import sys
import threading
import traceback
from pathlib import Path

import tkinter as tk
import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

from gui.roi_editor import ROIEditorWindow
from visualization.response_plots import show_plots, show_region_plots, show_spatial_response_map


# ── helpers ───────────────────────────────────────────────────────────────────

def _detect_zplanes(folder: str) -> list[str]:
    """Scan a session folder and return the z-plane names it contains."""
    import re
    path = Path(folder)
    if not path.is_dir():
        return []
    zs = set()
    for f in path.iterdir():
        if f.suffix == ".tif":
            m = re.findall(r"(\d{6})\.ome", f.stem)
            if m:
                zs.add(f"z{int(m[0])}")
    return sorted(zs)


def _frame_layout(fp: float, pre_s: float, bl_s: float, st_s: float) -> dict:
    pre_f  = round(pre_s / fp)
    base_f = round(bl_s  / fp)
    stim_f = round(st_s  / fp)
    ses_f  = pre_f + base_f + stim_f
    return dict(pre_f=pre_f, base_f=base_f, stim_f=stim_f, ses_f=ses_f)


def _read_provenance(project_dir: str) -> dict:
    """Load provenance.yaml from a project folder. Returns empty dict if not found."""
    import yaml
    from collections import defaultdict
    p = Path(project_dir) / "provenance.yaml"
    if p.exists():
        with open(p, "r") as f:
            return defaultdict(lambda: None, yaml.safe_load(f) or {})
    return defaultdict(lambda: None)


def _stage_status(prov: dict) -> dict[str, bool]:
    """Return which pipeline stages are already complete based on provenance."""
    mc_done   = bool(prov.get("rigid_motion_correction"))
    cnmf_done = bool(prov.get("source_extraction"))
    return dict(mc=mc_done, cnmf=cnmf_done)


# ── animal row widget ─────────────────────────────────────────────────────────

class AnimalRow(ctk.CTkFrame):
    """One row per animal; session entries are built dynamically from n_stims."""

    def __init__(self, parent, index: int, on_remove, n_stims: int = 2, **kw):
        super().__init__(parent, **kw)
        self.index = index
        self._sess_vars: list[ctk.StringVar] = []
        self._sess_frames: list[ctk.CTkFrame] = []

        ctk.CTkLabel(self, text=f"Animal {index + 1}",
                     width=78, font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=10, sticky="n", pady=6)

        self._rows_frame = ctk.CTkFrame(self, fg_color="transparent")
        self._rows_frame.grid(row=0, column=1, sticky="nsew", padx=2)

        ctk.CTkButton(self, text="✕", width=30, height=30,
                      fg_color="#c0392b", hover_color="#922b21",
                      command=on_remove).grid(row=0, column=2, padx=8, sticky="n")

        for _ in range(n_stims):
            self._add_row()

    # ── internal row management ────────────────────────────────────────────

    def _add_row(self):
        idx = len(self._sess_vars)
        var = ctk.StringVar()
        self._sess_vars.append(var)

        f = ctk.CTkFrame(self._rows_frame, fg_color="transparent")
        f.pack(fill="x", pady=1)
        self._sess_frames.append(f)

        ctk.CTkLabel(f, text=f"Session {idx + 1}:", width=74).pack(side="left")
        ctk.CTkEntry(f, textvariable=var, width=270,
                     placeholder_text=f"Stimulus {idx + 1} folder").pack(side="left", padx=4)
        ctk.CTkButton(f, text="Browse", width=72,
                      command=lambda v=var: self._browse(v)).pack(side="left")

    def _remove_last_row(self):
        if len(self._sess_vars) <= 1:
            return
        self._sess_vars.pop()
        self._sess_frames.pop().destroy()

    def _browse(self, var: ctk.StringVar):
        path = filedialog.askdirectory(title="Select session folder")
        if path:
            var.set(path)

    # ── public API ────────────────────────────────────────────────────────

    def set_n_stims(self, n: int):
        """Add or remove session rows to match n."""
        while len(self._sess_vars) < n:
            self._add_row()
        while len(self._sess_vars) > n:
            self._remove_last_row()

    def get_paths(self) -> list[str]:
        return [v.get().strip() for v in self._sess_vars]

    def set_paths(self, paths: list[str]):
        self.set_n_stims(len(paths))
        for var, path in zip(self._sess_vars, paths):
            var.set(path)


# ── main window ───────────────────────────────────────────────────────────────

class PipelineGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Luceo")
        self.geometry("860x760")
        self.minsize(720, 600)
        self.animal_rows: list[AnimalRow] = []
        self._build_ui()

    # ── build ──────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.tabs = ctk.CTkTabview(self)
        self.tabs.pack(fill="both", expand=True, padx=12, pady=12)
        for name in ("Animals & Data", "Recording", "Timing", "Run"):
            self.tabs.add(name)
        self._build_animals_tab()
        self._build_recording_tab()
        self._build_timing_tab()
        self._build_run_tab()

    # ── tab 1: animals & data ──────────────────────────────────────────────

    def _build_animals_tab(self):
        tab = self.tabs.tab("Animals & Data")

        # load project banner
        banner = ctk.CTkFrame(tab)
        banner.pack(fill="x", padx=14, pady=(14, 6))
        ctk.CTkLabel(banner,
                     text="Have a project folder from a previous run?",
                     anchor="w").pack(side="left", padx=12, pady=8)
        ctk.CTkButton(banner, text="Load existing project",
                      width=170, command=self._load_project).pack(
            side="right", padx=12, pady=8)

        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(4, 6))

        ctk.CTkLabel(top, text="Subject ID:", width=130, anchor="w").grid(
            row=0, column=0, pady=5, sticky="w")
        self.subject_var = ctk.StringVar()
        self.subject_var.trace_add("write", lambda *_: self._check_provenance())
        ctk.CTkEntry(top, textvariable=self.subject_var, width=180,
                     placeholder_text="e.g. ZH511").grid(
            row=0, column=1, padx=8, sticky="w")

        ctk.CTkLabel(top, text="Project Folder:", width=130, anchor="w").grid(
            row=1, column=0, pady=5, sticky="w")
        self.output_var = ctk.StringVar()
        self.output_var.trace_add("write", lambda *_: self._check_provenance())
        ctk.CTkEntry(top, textvariable=self.output_var, width=320,
                     placeholder_text="All stage outputs will be saved here").grid(
            row=1, column=1, padx=8, sticky="w")
        ctk.CTkButton(top, text="Browse", width=80,
                      command=self._browse_output).grid(row=1, column=2, padx=4)

        ctk.CTkLabel(top, text="Analysis output\n(optional):", width=130,
                     anchor="w", text_color="gray").grid(
            row=2, column=0, pady=5, sticky="w")
        self.analysis_out_var = ctk.StringVar()
        ctk.CTkEntry(top, textvariable=self.analysis_out_var, width=320,
                     placeholder_text="Default: <project folder>/analysis/ override here if needed").grid(
            row=2, column=1, padx=8, sticky="w")
        ctk.CTkButton(top, text="Browse", width=80,
                      command=self._browse_analysis_out).grid(row=2, column=2, padx=4)

        # ── number of stimuli (global, applies to all animals) ─────────────
        stims_row = ctk.CTkFrame(tab, fg_color="transparent")
        stims_row.pack(fill="x", padx=14, pady=(4, 2))
        ctk.CTkLabel(stims_row, text="Number of stimuli:", width=130,
                     anchor="w").pack(side="left")
        self.n_stims_var = ctk.StringVar(value="2")
        ctk.CTkSegmentedButton(
            stims_row, values=["1", "2", "3", "4"],
            variable=self.n_stims_var,
            command=self._on_n_stims_change,
            width=180,
        ).pack(side="left", padx=8)
        ctk.CTkLabel(stims_row,
                     text="one session folder per stimulus condition",
                     text_color="gray").pack(side="left", padx=4)

        # provenance status indicator
        self.prov_label = ctk.CTkLabel(tab, text="", text_color="gray")
        self.prov_label.pack(anchor="w", padx=18, pady=(0, 4))

        mode_row = ctk.CTkFrame(tab, fg_color="transparent")
        mode_row.pack(fill="x", padx=14, pady=4)
        ctk.CTkLabel(mode_row, text="Mode:").pack(side="left", padx=(0, 12))
        self.mode_var = ctk.StringVar(value="single")
        ctk.CTkRadioButton(mode_row, text="Single animal",
                           variable=self.mode_var, value="single",
                           command=self._on_mode_change).pack(side="left", padx=8)
        ctk.CTkRadioButton(mode_row, text="Multi animal",
                           variable=self.mode_var, value="multi",
                           command=self._on_mode_change).pack(side="left", padx=8)

        ctk.CTkLabel(tab,
                     text="Each session folder should contain the raw per-frame TIFFs from the microscope.",
                     text_color="gray").pack(anchor="w", padx=16, pady=(8, 0))

        self.animals_scroll = ctk.CTkScrollableFrame(tab, height=220)
        self.animals_scroll.pack(fill="both", expand=True, padx=12, pady=6)

        self.add_btn = ctk.CTkButton(tab, text="＋  Add Animal",
                                      width=150, command=self._add_animal)
        self.add_btn.pack(pady=4)

        self._add_animal()
        self._on_mode_change()

    def _on_n_stims_change(self, value=None):
        """Propagate stimulus count change to every animal row."""
        n = int(self.n_stims_var.get())
        for row in self.animal_rows:
            row.set_n_stims(n)

    def _on_mode_change(self):
        if self.mode_var.get() == "single":
            self.add_btn.pack_forget()
            while len(self.animal_rows) > 1:
                self.animal_rows.pop().destroy()
        else:
            self.add_btn.pack(pady=4)

    def _add_animal(self):
        n = int(self.n_stims_var.get())
        idx = len(self.animal_rows)
        row = AnimalRow(self.animals_scroll, idx, n_stims=n,
                        on_remove=lambda r=None: self._remove_animal(row))
        row.pack(fill="x", padx=4, pady=5)
        self.animal_rows.append(row)

    def _remove_animal(self, row: AnimalRow):
        if len(self.animal_rows) <= 1:
            return
        self.animal_rows.remove(row)
        row.destroy()
        for i, r in enumerate(self.animal_rows):
            r.index = i

    def _browse_output(self):
        path = filedialog.askdirectory(title="Project Folder")
        if path:
            self.output_var.set(path)

    def _browse_analysis_out(self):
        path = filedialog.askdirectory(title="Select analysis results folder  (optional)")
        if path:
            self.analysis_out_var.set(path)

    # ── project loading & provenance detection ─────────────────────────────

    def _load_project(self):
        """Let the user pick an existing project folder and restore state from provenance."""
        folder = filedialog.askdirectory(title="Select project folder  (e.g. ZH511/)")
        if not folder:
            return
        folder = Path(folder)
        prov = _read_provenance(str(folder))

        self.output_var.set(str(folder.parent))
        self.subject_var.set(folder.name)

        load_args = prov.get("load_data") or {}
        load_args = load_args.get("args") or {} if isinstance(load_args, dict) else {}
        multi_path = load_args.get("multi_path", [])
        ch_dict    = load_args.get("ch_dict", {})

        if multi_path:
            n = min(max(len(multi_path), 1), 4)
            self.n_stims_var.set(str(n))
            self._on_n_stims_change()
            self.animal_rows[0].set_paths([str(p) for p in multi_path])

        if ch_dict:
            self.mc_ch_var.set(ch_dict.get("mc_ch", "ch1"))
            self.func_ch_var.set(ch_dict.get("func_ch", "ch2"))

        mc_prov = prov.get("rigid_motion_correction") or {}
        if isinstance(mc_prov, dict) and mc_prov:
            self.z_planes_var.set(",".join(mc_prov.keys()))

        ap = prov.get("analysis_params") or {}
        if ap:
            if "z_planes"      in ap: self.z_planes_var.set(",".join(ap["z_planes"]))
            if "frame_period"  in ap: self.frame_period_var.set(str(ap["frame_period"]))
            if "pre_discard_s" in ap: self.pre_discard_var.set(str(ap["pre_discard_s"]))
            if "baseline_s"    in ap: self.baseline_var.set(str(ap["baseline_s"]))
            if "stim_s"        in ap: self.stim_var.set(str(ap["stim_s"]))
            if "threshold"     in ap: self.threshold_var.set(str(ap["threshold"]))
            if "cp_diameter" in ap:
                d = ap["cp_diameter"]
                if d == 0:
                    self.cp_diameter_auto_var.set(True)
                    self._on_diameter_auto_toggle()
                else:
                    self.cp_diameter_auto_var.set(False)
                    self.cp_diameter_var.set(str(d))
                    self._on_diameter_auto_toggle()
            if "cp_flow_threshold" in ap: self.cp_flow_var.set(str(ap["cp_flow_threshold"]))
            if "cp_cellprob"       in ap: self.cp_cellprob_var.set(str(ap["cp_cellprob"]))

        self._check_provenance()
        self.tabs.set("Run")

    def _sync_resume_curation(self):
        """Resume is a sub-option of CNMF: disabled and unticked while CNMF is off."""
        if not hasattr(self, "resume_curation"):
            return
        if self.do_cnmf.get():
            self.resume_curation.configure(state="normal")
        else:
            self.resume_curation.deselect()
            self.resume_curation.configure(state="disabled")

    def _check_provenance(self):
        if not hasattr(self, "do_mc"):
            return

        output  = self.output_var.get().strip()
        subject = self.subject_var.get().strip()
        if not output or not subject:
            self.prov_label.configure(text="")
            return

        project_dir = str(Path(output) / subject)
        prov   = _read_provenance(project_dir)
        status = _stage_status(prov)

        parts = []
        if status["mc"]:
            self.do_mc.deselect()
            parts.append("motion correction ✓")
        else:
            self.do_mc.select()

        if status["cnmf"]:
            self.do_cnmf.deselect()
            parts.append("CNMF ✓")
        else:
            self.do_cnmf.select()
        # select()/deselect() do not fire the checkbox command
        self._sync_resume_curation()

        results_dir = Path(project_dir) / "analysis"
        if (results_dir / "resp1.npy").exists():
            parts.append("analysis results ✓")

        if parts:
            self.prov_label.configure(
                text=f"  Existing project found - {',  '.join(parts)}",
                text_color="#5cb85c")
        else:
            self.prov_label.configure(
                text="  No existing project found at this location, will start fresh.",
                text_color="gray")

    # ── tab 2: recording ───────────────────────────────────────────────────

    def _build_recording_tab(self):
        tab = self.tabs.tab("Recording")
        ctk.CTkLabel(tab, text=" ").pack()

        def field(label, var, tip=""):
            f = ctk.CTkFrame(tab, fg_color="transparent")
            f.pack(fill="x", padx=22, pady=9)
            ctk.CTkLabel(f, text=label, width=220, anchor="w").pack(side="left")
            ctk.CTkEntry(f, textvariable=var, width=110).pack(side="left", padx=6)
            if tip:
                ctk.CTkLabel(f, text=tip, text_color="gray").pack(side="left")

        self.frame_period_var = ctk.StringVar(value="0.585")
        field("Frame period  (s / frame):", self.frame_period_var,
              "get the exact value from your .xml acquisition file")

        self.fr_label = ctk.CTkLabel(tab, text="", text_color="#5cb85c")
        self.fr_label.pack(anchor="w", padx=32)
        self.frame_period_var.trace_add("write", self._refresh)

        ctk.CTkLabel(tab, text="─" * 62, text_color="gray").pack(pady=8)

        self.z_planes_var = ctk.StringVar(value="z3")

        zrow = ctk.CTkFrame(tab, fg_color="transparent")
        zrow.pack(fill="x", padx=22, pady=9)
        ctk.CTkLabel(zrow, text="Z-plane(s) to analyse:", width=220, anchor="w").pack(side="left")
        ctk.CTkEntry(zrow, textvariable=self.z_planes_var, width=110).pack(side="left", padx=6)
        ctk.CTkLabel(zrow, text="comma-separated", text_color="gray").pack(side="left")
        ctk.CTkButton(zrow, text="Auto-detect", width=100,
                      command=self._autodetect_zplanes).pack(side="left", padx=10)

        self.mc_ch_var   = ctk.StringVar(value="ch1")
        self.func_ch_var = ctk.StringVar(value="ch2")
        field("Motion-correction channel:", self.mc_ch_var,
              "structural / anatomical  —  Prairie View Ch1 = red PMT (tdTomato)")
        field("Functional channel:",        self.func_ch_var,
              "calcium indicator signal  —  Prairie View Ch2 = green PMT (GCaMP)")

        ctk.CTkLabel(tab, text="─" * 62, text_color="gray").pack(pady=8)
        ctk.CTkLabel(tab, text="Cellpose ROI detection",
                     font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=22)
        ctk.CTkLabel(tab,
                     text="These settings affect which pixels are identified as neurons before CNMF.",
                     text_color="gray").pack(anchor="w", padx=22, pady=(2, 6))

        self.cp_diameter_var      = ctk.StringVar(value="15")
        self.cp_diameter_auto_var = ctk.BooleanVar(value=False)
        self.cp_flow_var          = ctk.StringVar(value="2.0")
        self.cp_cellprob_var      = ctk.StringVar(value="-1.0")

        # Cell diameter row,  manual entry + Auto toggle
        drow = ctk.CTkFrame(tab, fg_color="transparent")
        drow.pack(fill="x", padx=22, pady=9)
        ctk.CTkLabel(drow, text="Cell diameter  (px):", width=220, anchor="w").pack(side="left")
        self._cp_diameter_entry = ctk.CTkEntry(
            drow, textvariable=self.cp_diameter_var, width=110)
        self._cp_diameter_entry.pack(side="left", padx=6)
        ctk.CTkCheckBox(
            drow, text="Auto",
            variable=self.cp_diameter_auto_var,
            command=self._on_diameter_auto_toggle,
        ).pack(side="left", padx=8)
        ctk.CTkLabel(drow,
                     text="Auto = let Cellpose estimate  |  or enter a fixed px value",
                     text_color="gray").pack(side="left")
        field("Flow threshold:",               self.cp_flow_var,
              "shape strictness - lower = stricter  (Cellpose default: 0.4;  current: 2.0 = permissive)")
        field("Cell probability threshold:",   self.cp_cellprob_var,
              "detection sensitivity - lower = more cells  (Cellpose default: 0.0;  current: −1.0)")

        self._refresh()

    def _on_diameter_auto_toggle(self):
        auto = self.cp_diameter_auto_var.get()
        if auto:
            self._cp_diameter_entry.configure(
                state="disabled",
                fg_color="gray70",
                text_color="gray50",
            )
        else:
            self._cp_diameter_entry.configure(
                state="normal",
                fg_color=["#F9F9FA", "#343638"],
                text_color=["gray10", "#DCE4EE"],
            )

    def _autodetect_zplanes(self):
        if not self.animal_rows:
            messagebox.showinfo("Auto-detect", "Add an animal and set Session 1 first.")
            return
        paths = self.animal_rows[0].get_paths()
        if not paths or not paths[0]:
            messagebox.showinfo("Auto-detect", "Set the Session 1 path for Animal 1 first.")
            return
        zs = _detect_zplanes(paths[0])
        if zs:
            self.z_planes_var.set(",".join(zs))
            messagebox.showinfo("Auto-detect", f"Found: {', '.join(zs)}")
        else:
            messagebox.showwarning("Auto-detect",
                                   "No z-planes detected - check the folder contains Bruker TIFFs.")

    # ── tab 3: timing ──────────────────────────────────────────────────────

    def _build_timing_tab(self):
        tab = self.tabs.tab("Timing")
        ctk.CTkLabel(tab, text=" ").pack()
        ctk.CTkLabel(tab,
                     text="Enter all times in seconds.  Frame counts update automatically.",
                     text_color="gray").pack(anchor="w", padx=22)

        def tfield(label, var, default, tip=""):
            f = ctk.CTkFrame(tab, fg_color="transparent")
            f.pack(fill="x", padx=22, pady=9)
            ctk.CTkLabel(f, text=label, width=210, anchor="w").pack(side="left")
            ctk.CTkEntry(f, textvariable=var, width=80).pack(side="left", padx=6)
            ctk.CTkLabel(f, text="s", width=14).pack(side="left")
            if tip:
                ctk.CTkLabel(f, text=f"  {tip}", text_color="gray").pack(side="left")
            var.trace_add("write", self._refresh)

        self.pre_discard_var = ctk.StringVar(value="30")
        tfield("Pre-baseline discard:", self.pre_discard_var, "30",
               "dropped from the start of each session  (default 30 s)")

        self.baseline_var = ctk.StringVar(value="30")
        tfield("Baseline window:", self.baseline_var, "30",
               "used to compute z-score mean and SD  (default 30 s)")

        self.stim_var = ctk.StringVar(value="180")
        tfield("Stimulus duration:", self.stim_var, "180",
               "each stimulus delivery period  (set to match your protocol)")

        ctk.CTkLabel(tab, text="─" * 62, text_color="gray").pack(pady=6)

        self.layout_label = ctk.CTkLabel(tab, text="", justify="left",
                                          text_color="#5cb85c",
                                          font=ctk.CTkFont(family="Courier"))
        self.layout_label.pack(anchor="w", padx=30, pady=4)

        ctk.CTkLabel(tab, text="─" * 62, text_color="gray").pack(pady=6)

        thresh_row = ctk.CTkFrame(tab, fg_color="transparent")
        thresh_row.pack(fill="x", padx=22, pady=9)
        ctk.CTkLabel(thresh_row, text="Responder threshold (z-score):",
                     width=210, anchor="w").pack(side="left")
        self.threshold_var = ctk.StringVar(value="1.64")
        ctk.CTkEntry(thresh_row, textvariable=self.threshold_var,
                     width=80).pack(side="left", padx=6)
        ctk.CTkLabel(thresh_row,
                     text="  1.64 ≈ one-tailed p < 0.05  (standard in the calcium imaging literature)",
                     text_color="gray").pack(side="left")

        self._refresh()

    def _refresh(self, *_):
        try:
            fp  = float(self.frame_period_var.get())
            pre = float(self.pre_discard_var.get())
            bl  = float(self.baseline_var.get())
            st  = float(self.stim_var.get())
            assert fp > 0 and pre >= 0 and bl > 0 and st > 0
        except Exception:
            if hasattr(self, "fr_label"):
                self.fr_label.configure(text="  → invalid")
            if hasattr(self, "layout_label"):
                self.layout_label.configure(text="  (invalid input)")
            return

        if hasattr(self, "fr_label"):
            self.fr_label.configure(text=f"  → {1 / fp:.4f} Hz")

        if hasattr(self, "layout_label"):
            d = _frame_layout(fp, pre, bl, st)
            pf, bf, sf, se = d["pre_f"], d["base_f"], d["stim_f"], d["ses_f"]
            self.layout_label.configure(text=(
                f"  Each session is analysed independently from local frame 0:\n\n"
                f"  discard   frames  0 – {pf-1}        ({pf} frames,  ~{pre:.0f} s)\n"
                f"  baseline  frames  {pf} – {pf+bf-1}      ({bf} frames,  ~{bl:.0f} s)\n"
                f"  stimulus  frames  {pf+bf} – {pf+bf+sf-1}   ({sf} frames,  ~{st:.0f} s)\n\n"
                f"  Expected per session: {se} frames  ({pf} + {bf} + {sf})\n"
                f"  Actual count read from recording - mismatch shown in run log."
            ))

    # ── tab 4: run ─────────────────────────────────────────────────────────

    def _build_run_tab(self):
        tab = self.tabs.tab("Run")
        ctk.CTkLabel(tab, text=" ").pack()

        stages = ctk.CTkFrame(tab)
        stages.pack(fill="x", padx=20, pady=6)
        ctk.CTkLabel(stages, text="Pipeline stages:",
                     font=ctk.CTkFont(weight="bold")).pack(
            anchor="w", padx=12, pady=(8, 4))

        self.do_mc = ctk.CTkCheckBox(
            stages,
            text="Motion correction, slow, reads raw TIFFs, skip if already done")
        self.do_mc.pack(anchor="w", padx=16, pady=4)
        self.do_mc.select()

        self.do_cnmf = ctk.CTkCheckBox(
            stages,
            text="Source extraction / CNMF,  slow, opens interactive ROI editor, skip if already done",
            command=self._sync_resume_curation)
        self.do_cnmf.pack(anchor="w", padx=16, pady=4)
        self.do_cnmf.select()

        self.resume_curation = ctk.CTkCheckBox(
            stages,
            text="Resume previous ROI curation,  reopen the editor with the neurons you "
                 "added / removed last time instead of a fresh Cellpose detection")
        self.resume_curation.pack(anchor="w", padx=40, pady=(0, 4))
        self._sync_resume_curation()

        self.do_subregion_setup = ctk.CTkCheckBox(
            stages,
            text="Sub-region setup,  (re)draw the AP outline (rest = NTS) without re-running CNMF")
        self.do_subregion_setup.pack(anchor="w", padx=16, pady=4)

        self.do_neuron_curation = ctk.CTkCheckBox(
            stages,
            text="Neuron curation,  re-open neuron viewer on saved CNMF, accept / reject without re-running")
        self.do_neuron_curation.pack(anchor="w", padx=16, pady=4)

        self.do_multiplane_review = ctk.CTkCheckBox(
            stages,
            text="Multi-plane duplicate review, 3-D map across z-planes, detect and resolve duplicate neurons  (2+ planes)")
        self.do_multiplane_review.pack(anchor="w", padx=16, pady=4)

        self.do_analysis = ctk.CTkCheckBox(
            stages,
            text="Stimulus response analysis,  fast, re-run this after changing timing or threshold")
        self.do_analysis.pack(anchor="w", padx=16, pady=4)
        self.do_analysis.select()

        self.do_subregion = ctk.CTkCheckBox(
            stages,
            text="Sub-region analysis (AP vs NTS),  optional, requires the AP outlined in the ROI editor")
        self.do_subregion.pack(anchor="w", padx=16, pady=4)

        self.do_population = ctk.CTkCheckBox(
            stages,
            text="Co-activity clustering  (work in progress),  Sherringtonian view: group "
                 "individual neurons into functional types (K-Means / GMM / Hierarchical)")
        self.do_population.pack(anchor="w", padx=16, pady=4)

        self.do_manifold = ctk.CTkCheckBox(
            stages,
            text="Neural manifold analysis  (work in progress),  Hopfieldian view: population "
                 "state space,  coding dimensions, states / attractors, topology  (Ebitz & Hayden 2021)")
        self.do_manifold.pack(anchor="w", padx=16, pady=4)

        def _sync_analysis_dependent():
            state = "normal" if self.do_analysis.get() else "disabled"
            for cb in (self.do_population, self.do_manifold):
                cb.configure(state=state)
                if state == "disabled":
                    cb.deselect()

        self.do_analysis.configure(command=_sync_analysis_dependent)

        ctk.CTkLabel(tab,
                     text="Tip: to iterate on timing parameters, uncheck the first two and only re-run analysis.\n"
                          "Results are saved to  <project folder>/analysis/  (or the custom analysis folder set in tab 1).",
                     text_color="gray", wraplength=740).pack(anchor="w", padx=22, pady=4)

        self.run_btn = ctk.CTkButton(tab, text="▶   Run",
                                      height=44,
                                      font=ctk.CTkFont(size=15, weight="bold"),
                                      command=self._start_run)
        self.run_btn.pack(pady=10)

        ctk.CTkLabel(tab, text="Log:").pack(anchor="w", padx=20)
        self.log_box = ctk.CTkTextbox(tab, state="disabled",
                                       font=ctk.CTkFont(family="Courier", size=12))
        self.log_box.pack(fill="both", expand=True, padx=20, pady=(2, 14))

    # ── parameter collection ───────────────────────────────────────────────

    def _collect(self) -> dict:
        errs = []

        subject = self.subject_var.get().strip()
        if not subject:
            errs.append("Subject ID is required.")

        output = self.output_var.get().strip()
        if not output:
            errs.append("Output / project folder is required.")

        n_stims = int(self.n_stims_var.get())
        animals = []
        for r in self.animal_rows:
            paths = r.get_paths()
            if not all(paths):
                errs.append(
                    f"All {n_stims} session folder(s) are required for Animal {r.index + 1}.")
            animals.append(paths)

        def fval(var, name):
            try:
                v = float(var.get())
                assert v > 0
                return v
            except Exception:
                errs.append(f"{name} must be a positive number.")
                return 1.0

        fp        = fval(self.frame_period_var, "Frame period")
        pre_s     = fval(self.pre_discard_var,  "Pre-discard time")
        base_s    = fval(self.baseline_var,      "Baseline time")
        stim_s    = fval(self.stim_var,          "Stimulus duration")
        threshold = fval(self.threshold_var,     "Responder threshold")

        z_planes = [z.strip() for z in self.z_planes_var.get().split(",") if z.strip()]
        if not z_planes:
            errs.append("At least one z-plane is required.")

        def ival(var, name, lo=None, hi=None):
            try:
                v = int(float(var.get()))
                if lo is not None: assert v >= lo
                if hi is not None: assert v <= hi
                return v
            except Exception:
                errs.append(f"{name} must be a positive integer.")
                return 1

        cp_diameter  = 0 if self.cp_diameter_auto_var.get() else ival(self.cp_diameter_var, "Cell diameter", lo=1)
        try:
            cp_flow      = float(self.cp_flow_var.get())
            cp_cellprob  = float(self.cp_cellprob_var.get())
        except ValueError:
            errs.append("Flow threshold and cell probability threshold must be numbers.")
            cp_flow, cp_cellprob = 2.0, -1.0

        return dict(
            subject=subject, output=output, animals=animals,
            analysis_out=self.analysis_out_var.get().strip(),
            n_stims=n_stims,
            frame_period=fp, pre_discard_s=pre_s, baseline_s=base_s,
            stim_s=stim_s, threshold=threshold, z_planes=z_planes,
            ch_dict={"mc_ch": self.mc_ch_var.get().strip(),
                     "func_ch": self.func_ch_var.get().strip()},
            cellpose=dict(diameter=cp_diameter,
                          flow_threshold=cp_flow,
                          cellprob_threshold=cp_cellprob),
            errors=errs,
        )

    # ── run ────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", msg + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")
        self.update_idletasks()

    def _start_run(self):
        p = self._collect()
        if p["errors"]:
            messagebox.showerror("Input error", "\n".join(p["errors"]))
            return
        self.log_box.configure(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.configure(state="disabled")
        self.run_btn.configure(state="disabled", text="Running…")
        threading.Thread(target=self._run_pipeline, args=(p,), daemon=True).start()

    # ── reference-channel loading for the sub-region view ─────────────────────

    @staticmethod
    def _channel_list(provenance, z) -> list[str]:
        """Every channel name present for this z-plane, motion-corrected or not.

        combine_tiffs writes one combined TIFF per channel found in the raw
        acquisition, so channels outside ch_dict (e.g. a tdTomato channel that
        is neither the functional nor the motion-correction channel) are still
        on disk and can be shown as an anatomical reference.
        """
        chans: list[str] = []
        sessions = ((provenance.get('load_data') or {}).get('filenames') or {})
        for sess in sessions.values():
            for ch in ((sess or {}).get(z) or {}):
                if ch not in chans:
                    chans.append(ch)
        rig = (((provenance.get('rigid_motion_correction') or {}).get(z) or {})
               .get('filenames') or {})
        for ch in rig:
            if ch not in chans:
                chans.append(ch)
        return sorted(chans)

    def _gray_to_bkg(self, img, target_hw, tag=""):
        """Resize a 2-D projection to target_hw and replicate to three planes.

        Deliberately keeps **float32** precision.  The old version divided by
        img.max() and cast to uint8 here, before the editor's contrast sliders
        ever saw the data.  On a dim channel with one bright outlier that
        collapses the whole low end onto a handful of integer levels — most of
        the frame lands on exactly 0 — which both destroys faint structure and
        makes the Dark clip slider a no-op, since every low percentile then
        returns the same value.  The editor's _stretch() does its own
        percentile normalisation, so quantising early buys nothing.
        """
        import cv2 as cv

        img    = np.asarray(img, dtype=np.float32)
        th, tw = target_hw
        if img.shape != (th, tw):
            self._log(f"  {tag} dims {img.shape} differ from functional "
                      f"{(th, tw)},  resizing to match.")
            img = cv.resize(img, (tw, th), interpolation=cv.INTER_LINEAR)

        pct = np.percentile(img, [0, 1, 25, 50, 75, 99, 100])
        self._log(f"  {tag} intensity 0/1/25/50/75/99/100 pct: "
                  + "  ".join(f"{v:.1f}" for v in pct))
        return np.repeat(img[:, :, None], 3, axis=2)

    @staticmethod
    def _reduce_memmap(Yr, dims, stat, chunk=8192):
        """Per-pixel temporal statistic over a CaImAn memmap.

        `Yr` is (n_pixels, T), so each pixel's time course is one row and any
        temporal statistic is a row-wise reduction.  Chunking over pixels keeps
        peak memory at chunk x T floats instead of the whole movie.
        """
        n   = Yr.shape[0]
        out = np.empty(n, dtype=np.float32)
        for i in range(0, n, chunk):
            blk = np.asarray(Yr[i:i + chunk], dtype=np.float32)
            out[i:i + chunk] = (blk.mean(axis=1) if stat == "mean"
                                else np.percentile(blk, 99, axis=1))
        return out.reshape(dims, order='F')

    @staticmethod
    def _reduce_stack(stack, stat):
        """Same statistic over an in-memory (T, h, w) stack."""
        if stack.ndim == 2:
            return stack
        return (stack.mean(axis=0) if stat == "mean"
                else np.percentile(stack, 99, axis=0))

    def _channel_background(self, provenance, z, ch, target_hw, stat="mean"):
        """Temporal projection of `ch` as ((h, w, 3) uint8, note).

        `stat` is "mean" (best SNR on a static label such as tdTomato — averaging
        T frames cuts shot noise by sqrt(T)) or "p99" (the same 99th-percentile
        statistic the functional summary image uses, which sees past motion-
        correction border fill and vessel shadows).  Display only: nothing here
        feeds the analysis, the seeding image is built separately in
        pipeline._identify_rois.

        Prefers the motion-corrected memmap.  Channels that were never motion
        corrected fall back to their raw combined TIFF.  Returns (None, "") when
        the channel cannot be loaded.
        """
        import caiman as cm
        from tifffile import imread as _imread

        root = Path(provenance.get('output_dir') or '.')

        rig = (((provenance.get('rigid_motion_correction') or {}).get(z) or {})
               .get('filenames') or {})
        path = rig.get(ch)
        if path:
            path = Path(path)
            if not path.is_absolute():
                path = root / z / path.name
            if path.exists():
                try:
                    Yr, dims, _T = cm.load_memmap(str(path))
                    img = self._reduce_memmap(Yr, dims, stat)
                except Exception:
                    # not a CaImAn memmap — fall back to loading the movie whole
                    img = self._reduce_stack(
                        np.asarray(cm.load(str(path)), dtype=np.float32), stat)
                return (self._gray_to_bkg(img, target_hw, tag=f"{ch} {stat}"),
                        "motion-corrected")

        sessions = ((provenance.get('load_data') or {}).get('filenames') or {})
        for sess in sessions.values():
            raw = ((sess or {}).get(z) or {}).get(ch)
            if not raw:
                continue
            raw = Path(raw)
            if not raw.is_absolute():
                raw = root / raw.name
            if not raw.exists():
                continue
            stack = np.asarray(_imread(str(raw))).squeeze().astype(np.float32)
            return (self._gray_to_bkg(self._reduce_stack(stack, stat),
                                      target_hw, tag=f"{ch} {stat}"),
                    "raw, not motion-corrected")

        return None, ""

    def _roi_editor_for_pipeline(self, output_dir, mc_corr_file, z, roi_masks, roi_img_bkg, roi_img_mask,
                                 subregion_only=False):
        """Called from the worker thread. Shows ROIEditorWindow on the main thread and blocks until Finish.

        subregion_only, the ROIs are an already-curated set being reopened to
        define sub-regions, so the saved curation record is extended rather than
        replaced (its detected / added / removed history is kept).
        """
        import yaml
        from analysis.roi_curation import load_curation, save_curation

        # a plane with no detected ROIs can arrive as a 1-D empty array (older
        # saved concat_roi-masks.npy files); give it the (pixels, 0) shape
        roi_masks = np.asarray(roi_masks)
        if roi_masks.ndim != 2:
            roi_masks = np.zeros((roi_img_bkg.shape[0] * roi_img_bkg.shape[1], 0), dtype=bool)
        if not Path(mc_corr_file).is_absolute():
            mc_corr_file = str(Path(output_dir) / Path(mc_corr_file).name)

        ds_path = Path(output_dir) / 'display_settings.yaml'
        if not hasattr(self, '_display_settings'):
            if ds_path.exists():
                with open(ds_path, 'r') as f:
                    self._display_settings = yaml.safe_load(f) or {}
            else:
                self._display_settings = {}

        # ── sub-region reference channel ──────────────────────────────────────
        # Defaults to the motion-correction channel, but every channel found in
        # the acquisition is offered so the user can pick whichever one carries
        # the anatomical label (tdTomato) for AP / NTS orientation.
        target_hw  = roi_img_bkg.shape[:2]
        provenance = getattr(self, '_provenance', None)
        channels, ch_default, ch_luts = [], None, {}
        if provenance is not None:
            try:
                ch_dict = ((provenance.get('load_data') or {}).get('args') or {}).get('ch_dict', {})
                ch_default = ch_dict.get('mc_ch')
                channels   = self._channel_list(provenance, z)
                # Prairie View puts the structural label on the red PMT and the
                # calcium indicator on the green one; mirror that in the viewer.
                if ch_dict.get('mc_ch'):
                    ch_luts[ch_dict['mc_ch']] = "Red"
                if ch_dict.get('func_ch'):
                    ch_luts[ch_dict['func_ch']] = "Green"
                self._log(f"  Sub-region reference channels available: {', '.join(channels) or 'none'}")
            except Exception as _e:
                self._log(f"  Warning: could not enumerate channels ({_e})")

        def _load_channel(ch, stat="mean"):
            try:
                return self._channel_background(provenance, z, ch, target_hw, stat)
            except Exception as _e:
                self._log(f"  Warning: could not load channel {ch} ({stat}) "
                          f"for sub-region view ({_e})")
                return None, ""

        mc_img_bkg, mc_note = None, ""
        if provenance is not None and ch_default:
            mc_img_bkg, mc_note = _load_channel(ch_default, "mean")
        if mc_img_bkg is None:
            # Fall back to the motion-corrected file the pipeline handed us.  The
            # channel picker is dropped here: if provenance could not serve the
            # default channel it cannot be trusted to serve the others either.
            channels = []
            try:
                import caiman as cm
                mc_img_bkg = self._gray_to_bkg(
                    np.mean(cm.load(mc_corr_file), axis=0), target_hw, tag="MC channel")
                mc_note = "motion-corrected"
                ch_default = ch_default or "MC"
            except Exception as _e:
                self._log(f"  Warning: could not load MC channel for sub-region view ({_e})")

        # ── previous curation of this plane ───────────────────────────────────
        # The AP outline is restored either way (the anatomy has not moved).
        # Stable ids are only reused when these ROIs are that curation's output.
        prev = (None, None)
        try:
            prev = load_curation(output_dir, z)
        except Exception as _e:
            self._log(f"  Warning: could not read previous ROI curation for {z} ({_e})")
        # a resumed plane uses the record its masks were rebuilt from, whose id
        # list matches those masks column for column
        prev_rec = getattr(self, '_resumed_planes', {}).get(z) or prev[0] or {}
        resumed = z in getattr(self, '_resumed_planes', {})
        reuse_ids = ((subregion_only or resumed)
                     and len(prev_rec.get("final_ids") or []) == roi_masks.shape[1])
        ap_polygon = prev_rec.get("ap_polygon")
        if ap_polygon:
            self._log(f"  {z}: AP outline restored from the previous curation.")

        result_holder = [None]
        done = threading.Event()

        def _show():
            def _on_finish(masks, bkg, mask_img, settings, sreg_masks=None, record=None):
                result_holder[0] = (masks, bkg, mask_img, settings, sreg_masks, record)
                done.set()
            try:
                ROIEditorWindow(self, z, roi_img_bkg, roi_img_mask, roi_masks, mc_corr_file,
                                _on_finish, display_settings=self._display_settings,
                                mc_img_bkg=mc_img_bkg,
                                channels=channels, channel_loader=_load_channel,
                                channel_default=ch_default, channel_note=mc_note,
                                channel_luts=ch_luts,
                                roi_ids=prev_rec.get("final_ids") if reuse_ids else None,
                                n_detected=prev_rec.get("n_detected") if reuse_ids else None,
                                next_id=prev_rec.get("next_id") if reuse_ids else None,
                                ap_polygon=ap_polygon)
            except Exception as e:
                # without this the worker thread waits forever on a window that
                # never opened and the pipeline hangs silently
                result_holder[0] = e
                self._log(traceback.format_exc())
                done.set()

        self.after(0, _show)
        done.wait()

        if isinstance(result_holder[0], Exception):
            raise RuntimeError(f"ROI editor for {z} failed to open") from result_holder[0]
        new_masks, new_bkg, new_mask_img, settings, sreg_masks, record = result_holder[0]

        self._display_settings = settings
        with open(ds_path, 'w') as f:
            yaml.dump(settings, f)

        roi_masks_file = Path(output_dir) / 'concat_roi-masks.npy'
        np.save(roi_masks_file, new_masks)

        # (AP, NTS) masks, region 0 is AP, region 1 is everything outside it
        sreg_file = Path(output_dir) / f'subregion_masks_{z}.npy'
        if sreg_masks and any(m is not None for m in sreg_masks):
            h, w = roi_img_bkg.shape[:2]
            fill = np.zeros((h, w), dtype=bool)
            sreg_arr = np.stack([m if m is not None else fill for m in sreg_masks])
            np.save(sreg_file, sreg_arr)
        elif sreg_file.exists():
            # finished without an AP on purpose, a stale file would silently
            # keep classifying this plane with the old outline.  Set aside, not
            # deleted, so older hand-drawn regions are never lost.
            bak = sreg_file.with_name(f'subregion_masks_{z}.cleared.npy')
            sreg_file.replace(bak)
            self._log(f"  {z}: no AP defined, previous sub-regions moved to {bak.name}")

        if record is not None:
            try:
                s = save_curation(output_dir, z, record,
                                  previous=prev if reuse_ids else None)
                ac = s["after_curation"]
                self._log(
                    f"  {z} curation: detected {s['n_detected']}  →  final {s['n_final']}  "
                    f"(added {len(s['added_ids'])}, removed by hand "
                    f"{len(s['removed_manual_ids'])}, excluded by region "
                    f"{len(s['removed_region_ids'])})")
                if s["ap_polygon"]:
                    d = ac["all"]["density"]
                    self._log(
                        f"    AP: {ac['all']['count']['AP']} neurons ({d['AP']} / 10k px²)   "
                        f"NTS: {ac['all']['count']['NTS']} neurons ({d['NTS']} / 10k px²)")
                self._log(f"    Record + screenshots: {Path(output_dir) / 'roi_curation'}")
                self._write_curation_summary(Path(output_dir).parent)
            except Exception:
                self._log(f"  ⚠ Could not save ROI curation record for {z}:")
                self._log(traceback.format_exc())

        return new_masks, roi_masks_file, new_bkg, new_mask_img

    def _load_previous_curation(self, plane_dir, z):
        """initial_rois_fn for source_extraction: the ROIs saved by this plane's
        last curation, or None (→ Cellpose runs as usual)."""
        from analysis.roi_curation import load_curated_masks
        try:
            got = load_curated_masks(plane_dir, z)
        except Exception as _e:
            self._log(f"  Warning: could not load previous curation for {z} ({_e}), "
                      "running Cellpose.")
            return None
        if got is None:
            self._log(f"  {z}: no previous curation saved, running Cellpose.")
            return None
        masks, rec = got
        self._resumed_planes[z] = rec
        self._log(
            f"  {z}: resuming previous curation, {masks.shape[1]} ROIs "
            f"({len(rec.get('added_ids', []))} added, "
            f"{len(rec.get('removed_manual_ids', [])) + len(rec.get('removed_region_ids', []))} "
            f"removed last time). Cellpose skipped.")
        return masks

    # ── curation yield ────────────────────────────────────────────────────────

    def _write_curation_summary(self, animal_dir):
        from analysis.roi_curation import write_summary
        try:
            path = write_summary(animal_dir)
            if path is not None:
                self._log(f"    Curation summary: {path}")
        except Exception as _e:
            self._log(f"  Warning: could not write curation summary ({_e})")

    def _report_curation_yield(self, provenance, z, responsive=None, cnm=None):
        """Track detected vs hand-added neurons through CNMF / acceptance /
        responsiveness, for one plane.  Silent when the plane has no curation
        record (curated before records existed)."""
        from analysis.roi_curation import update_yield, format_yield
        from pipeline_funcs import _load_is_cell
        try:
            se = (provenance.get('source_extraction') or {}).get(z) or {}
            cnm_file = (se.get('filenames') or {}).get('cnm_file')
            if not cnm_file or not Path(cnm_file).exists():
                return
            plane_dir = Path(cnm_file).parent
            if not (plane_dir / 'roi_curation' / f'roi_curation_{z}.json').exists():
                return
            if cnm is None:
                from caiman.source_extraction.cnmf import cnmf as _cnmf_module
                cnm = _cnmf_module.load_CNMF(str(cnm_file))
            y = update_yield(plane_dir, z, cnm.estimates.A,
                             is_cell=_load_is_cell(cnm_file, z),
                             responsive=responsive)
            if y is not None:
                for line in format_yield(z, y):
                    self._log(line)
                self._write_curation_summary(plane_dir.parent)
        except Exception as _e:
            self._log(f"  Warning: curation yield for {z} unavailable ({_e})")

    def _neuron_viewer_for_pipeline(self, cnm, mean_img, provenance=None, z=None):
        """Called from the worker thread. Opens NeuronViewerWindow on the main thread
        and blocks until the user clicks Save & Close."""
        from gui.neuron import Neuron
        from gui.neuron_viewer import NeuronViewerWindow

        self._log("  Building neuron list from CNMF estimates …")
        neurons = Neuron.build_all(cnm.estimates, dims=cnm.dims)   # pure numpy, safe on worker thread
        n_fail = sum(1 for n in neurons if n.passed_qc is False)
        if any(n.passed_qc is not None for n in neurons):
            self._log(f"  CaImAn check: {len(neurons) - n_fail} passed, {n_fail} failed "
                      f"(failed start rejected; review them in the viewer)")
        else:
            self._log("  CaImAn check: no pass/fail saved in this CNMF file, all start accepted")
        self._log(f"  Neuron viewer: {len(neurons)} components — opening …")

        qc_thresholds = None
        try:
            qc_thresholds = {key: cnm.params.get('quality', key)
                             for key in ('min_SNR', 'rval_thr', 'SNR_lowest', 'rval_lowest')}
        except Exception:
            pass

        # Build timing_info for trace annotations if we have session data
        timing_info = None
        if provenance is not None and z is not None:
            try:
                from pipeline_funcs import _session_lengths
                ses_lens = _session_lengths(provenance, z)
                fp    = getattr(self, '_current_fp',     0.033)
                pre_s = getattr(self, '_current_pre_s',  30.0)
                base_s = getattr(self, '_current_base_s', 30.0)
                stim_s = getattr(self, '_current_stim_s', 180.0)
                timing_info = dict(
                    fp=fp,
                    pre_f=round(pre_s  / fp),
                    base_f=round(base_s / fp),
                    stim_f=round(stim_s / fp),
                    session_lengths=ses_lens,
                )
            except Exception as _e:
                self._log(f"  (timing annotations unavailable: {_e})")

        result_holder = [None]
        done = threading.Event()

        def _show():
            def _on_close(is_cell):
                n_acc = int(is_cell.sum())
                result_holder[0] = is_cell
                self._log(f"  Neuron viewer closed,  {n_acc} / {len(neurons)} accepted.")
                done.set()
            NeuronViewerWindow(
                self, neurons, mean_img,
                frame_period=getattr(self, '_current_fp', 0.033),
                on_close=_on_close,
                timing_info=timing_info,
                qc_thresholds=qc_thresholds,
            )

        self.after(0, _show)
        done.wait()
        return result_holder[0]

    def _multiplane_viewer_for_pipeline(self, provenance, z_planes):
        """Load all z-planes, open ZPlaneViewerWindow on the main thread, block until done."""
        import caiman as cm
        from caiman.source_extraction.cnmf import cnmf as _cnmf_module
        from gui.neuron import Neuron
        from gui.zplane_viewer import ZPlaneViewerWindow
        from pipeline_funcs import _load_is_cell

        plane_data = []
        ch_dict = provenance['load_data']['args']['ch_dict']
        for z in z_planes:
            se = (provenance.get('source_extraction') or {}).get(z)
            if not se:
                self._log(f"  ⚠ No CNMF results for {z},  skipping.")
                continue
            cnm_file = se['filenames']['cnm_file']
            if not Path(cnm_file).exists():
                self._log(f"  ⚠ CNMF file not found: {cnm_file}")
                continue

            func_corr_file = provenance['rigid_motion_correction'][z]['filenames'][ch_dict['func_ch']]
            if not Path(func_corr_file).is_absolute():
                func_corr_file = str(Path(func_corr_file))

            self._log(f"  Loading {z} …")
            cnm      = _cnmf_module.load_CNMF(cnm_file)
            neurons  = Neuron.build_all(cnm.estimates, dims=cnm.dims)
            is_cell  = _load_is_cell(cnm_file, z)
            if is_cell is None:
                is_cell = np.ones(len(neurons), dtype=bool)
            func_lc = np.percentile(cm.load(func_corr_file), 99, axis=0)

            plane_data.append(dict(
                z=z, neurons=neurons, is_cell=is_cell, mean_image=func_lc))

        if len(plane_data) < 2:
            self._log("  Multi-plane review needs at least 2 z-planes with CNMF,  skipping.")
            return {}

        result_holder = [None]
        done = threading.Event()

        def _show():
            def _on_close(updated):
                result_holder[0] = updated
                done.set()
            ZPlaneViewerWindow(self, plane_data, on_close=_on_close)

        self.after(0, _show)
        done.wait()
        return result_holder[0] or {}

    def _run_pipeline(self, p: dict):
        class _StdoutCapture:
            def __init__(self, fn):
                self._fn = fn
                self._buf = ""
            def write(self, s):
                self._buf += s
                while "\n" in self._buf:
                    line, self._buf = self._buf.split("\n", 1)
                    self._fn(line)
            def flush(self): pass

        old_stdout, sys.stdout = sys.stdout, _StdoutCapture(self._log)
        try:
            self._pipeline_body(p)
        except Exception:
            self._log("\n── ERROR ──")
            self._log(traceback.format_exc())
        finally:
            sys.stdout = old_stdout
            self.after(0, lambda: (
                self.run_btn.configure(state="normal", text="▶   Run"),
                self._check_provenance(),
            ))

    def _pipeline_body(self, p: dict):
        from pipeline import (init, load_data, affine_motion_correction,
                               rigid_motion_correction, source_extraction,
                               _get_provenance, _save_provenance)
        from pipeline_funcs import (get_stims_n, get_resp_n,
                                    get_region_labels, get_spatial_response_data)

        fp, pre_s, base_s, stim_s = (p["frame_period"], p["pre_discard_s"],
                                      p["baseline_s"],   p["stim_s"])
        self._current_fp     = fp      # made available to _neuron_viewer_for_pipeline
        self._current_pre_s  = pre_s
        self._current_base_s = base_s
        self._current_stim_s = stim_s
        threshold = p["threshold"]
        n_stims   = p["n_stims"]

        d = _frame_layout(fp, pre_s, base_s, stim_s)
        stim_onset_idx = d["base_f"]

        self._log(
            f"Frame layout: {d['pre_f']} discard  +  {d['base_f']} baseline  "
            f"+  {d['stim_f']} stim  =  {d['ses_f']} per session  "
            f"({n_stims} session(s) × {d['ses_f']} = {n_stims * d['ses_f']} total)"
        )

        def _record_analysis_params(provenance, n_stims_val):
            provenance['analysis_params'] = dict(
                frame_period=fp, pre_discard_s=pre_s,
                baseline_s=base_s, stim_s=stim_s, threshold=threshold,
                n_stims=n_stims_val, z_planes=p["z_planes"],
                cp_diameter=p["cellpose"]["diameter"],
                cp_flow_threshold=p["cellpose"]["flow_threshold"],
                cp_cellprob=p["cellpose"]["cellprob_threshold"],
            )
            _save_provenance(provenance)

        _all_stims_n: list[list[np.ndarray]] = []   # [animal][stim_idx] → (K, T)
        _animal_labels: list[str] = []              # aligned with _all_stims_n
        _z_ids:  list[np.ndarray] = []
        _all_region_labels: list[np.ndarray] = []
        _all_spatial = []

        for i, sessions in enumerate(p["animals"]):
            label = (f"{p['subject']}_animal{i + 1}"
                     if len(p["animals"]) > 1 else p["subject"])
            out_dir = str(Path(p["output"]) / label)
            self._log(f"\n── Animal {i + 1}  ({label})")

            provenance = init(out_dir)
            # made available to _roi_editor_for_pipeline, which is called from
            # inside source_extraction and has no provenance of its own
            self._provenance = provenance

            # Persist the run configuration immediately, before any processing,
            # so reopening this project later recovers the settings actually used
            # even if the run is interrupted or only an earlier stage is selected.
            _record_analysis_params(provenance, n_stims)

            if self.do_mc.get():
                for z in p["z_planes"]:
                    self._log(f"  Loading data ({z}) …")
                    provenance, data = load_data(
                        provenance, sessions, p["ch_dict"], z)
                    self._log(f"  Affine correction ({z}) …")
                    provenance, affcorr = affine_motion_correction(
                        provenance, z, data)
                    self._log(f"  Rigid correction ({z}) …")
                    provenance, _ = rigid_motion_correction(
                        provenance, z, affcorr)
            else:
                self._log("  Skipping motion correction,  loading saved provenance.")
                provenance = _get_provenance(out_dir)
                self._provenance = provenance

            self._provenance = provenance

            self._resumed_planes = {}              # z -> record the ROIs came from
            if self.do_cnmf.get():
                roi_fn = self._roi_editor_for_pipeline
                resume_fn = self._load_previous_curation if self.resume_curation.get() else None
                cp = p["cellpose"]
                self._log(
                    f"  Cellpose params,  diameter: {cp['diameter']}  "
                    f"flow_threshold: {cp['flow_threshold']}  "
                    f"cellprob_threshold: {cp['cellprob_threshold']}")
                for z in p["z_planes"]:
                    self._log(
                        f"  Source extraction ({z}),  "
                        "ROI editor will open in a popup window …")
                    def _make_viewer_fn(z_name):
                        def _fn(cnm, img):
                            return self._neuron_viewer_for_pipeline(
                                cnm, img, provenance, z_name)
                        return _fn
                    provenance = source_extraction(
                        provenance, None, z, None,
                        roi_editor_fn=roi_fn,
                        neuron_viewer_fn=_make_viewer_fn(z),
                        initial_rois_fn=resume_fn,
                        idroi_params=cp,
                        runcnmf_params=dict(fr=1 / fp))
                    self._report_curation_yield(provenance, z)
            else:
                self._log("  Skipping CNMF,  using saved results.")

            if self.do_subregion_setup.get():
                self._log("  Sub-region setup, ROI editor will open for sub-region definition …")
                self._provenance = provenance
                try:
                    import caiman as cm
                    import cv2 as _cv2
                    ch_dict = provenance['load_data']['args']['ch_dict']
                    for z in p["z_planes"]:
                        se = (provenance.get('source_extraction') or {}).get(z)
                        if not se:
                            self._log(f"  ⚠ No CNMF results found for {z},  run CNMF first.")
                            continue
                        roi_masks_file = se['filenames']['roi_masks_file']
                        if not Path(roi_masks_file).is_absolute():
                            roi_masks_file = str(Path(out_dir) / Path(roi_masks_file).name)
                        if not Path(roi_masks_file).exists():
                            candidate = str(Path(out_dir) / z / Path(roi_masks_file).name)
                            if Path(candidate).exists():
                                roi_masks_file = candidate
                        if not Path(roi_masks_file).exists():
                            self._log(f"  ⚠ ROI masks file not found: {roi_masks_file}")
                            continue
                        roi_masks = np.load(roi_masks_file)

                        func_corr_file = provenance['rigid_motion_correction'][z]['filenames'][ch_dict['func_ch']]
                        mc_corr_file   = provenance['rigid_motion_correction'][z]['filenames'][ch_dict['mc_ch']]
                        if not Path(func_corr_file).is_absolute():
                            func_corr_file = str(Path(out_dir) / z / Path(func_corr_file).name)
                        if not Path(mc_corr_file).is_absolute():
                            mc_corr_file = str(Path(out_dir) / z / Path(mc_corr_file).name)

                        func_movie = cm.load(func_corr_file)
                        func_lc = np.array(func_movie).max(axis=0)
                        scale = func_lc.max()
                        if scale > 0:
                            roi_img_bkg = (func_lc * 190 / scale).astype(np.uint8)
                        else:
                            roi_img_bkg = np.zeros(func_lc.shape, dtype=np.uint8)
                        roi_img_bkg = _cv2.cvtColor(roi_img_bkg, _cv2.COLOR_GRAY2RGB)

                        roi_img_mask = np.zeros([*func_lc.shape, 3], dtype=np.uint8)
                        for mask_col in roi_masks.T:
                            roi_img_mask[mask_col.reshape(func_lc.shape, order='F')] = (
                                np.tile(65 * np.random.rand(1, 3), (int(mask_col.sum()), 1)).astype(int))

                        self._roi_editor_for_pipeline(
                            Path(out_dir) / z, mc_corr_file, z,
                            roi_masks, roi_img_bkg, roi_img_mask,
                            subregion_only=True)
                        self._report_curation_yield(provenance, z)
                except Exception:
                    self._log("  ⚠ Sub-region setup failed:")
                    self._log(traceback.format_exc())

            if self.do_neuron_curation.get():
                self._log("  Neuron curation,  loading saved CNMF, opening viewer …")
                try:
                    import caiman as cm
                    from caiman.source_extraction.cnmf import cnmf as _cnmf_module
                    ch_dict = provenance['load_data']['args']['ch_dict']
                    for z in p["z_planes"]:
                        se = (provenance.get('source_extraction') or {}).get(z)
                        if not se:
                            self._log(f"  ⚠ No CNMF results found for {z}, run CNMF first.")
                            continue
                        cnm_file = se['filenames']['cnm_file']
                        if not Path(cnm_file).exists():
                            self._log(f"  ⚠ CNMF file not found: {cnm_file}")
                            continue

                        func_corr_file = provenance['rigid_motion_correction'][z]['filenames'][ch_dict['func_ch']]
                        if not Path(func_corr_file).is_absolute():
                            func_corr_file = str(Path(out_dir) / z / Path(func_corr_file).name)

                        self._log(f"    {z}: computing background image …")
                        func_lc = np.percentile(cm.load(func_corr_file), 99, axis=0)

                        self._log(f"    {z}: loading CNMF …")
                        cnm = _cnmf_module.load_CNMF(cnm_file)

                        is_cell = self._neuron_viewer_for_pipeline(cnm, func_lc, provenance, z)
                        if is_cell is not None:
                            is_cell_file = Path(cnm_file).parent / f'concat_{z}_is_cell.npy'
                            np.save(is_cell_file, is_cell)
                            provenance['source_extraction'][z]['filenames']['is_cell_file'] = str(is_cell_file)
                            _save_provenance(provenance)
                            self._report_curation_yield(provenance, z, cnm=cnm)
                except Exception:
                    self._log("  ⚠ Neuron curation failed:")
                    self._log(traceback.format_exc())

            if self.do_multiplane_review.get():
                if len(p["z_planes"]) < 2:
                    self._log("  Multi-plane review skipped,  only one z-plane configured.")
                else:
                    self._log("  Multi-plane duplicate review, loading all z-planes …")
                    try:
                        updated = self._multiplane_viewer_for_pipeline(
                            provenance, p["z_planes"])
                        for z, new_is_cell in updated.items():
                            se = (provenance.get('source_extraction') or {}).get(z)
                            if not se:
                                continue
                            cnm_file    = se['filenames']['cnm_file']
                            is_cell_file = Path(cnm_file).parent / f'concat_{z}_is_cell.npy'
                            np.save(is_cell_file, new_is_cell)
                            provenance['source_extraction'][z]['filenames']['is_cell_file'] = str(is_cell_file)
                            n_acc = int(new_is_cell.sum())
                            self._log(f"  {z}: {n_acc} / {len(new_is_cell)} neurons accepted")
                        _save_provenance(provenance)
                        for z in updated:
                            self._report_curation_yield(provenance, z)
                    except Exception:
                        self._log("  ⚠ Multi-plane review failed:")
                        self._log(traceback.format_exc())

            if self.do_analysis.get():
                self._log("  Computing stimulus responses …")
                for z in p["z_planes"]:
                    mc_z = ((provenance.get('rigid_motion_correction') or {})
                            .get(z) or {})
                    counts = mc_z.get('session_frame_counts')
                    if counts:
                        expected = d['ses_f']
                        status = "✓" if all(c == expected for c in counts) else "⚠ MISMATCH"
                        self._log(
                            f"    {z}: {counts} frames recorded  "
                            f"(timing params expect {expected} each)  {status}"
                        )
                stims_n, z_ids = get_stims_n(
                    provenance,
                    frame_period=fp,
                    pre_discard_s=pre_s,
                    baseline_s=base_s,
                    stim_s=stim_s,
                )
                _record_analysis_params(provenance, len(stims_n))

                # Accumulate per-animal stims,  each is a list of N arrays
                _all_stims_n.append(stims_n)
                _animal_labels.append(label)
                _z_ids.append(z_ids)

                # detected vs hand-added: how many respond?  Same criterion as
                # get_resp_n, median z-score over the stim window above threshold
                # for any stimulus.  Rows follow get_stims_n's plane order.
                _responds = np.any(
                    [np.median(s[:, stim_onset_idx:], axis=1) > threshold
                     for s in stims_n], axis=0)
                for z in provenance['source_extraction'].keys():
                    _rows = z_ids == int(str(z).replace('z', ''))
                    self._report_curation_yield(provenance, z,
                                                responsive=_responds[_rows])
                self._log("  Building spatial response data …")
                _all_spatial.append((label, get_spatial_response_data(
                    provenance, fp, pre_s, base_s, stim_s,
                    subregion_dir=out_dir)))

                # AP / NTS label per neuron whenever an AP outline exists, so it is
                # saved in neurons.csv even when the sub-region plots are not requested
                sreg_files = {z: Path(out_dir) / z / f'subregion_masks_{z}.npy'
                              for z in p["z_planes"]}
                rlabels = np.full(stims_n[0].shape[0], -1, dtype=int)
                if any(f.exists() for f in sreg_files.values()):
                    self._log("  Classifying neurons by sub-region …")
                    missing = [z for z, f in sreg_files.items() if not f.exists()]
                    if missing:
                        self._log(
                            f"  ⚠ No AP outline for {', '.join(missing)}, their neurons stay\n"
                            "    unclassified. Run 'Sub-region setup' to add them; planes\n"
                            "    already outlined reopen with their AP restored.")
                    try:
                        rlabels = get_region_labels(provenance, out_dir)
                    except Exception:
                        self._log("  ⚠ Sub-region classification failed:")
                        self._log(traceback.format_exc())
                    else:
                        n_a = int((rlabels == 0).sum())
                        n_b = int((rlabels == 1).sum())
                        n_none = int((rlabels == -1).sum())
                        self._log(
                            f"    AP: {n_a}   NTS: {n_b}   "
                            f"Unclassified: {n_none}"
                        )
                        if n_a == 0 and n_b == 0:
                            self._log(
                                "  ⚠ All neurons are unclassified,  check that the\n"
                                "    sub-region polygons cover the neurons on the canvas."
                            )
                elif self.do_subregion.get():
                    self._log(
                        "  ⚠ No subregion_masks_*.npy files found,  expected at:\n"
                        + "\n".join(f"      {f}" for f in sreg_files.values())
                        + "\n    Run 'Sub-region setup' (or CNMF) and outline the AP\n"
                          "    in the ROI editor, then re-run analysis."
                    )
                _all_region_labels.append(rlabels)

        if not ((self.do_analysis.get() or self.do_population.get() or
                 self.do_manifold.get()) and _all_stims_n):
            self._log("\nDone.")
            return

        self._log("\nClassifying responders across all animals …")

        # Stack across animals: each stim index has one big array
        N = len(_all_stims_n[0])
        combined_stims_n = [
            np.vstack([animal_stims[j] for animal_stims in _all_stims_n])
            for j in range(N)
        ]
        z_ids_all = np.concatenate(_z_ids)
        region_labels_all = (np.concatenate(_all_region_labels) if _all_region_labels
                             else np.full(len(z_ids_all), -1, dtype=int))
        mouse_labels_all = np.repeat(_animal_labels,
                                     [s[0].shape[0] for s in _all_stims_n])

        # Extract centroids aligned with combined_stims_n (for manifold viewer)
        manifold_centroids = None
        if _all_spatial:
            try:
                all_centers = [zd['centers'] for _, sd in _all_spatial
                               for zd in sd if zd.get('centers') is not None]
                if all_centers:
                    manifold_centroids = np.vstack(all_centers)  # (K_total, 2)
            except Exception:
                pass

        stim_names = [f"Stimulus {j + 1}" for j in range(N)]

        out_dir = str(Path(p["output"]) / p["subject"])
        results_parent = p["analysis_out"] if p["analysis_out"] else out_dir
        # Derive results_dir path (same as _save_results would return)
        results_dir = str(Path(results_parent) / "analysis")
        Path(results_dir).mkdir(parents=True, exist_ok=True)

        if self.do_analysis.get():
            resp_n, nums, group_sizes, z_ids_resp = get_resp_n(
                combined_stims_n, z_ids_all,
                stim_onset_idx=stim_onset_idx,
                threshold=threshold,
            )

            n_total = resp_n[0].shape[0]
            if N == 2:
                self._log(
                    f"Stim-1 only: {nums[0]}   Both: {nums[1]}   Stim-2 only: {nums[2]}   "
                    f"Total responsive: {n_total} / {combined_stims_n[0].shape[0]}"
                )
            else:
                parts = "   ".join(
                    f"Stim-{j+1} resp: {nums[j]}" for j in range(N))
                self._log(
                    f"{parts}   "
                    f"Total responsive (unique): {n_total} / {combined_stims_n[0].shape[0]}"
                )

            results_dir = self._save_results(
                results_parent, resp_n, nums, z_ids_resp, group_sizes, p,
                stim_onset_idx, d["ses_f"],
                stims_n=combined_stims_n, z_ids=z_ids_all,
                region_labels=region_labels_all, mouse_labels=mouse_labels_all)
            self._log(f"Results saved to  {results_dir}")

            self._log("Opening figures …")
            self.after(0, lambda: show_plots(
                resp_n, nums, group_sizes,
                stim_onset_idx, d["ses_f"],
                fp, pre_s, stim_s, results_dir,
                stim_names=stim_names))

            for _lbl, _sd in _all_spatial:
                self.after(0, lambda lbl=_lbl, sd=_sd: show_spatial_response_map(
                    lbl, sd, threshold, results_dir))

            # optional sub-region analysis
            if self.do_subregion.get() and _all_region_labels:
                region_results = {}
                for reg_idx, reg_name in [(0, "AP"), (1, "NTS")]:
                    mask = region_labels_all == reg_idx
                    n_total_r = int(mask.sum())
                    if n_total_r == 0:
                        continue
                    stims_region = [s[mask] for s in combined_stims_n]
                    r_n, nums_r, gsizes_r, _ = get_resp_n(
                        stims_region, z_ids_all[mask],
                        stim_onset_idx=stim_onset_idx,
                        threshold=threshold,
                    )
                    region_results[reg_name] = {
                        'resp_n': r_n, 'nums': nums_r,
                        'group_sizes': gsizes_r, 'n_total': n_total_r,
                    }
                    if N == 2:
                        self._log(
                            f"{reg_name} — total: {n_total_r}  "
                            f"stim1-only: {nums_r[0]}  both: {nums_r[1]}  stim2-only: {nums_r[2]}"
                        )
                    else:
                        parts = "   ".join(
                            f"stim{j+1} resp: {nums_r[j]}" for j in range(N))
                        self._log(f"{reg_name} — total: {n_total_r}  {parts}")

                if region_results:
                    self._log(
                        f"  Opening sub-region figures for: {', '.join(region_results.keys())}")
                    self.after(0, lambda rr=region_results: show_region_plots(
                        rr, stim_onset_idx, d["ses_f"], fp, pre_s, stim_s, results_dir,
                        stim_names=stim_names))
                else:
                    self._log(
                        "  ⚠ Sub-region plots skipped, no neurons were classified into any region.")

        # ── population analysis (optional) ────────────────────────────────────
        if self.do_population.get() and combined_stims_n:
            self._log("  Running population analysis …")
            from analysis.population import run_population_analysis
            from gui.population_viewer import PopulationViewerWindow
            import traceback as _tb
            try:
                pop_results = run_population_analysis(
                    combined_stims_n,
                    stim_onset_idx=stim_onset_idx,
                    results_dir=results_dir,
                    fp=fp,
                    stim_names=stim_names,
                    centroids=manifold_centroids,
                    z_ids=z_ids_all,
                )
                self.after(0, lambda r=pop_results: PopulationViewerWindow(
                    self, r, results_dir))
                self._log("  Population viewer opened.")
            except Exception:
                self._log("  Population analysis failed:")
                self._log(_tb.format_exc())

        # ── neural manifold analysis (optional) ───────────────────────────────
        if self.do_manifold.get() and combined_stims_n:
            self._log("  Running neural manifold analysis …")
            from analysis.manifold import run_manifold_analysis
            from gui.manifold_viewer import ManifoldViewerWindow
            import traceback as _tb_mfld
            try:
                mfld_results = run_manifold_analysis(
                    combined_stims_n,
                    stim_onset_idx=stim_onset_idx,
                    results_dir=results_dir,
                    fp=fp,
                    stim_names=stim_names,
                    centroids=manifold_centroids,
                )
                mfld_results["z_ids_all"] = z_ids_all
                self.after(0, lambda r=mfld_results: ManifoldViewerWindow(self, r))
                self._log("  Manifold viewer opened.")
            except Exception:
                self._log("  Manifold analysis failed:")
                self._log(_tb_mfld.format_exc())

        self._log("Done.")

    # ── save results ───────────────────────────────────────────────────────

    def _save_results(self, out_dir: str, resp_n, nums, z_ids_resp, group_sizes,
                      params: dict, stim_onset_idx: int, ses_f: int, *,
                      stims_n, z_ids, region_labels, mouse_labels) -> str:
        """Save responder arrays plus the all-neuron table and traces.

        stims_n / z_ids / region_labels / mouse_labels cover every neuron, in
        the same row order; they become neurons.csv and traces_stim<j>.npy
        (see analysis.results_io), which combine_mice.py reads.
        """
        import yaml
        from analysis.results_io import save_neuron_results, REGIONS

        results_dir = Path(out_dir) / "analysis"
        results_dir.mkdir(parents=True, exist_ok=True)

        N = len(resp_n)
        for j, r in enumerate(resp_n):
            np.save(results_dir / f"resp{j + 1}.npy", r)
        np.save(results_dir / "nums.npy",        np.array(nums))
        np.save(results_dir / "group_sizes.npy", np.array(group_sizes))
        np.save(results_dir / "z_ids_resp.npy",  z_ids_resp)

        table = save_neuron_results(results_dir, stims_n, z_ids, region_labels,
                                    mouse_labels, stim_onset_idx, params["threshold"])

        saved_params = dict(
            frame_period   = params["frame_period"],
            pre_discard_s  = params["pre_discard_s"],
            baseline_s     = params["baseline_s"],
            stim_s         = params["stim_s"],
            threshold      = params["threshold"],
            n_stims        = N,
            stim_onset_idx = stim_onset_idx,
            ses_f          = ses_f,
            z_planes       = params["z_planes"],
            mice           = list(dict.fromkeys(table["mouse"])),
            # all accepted neurons; n_responders is the unique responder count
            n_total_neurons = len(table),
            n_responders    = int(table["responder"].sum()),
        )
        if N == 2:
            saved_params.update(
                n_stim1_only    = int(nums[0]),
                n_both          = int(nums[1]),
                n_stim2_only    = int(nums[2]),
            )
        for reg in REGIONS:
            in_reg = table["region"] == reg
            if in_reg.any():
                saved_params[f"n_neurons_{reg}"]    = int(in_reg.sum())
                saved_params[f"n_responders_{reg}"] = int(table.loc[in_reg, "responder"].sum())
        with open(results_dir / "params.yaml", "w") as f:
            yaml.safe_dump(saved_params, f, default_flow_style=False)

        return str(results_dir)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
