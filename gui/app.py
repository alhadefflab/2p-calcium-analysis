"""
Luceo — 2P Calcium Imaging Pipeline
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
            m = re.findall(r"(\d{6})ome", f.stem)
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
    def __init__(self, parent, index: int, on_remove, **kw):
        super().__init__(parent, **kw)
        self.index = index

        ctk.CTkLabel(self, text=f"Animal {index + 1}",
                     width=78, font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, rowspan=2, padx=10)

        self.sess1_var = ctk.StringVar()
        self.sess2_var = ctk.StringVar()

        for row_idx, (label, var, ph) in enumerate([
            ("Session 1:", self.sess1_var, "Stimulus 1 folder  (e.g. fructose)"),
            ("Session 2:", self.sess2_var, "Stimulus 2 folder  (e.g. glucose)"),
        ]):
            ctk.CTkLabel(self, text=label, width=74).grid(
                row=row_idx, column=1, padx=(0, 4),
                pady=(5 if row_idx == 0 else 2, 5 if row_idx == 1 else 2), sticky="w")
            ctk.CTkEntry(self, textvariable=var, width=270,
                         placeholder_text=ph).grid(
                row=row_idx, column=2, padx=4,
                pady=(5 if row_idx == 0 else 2, 5 if row_idx == 1 else 2))
            ctk.CTkButton(self, text="Browse", width=72,
                          command=lambda v=var: self._browse(v)).grid(
                row=row_idx, column=3, padx=4)

        ctk.CTkButton(self, text="✕", width=30, height=30,
                      fg_color="#c0392b", hover_color="#922b21",
                      command=on_remove).grid(row=0, column=4, rowspan=2, padx=8)

    def _browse(self, var: ctk.StringVar):
        path = filedialog.askdirectory(title="Select session folder")
        if path:
            var.set(path)

    def get_paths(self) -> tuple[str, str]:
        return self.sess1_var.get().strip(), self.sess2_var.get().strip()

    def set_paths(self, s1: str, s2: str):
        self.sess1_var.set(s1)
        self.sess2_var.set(s2)


# ── main window ───────────────────────────────────────────────────────────────

class PipelineGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Luceo")
        self.geometry("860x720")
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
                     placeholder_text="Default: <project folder>/analysis/  — override here if needed").grid(
            row=2, column=1, padx=8, sticky="w")
        ctk.CTkButton(top, text="Browse", width=80,
                      command=self._browse_analysis_out).grid(row=2, column=2, padx=4)

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

    def _on_mode_change(self):
        if self.mode_var.get() == "single":
            self.add_btn.pack_forget()
            while len(self.animal_rows) > 1:
                self.animal_rows.pop().destroy()
        else:
            self.add_btn.pack(pady=4)

    def _add_animal(self):
        idx = len(self.animal_rows)
        row = AnimalRow(self.animals_scroll, idx,
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

        # Always derive output and subject from the folder the user selected,
        # not from the path stored in provenance (which may be from another PC).
        self.output_var.set(str(folder.parent))
        self.subject_var.set(folder.name)

        # restore session paths
        load_args = prov.get("load_data") or {}
        load_args = load_args.get("args") or {} if isinstance(load_args, dict) else {}
        multi_path = load_args.get("multi_path", [])
        ch_dict    = load_args.get("ch_dict", {})

        if len(multi_path) >= 2:
            self.animal_rows[0].set_paths(str(multi_path[0]), str(multi_path[1]))

        if ch_dict:
            self.mc_ch_var.set(ch_dict.get("mc_ch", "ch1"))
            self.func_ch_var.set(ch_dict.get("func_ch", "ch2"))

        # restore z-planes
        mc_prov = prov.get("rigid_motion_correction") or {}
        if isinstance(mc_prov, dict) and mc_prov:
            self.z_planes_var.set(",".join(mc_prov.keys()))

        # restore timing params if a previous analysis was saved
        ap = prov.get("analysis_params") or {}
        if ap:
            if "frame_period"  in ap: self.frame_period_var.set(str(ap["frame_period"]))
            if "pre_discard_s" in ap: self.pre_discard_var.set(str(ap["pre_discard_s"]))
            if "baseline_s"    in ap: self.baseline_var.set(str(ap["baseline_s"]))
            if "stim_s"        in ap: self.stim_var.set(str(ap["stim_s"]))
            if "threshold"     in ap: self.threshold_var.set(str(ap["threshold"]))

        self._check_provenance()
        self.tabs.set("Run")

    def _check_provenance(self):
        """
        Read provenance for the current output+subject and update the stage
        checkboxes and status label to reflect what is already done.
        """
        if not hasattr(self, "do_mc"):
            return  # Run tab not built yet

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

        # check for saved analysis results
        results_dir = Path(project_dir) / "analysis"
        if (results_dir / "resp1.npy").exists():
            parts.append("analysis results ✓")

        if parts:
            self.prov_label.configure(
                text=f"  Existing project found — {',  '.join(parts)}",
                text_color="#5cb85c")
        else:
            self.prov_label.configure(
                text="  No existing project found at this location — will start fresh.",
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
        field("Motion-correction channel:", self.mc_ch_var,   "structural / anatomical")
        field("Functional channel:",        self.func_ch_var, "calcium indicator signal")

        self._refresh()

    def _autodetect_zplanes(self):
        if not self.animal_rows:
            messagebox.showinfo("Auto-detect", "Add an animal and set Session 1 first.")
            return
        sess1, _ = self.animal_rows[0].get_paths()
        if not sess1:
            messagebox.showinfo("Auto-detect", "Set the Session 1 path for Animal 1 first.")
            return
        zs = _detect_zplanes(sess1)
        if zs:
            self.z_planes_var.set(",".join(zs))
            messagebox.showinfo("Auto-detect", f"Found: {', '.join(zs)}")
        else:
            messagebox.showwarning("Auto-detect",
                                   "No z-planes detected — check the folder contains Bruker TIFFs.")

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
                f"  Actual count read from recording — mismatch shown in run log."
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
            text="Motion correction  —  slow, reads raw TIFFs, skip if already done")
        self.do_mc.pack(anchor="w", padx=16, pady=4)
        self.do_mc.select()

        self.do_cnmf = ctk.CTkCheckBox(
            stages,
            text="Source extraction / CNMF  —  slow, opens interactive ROI editor, skip if already done")
        self.do_cnmf.pack(anchor="w", padx=16, pady=4)
        self.do_cnmf.select()

        self.do_subregion_setup = ctk.CTkCheckBox(
            stages,
            text="Sub-region setup  —  (re)define sub-regions without re-running CNMF")
        self.do_subregion_setup.pack(anchor="w", padx=16, pady=4)

        self.do_analysis = ctk.CTkCheckBox(
            stages,
            text="Stimulus response analysis  —  fast, re-run this after changing timing or threshold")
        self.do_analysis.pack(anchor="w", padx=16, pady=4)
        self.do_analysis.select()

        self.do_subregion = ctk.CTkCheckBox(
            stages,
            text="Sub-region analysis  —  optional, requires regions defined in the ROI editor")
        self.do_subregion.pack(anchor="w", padx=16, pady=(4, 10))

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

        animals = []
        for r in self.animal_rows:
            s1, s2 = r.get_paths()
            if not s1 or not s2:
                errs.append(f"Both session folders are required for Animal {r.index + 1}.")
            animals.append((s1, s2))

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

        return dict(
            subject=subject, output=output, animals=animals,
            analysis_out=self.analysis_out_var.get().strip(),
            frame_period=fp, pre_discard_s=pre_s, baseline_s=base_s,
            stim_s=stim_s, threshold=threshold, z_planes=z_planes,
            ch_dict={"mc_ch": self.mc_ch_var.get().strip(),
                     "func_ch": self.func_ch_var.get().strip()},
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

    def _roi_editor_for_pipeline(self, output_dir, mc_corr_file, z, roi_masks, roi_img_bkg, roi_img_mask):
        """Called from the worker thread. Shows ROIEditorWindow on the main thread and blocks until Finish."""
        import yaml
        # Resolve relative mmap path — provenance may store it relative to the project root
        if not Path(mc_corr_file).is_absolute():
            mc_corr_file = str(Path(output_dir) / Path(mc_corr_file).name)

        # Load persisted display settings (carry over from previous z-plane)
        ds_path = Path(output_dir) / 'display_settings.yaml'
        if not hasattr(self, '_display_settings'):
            if ds_path.exists():
                with open(ds_path, 'r') as f:
                    self._display_settings = yaml.safe_load(f) or {}
            else:
                self._display_settings = {}

        # Pre-compute structural (MC) channel mean frame for anatomical orientation.
        # Use cm.load() — the same API used by _identify_rois for the functional
        # channel — so both images come from the same CaImAn code path.
        mc_img_bkg = None
        try:
            import caiman as cm
            import cv2 as cv
            from PIL import Image as _PILImg
            mc_movie = cm.load(mc_corr_file)        # shape (T, d1, d2)
            mc_mean  = np.mean(mc_movie, axis=0)    # (d1, d2)
            scale    = mc_mean.max()
            if scale > 0:
                mc_gray = (mc_mean * 190 / scale).astype(np.uint8)
                func_h, func_w = roi_img_bkg.shape[:2]
                mc_h, mc_w = mc_gray.shape
                if (mc_h, mc_w) != (func_h, func_w):
                    self._log(
                        f"  MC channel dims {(mc_h, mc_w)} differ from functional "
                        f"{(func_h, func_w)} — resizing MC background to match.")
                    mc_gray = np.array(
                        _PILImg.fromarray(mc_gray).resize(
                            (func_w, func_h), _PILImg.BILINEAR))
                else:
                    self._log(
                        f"  MC background dims match functional ({func_h}×{func_w}) — no resize needed.")
                mc_img_bkg = cv.cvtColor(mc_gray, cv.COLOR_GRAY2RGB)
        except Exception as _e:
            self._log(f"  Warning: could not load MC channel for sub-region view ({_e})")

        result_holder = [None]
        done = threading.Event()

        def _show():
            def _on_finish(masks, bkg, mask_img, settings, sreg_masks=None):
                result_holder[0] = (masks, bkg, mask_img, settings, sreg_masks)
                done.set()
            ROIEditorWindow(self, z, roi_img_bkg, roi_img_mask, roi_masks, mc_corr_file,
                            _on_finish, display_settings=self._display_settings,
                            mc_img_bkg=mc_img_bkg)

        self.after(0, _show)
        done.wait()

        new_masks, new_bkg, new_mask_img, settings, sreg_masks = result_holder[0]

        # Persist settings for next z-plane and save to project folder
        self._display_settings = settings
        with open(ds_path, 'w') as f:
            yaml.dump(settings, f)

        roi_masks_file = Path(output_dir) / 'concat_roi-masks.npy'
        np.save(roi_masks_file, new_masks)

        if sreg_masks and any(m is not None for m in sreg_masks):
            h, w = roi_img_bkg.shape[:2]
            fill = np.zeros((h, w), dtype=bool)
            sreg_arr = np.stack([m if m is not None else fill for m in sreg_masks])
            np.save(Path(output_dir) / f'subregion_masks_{z}.npy', sreg_arr)

        return new_masks, roi_masks_file, new_bkg, new_mask_img

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
        from pipeline_funcs import (get_stims1_stims2, get_resp1_resp2,
                                    get_region_labels, get_spatial_response_data)

        fp, pre_s, base_s, stim_s = (p["frame_period"], p["pre_discard_s"],
                                      p["baseline_s"],   p["stim_s"])
        threshold = p["threshold"]

        d = _frame_layout(fp, pre_s, base_s, stim_s)
        stim_onset_idx = d["base_f"]

        self._log(
            f"Frame layout: {d['pre_f']} discard  +  {d['base_f']} baseline  "
            f"+  {d['stim_f']} stim  =  {d['ses_f']} per session  "
            f"({2 * d['ses_f']} total)"
        )

        _all_stims1, _all_stims2, _z_ids, _all_region_labels = [], [], [], []
        _all_spatial = []   # list of (label, spatial_data) for spatial response maps

        for i, (sess1, sess2) in enumerate(p["animals"]):
            label = (f"{p['subject']}_animal{i + 1}"
                     if len(p["animals"]) > 1 else p["subject"])
            out_dir = str(Path(p["output"]) / label)
            self._log(f"\n── Animal {i + 1}  ({label})")

            provenance = init(out_dir)

            if self.do_mc.get():
                for z in p["z_planes"]:
                    self._log(f"  Loading data ({z}) …")
                    provenance, data = load_data(
                        provenance, [sess1, sess2], p["ch_dict"], z)
                    self._log(f"  Affine correction ({z}) …")
                    provenance, affcorr = affine_motion_correction(
                        provenance, z, data)
                    self._log(f"  Rigid correction ({z}) …")
                    provenance, _ = rigid_motion_correction(
                        provenance, z, affcorr)
            else:
                self._log("  Skipping motion correction — loading saved provenance.")
                provenance = _get_provenance(out_dir)

            if self.do_cnmf.get():
                roi_fn = self._roi_editor_for_pipeline
                for z in p["z_planes"]:
                    self._log(
                        f"  Source extraction ({z}) — "
                        "ROI editor will open in a popup window …")
                    provenance = source_extraction(
                        provenance, None, z, None, roi_editor_fn=roi_fn)
            else:
                self._log("  Skipping CNMF — using saved results.")

            if self.do_subregion_setup.get():
                self._log("  Sub-region setup — ROI editor will open for sub-region definition …")
                try:
                    import caiman as cm
                    import cv2 as _cv2
                    ch_dict = provenance['load_data']['args']['ch_dict']
                    for z in p["z_planes"]:
                        se = (provenance.get('source_extraction') or {}).get(z)
                        if not se:
                            self._log(f"  ⚠ No CNMF results found for {z} — run CNMF first.")
                            continue
                        roi_masks_file = se['filenames']['roi_masks_file']
                        if not Path(roi_masks_file).is_absolute():
                            roi_masks_file = str(Path(out_dir) / Path(roi_masks_file).name)
                        if not Path(roi_masks_file).exists():
                            # try inside z sub-folder
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
                            roi_masks, roi_img_bkg, roi_img_mask)
                except Exception:
                    self._log("  ⚠ Sub-region setup failed:")
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
                            f"    {z}: {counts[0]} + {counts[1]} frames recorded  "
                            f"(timing params expect {expected} each)  {status}"
                        )
                stims1, stims2, z_ids = get_stims1_stims2(
                    provenance,
                    frame_period=fp,
                    pre_discard_s=pre_s,
                    baseline_s=base_s,
                    stim_s=stim_s,
                )
                provenance['analysis_params'] = dict(
                    frame_period=fp, pre_discard_s=pre_s,
                    baseline_s=base_s, stim_s=stim_s, threshold=threshold,
                )
                _save_provenance(provenance)
                _all_stims1.append(stims1)
                _all_stims2.append(stims2)
                _z_ids.append(z_ids)
                self._log("  Building spatial response data …")
                _all_spatial.append((label, get_spatial_response_data(
                    provenance, fp, pre_s, base_s, stim_s,
                    subregion_dir=out_dir)))

                if self.do_subregion.get():
                    self._log("  Classifying neurons by sub-region …")
                    # Check whether any subregion mask files exist for this animal.
                    # Files are stored in the per-z subfolder alongside other CNMF outputs.
                    found_any = any(
                        (Path(out_dir) / z / f'subregion_masks_{z}.npy').exists()
                        for z in p["z_planes"]
                    )
                    if not found_any:
                        self._log(
                            "  ⚠ No subregion_masks_*.npy files found — expected at:\n"
                            + "\n".join(
                                f"      {Path(out_dir) / z / f'subregion_masks_{z}.npy'}"
                                for z in p["z_planes"])
                            + "\n    Run 'Sub-region setup' (or CNMF) and define both regions\n"
                              "    in the ROI editor, then re-run analysis."
                        )
                        _all_region_labels.append(np.full(stims1.shape[0], -1, dtype=int))
                    else:
                        rlabels = get_region_labels(provenance, out_dir)
                        _all_region_labels.append(rlabels)
                        n_a = int((rlabels == 0).sum())
                        n_b = int((rlabels == 1).sum())
                        n_none = int((rlabels == -1).sum())
                        self._log(
                            f"    Region A: {n_a}   Region B: {n_b}   "
                            f"Unclassified: {n_none}"
                        )
                        if n_a == 0 and n_b == 0:
                            self._log(
                                "  ⚠ All neurons are unclassified — check that the\n"
                                "    sub-region polygons cover the neurons on the canvas."
                            )
                else:
                    _all_region_labels.append(np.full(stims1.shape[0], -1, dtype=int))

        if not (self.do_analysis.get() and _all_stims1):
            self._log("\nDone.")
            return

        self._log("\nClassifying responders across all animals …")
        all_stims1 = np.vstack(_all_stims1)
        all_stims2 = np.vstack(_all_stims2)
        z_ids_all  = np.concatenate(_z_ids)

        resp1, resp2, nums, z_ids_sel = get_resp1_resp2(
            all_stims1, all_stims2, z_ids_all,
            stim_onset_idx=stim_onset_idx,
            threshold=threshold,
        )

        n_total = sum(nums)
        self._log(
            f"Stim-1 only: {nums[0]}   Both: {nums[1]}   Stim-2 only: {nums[2]}   "
            f"Total responsive: {n_total} / {all_stims1.shape[0]}"
        )

        # save results — use custom analysis output folder if the user specified one
        out_dir = str(Path(p["output"]) / p["subject"])
        results_parent = p["analysis_out"] if p["analysis_out"] else out_dir
        results_dir = self._save_results(
            results_parent, resp1, resp2, nums, z_ids_sel, p,
            stim_onset_idx, d["ses_f"])
        self._log(f"Results saved to  {results_dir}")

        self._log("Opening figures …")
        self.after(0, lambda: show_plots(
            resp1, resp2, nums, stim_onset_idx, d["ses_f"],
            fp, pre_s, stim_s, results_dir))

        for _lbl, _sd in _all_spatial:
            self.after(0, lambda lbl=_lbl, sd=_sd: show_spatial_response_map(
                lbl, sd, threshold, results_dir))

        # optional sub-region analysis
        if self.do_subregion.get() and _all_region_labels:
            region_labels_all = np.concatenate(_all_region_labels)
            region_results = {}
            for reg_idx, reg_name in [(0, "Region A"), (1, "Region B")]:
                mask = region_labels_all == reg_idx
                n_total_r = int(mask.sum())
                if n_total_r == 0:
                    continue
                r1_r, r2_r, nums_r, _ = get_resp1_resp2(
                    all_stims1[mask], all_stims2[mask],
                    z_ids_all[mask],
                    stim_onset_idx=stim_onset_idx,
                    threshold=threshold,
                )
                region_results[reg_name] = {
                    'resp1': r1_r, 'resp2': r2_r,
                    'nums': nums_r, 'n_total': n_total_r,
                }
                self._log(
                    f"{reg_name} — total: {n_total_r}  "
                    f"stim1-only: {nums_r[0]}  both: {nums_r[1]}  stim2-only: {nums_r[2]}"
                )
            if region_results:
                self._log(
                    f"  Opening sub-region figures for: {', '.join(region_results.keys())}")
                self.after(0, lambda rr=region_results: show_region_plots(
                    rr, stim_onset_idx, d["ses_f"], fp, pre_s, stim_s, results_dir))
            else:
                self._log(
                    "  ⚠ Sub-region plots skipped — no neurons were classified into any region.")

        self._log("Done.")

    # ── save results ───────────────────────────────────────────────────────

    def _save_results(self, out_dir: str, resp1, resp2, nums, z_ids_sel,
                      params: dict, stim_onset_idx: int, ses_f: int) -> str:
        import yaml

        results_dir = Path(out_dir) / "analysis"
        results_dir.mkdir(parents=True, exist_ok=True)

        np.save(results_dir / "resp1.npy",    resp1)
        np.save(results_dir / "resp2.npy",    resp2)
        np.save(results_dir / "nums.npy",     np.array(nums))
        np.save(results_dir / "z_ids_stim1.npy", z_ids_sel[0])
        np.save(results_dir / "z_ids_both.npy",  z_ids_sel[1])
        np.save(results_dir / "z_ids_stim2.npy", z_ids_sel[2])

        saved_params = dict(
            frame_period   = params["frame_period"],
            pre_discard_s  = params["pre_discard_s"],
            baseline_s     = params["baseline_s"],
            stim_s         = params["stim_s"],
            threshold      = params["threshold"],
            stim_onset_idx = stim_onset_idx,
            ses_f          = ses_f,
            z_planes       = params["z_planes"],
            n_stim1_only   = int(nums[0]),
            n_both         = int(nums[1]),
            n_stim2_only   = int(nums[2]),
            n_total_neurons= int(sum(nums)),
        )
        with open(results_dir / "params.yaml", "w") as f:
            yaml.safe_dump(saved_params, f, default_flow_style=False)

        return str(results_dir)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
