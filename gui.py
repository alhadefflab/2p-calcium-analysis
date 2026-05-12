"""
2P Calcium Imaging Pipeline — GUI
Run:  python gui.py
Requires: pip install customtkinter
"""
import sys
import threading
import traceback
from pathlib import Path

import customtkinter as ctk
from tkinter import filedialog, messagebox
import numpy as np

ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")


# ── helpers ──────────────────────────────────────────────────────────────────

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
                row=row_idx, column=1, padx=(0, 4), pady=(4 if row_idx == 0 else 2, 4 if row_idx == 1 else 2), sticky="w")
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


# ── main window ───────────────────────────────────────────────────────────────

class PipelineGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("2P Calcium Imaging Pipeline")
        self.geometry("860x700")
        self.minsize(720, 580)
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

        top = ctk.CTkFrame(tab, fg_color="transparent")
        top.pack(fill="x", padx=14, pady=(14, 6))

        ctk.CTkLabel(top, text="Subject ID:", width=130, anchor="w").grid(
            row=0, column=0, pady=5, sticky="w")
        self.subject_var = ctk.StringVar()
        ctk.CTkEntry(top, textvariable=self.subject_var, width=180,
                     placeholder_text="e.g. ZH511").grid(
            row=0, column=1, padx=8, sticky="w")

        ctk.CTkLabel(top, text="Output folder:", width=130, anchor="w").grid(
            row=1, column=0, pady=5, sticky="w")
        self.output_var = ctk.StringVar()
        ctk.CTkEntry(top, textvariable=self.output_var, width=320,
                     placeholder_text="Where to save results").grid(
            row=1, column=1, padx=8, sticky="w")
        ctk.CTkButton(top, text="Browse", width=80,
                      command=self._browse_output).grid(row=1, column=2, padx=4)

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

        self.animals_scroll = ctk.CTkScrollableFrame(tab, height=250)
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
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

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
        # Use the first session folder of the first animal to detect z-planes
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
                f"  Per session : {se} frames  "
                f"({pf} discard + {bf} baseline + {sf} stim)\n"
                f"  Full file   : {2 * se} frames  (session 1 + session 2)\n\n"
                f"  Session 1   discard  0–{pf-1}   "
                f"baseline  {pf}–{pf+bf-1}   "
                f"stimulus  {pf+bf}–{se-1}\n"
                f"  Session 2   discard  {se}–{se+pf-1}   "
                f"baseline  {se+pf}–{se+pf+bf-1}   "
                f"stimulus  {se+pf+bf}–{2*se-1}"
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

        self.do_analysis = ctk.CTkCheckBox(
            stages,
            text="Stimulus response analysis  —  fast, re-run this after changing timing or threshold")
        self.do_analysis.pack(anchor="w", padx=16, pady=(4, 10))
        self.do_analysis.select()

        ctk.CTkLabel(tab,
                     text="Tip: to iterate on timing parameters, uncheck the first two and only re-run analysis.",
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
            errs.append("Output folder is required.")

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
            self.after(0, lambda: self.run_btn.configure(
                state="normal", text="▶   Run"))

    def _pipeline_body(self, p: dict):
        from pipeline import (init, load_data, affine_motion_correction,
                               rigid_motion_correction, source_extraction,
                               _get_provenance)
        from pipeline_funcs import get_stims1_stims2, get_resp1_resp2

        fp, pre_s, base_s, stim_s = (p["frame_period"], p["pre_discard_s"],
                                      p["baseline_s"],   p["stim_s"])
        threshold = p["threshold"]

        d = _frame_layout(fp, pre_s, base_s, stim_s)
        stim_onset_idx = d["base_f"]  # index inside stims1/stims2 where stimulus starts

        self._log(
            f"Frame layout: {d['pre_f']} discard  +  {d['base_f']} baseline  "
            f"+  {d['stim_f']} stim  =  {d['ses_f']} per session  "
            f"({2 * d['ses_f']} total)"
        )

        _all_stims1, _all_stims2, _z_ids = [], [], []

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
                for z in p["z_planes"]:
                    self._log(
                        f"  Source extraction ({z}) — "
                        "watch for the ROI editor window in the background …")
                    provenance = source_extraction(
                        provenance, None, z, None)
            else:
                self._log("  Skipping CNMF — using saved results.")

            if self.do_analysis.get():
                self._log("  Computing stimulus responses …")
                stims1, stims2, z_ids = get_stims1_stims2(
                    provenance,
                    frame_period=fp,
                    pre_discard_s=pre_s,
                    baseline_s=base_s,
                    stim_s=stim_s,
                )
                _all_stims1.append(stims1)
                _all_stims2.append(stims2)
                _z_ids.append(z_ids)

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
        self._log("Opening figures …")
        self.after(0, lambda: self._show_plots(
            resp1, resp2, nums, stim_onset_idx, d["ses_f"], fp, pre_s))
        self._log("Done.")

    # ── plots ──────────────────────────────────────────────────────────────

    def _show_plots(self, resp1, resp2, nums,
                    stim_onset_idx: int, ses_f: int, fp: float, pre_s: float):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec

        # heatmap
        fig = plt.figure(constrained_layout=True, figsize=(10, 5))
        gs  = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 10, 10])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])

        n = resp1.shape[0]
        ax0.set_xlim(0, 1); ax0.set_ylim(0, n); ax0.invert_yaxis()
        ax0.xaxis.set_visible(False)
        ax0.axhspan(0,               nums[0],             facecolor="#4fa1ca")
        ax0.axhspan(nums[0],         nums[0] + nums[1],   facecolor="#bb70b6")
        ax0.axhspan(nums[0]+nums[1], n,                   facecolor="#110979")
        ax0.set_yticks([0, max(n - 1, 0)], [1, n])
        ax0.set_ylabel("Neuron #")

        im = ax1.imshow(resp1, aspect="auto", vmin=0, vmax=8)
        ax2.imshow(resp2, aspect="auto", vmin=0, vmax=8)

        # x-tick positions & labels in seconds
        pre_f = round(pre_s / fp)
        tick_frames = [0, stim_onset_idx, resp1.shape[1]]
        tick_labels = [
            f"{round(pre_s)}",
            f"{round((pre_f + stim_onset_idx) * fp)}",
            f"{round((pre_f + resp1.shape[1]) * fp)}",
        ]
        for ax, title in ((ax1, "Stimulus 1"), (ax2, "Stimulus 2")):
            ax.axvline(stim_onset_idx, color="w", lw=0.8, ls="--")
            ax.set_title(title)
            ax.yaxis.set_visible(False)
            ax.set_xlabel("Time (s from session start)")
            ax.set_xticks(tick_frames, tick_labels)
        fig.colorbar(im, ax=[ax1, ax2], shrink=0.5, label="z-score")
        fig.suptitle("Responder heatmap")

        # bar chart
        fig2, ax = plt.subplots(figsize=(4, 4))
        pct = [v / n * 100 for v in nums] if n else [0, 0, 0]
        ax.bar(["Stim 1\nonly", "Both", "Stim 2\nonly"], pct,
               color=["#4fa1ca", "#bb70b6", "#110979"])
        ax.set_ylabel("% responsive neurons")
        ax.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        fig2.suptitle("Responder breakdown", y=1.02)

        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
