"""
2P Calcium Imaging Pipeline — GUI
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


def _canvas_to_image(cx: float, cy: float,
                     scale_x: float, scale_y: float) -> tuple:
    """Convert canvas pixel (cx, cy) to image pixel (ix, iy) using per-axis scales."""
    return int(cx / scale_x), int(cy / scale_y)


# ── ROI curation window ───────────────────────────────────────────────────────

class ROIEditorWindow(ctk.CTkToplevel):
    """Integrated ROI curation: remove, add, and region-exclusion in one window."""

    _REMOVE = "remove"
    _ADD    = "add"
    _REGION = "region"

    _INSTRUCTIONS = {
        "remove": (
            "RIGHT-CLICK on a colored patch to remove that neuron.\n\n"
            "The patch disappears immediately.\n\n"
            "Use Undo to restore the last change."
        ),
        "add": (
            "LEFT-CLICK and DRAG to paint a new neuron.\n\n"
            "Fill the entire soma — do not just trace the outline.\n\n"
            "Release the mouse to confirm."
        ),
        "region": (
            "LEFT-CLICK to place polygon vertices around the region to KEEP "
            "(e.g. draw around the DVC).\n\n"
            "RIGHT-CLICK to close the polygon.\n\n"
            "Neurons whose centres fall outside are removed."
        ),
    }

    def __init__(self, parent, z, roi_img_bkg, roi_img_mask, roi_masks, mc_corr_file, on_finish,
                 display_settings=None):
        super().__init__(parent)
        self.title(f"ROI Curation — {z}")
        self.resizable(True, True)
        self.lift()
        self.focus_force()

        self._on_finish = on_finish
        self._roi_masks = roi_masks.copy()
        self._roi_bkg   = roi_img_bkg.copy()
        self._roi_msk   = roi_img_mask.copy()

        h, w = roi_img_bkg.shape[:2]
        self._ih, self._iw = h, w
        # initial scales — updated when window maximises and Configure fires
        _s = 500 / max(h, w)
        self._scale_x = _s
        self._scale_y = _s
        self._dh = int(h * _s)
        self._dw = int(w * _s)

        self._mode      = None
        self._new_mask  = None
        self._add_col   = None
        self._poly_pts  = []
        self._poly_ids  = []
        self._history   = []
        self._img_id    = None
        self._ref_id    = None

        ds = display_settings or {}
        self._gamma_var  = tk.DoubleVar(value=ds.get("gamma",   1.36))
        self._lo_var     = tk.DoubleVar(value=ds.get("lo_pct",  26.7))
        self._hi_var     = tk.DoubleVar(value=ds.get("hi_pct",  98.8))

        self._build_ui()
        self._set_mode(self._REMOVE)
        self.after(50, lambda: self.state('zoomed'))  # open maximised

    # ── build ─────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = ctk.CTkFrame(self)
        outer.pack(fill="both", expand=True, padx=10, pady=10)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=1)  # canvas area expands
        outer.columnconfigure(1, weight=0)  # panel stays fixed

        # ── canvas area (left two thirds) ─────────────────────────────────────
        canvas_area = ctk.CTkFrame(outer)
        canvas_area.grid(row=0, column=0, padx=(0, 10), sticky="nsew")
        canvas_area.rowconfigure(1, weight=1)
        canvas_area.columnconfigure(0, weight=1)
        canvas_area.columnconfigure(1, weight=1)

        ctk.CTkLabel(canvas_area, text="Reference  (no ROIs)",
                     font=ctk.CTkFont(size=11)).grid(row=0, column=0, pady=(4, 2))
        ctk.CTkLabel(canvas_area, text="ROIs  (interactive)",
                     font=ctk.CTkFont(size=11)).grid(row=0, column=1, pady=(4, 2))

        self._canvas_ref = tk.Canvas(canvas_area, bg="black", highlightthickness=0)
        self._canvas_ref.grid(row=1, column=0, sticky="nsew", padx=(0, 4))

        self._canvas = tk.Canvas(canvas_area, bg="black", highlightthickness=0)
        self._canvas.grid(row=1, column=1, sticky="nsew")
        self._canvas.bind("<Configure>",       self._on_canvas_resize)
        self._canvas.bind("<Button-3>",        self._on_right)
        self._canvas.bind("<Button-1>",        self._on_left_dn)
        self._canvas.bind("<B1-Motion>",       self._on_left_mv)
        self._canvas.bind("<ButtonRelease-1>", self._on_left_up)

        panel = ctk.CTkFrame(outer, width=220)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)

        ctk.CTkLabel(panel, text="Mode",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(12, 6), padx=10)

        self._mode_btns = {}
        for key, lbl in [(self._REMOVE, "Remove Neurons"),
                          (self._ADD,    "Add Neuron"),
                          (self._REGION, "Exclude Region")]:
            b = ctk.CTkButton(panel, text=lbl, width=190,
                               command=lambda k=key: self._set_mode(k))
            b.pack(pady=3, padx=10)
            self._mode_btns[key] = b

        ctk.CTkFrame(panel, height=2, fg_color="gray40").pack(fill="x", padx=10, pady=10)

        self._instr = ctk.CTkLabel(panel, text="", wraplength=200,
                                    justify="left", anchor="nw")
        self._instr.pack(padx=10, fill="x")

        ctk.CTkFrame(panel, height=2, fg_color="gray40").pack(fill="x", padx=10, pady=10)

        self._status = ctk.CTkLabel(panel, text="", wraplength=200,
                                     text_color="#aaaaaa", anchor="w")
        self._status.pack(padx=10, fill="x")

        # ── display settings sliders ──────────────────────────────────────────
        ctk.CTkFrame(panel, height=2, fg_color="gray40").pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(panel, text="Display Settings",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(0, 4))

        def _make_slider(label, var, from_, to, steps):
            row = ctk.CTkFrame(panel, fg_color="transparent")
            row.pack(fill="x", padx=10, pady=2)
            val_lbl = ctk.CTkLabel(row, width=38, anchor="e",
                                   text=f"{var.get():.2f}")
            def _on_change(v, lbl=val_lbl, variable=var):
                variable.set(float(v))
                lbl.configure(text=f"{float(v):.2f}")
                self._refresh_canvas()
            ctk.CTkLabel(row, text=label, width=82, anchor="w").pack(side="left")
            ctk.CTkSlider(row, from_=from_, to=to, number_of_steps=steps,
                          variable=var, command=_on_change,
                          width=80).pack(side="left", padx=4)
            val_lbl.pack(side="left")

        _make_slider("Gamma",      self._gamma_var, 0.2, 1.5, 130)
        _make_slider("Dark clip%", self._lo_var,    0.0, 30.0, 300)
        _make_slider("Bright clip%", self._hi_var,  70.0, 100.0, 300)
        # ─────────────────────────────────────────────────────────────────────

        ctk.CTkButton(panel, text="Finish ✓", width=190,
                       fg_color="#2d6a2d", hover_color="#1e4d1e",
                       command=self._do_finish).pack(side="bottom", padx=10, pady=4)
        ctk.CTkButton(panel, text="Undo", width=190,
                       command=self._undo).pack(side="bottom", padx=10, pady=4)

        self.protocol("WM_DELETE_WINDOW", self._do_finish)

    # ── mode ──────────────────────────────────────────────────────────────────

    def _set_mode(self, mode):
        self._mode    = mode
        self._new_mask = None
        self._add_col  = None
        for pid in self._poly_ids:
            self._canvas.delete(pid)
        self._poly_pts = []
        self._poly_ids = []

        for k, btn in self._mode_btns.items():
            btn.configure(fg_color="#1a5276" if k == mode else ("#3b8ed0", "#1f6aa5"))
        self._instr.configure(text=self._INSTRUCTIONS.get(mode, ""))
        self._status.configure(text="")

    # ── canvas ────────────────────────────────────────────────────────────────

    def _on_canvas_resize(self, event):
        if hasattr(self, '_resize_job'):
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._apply_resize)

    def _apply_resize(self):
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w > 1 and h > 1:
            self._dw = w
            self._dh = h
            self._scale_x = w / self._iw
            self._scale_y = h / self._ih
            self._refresh_canvas()

    def _bright_bkg(self) -> np.ndarray:
        """Contrast-stretch + gamma lift using slider-controlled parameters."""
        bkg   = self._roi_bkg.astype(np.float32)
        lo    = np.percentile(bkg, self._lo_var.get())
        hi    = np.percentile(bkg, self._hi_var.get())
        gamma = self._gamma_var.get()
        if hi > lo:
            bkg = np.clip((bkg - lo) / (hi - lo), 0, 1)
        else:
            bkg = np.zeros_like(bkg)
        bkg = np.power(bkg, gamma) * 255
        return np.clip(bkg, 0, 255).astype(np.uint8)

    def _refresh_canvas(self):
        from PIL import Image as PILImage, ImageTk
        bkg_bright = self._bright_bkg()
        dw, dh = self._dw, self._dh

        # reference: bright background only
        pil_ref = PILImage.fromarray(bkg_bright).resize((dw, dh), PILImage.BILINEAR)
        self._tk_ref = ImageTk.PhotoImage(pil_ref)
        if self._ref_id is None:
            self._ref_id = self._canvas_ref.create_image(0, 0, anchor="nw", image=self._tk_ref)
        else:
            self._canvas_ref.itemconfig(self._ref_id, image=self._tk_ref)

        # interactive: bright background + ROI colour overlay
        combined = np.clip(
            bkg_bright.astype(np.int16) + self._roi_msk.astype(np.int16), 0, 255
        ).astype(np.uint8)
        pil_roi = PILImage.fromarray(combined).resize((dw, dh), PILImage.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(pil_roi)
        if self._img_id is None:
            self._img_id = self._canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        else:
            self._canvas.itemconfig(self._img_id, image=self._tk_img)
        self._canvas.tag_lower(self._img_id)

    def _c2i(self, cx, cy):
        return _canvas_to_image(cx, cy, self._scale_x, self._scale_y)

    def _flat(self, ix, iy):
        return ix * self._ih + iy

    # ── remove ────────────────────────────────────────────────────────────────

    def _on_right(self, event):
        if self._mode == self._REMOVE:
            ix, iy = self._c2i(event.x, event.y)
            if not (0 <= ix < self._iw and 0 <= iy < self._ih):
                return
            flat = self._flat(ix, iy)
            if flat >= self._roi_masks.shape[0]:
                return
            row = self._roi_masks[flat]
            if row.sum() != 1:
                return
            self._push_history()
            nidx = int(np.argmax(row))
            pxs = self._roi_masks[:, nidx].reshape((self._ih, self._iw), order='F')
            self._roi_msk[pxs] = 0
            self._roi_masks = np.delete(self._roi_masks, nidx, 1)
            self._status.configure(text=f"Removed. Total: {self._roi_masks.shape[1]}")
            self._refresh_canvas()
        elif self._mode == self._REGION:
            self._close_polygon()

    # ── add ───────────────────────────────────────────────────────────────────

    def _on_left_dn(self, event):
        if self._mode == self._ADD:
            self._new_mask = np.zeros((self._ih, self._iw), dtype=bool)
            self._add_col  = tuple(np.random.randint(40, 210, 3).tolist())
            self._paint(event.x, event.y)
        elif self._mode == self._REGION:
            self._add_poly_pt(event.x, event.y)

    def _on_left_mv(self, event):
        if self._mode == self._ADD and self._new_mask is not None:
            self._paint(event.x, event.y)

    def _paint(self, cx, cy):
        ix, iy = self._c2i(cx, cy)
        br = 5
        for dx in range(-br, br + 1):
            for dy in range(-br, br + 1):
                px, py = ix + dx, iy + dy
                if 0 <= px < self._iw and 0 <= py < self._ih:
                    self._new_mask[py, px] = True
        r = max(2, int(br * min(self._scale_x, self._scale_y)))
        col = "#{:02x}{:02x}{:02x}".format(*self._add_col)
        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                  fill=col, outline=col, tags="paint")

    def _on_left_up(self, event):
        if self._mode != self._ADD or self._new_mask is None:
            return
        if not self._new_mask.any():
            self._new_mask = None
            return
        confirmed = messagebox.askyesno(
            "Add neuron", "Add this painted region as a new neuron?", parent=self)
        self._canvas.delete("paint")
        if confirmed:
            self._push_history()
            flat_col = self._new_mask.flatten('F').reshape(-1, 1)
            self._roi_masks = np.concatenate([self._roi_masks, flat_col], axis=1)
            self._roi_msk[self._new_mask] = np.array(self._add_col, dtype=np.uint8)
            self._status.configure(text=f"Added. Total: {self._roi_masks.shape[1]}")
        self._new_mask = None
        self._add_col  = None
        self._refresh_canvas()

    # ── region exclusion ──────────────────────────────────────────────────────

    def _add_poly_pt(self, cx, cy):
        dot = self._canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                        fill="yellow", outline="yellow", tags="poly")
        self._poly_ids.append(dot)
        if self._poly_pts:
            px, py = self._poly_pts[-1]
            ln = self._canvas.create_line(px, py, cx, cy,
                                           fill="yellow", width=2, tags="poly")
            self._poly_ids.append(ln)
        self._poly_pts.append((cx, cy))
        self._status.configure(
            text=f"{len(self._poly_pts)} point(s). Right-click to close.")

    def _close_polygon(self):
        if len(self._poly_pts) < 3:
            self._status.configure(text="Need at least 3 points first.")
            return
        px, py = self._poly_pts[-1]
        fx, fy = self._poly_pts[0]
        self._poly_ids.append(
            self._canvas.create_line(px, py, fx, fy,
                                      fill="yellow", width=2, tags="poly"))

        from PIL import Image as PILImage, ImageDraw
        poly_img = [_canvas_to_image(cx, cy, self._scale_x, self._scale_y)
                    for cx, cy in self._poly_pts]
        pmask = PILImage.new('L', (self._iw, self._ih), 0)
        ImageDraw.Draw(pmask).polygon(poly_img, fill=255)
        inside = np.array(pmask, dtype=bool)

        n = self._roi_masks.shape[1]
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            pxs = self._roi_masks[:, i].reshape((self._ih, self._iw), order='F')
            ys, xs = np.where(pxs)
            if len(xs) == 0:
                keep[i] = False
                continue
            keep[i] = inside[int(ys.mean()), int(xs.mean())]

        removed = int((~keep).sum())
        if removed > 0 and messagebox.askyesno(
                "Exclude region",
                f"Remove {removed} neuron(s) outside the polygon?",
                parent=self):
            self._push_history()
            for i in np.where(~keep)[0]:
                pxs = self._roi_masks[:, i].reshape((self._ih, self._iw), order='F')
                self._roi_msk[pxs] = 0
            self._roi_masks = self._roi_masks[:, keep]
            self._status.configure(
                text=f"Excluded {removed}. Total: {self._roi_masks.shape[1]}")
            self._refresh_canvas()
        elif removed == 0:
            self._status.configure(text="All neurons are inside the polygon.")

        for pid in self._poly_ids:
            self._canvas.delete(pid)
        self._poly_ids = []
        self._poly_pts = []

    # ── undo / finish ─────────────────────────────────────────────────────────

    def _push_history(self):
        self._history.append((self._roi_masks.copy(), self._roi_msk.copy()))
        if len(self._history) > 20:
            self._history.pop(0)

    def _undo(self):
        if not self._history:
            self._status.configure(text="Nothing to undo.")
            return
        self._roi_masks, self._roi_msk = self._history.pop()
        self._status.configure(text=f"Undone. Total: {self._roi_masks.shape[1]}")
        self._refresh_canvas()

    def _do_finish(self):
        clean = self._roi_masks[:, ~(self._roi_masks.sum(axis=0) == 0)]
        settings = {
            "gamma":  round(self._gamma_var.get(), 3),
            "lo_pct": round(self._lo_var.get(),    3),
            "hi_pct": round(self._hi_var.get(),    3),
        }
        self._on_finish(clean, self._roi_bkg, self._roi_msk, settings)
        self.destroy()


# ── main window ───────────────────────────────────────────────────────────────

class PipelineGUI(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("2P Calcium Imaging Pipeline")
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

        result_holder = [None]
        done = threading.Event()

        def _show():
            def _on_finish(masks, bkg, mask_img, settings):
                result_holder[0] = (masks, bkg, mask_img, settings)
                done.set()
            ROIEditorWindow(self, z, roi_img_bkg, roi_img_mask, roi_masks, mc_corr_file,
                            _on_finish, display_settings=self._display_settings)

        self.after(0, _show)
        done.wait()

        new_masks, new_bkg, new_mask_img, settings = result_holder[0]

        # Persist settings for next z-plane and save to project folder
        self._display_settings = settings
        with open(ds_path, 'w') as f:
            yaml.dump(settings, f)

        roi_masks_file = Path(output_dir) / 'concat_roi-masks.npy'
        np.save(roi_masks_file, new_masks)
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
                               _get_provenance)
        from pipeline_funcs import get_stims1_stims2, get_resp1_resp2

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
                roi_fn = self._roi_editor_for_pipeline
                for z in p["z_planes"]:
                    self._log(
                        f"  Source extraction ({z}) — "
                        "ROI editor will open in a popup window …")
                    provenance = source_extraction(
                        provenance, None, z, None, roi_editor_fn=roi_fn)
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

        # save results — use custom analysis output folder if the user specified one
        out_dir = str(Path(p["output"]) / p["subject"])
        results_parent = p["analysis_out"] if p["analysis_out"] else out_dir
        results_dir = self._save_results(
            results_parent, resp1, resp2, nums, z_ids_sel, p,
            stim_onset_idx, d["ses_f"])
        self._log(f"Results saved to  {results_dir}")

        self._log("Opening figures …")
        self.after(0, lambda: self._show_plots(
            resp1, resp2, nums, stim_onset_idx, d["ses_f"],
            fp, pre_s, results_dir))
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

    # ── plots ──────────────────────────────────────────────────────────────

    def _show_plots(self, resp1, resp2, nums,
                    stim_onset_idx: int, ses_f: int,
                    fp: float, pre_s: float, results_dir: str):
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
        fig.savefig(Path(results_dir) / "heatmap.png", dpi=150, bbox_inches="tight")

        # bar chart
        fig2, ax = plt.subplots(figsize=(4, 4))
        pct = [v / n * 100 for v in nums] if n else [0, 0, 0]
        ax.bar(["Stim 1\nonly", "Both", "Stim 2\nonly"], pct,
               color=["#4fa1ca", "#bb70b6", "#110979"])
        ax.set_ylabel("% responsive neurons")
        ax.spines[["top", "right"]].set_visible(False)
        fig2.tight_layout()
        fig2.savefig(Path(results_dir) / "breakdown.png", dpi=150, bbox_inches="tight")

        plt.show()


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    app = PipelineGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
