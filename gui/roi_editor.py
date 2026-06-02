import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
import numpy as np


def _canvas_to_image(cx: float, cy: float,
                     scale_x: float, scale_y: float) -> tuple:
    """Convert canvas pixel (cx, cy) to image pixel (ix, iy) using per-axis scales."""
    return int(cx / scale_x), int(cy / scale_y)


# ── ROI curation window ───────────────────────────────────────────────────────

class ROIEditorWindow(ctk.CTkToplevel):
    """Integrated ROI curation: remove, add, and region-exclusion in one window."""

    _REMOVE    = "remove"
    _ADD       = "add"
    _REGION    = "region"
    _SUBREGION = "subregion"

    _INSTRUCTIONS = {
        "remove": (
            "RIGHT-CLICK on a colored patch to remove that neuron.\n\n"
            "The patch disappears immediately.\n\n"
            "Use Undo to restore the last change."
        ),
        "add": (
            "LEFT-CLICK and DRAG to trace the outline of a neuron.\n\n"
            "Release the mouse to confirm — the interior fills automatically."
        ),
        "region": (
            "LEFT-CLICK to place polygon vertices around the region to KEEP "
            "(e.g. draw around the DVC).\n\n"
            "RIGHT-CLICK to close the polygon.\n\n"
            "Neurons whose centres fall outside are removed."
        ),
        "subregion": (
            "Define two sub-regions (A=yellow, B=cyan).\n\n"
            "LEFT-CLICK to place vertices. RIGHT-CLICK to confirm each region.\n\n"
            "After both are drawn, use Snap Boundaries if needed, then Finish."
        ),
    }

    def __init__(self, parent, z, roi_img_bkg, roi_img_mask, roi_masks, mc_corr_file, on_finish,
                 display_settings=None, mc_img_bkg=None):
        super().__init__(parent)
        self.title(f"ROI Curation — {z}")
        self.resizable(True, True)
        self.lift()
        self.focus_force()

        self._on_finish = on_finish
        self._roi_masks = roi_masks.copy()
        self._roi_bkg   = roi_img_bkg.copy()
        self._roi_msk   = roi_img_mask.copy()
        self._mc_bkg    = mc_img_bkg  # structural channel for sub-region orientation

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

        # sub-region state
        self._sreg_polys      = [[], []]   # canvas-coord vertices per region
        self._sreg_canvas_ids = [[], []]   # canvas item IDs per region
        self._sreg_masks      = [None, None]  # bool arrays (h, w) per region
        self._sreg_cur        = 0          # which region is being drawn (0=A, 1=B)

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

        self._lbl_ref = ctk.CTkLabel(canvas_area, text="Reference  (no ROIs)",
                                     font=ctk.CTkFont(size=11))
        self._lbl_ref.grid(row=0, column=0, pady=(4, 2))
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
        for key, lbl in [(self._REMOVE,    "Remove Neurons"),
                          (self._ADD,       "Add Neuron"),
                          (self._REGION,    "Exclude Region"),
                          (self._SUBREGION, "Define Sub-Regions")]:
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
        self._snap_btn = ctk.CTkButton(panel, text="Snap Boundaries", width=190,
                                        state="disabled",
                                        command=self._sreg_snap)
        self._snap_btn.pack(side="bottom", padx=10, pady=4)
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

        if mode == self._SUBREGION:
            # clear any in-progress polygon drawing for the current region
            for pid in self._sreg_canvas_ids[self._sreg_cur]:
                self._canvas.delete(pid)
            self._sreg_canvas_ids[self._sreg_cur] = []
            self._sreg_polys[self._sreg_cur] = []
            # keep _sreg_masks intact so confirmed regions survive mode switches
            ref_lbl = ("Structural channel  (anatomy / MC)"
                       if self._mc_bkg is not None else "Reference  (no ROIs)")
        else:
            ref_lbl = "Reference  (no ROIs)"

        if hasattr(self, '_lbl_ref'):
            self._lbl_ref.configure(text=ref_lbl)

        for k, btn in self._mode_btns.items():
            btn.configure(fg_color="#1a5276" if k == mode else ("#3b8ed0", "#1f6aa5"))
        self._instr.configure(text=self._INSTRUCTIONS.get(mode, ""))
        self._status.configure(text="")
        self._refresh_canvas()

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

        # Interactive canvas always uses the functional channel so ROIs stay
        # visually aligned with the background they were detected on.
        func_bright = self._bright_bkg()
        dw, dh = self._dw, self._dh

        # build region overlay (yellow=A, cyan=B) for sub-region mode
        sreg_overlay = np.zeros((self._ih, self._iw, 3), dtype=np.int16)
        if self._mode == self._SUBREGION:
            colors = [(80, 80, 0), (0, 60, 80)]
            for mask, col in zip(self._sreg_masks, colors):
                if mask is not None:
                    sreg_overlay[mask] = col

        # ── reference canvas ──────────────────────────────────────────────────
        # In sub-region mode show the structural (MC) channel so the user can
        # orient anatomically.  Fall back to functional if MC is unavailable.
        if self._mode == self._SUBREGION and self._mc_bkg is not None:
            mc_f  = self._mc_bkg.astype(np.float32)
            lo    = np.percentile(mc_f, self._lo_var.get())
            hi    = np.percentile(mc_f, self._hi_var.get())
            gamma = self._gamma_var.get()
            if hi > lo:
                mc_f = np.clip((mc_f - lo) / (hi - lo), 0, 1)
            else:
                mc_f = np.zeros_like(mc_f)
            ref_bright = np.clip(np.power(mc_f, gamma) * 255, 0, 255).astype(np.uint8)
        else:
            ref_bright = func_bright

        ref_base = np.clip(
            ref_bright.astype(np.int16) + sreg_overlay, 0, 255
        ).astype(np.uint8)
        pil_ref = PILImage.fromarray(ref_base).resize((dw, dh), PILImage.BILINEAR)
        self._tk_ref = ImageTk.PhotoImage(pil_ref)
        if self._ref_id is None:
            self._ref_id = self._canvas_ref.create_image(0, 0, anchor="nw", image=self._tk_ref)
        else:
            self._canvas_ref.itemconfig(self._ref_id, image=self._tk_ref)

        # ── interactive canvas ────────────────────────────────────────────────
        # Always functional channel + ROI overlay so ROIs remain correctly placed.
        combined = np.clip(
            func_bright.astype(np.int16) + self._roi_msk.astype(np.int16) + sreg_overlay,
            0, 255
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
        elif self._mode == self._SUBREGION:
            self._sreg_close_region()

    # ── add ───────────────────────────────────────────────────────────────────

    def _on_left_dn(self, event):
        if self._mode == self._ADD:
            self._new_mask = np.zeros((self._ih, self._iw), dtype=bool)
            self._add_col  = tuple(np.random.randint(40, 210, 3).tolist())
            self._paint(event.x, event.y)
        elif self._mode == self._REGION:
            self._add_poly_pt(event.x, event.y)
        elif self._mode == self._SUBREGION and self._sreg_cur <= 1:
            self._sreg_add_pt(event.x, event.y)

    def _on_left_mv(self, event):
        if self._mode == self._ADD and self._new_mask is not None:
            self._paint(event.x, event.y)

    def _paint(self, cx, cy):
        ix, iy = self._c2i(cx, cy)
        br = 2
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
            from scipy.ndimage import binary_fill_holes
            self._new_mask = binary_fill_holes(self._new_mask)
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

    # ── sub-region definition ─────────────────────────────────────────────────

    def _sreg_add_pt(self, cx, cy):
        ri = self._sreg_cur
        col = "yellow" if ri == 0 else "cyan"
        dot = self._canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                        fill=col, outline=col, tags="sreg")
        self._sreg_canvas_ids[ri].append(dot)
        if self._sreg_polys[ri]:
            px, py = self._sreg_polys[ri][-1]
            ln = self._canvas.create_line(px, py, cx, cy,
                                           fill=col, width=2, tags="sreg")
            self._sreg_canvas_ids[ri].append(ln)
        self._sreg_polys[ri].append((cx, cy))
        region_name = "Region A" if ri == 0 else "Region B"
        self._status.configure(
            text=f"{region_name}: {len(self._sreg_polys[ri])} point(s). Right-click to confirm.")

    def _sreg_close_region(self):
        ri = self._sreg_cur
        pts = self._sreg_polys[ri]
        if len(pts) < 3:
            self._status.configure(text="Need at least 3 points first.")
            return

        col = "yellow" if ri == 0 else "cyan"
        px, py = pts[-1]
        fx, fy = pts[0]
        close_ln = self._canvas.create_line(px, py, fx, fy,
                                             fill=col, width=2, tags="sreg")
        self._sreg_canvas_ids[ri].append(close_ln)

        from PIL import Image as PILImage, ImageDraw
        img_pts = [_canvas_to_image(cx, cy, self._scale_x, self._scale_y) for cx, cy in pts]
        pmask = PILImage.new('L', (self._iw, self._ih), 0)
        ImageDraw.Draw(pmask).polygon(img_pts, fill=255)
        self._sreg_masks[ri] = np.array(pmask, dtype=bool)

        self._sreg_cur += 1
        if self._sreg_cur == 1:
            self._status.configure(text="Region A confirmed. Now draw Region B (cyan).")
        else:
            self._status.configure(
                text="Both regions defined. Use Snap Boundaries if edges are close, then Finish.")
            if hasattr(self, '_snap_btn'):
                self._snap_btn.configure(state="normal")
        self._refresh_canvas()

    def _sreg_snap(self, threshold_px: int = 20):
        """Snap vertices of Region A and Region B that are within threshold_px of
        each other to their exact midpoint, closing any tiny gap at the shared border.
        Both region masks are recomputed after snapping."""
        if self._sreg_cur < 2:
            self._status.configure(text="Define both regions first.")
            return

        pts_a = list(self._sreg_polys[0])
        pts_b = list(self._sreg_polys[1])

        new_a = list(pts_a)
        new_b = list(pts_b)
        snapped = 0

        for i, (ax, ay) in enumerate(pts_a):
            for j, (bx, by) in enumerate(pts_b):
                dist = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
                if dist <= threshold_px:
                    mx = (ax + bx) / 2
                    my = (ay + by) / 2
                    new_a[i] = (mx, my)
                    new_b[j] = (mx, my)
                    snapped += 1

        if snapped == 0:
            self._status.configure(
                text=f"No vertex pairs within {threshold_px} px — try moving vertices closer first.")
            return

        from PIL import Image as PILImage, ImageDraw

        for ri, new_pts, col in [(0, new_a, "yellow"), (1, new_b, "cyan")]:
            # Clear existing canvas items for this region
            for pid in self._sreg_canvas_ids[ri]:
                self._canvas.delete(pid)
            self._sreg_canvas_ids[ri] = []
            self._sreg_polys[ri] = new_pts

            # Redraw polygon on the interactive canvas
            for k, (cx, cy) in enumerate(new_pts):
                dot = self._canvas.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                                fill=col, outline=col, tags="sreg")
                self._sreg_canvas_ids[ri].append(dot)
                if k > 0:
                    px, py = new_pts[k - 1]
                    ln = self._canvas.create_line(px, py, cx, cy,
                                                   fill=col, width=2, tags="sreg")
                    self._sreg_canvas_ids[ri].append(ln)
            if len(new_pts) >= 2:
                px, py = new_pts[-1]
                fx, fy = new_pts[0]
                close_ln = self._canvas.create_line(px, py, fx, fy,
                                                     fill=col, width=2, tags="sreg")
                self._sreg_canvas_ids[ri].append(close_ln)

            # Recompute mask from updated polygon
            img_pts = [_canvas_to_image(cx, cy, self._scale_x, self._scale_y)
                       for cx, cy in new_pts]
            pmask = PILImage.new('L', (self._iw, self._ih), 0)
            ImageDraw.Draw(pmask).polygon(img_pts, fill=255)
            self._sreg_masks[ri] = np.array(pmask, dtype=bool)

        self._status.configure(
            text=f"Snapped {snapped} vertex pair(s). Masks updated.")
        self._refresh_canvas()

    # ── undo / finish ─────────────────────────────────────────────────────────

    def _push_history(self):
        self._history.append((self._roi_masks.copy(), self._roi_msk.copy()))
        if len(self._history) > 20:
            self._history.pop(0)

    def _undo(self):
        if self._mode == self._SUBREGION:
            self._sreg_undo()
            return
        if not self._history:
            self._status.configure(text="Nothing to undo.")
            return
        self._roi_masks, self._roi_msk = self._history.pop()
        self._status.configure(text=f"Undone. Total: {self._roi_masks.shape[1]}")
        self._refresh_canvas()

    def _sreg_undo(self):
        ri = self._sreg_cur  # 0, 1, or 2

        # If actively drawing (ri < 2) and there are in-progress vertices, clear them.
        # NOTE: _sreg_polys only has indices 0 and 1, so guard with ri < 2 before indexing.
        if ri < 2 and self._sreg_polys[ri]:
            for pid in self._sreg_canvas_ids[ri]:
                self._canvas.delete(pid)
            self._sreg_canvas_ids[ri] = []
            self._sreg_polys[ri] = []
            self._status.configure(
                text=f"{'Region A' if ri == 0 else 'Region B'} drawing cleared.")
            return

        # Step back one confirmed region (works for ri == 1 or ri == 2)
        if ri > 0:
            prev = ri - 1          # the last-confirmed region index
            self._sreg_cur = prev
            self._sreg_masks[prev] = None
            for pid in self._sreg_canvas_ids[prev]:
                self._canvas.delete(pid)
            self._sreg_canvas_ids[prev] = []
            self._sreg_polys[prev] = []
            if hasattr(self, '_snap_btn'):
                self._snap_btn.configure(state="disabled")
            self._status.configure(
                text=f"{'Region A' if prev == 0 else 'Region B'} removed — redraw it.")
            self._refresh_canvas()
            return

        self._status.configure(text="Nothing to undo.")

    def _do_finish(self):
        clean = self._roi_masks[:, ~(self._roi_masks.sum(axis=0) == 0)]
        settings = {
            "gamma":  round(self._gamma_var.get(), 3),
            "lo_pct": round(self._lo_var.get(),    3),
            "hi_pct": round(self._hi_var.get(),    3),
        }
        sreg = self._sreg_masks if any(m is not None for m in self._sreg_masks) else None
        self._on_finish(clean, self._roi_bkg, self._roi_msk, settings, sreg)
        self.destroy()
