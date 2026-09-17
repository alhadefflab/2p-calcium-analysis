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

    # Reference-panel LUT button labels.
    _GRAY  = "Gray"
    _MERGE = "Merge"
    _RED   = "Red"
    _GREEN = "Green"

    # internal LUT modes.  Gray shows the dropdown's channel; Red is always the
    # structural channel (tdTomato) and Green always the functional one (GCaMP),
    # whatever the dropdown is on; Merge overlays the two.
    _LUT_GRAY  = "gray"
    _LUT_RED   = "red"
    _LUT_GREEN = "green"
    _LUT_MERGE = "merge"

    # polygon editing hit radii, in canvas pixels
    _VTX_HIT  = 9
    _EDGE_HIT = 6

    # Reference-panel temporal statistic.  Applies to the LEFT panel only — the
    # functional image on the right (and the green half of Merge) keeps whatever
    # projection the pipeline produced, so mean-tdTomato beside p99-GCaMP stays
    # available as a comparison.
    _MEAN = "Mean"
    _P99  = "P99"
    _STAT = {_MEAN: "mean", _P99: "p99"}

    # Density overlay peak blend strength (0-1).  Alpha-blended rather than
    # added, so it cannot clip against a bright background.
    _DENS_ALPHA = 0.55

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
            "Neurons whose centres fall outside are removed.\n\n"
            "Adjust while drawing:\n"
            "• drag a vertex to move it\n"
            "• Shift+click an edge to insert a vertex\n"
            "• Shift+right-click a vertex to delete it\n"
            "• Undo / Ctrl+Z / Backspace: remove the last step\n"
            "• Esc: cancel the polygon"
        ),
        "subregion": (
            "Outline the AREA POSTREMA (yellow).\n\n"
            "LEFT-CLICK to place vertices. RIGHT-CLICK to confirm.\n\n"
            "Every kept neuron outside the AP outline is NTS, excluded "
            "neurons are already gone.\n\n"
            "Adjust at any time, also after confirming:\n"
            "• drag a vertex to move it\n"
            "• click an edge to insert a vertex (Shift+click while drawing)\n"
            "• right-click a vertex to delete it (Shift+right-click while drawing)\n"
            "• Undo / Ctrl+Z: one step back (a confirmed outline reopens)\n"
            "• Backspace: remove the last point · Esc: cancel drawing\n\n"
            "Reference Panel below sets what the left image shows."
        ),
    }

    # ROI rendering on the interactive panel
    _STYLE_FILL    = "Fill"
    _STYLE_OUTLINE = "Outline"
    _STYLE_BOTH    = "Both"

    # AP / NTS outline colours (sub-region mode), as RGB
    _AP_RGB  = (255, 226, 61)
    _NTS_RGB = (61, 224, 255)

    def __init__(self, parent, z, roi_img_bkg, roi_img_mask, roi_masks, mc_corr_file, on_finish,
                 display_settings=None, mc_img_bkg=None,
                 channels=None, channel_loader=None, channel_default=None,
                 channel_note="", channel_luts=None,
                 roi_ids=None, n_detected=None, next_id=None, ap_polygon=None):
        super().__init__(parent)
        self.title(f"ROI Curation — {z}")
        self.resizable(True, True)
        self.lift()
        self.focus_force()

        self._z         = z
        self._on_finish = on_finish
        roi_masks = np.asarray(roi_masks)
        if roi_masks.ndim != 2:            # no ROIs detected on this plane
            roi_masks = np.zeros((roi_img_bkg.shape[0] * roi_img_bkg.shape[1], 0), dtype=bool)
        self._roi_masks = roi_masks.copy()
        self._roi_bkg   = roi_img_bkg.copy()
        self._roi_msk   = roi_img_mask.copy()
        self._mc_bkg    = mc_img_bkg  # structural channel for sub-region orientation

        # ── curation bookkeeping ──────────────────────────────────────────────
        # Every ROI carries a stable id so the record can say which neurons were
        # added or removed.  Detected ROIs are 0..n_detected-1; added ones
        # continue from next_id.  A reopened editor (sub-region setup) passes the
        # ids saved last time, so they stay consistent across sessions.
        n0 = roi_masks.shape[1]
        self._ids = (np.asarray(roi_ids, dtype=int).copy()
                     if roi_ids is not None and len(roi_ids) == n0
                     else np.arange(n0))
        self._n_detected = int(n_detected) if n_detected is not None else n0
        self._next_id    = max(int(next_id or 0), int(self._ids.max(initial=-1)) + 1,
                               self._n_detected)
        self._removal_reason = {}        # {id: 'manual' | 'region'}
        self._excl_polys     = []        # exclusion polygons actually applied
        self._keep_mask      = np.ones(roi_img_bkg.shape[:2], dtype=bool)

        # reference-channel selection (sub-region mode only) — lets the user
        # pick whichever channel carries the anatomical label (e.g. tdTomato)
        self._channels  = list(channels or [])
        self._ch_loader = channel_loader
        self._ch_cur    = channel_default
        self._ch_note   = channel_note
        # cached per (channel, statistic) — each pair is computed at most once
        self._ch_cache  = {}
        self._stat_cur  = self._MEAN
        if channel_default and mc_img_bkg is not None:
            self._ch_cache[(channel_default, "mean")] = (mc_img_bkg, channel_note)

        # {channel: LUT name}, from the rig's PMT assignment — structural on the
        # red detector, functional on the green one.  Anything unmapped is shown
        # red, matching the old behaviour.
        self._ch_luts   = dict(channel_luts or {})
        self._struct_ch = channel_default      # the red/structural channel
        self._func_ch   = next((c for c, l in self._ch_luts.items() if l == self._GREEN), None)
        self._lut_mode  = self._LUT_RED

        # neuron-density overlay (AP/NTS boundary criterion 1)
        self._cent_cache = None
        self._dens_cache = None
        self._sreg_seen  = False   # first entry into sub-region mode defaults to Merge

        h, w = roi_img_bkg.shape[:2]
        self._ih, self._iw = h, w
        # initial scales — updated when window maximises and Configure fires.
        # The two canvases are tracked separately: they are gridded with
        # different padding, so their pixel sizes are not identical and a click
        # must be converted using the scale of the canvas it landed on.
        _s = 500 / max(h, w)
        self._scale_x = _s
        self._scale_y = _s
        self._dh = int(h * _s)
        self._dw = int(w * _s)
        self._ref_scale_x = _s
        self._ref_scale_y = _s
        self._ref_dh = self._dh
        self._ref_dw = self._dw

        self._mode      = None
        self._new_mask  = None
        self._add_col   = None
        # polygon vertices are stored in IMAGE coordinates, so they survive a
        # window resize and can be drawn on either canvas at its own scale
        self._poly_pts  = []
        self._poly_ids  = []       # list of (canvas, item_id)
        self._history   = []
        self._img_id    = None
        self._ref_id    = None

        # sub-region state: one AP polygon; NTS is everything kept outside it
        self._ap_pts  = []         # IMAGE-coord vertices (in progress or confirmed)
        self._ap_ids  = []         # (canvas, item_id) pairs
        self._ap_mask = None       # bool (h, w) once confirmed

        # vertex-level undo for the polygon being edited: (mode, points, closed)
        # snapshots taken before every point / drag / insert / delete / confirm
        self._vtx_hist = []
        self._drag     = None      # (vertex index, start point, inserted) while dragging

        # ROI label image for hover lookup and outline rendering: rebuilt
        # lazily after any mask edit (see _invalidate_density)
        self._lab_cache     = None
        self._outline_cache = {}
        self._hover_id      = None     # stable id currently highlighted

        # the label image as it was on entry, for the curation record
        from analysis.roi_curation import label_image
        self._initial_labels = label_image(self._roi_masks, h, w, self._ids)[0]

        ds = display_settings or {}
        # bright clip is only usable in 90-100; clamp settings saved when the
        # slider still went down to 70
        _bright = lambda v: min(100.0, max(90.0, float(v)))
        self._gamma_var  = tk.DoubleVar(value=ds.get("gamma",   1.36))
        self._lo_var     = tk.DoubleVar(value=ds.get("lo_pct",  26.7))
        self._hi_var     = tk.DoubleVar(value=_bright(ds.get("hi_pct", 98.8)))

        # The reference panel gets its own contrast.  One shared set cannot serve
        # a bright GCaMP image and a faint tdTomato halo at once, settings that
        # make the functional channel readable crush exactly the neuropil signal
        # the AP/NTS boundary is read from.
        #
        # Defaults lift dim signal (gamma < 1, almost no dark clipping) without
        # going to the other extreme.  
        # No single setting serves the core and the edge at once, that is what the
        # sliders are for; the default just should not start at an extreme.
        self._ref_gamma_var = tk.DoubleVar(value=ds.get("ref_gamma",  0.80))
        self._ref_lo_var    = tk.DoubleVar(value=ds.get("ref_lo_pct",  1.50))
        self._ref_hi_var    = tk.DoubleVar(value=_bright(ds.get("ref_hi_pct", 99.50)))
        self._dens_sigma_var = tk.DoubleVar(value=ds.get("dens_sigma", 12.0))
        self._dens_level_var = tk.DoubleVar(value=ds.get("dens_level", 0.50))
        self._roi_style_var  = tk.StringVar(value=ds.get("roi_style", self._STYLE_BOTH))
        self._ref_outline_var = tk.BooleanVar(value=bool(ds.get("ref_outlines", False)))

        self._build_ui()
        self._set_mode(self._REMOVE)

        # an AP outline from an earlier curation of this plane is restored, so
        # reopening a plane does not mean redrawing it
        if ap_polygon is not None and len(ap_polygon) >= 3:
            self._ap_pts  = [(float(x), float(y)) for x, y in ap_polygon]
            self._ap_mask = self._poly_mask(self._ap_pts)
            self._redraw_polys()
            self._refresh_canvas()
            self._status.configure(
                text="AP outline restored from the previous curation of this "
                     "plane. In Define Sub-Regions, Undo clears it.")

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

        # what is under the mouse, answers "is there a neuron here?"
        self._hover_lbl = ctk.CTkLabel(canvas_area, text="", anchor="w",
                                       fg_color="#1b1b1b", corner_radius=4, height=26,
                                       font=ctk.CTkFont(size=13, weight="bold"))
        self._hover_lbl.grid(row=2, column=0, columnspan=2, sticky="ew",
                             padx=6, pady=(4, 0))

        # Both canvases take the same handlers, so regions can be drawn on the
        # reference (anatomical) image as well as the interactive one.  Which
        # canvas a click came from is read from event.widget.
        for cv in (self._canvas, self._canvas_ref):
            cv.bind("<Configure>",       self._on_canvas_resize)
            cv.bind("<Button-3>",        self._on_right)
            cv.bind("<Button-1>",        self._on_left_dn)
            cv.bind("<B1-Motion>",       self._on_left_mv)
            cv.bind("<ButtonRelease-1>", self._on_left_up)
            cv.bind("<Motion>",          self._on_motion)
            cv.bind("<Leave>",           self._on_leave)

        # 250 not 220: the scrollable body's scrollbar needs ~20 px on top of the
        # 190-wide controls plus their 10 px padding.
        panel = ctk.CTkFrame(outer, width=250)
        panel.grid(row=0, column=1, sticky="nsew")
        panel.grid_propagate(False)
        panel.pack_propagate(False)

        # The action buttons live in a fixed footer packed BEFORE the body, so
        # they always reserve their space.  Previously everything shared one
        # frame and the bottom-packed buttons were squeezed out of view as
        # controls were added above them.
        footer = ctk.CTkFrame(panel, fg_color="transparent")
        footer.pack(side="bottom", fill="x", pady=(4, 8))
        ctk.CTkButton(footer, text="Finish ✓", width=190,
                      fg_color="#2d6a2d", hover_color="#1e4d1e",
                      command=self._do_finish).pack(side="bottom", padx=10, pady=3)
        ctk.CTkButton(footer, text="Undo", width=190,
                      command=self._undo).pack(side="bottom", padx=10, pady=3)

        # Everything else scrolls, so the panel can hold more controls than fit.
        body = ctk.CTkScrollableFrame(panel, fg_color="transparent")
        body.pack(side="top", fill="both", expand=True)

        ctk.CTkLabel(body, text="Mode",
                     font=ctk.CTkFont(size=13, weight="bold")).pack(pady=(4, 6), padx=10)

        self._mode_btns = {}
        for key, lbl in [(self._REMOVE,    "Remove Neurons"),
                          (self._ADD,       "Add Neuron"),
                          (self._REGION,    "Exclude Region"),
                          (self._SUBREGION, "Define Sub-Regions")]:
            b = ctk.CTkButton(body, text=lbl, width=190,
                               command=lambda k=key: self._set_mode(k))
            b.pack(pady=3, padx=10)
            self._mode_btns[key] = b

        ctk.CTkFrame(body, height=2, fg_color="gray40").pack(fill="x", padx=10, pady=10)

        self._instr = ctk.CTkLabel(body, text="", wraplength=190,
                                    justify="left", anchor="nw")
        self._instr.pack(padx=10, fill="x")

        ctk.CTkFrame(body, height=2, fg_color="gray40").pack(fill="x", padx=10, pady=10)

        self._status = ctk.CTkLabel(body, text="", wraplength=190,
                                     text_color="#aaaaaa", anchor="w")
        self._status.pack(padx=10, fill="x")

        # ── display settings sliders ──────────────────────────────────────────
        ctk.CTkFrame(body, height=2, fg_color="gray40").pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(body, text="Display Settings",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(0, 4))

        def _make_slider(label, var, from_, to, steps, parent=None):
            row = ctk.CTkFrame(parent if parent is not None else body,
                               fg_color="transparent")
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

        ctk.CTkLabel(body, text="ROI panel (functional)", text_color="#888888",
                     anchor="w").pack(padx=10, fill="x")
        _make_slider("Gamma",      self._gamma_var, 0.2, 1.5, 130)
        _make_slider("Dark clip%", self._lo_var,    0.0, 30.0, 300)
        _make_slider("Bright clip%", self._hi_var,  90.0, 100.0, 100)

        # Outlines stay visible on a bright background, where the additive
        # colour fill washes out.
        ctk.CTkLabel(body, text="ROIs drawn as", text_color="#888888",
                     anchor="w").pack(padx=10, pady=(6, 0), fill="x")
        ctk.CTkSegmentedButton(
            body, values=[self._STYLE_FILL, self._STYLE_OUTLINE, self._STYLE_BOTH],
            variable=self._roi_style_var,
            command=lambda _v: self._refresh_canvas()).pack(padx=10, pady=(2, 2), fill="x")
        ctk.CTkCheckBox(body, text="Outlines on reference too",
                        variable=self._ref_outline_var,
                        command=self._refresh_canvas).pack(padx=10, pady=(4, 2), anchor="w")
        # ─────────────────────────────────────────────────────────────────────

        # ── reference panel: channel + LUT (sub-region mode) ──────────────────
        ctk.CTkFrame(body, height=2, fg_color="gray40").pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(body, text="Reference Panel",
                     font=ctk.CTkFont(size=12, weight="bold")).pack(padx=10, pady=(0, 4))

        self._ch_menu = None
        if self._ch_loader is not None and len(self._channels) > 1:
            self._ch_var = tk.StringVar(
                value=self._ch_cur if self._ch_cur in self._channels else self._channels[0])
            self._ch_menu = ctk.CTkOptionMenu(
                body, values=self._channels, variable=self._ch_var,
                width=190, command=self._on_channel_change)
            self._ch_menu.pack(padx=10, pady=2)

        self._ref_color_var = tk.StringVar()
        self._ref_color_btn = ctk.CTkSegmentedButton(
            body, values=[self._GRAY, self._RED, self._GREEN, self._MERGE],
            variable=self._ref_color_var, command=self._on_lut_change)
        self._ref_color_btn.pack(padx=10, pady=(4, 2), fill="x")

        self._stat_var = tk.StringVar(value=self._MEAN)
        self._stat_btn = ctk.CTkSegmentedButton(
            body, values=[self._MEAN, self._P99],
            variable=self._stat_var, command=self._on_stat_change)
        self._stat_btn.pack(padx=10, pady=(2, 2), fill="x")

        self._lut_caption = ctk.CTkLabel(
            body, text="", text_color="#888888", wraplength=190,
            justify="left", anchor="w")
        self._lut_caption.pack(padx=10, pady=(0, 4), fill="x")

        ctk.CTkLabel(body, text="Reference contrast", text_color="#888888",
                     anchor="w").pack(padx=10, pady=(4, 0), fill="x")
        _make_slider("Gamma",       self._ref_gamma_var, 0.2, 1.5,  130)
        _make_slider("Dark clip%",  self._ref_lo_var,    0.0, 30.0, 300)
        _make_slider("Bright clip%", self._ref_hi_var,  90.0, 100.0, 100)

        self._dens_var = tk.BooleanVar(value=False)
        ctk.CTkCheckBox(body, text="Neuron density", variable=self._dens_var,
                        command=self._on_density_toggle).pack(
            padx=10, pady=(8, 2), anchor="w")
        _make_slider("Smoothing", self._dens_sigma_var, 2.0, 40.0, 380)
        _make_slider("Contour",   self._dens_level_var, 0.05, 0.95, 90)
        ctk.CTkLabel(body,
                     text="Local density of detected ROIs, the same centres "
                          "get assigned to AP/NTS. Contour draws an iso-density "
                          "line to trace. AP is much denser than NTS "
                          "(Huang et al. 2024).",
                     text_color="#888888", wraplength=190,
                     justify="left", anchor="w").pack(padx=10, pady=(0, 6), fill="x")

        self._sync_lut_buttons()

        # keyboard shortcuts for polygon drawing
        self.bind("<Control-z>", lambda _e: self._undo())
        self.bind("<Control-Z>", lambda _e: self._undo())
        self.bind("<BackSpace>", lambda _e: self._remove_last_vertex())
        self.bind("<Escape>",    lambda _e: self._cancel_polygon())

        self.protocol("WM_DELETE_WINDOW", self._do_finish)

    # ── mode ──────────────────────────────────────────────────────────────────

    def _set_mode(self, mode):
        self._mode    = mode
        self._new_mask = None
        self._add_col  = None
        self._delete_ids(self._poly_ids)
        self._poly_pts = []
        self._poly_ids = []
        self._vtx_hist = []
        self._drag     = None

        # an unconfirmed AP outline is dropped on a mode switch; a confirmed one
        # survives
        if self._ap_mask is None and self._ap_pts:
            self._delete_ids(self._ap_ids)
            self._ap_ids = []
            self._ap_pts = []

        # Sub-region drawing defaults to Merge: it shows the anatomical landmark
        # and the neurons being partitioned in one image, which is the view the
        # AP/NTS boundary is judged from.
        if (mode == self._SUBREGION and self._mc_bkg is not None
                and not self._sreg_seen):
            self._sreg_seen = True
            self._lut_mode  = self._LUT_MERGE
            self._sync_lut_buttons()

        self._update_ref_label()
        # The reference controls apply in every mode now, so they stay enabled.
        # The statistic toggle still needs provenance-backed loading to be able
        # to recompute a projection.
        if getattr(self, '_stat_btn', None) is not None:
            self._stat_btn.configure(
                state="normal" if self._channels else "disabled")

        for k, btn in self._mode_btns.items():
            btn.configure(fg_color="#1a5276" if k == mode else ("#3b8ed0", "#1f6aa5"))
        self._instr.configure(text=self._INSTRUCTIONS.get(mode, ""))
        self._status.configure(text="")
        self._refresh_canvas()

    # ── reference channel ─────────────────────────────────────────────────────

    def _update_ref_label(self):
        if not hasattr(self, '_lbl_ref'):
            return
        mode = self._lut_mode
        if mode == self._LUT_GREEN:
            txt = f"Reference: {self._func_ch or 'functional'} (GCaMP) · Green"
        elif self._mc_bkg is None:
            txt = "Reference  (no ROIs)"
        elif mode == self._LUT_MERGE:
            txt = (f"Reference: {self._struct_ch or 'structural'} + "
                   f"{self._func_ch or 'functional'} · {self._stat_cur} · Merge")
        elif mode == self._LUT_RED:
            txt = f"Reference: {self._struct_ch or 'structural'} (tdTomato) · {self._stat_cur} · Red"
        else:
            txt = (f"Reference: {self._ch_cur or 'structural'} · {self._stat_cur} · Gray"
                   + (f"  ({self._ch_note})" if self._ch_note else ""))
        self._lbl_ref.configure(text=txt)

    _LUT_LABEL = {"gray": "Gray", "red": "Red", "green": "Green", "merge": "Merge"}

    def _sync_lut_buttons(self):
        """Point the LUT buttons at the current mode and refresh the caption."""
        self._ref_color_var.set(self._LUT_LABEL[self._lut_mode])
        self._lut_caption.configure(
            text="Gray: the channel in the dropdown.  Red: tdTomato.  "
                 "Green: GCaMP.  Merge: both.\n"
                 "Mean: best SNR on a static label.  P99: matches the "
                 "functional panel.\nLeft panel only.")

    def _on_lut_change(self, value):
        self._lut_mode = {v: k for k, v in self._LUT_LABEL.items()}.get(value, self._LUT_RED)
        self._update_ref_label()
        self._refresh_canvas()

    def _func_ref_bright(self):
        """Functional channel (GCaMP) with the reference panel's own contrast.

        Uses the dropdown's loaded projection of the functional channel when one
        is cached (so Mean / P99 apply), otherwise the functional image the ROI
        panel shows.  Never triggers a load.
        """
        stat = self._STAT.get(self._stat_var.get(), "mean")
        img = None
        if self._func_ch is not None:
            img = self._ch_cache.get((self._func_ch, stat), (None, ""))[0]
        return self._ref_stretch(img if img is not None else self._roi_bkg)

    def _apply_ref_lut(self, ref, func):
        """Colourise the reference panel.

        PMT data carries intensity only — the red/green you see on the rig and in
        papers is a display LUT, which Prairie View applies to the Ch1 (red,
        570-640 nm → tdTomato) and Ch2 (green, 500-550 nm → GCaMP) detectors.
        Same idea here: purely a display choice, the arrays stay untouched.

        `ref` is the selected channel, `func` the functional one, both already
        contrast-stretched (h, w, 3) uint8 with the grey value replicated across
        the three planes — so plane 0 of either is its intensity image.
        """
        if self._lut_mode == self._LUT_GRAY:
            return ref

        out = np.zeros_like(ref)
        if self._lut_mode == self._LUT_RED:
            struct = ref if self._ch_cur == self._struct_ch else self._struct_bright()
            out[..., 0] = (struct if struct is not None else ref)[..., 0]
            return out
        if self._lut_mode == self._LUT_GREEN:
            out[..., 1] = self._func_ref_bright()[..., 0]
            return out

        # Merge: structural red + functional green, whichever channel the
        # dropdown happens to be on.
        struct = ref if self._ch_cur == self._struct_ch else self._struct_bright()
        if struct is not None:
            out[..., 0] = struct[..., 0]
        out[..., 1] = func[..., 0]
        return out

    def _struct_bright(self):
        """Contrast-stretched structural channel, independent of the dropdown.

        Prefers the current statistic, then falls back to the mean, which is
        always preloaded — so switching to Merge never blocks on a P99 that has
        not been computed yet.
        """
        stat = self._STAT.get(self._stat_var.get(), "mean")
        for key in ((self._struct_ch, stat), (self._struct_ch, "mean")):
            img, _ = self._ch_cache.get(key, (None, ""))
            if img is not None:
                return self._ref_stretch(img)
        return None

    def _load_ref(self, ch) -> bool:
        """Put (channel, current statistic) on the reference panel.

        Each pair is computed once and cached; P99 needs a partial sort per
        pixel, so the first switch can take a moment.
        """
        if ch is None or self._ch_loader is None:
            return False

        stat_lbl = self._stat_var.get()
        key      = (ch, self._STAT.get(stat_lbl, "mean"))
        if key not in self._ch_cache:
            self._status.configure(text=f"Loading {ch} · {stat_lbl} …")
            self.update_idletasks()
            self._ch_cache[key] = self._ch_loader(*key) or (None, "")

        img, note = self._ch_cache[key]
        if img is None:
            self._status.configure(text=f"{ch} · {stat_lbl} could not be loaded.")
            return False

        self._ch_cur, self._mc_bkg, self._ch_note = ch, img, note
        self._stat_cur = stat_lbl
        self._status.configure(text=f"Reference: {ch} · {stat_lbl} ({note}).")
        self._sync_lut_buttons()
        self._update_ref_label()
        self._refresh_canvas()
        return True

    def _on_channel_change(self, ch):
        """Swap the reference image to another acquisition channel."""
        if not self._load_ref(ch) and self._ch_cur:
            self._ch_var.set(self._ch_cur)      # revert to what is on screen

    def _on_stat_change(self, _value):
        """Swap the reference image between the mean and the 99th percentile."""
        if not self._load_ref(self._ch_cur):
            self._stat_var.set(self._stat_cur)  # revert to what is on screen

    # ── canvas ────────────────────────────────────────────────────────────────

    def _on_canvas_resize(self, event):
        if hasattr(self, '_resize_job'):
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._apply_resize)

    def _apply_resize(self):
        changed = False
        w, h = self._canvas.winfo_width(), self._canvas.winfo_height()
        if w > 1 and h > 1:
            self._dw, self._dh = w, h
            self._scale_x = w / self._iw
            self._scale_y = h / self._ih
            changed = True

        rw, rh = self._canvas_ref.winfo_width(), self._canvas_ref.winfo_height()
        if rw > 1 and rh > 1:
            self._ref_dw, self._ref_dh = rw, rh
            self._ref_scale_x = rw / self._iw
            self._ref_scale_y = rh / self._ih
            changed = True

        if changed:
            self._refresh_canvas()
            self._redraw_polys()

    def _redraw_polys(self):
        """Re-render polygon overlays on both canvases.

        Vertices live in image coordinates, so their canvas positions change
        whenever a canvas is resized and the items have to be rebuilt.
        """
        self._delete_ids(self._poly_ids)
        self._poly_ids = []
        for k, pt in enumerate(self._poly_pts):
            self._poly_ids += self._draw_vertex(*pt, "yellow", "poly")
            if k > 0:
                self._poly_ids += self._draw_edge(
                    self._poly_pts[k - 1], pt, "yellow", "poly")

        self._delete_ids(self._ap_ids)
        self._ap_ids = []
        pts = self._ap_pts
        for k, pt in enumerate(pts):
            self._ap_ids += self._draw_vertex(*pt, "yellow", "sreg")
            if k > 0:
                self._ap_ids += self._draw_edge(pts[k - 1], pt, "yellow", "sreg")
        # a confirmed outline keeps its closing edge
        if self._ap_mask is not None and len(pts) >= 2:
            self._ap_ids += self._draw_edge(pts[-1], pts[0], "yellow", "sreg")

    # ── neuron density (AP/NTS boundary criterion) ────────────────────────────

    def _centroids(self) -> np.ndarray:
        """(N, 2) array of ROI centres as (row, col), cached per mask edit.

        Same extraction the region-exclusion test uses, hoisted out so the
        density map does not recompute it on every redraw.
        """
        if self._cent_cache is not None:
            return self._cent_cache
        pts = []
        for i in range(self._roi_masks.shape[1]):
            pxs = self._roi_masks[:, i].reshape((self._ih, self._iw), order='F')
            ys, xs = np.where(pxs)
            if len(xs):
                pts.append((ys.mean(), xs.mean()))
        self._cent_cache = (np.array(pts, dtype=np.float32) if pts
                            else np.zeros((0, 2), dtype=np.float32))
        return self._cent_cache

    def _density_map(self) -> np.ndarray:
        """Kernel density of ROI centres, normalised to 0..1.

        This is the density of the neurons the boundary will actually partition,
        which is what makes it self-consistent with `get_region_labels` that
        assigns each ROI by its centre of mass, the same points estimated here.
        It is deliberately NOT the paper's measurement (they judged tdTomato soma
        density by eye); it is the same idea computed on the population under
        analysis.

        Normalised against a high percentile rather than the max, so one unusually
        tight clump cannot flatten the rest of the field.  Cached against the
        smoothing sigma and the current mask set.
        """
        from scipy.ndimage import gaussian_filter

        sigma = float(self._dens_sigma_var.get())
        key   = (sigma, self._roi_masks.shape[1])
        if self._dens_cache is not None and self._dens_cache[0] == key:
            return self._dens_cache[1]

        counts = np.zeros((self._ih, self._iw), dtype=np.float32)
        cents  = self._centroids()
        if len(cents):
            rr = np.clip(cents[:, 0].astype(int), 0, self._ih - 1)
            cc = np.clip(cents[:, 1].astype(int), 0, self._iw - 1)
            np.add.at(counts, (rr, cc), 1.0)
        dens = gaussian_filter(counts, sigma=sigma, mode='nearest')
        hi   = float(np.percentile(dens, 99.5))
        dens = np.clip(dens / hi, 0, 1) if hi > 0 else dens
        self._dens_cache = (key, dens)
        return dens

    def _density_contour(self) -> np.ndarray:
        """1-px boolean outline of the current iso-density level."""
        from scipy.ndimage import binary_dilation

        inside = self._density_map() >= float(self._dens_level_var.get())
        return binary_dilation(inside) & ~inside

    def _blend_density(self, img) -> np.ndarray:
        """Alpha-blend the density field over `img`, plus its iso-contour.

        Blending rather than adding is the whole point: an additive tint clips
        against an already-bright background, so it vanished exactly where
        density peaked — on the dense band it contributed +5 of an intended +110.
        Blending is bounded by construction and reads the same on any ground.
        The tint is blue-cyan so it can never be confused with the red or green
        channel LUTs, and the contour is the line you would actually trace.
        """
        dens = self._density_map()[:, :, None]
        a    = dens * self._DENS_ALPHA
        out  = (img.astype(np.float32) * (1.0 - a)
                + np.array([70, 150, 255], dtype=np.float32) * a)
        out[self._density_contour()] = (235, 250, 255)
        return np.clip(out, 0, 255).astype(np.uint8)

    def _invalidate_density(self):
        """Drop cached centroids/density/labels after any change to the mask set."""
        self._cent_cache = None
        self._dens_cache = None
        self._lab_cache  = None
        self._outline_cache = {}
        self._clear_hover()

    def _on_density_toggle(self):
        self._refresh_canvas()

    # ── ROI labels, outlines, hover ───────────────────────────────────────────

    def _labels(self):
        """(labels, counts): column index of the ROI on each pixel (-1 none;
        smallest ROI on overlaps) and how many ROIs cover it.  Cached."""
        if self._lab_cache is None:
            from analysis.roi_curation import label_image
            self._lab_cache = label_image(self._roi_masks, self._ih, self._iw)
        return self._lab_cache

    def _col_at(self, ix, iy):
        """(column index or -1, number of ROIs covering) at an image pixel."""
        if not (0 <= ix < self._iw and 0 <= iy < self._ih):
            return -1, 0
        lab, cnt = self._labels()
        return int(lab[iy, ix]), int(cnt[iy, ix])

    def _outline_rgb(self, dw, dh):
        """(boundary mask, colour image) at display resolution, cached.

        Computed on the nearest-neighbour upscaled label image, so lines are
        crisp 2 px at any zoom instead of a blurred 1 px image-space edge.
        Colours are per stable id (golden-ratio hues, so neighbours differ and
        a neuron keeps its colour across edits); in sub-region mode with an AP
        outline they switch to AP yellow / NTS cyan instead.
        """
        from PIL import Image as PILImage
        region = self._mode == self._SUBREGION and self._ap_mask is not None
        key = (dw, dh, region)
        if key in self._outline_cache:
            return self._outline_cache[key]

        lab, _ = self._labels()
        L = np.asarray(PILImage.fromarray(lab).resize((dw, dh), PILImage.NEAREST))
        edge = np.zeros(L.shape, dtype=bool)
        for s in (1, 2):
            dx = L[:, s:] != L[:, :-s]
            dy = L[s:, :] != L[:-s, :]
            edge[:, s:] |= dx
            edge[:, :-s] |= dx
            edge[s:, :] |= dy
            edge[:-s, :] |= dy
        edge &= L >= 0          # inner edge only: the line sits on the neuron

        n = self._roi_masks.shape[1]
        if region:
            cents = self._centroids()
            # _centroids skips empty columns; map back through non-empty ones
            nonempty = np.where(self._roi_masks.any(axis=0))[0]
            pal = np.tile(np.array(self._NTS_RGB, np.uint8), (max(n, 1), 1))
            if len(cents):
                rr = np.clip(cents[:, 0].astype(int), 0, self._ih - 1)
                cc = np.clip(cents[:, 1].astype(int), 0, self._iw - 1)
                pal[nonempty[self._ap_mask[rr, cc]]] = self._AP_RGB
        else:
            hue = (self._ids * 0.6180339887) % 1.0 if n else np.zeros(0)
            pal = self._hsv_to_rgb(hue, 0.85, 1.0)
            if not n:
                pal = np.zeros((1, 3), np.uint8)
        colour = pal[np.clip(L, 0, None)]
        self._outline_cache[key] = (edge, colour)
        return edge, colour

    @staticmethod
    def _hsv_to_rgb(h, s, v) -> np.ndarray:
        """Vectorised HSV → uint8 RGB for an array of hues."""
        h = np.asarray(h, dtype=np.float32)
        i = np.floor(h * 6).astype(int) % 6
        f = h * 6 - np.floor(h * 6)
        p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
        v = np.full_like(h, v)
        r = np.choose(i, [v, q, p, p, t, v])
        g = np.choose(i, [t, v, v, q, p, p])
        b = np.choose(i, [p, p, t, v, v, q])
        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    def _draw_outlines(self, pil_img):
        """Paint ROI outlines onto a display-size PIL image."""
        from PIL import Image as PILImage
        dw, dh = pil_img.size
        edge, colour = self._outline_rgb(dw, dh)
        arr = np.array(pil_img)
        arr[edge] = colour[edge]
        return PILImage.fromarray(arr)

    def _on_motion(self, event):
        self._hover_neuron(event)
        # polygon handles: move cursor on a vertex, insert cursor on an edge
        pts, closed = self._active_poly()
        if pts:
            cv = event.widget
            if self._vertex_at(cv, event.x, event.y) is not None:
                cv.configure(cursor="fleur")
            elif self._edge_at(cv, event.x, event.y) is not None and (
                    closed or getattr(event, "state", 0) & 0x0001):
                cv.configure(cursor="plus")
            elif self._mode != self._REMOVE:
                cv.configure(cursor="")

    def _hover_neuron(self, event):
        if self._mode == self._ADD and self._new_mask is not None:
            return                                  # painting: stay out of the way
        ix, iy = self._c2i_on(event.widget, event.x, event.y)
        col, cnt = self._col_at(ix, iy)
        sid = int(self._ids[col]) if col >= 0 else None
        if sid == self._hover_id:
            return
        self._clear_hover()
        if col < 0:
            self._hover_lbl.configure(text="No neuron under cursor", text_color="#888888")
            return

        self._hover_id = sid
        origin = "added" if sid >= self._n_detected else "detected"
        size   = int(self._roi_masks[:, col].sum())
        where  = ""
        if self._ap_mask is not None:
            r, c = self._roi_centre(col)
            where = "  ·  AP" if self._ap_mask[r, c] else "  ·  NTS"
        extra  = f"  ·  {cnt} overlapping: the smallest is picked" if cnt > 1 else ""
        action = "  ·  right-click to remove" if self._mode == self._REMOVE else ""
        self._hover_lbl.configure(
            text=f"● Neuron #{sid} ({origin}, {size} px){where}{extra}{action}",
            text_color="#ffe23d")

        # highlight on both canvases: dark halo under a bright line reads on any
        # background
        for cv in (self._canvas, self._canvas_ref):
            sx, sy = self._scales_for(cv)
            for seg in self._roi_contours(col):
                pts = [v for x, y in seg for v in (x * sx, y * sy)]
                if len(pts) < 4:
                    continue
                cv.create_line(*pts, fill="black", width=5, tags="hover")
                cv.create_line(*pts, fill="#ffe23d", width=2, tags="hover")
        if self._mode == self._REMOVE:
            event.widget.configure(cursor="hand2")

    def _on_leave(self, _event=None):
        self._clear_hover()
        self._hover_lbl.configure(text="")

    def _clear_hover(self):
        self._hover_id = None
        for cv in (getattr(self, "_canvas", None), getattr(self, "_canvas_ref", None)):
            if cv is not None:
                cv.delete("hover")
                cv.configure(cursor="")

    def _roi_centre(self, col):
        pxs = self._roi_masks[:, col].reshape((self._ih, self._iw), order='F')
        ys, xs = np.nonzero(pxs)
        return int(ys.mean()), int(xs.mean())

    def _roi_contours(self, col):
        """Image-space (x, y) contour lines of one ROI (pixel-edge aligned)."""
        from skimage.measure import find_contours
        pxs = self._roi_masks[:, col].reshape((self._ih, self._iw), order='F')
        ys, xs = np.nonzero(pxs)
        if not len(ys):
            return []
        y0, x0 = ys.min(), xs.min()
        crop = np.pad(pxs[y0:ys.max() + 1, x0:xs.max() + 1], 1).astype(float)
        # +0.5 moves from pixel centres to the canvas's pixel-corner coordinates
        return [np.column_stack([c[:, 1] + x0 - 0.5, c[:, 0] + y0 - 0.5])
                for c in find_contours(crop, 0.5)]

    def _stretch(self, img, gamma=None, lo_pct=None, hi_pct=None) -> np.ndarray:
        """Contrast-stretch + gamma lift.

        Defaults to the functional (ROI panel) sliders; pass explicit values to
        stretch the reference panel with its own, independent settings.
        """
        f     = np.asarray(img, dtype=np.float32)
        gamma = self._gamma_var.get() if gamma  is None else gamma
        lo    = np.percentile(f, self._lo_var.get() if lo_pct is None else lo_pct)
        hi    = np.percentile(f, self._hi_var.get() if hi_pct is None else hi_pct)
        if hi > lo:
            f = np.clip((f - lo) / (hi - lo), 0, 1)
        else:
            f = np.zeros_like(f)
        return np.clip(np.power(f, gamma) * 255, 0, 255).astype(np.uint8)

    def _ref_stretch(self, img) -> np.ndarray:
        """Stretch an image with the reference panel's own contrast settings."""
        return self._stretch(img,
                             gamma=self._ref_gamma_var.get(),
                             lo_pct=self._ref_lo_var.get(),
                             hi_pct=self._ref_hi_var.get())

    def _bright_bkg(self) -> np.ndarray:
        """The functional-channel background, contrast-stretched."""
        return self._stretch(self._roi_bkg)

    def _refresh_canvas(self):
        from PIL import Image as PILImage, ImageTk

        # Interactive canvas always uses the functional channel so ROIs stay
        # visually aligned with the background they were detected on.
        func_bright = self._bright_bkg()
        dw, dh = self._dw, self._dh

        # AP tint (yellow) for sub-region mode; NTS neurons are shown by their
        # cyan outlines rather than by tinting the whole rest of the field
        sreg_overlay = np.zeros((self._ih, self._iw, 3), dtype=np.int16)
        if self._mode == self._SUBREGION and self._ap_mask is not None:
            sreg_overlay[self._ap_mask] = (80, 80, 0)

        # ── reference canvas ──────────────────────────────────────────────────
        # Shows the selected channel through the chosen LUT in every mode, not
        # just sub-region: the structural channel is just as useful for judging
        # add/remove decisions.  Falls back to the functional channel if the
        # reference could not be loaded.
        if self._mc_bkg is not None:
            ref_bright = self._apply_ref_lut(self._ref_stretch(self._mc_bkg), func_bright)
        elif self._lut_mode == self._LUT_GREEN:
            ref_bright = self._apply_ref_lut(func_bright, func_bright)
        else:
            ref_bright = func_bright

        if self._dens_var.get():
            ref_bright = self._blend_density(ref_bright)

        ref_base = np.clip(
            ref_bright.astype(np.int16) + sreg_overlay, 0, 255
        ).astype(np.uint8)
        pil_ref = PILImage.fromarray(ref_base).resize(
            (self._ref_dw, self._ref_dh), PILImage.BILINEAR)
        if self._ref_outline_var.get():
            pil_ref = self._draw_outlines(pil_ref)
        self._tk_ref = ImageTk.PhotoImage(pil_ref)
        if self._ref_id is None:
            self._ref_id = self._canvas_ref.create_image(0, 0, anchor="nw", image=self._tk_ref)
        else:
            self._canvas_ref.itemconfig(self._ref_id, image=self._tk_ref)
        # keep the image behind any polygon drawn on this canvas
        self._canvas_ref.tag_lower(self._ref_id)

        # ── interactive canvas ────────────────────────────────────────────────
        # Always functional channel + ROI overlay so ROIs remain correctly placed.
        style = self._roi_style_var.get()
        fill  = (self._roi_msk.astype(np.int16) if style != self._STYLE_OUTLINE
                 else 0)
        combined = np.clip(
            func_bright.astype(np.int16) + fill + sreg_overlay, 0, 255
        ).astype(np.uint8)
        pil_roi = PILImage.fromarray(combined).resize((dw, dh), PILImage.BILINEAR)
        if style != self._STYLE_FILL:
            pil_roi = self._draw_outlines(pil_roi)
        self._tk_img = ImageTk.PhotoImage(pil_roi)
        if self._img_id is None:
            self._img_id = self._canvas.create_image(0, 0, anchor="nw", image=self._tk_img)
        else:
            self._canvas.itemconfig(self._img_id, image=self._tk_img)
        self._canvas.tag_lower(self._img_id)

    def _scales_for(self, canvas):
        """(scale_x, scale_y) of whichever canvas is being addressed."""
        if canvas is self._canvas_ref:
            return self._ref_scale_x, self._ref_scale_y
        return self._scale_x, self._scale_y

    def _c2i_on(self, canvas, cx, cy):
        """Canvas pixel → image pixel, using that canvas's own scale."""
        sx, sy = self._scales_for(canvas)
        return _canvas_to_image(cx, cy, sx, sy)

    def _i2c_on(self, canvas, ix, iy):
        """Image pixel → canvas pixel, using that canvas's own scale."""
        sx, sy = self._scales_for(canvas)
        return ix * sx, iy * sy

    def _draw_vertex(self, ix, iy, col, tag):
        """Draw one polygon vertex on both canvases; returns (canvas, id) pairs."""
        ids = []
        for cv in (self._canvas, self._canvas_ref):
            cx, cy = self._i2c_on(cv, ix, iy)
            ids.append((cv, cv.create_oval(cx - 4, cy - 4, cx + 4, cy + 4,
                                           fill=col, outline=col, tags=tag)))
        return ids

    def _draw_edge(self, p0, p1, col, tag):
        """Draw one polygon edge on both canvases; returns (canvas, id) pairs."""
        ids = []
        for cv in (self._canvas, self._canvas_ref):
            x0, y0 = self._i2c_on(cv, *p0)
            x1, y1 = self._i2c_on(cv, *p1)
            ids.append((cv, cv.create_line(x0, y0, x1, y1,
                                           fill=col, width=2, tags=tag)))
        return ids

    @staticmethod
    def _delete_ids(ids):
        """Delete a list of (canvas, item_id) pairs."""
        for cv, item in ids:
            cv.delete(item)

    def _flat(self, ix, iy):
        return ix * self._ih + iy

    # ── remove ────────────────────────────────────────────────────────────────

    def _on_right(self, event):
        if self._mode == self._REMOVE:
            ix, iy = self._c2i_on(event.widget, event.x, event.y)
            # same lookup the hover highlight uses, so what is highlighted is
            # what gets removed, including the smallest of overlapping ROIs
            nidx, _ = self._col_at(ix, iy)
            if nidx < 0:
                self._status.configure(text="No neuron there.")
                return
            self._push_history()
            sid = int(self._ids[nidx])
            pxs = self._roi_masks[:, nidx].reshape((self._ih, self._iw), order='F')
            self._roi_msk[pxs] = 0
            self._roi_masks = np.delete(self._roi_masks, nidx, 1)
            self._ids = np.delete(self._ids, nidx)
            self._removal_reason[sid] = "manual"
            self._invalidate_density()
            self._status.configure(
                text=f"Removed #{sid}. Total: {self._roi_masks.shape[1]}")
            self._refresh_canvas()
            self._on_motion(event)          # show what is under the cursor now
            return

        pts, closed = self._active_poly()
        if pts is None:
            return
        # right-click on a vertex deletes it: plain right-click once the AP is
        # confirmed (closing is no longer its job), Shift+right-click while drawing
        shift = getattr(event, "state", 0) & 0x0001
        k = self._vertex_at(event.widget, event.x, event.y)
        if k is not None and (closed or shift):
            self._delete_vertex(k)
            return
        if self._mode == self._REGION:
            self._close_polygon()
        elif not closed and len(self._ap_pts) >= 3:
            self._snapshot_poly()
            self._sreg_close_region()
        else:
            self._sreg_close_region()       # reports "need 3 points" / no-op

    # ── add ───────────────────────────────────────────────────────────────────

    def _on_left_dn(self, event):
        if self._mode == self._ADD:
            self._new_mask = np.zeros((self._ih, self._iw), dtype=bool)
            self._add_col  = tuple(np.random.randint(40, 210, 3).tolist())
            self._paint(event.widget, event.x, event.y)
            return

        pts, closed = self._active_poly()
        if pts is None:
            return
        cv, x, y = event.widget, event.x, event.y
        shift = getattr(event, "state", 0) & 0x0001

        # grab an existing vertex
        k = self._vertex_at(cv, x, y)
        if k is not None:
            self._snapshot_poly()
            self._drag = (k, pts[k], False)
            return

        # insert on an edge: plain click on a confirmed outline, Shift while drawing
        e = self._edge_at(cv, x, y)
        if e is not None and (closed or shift):
            self._snapshot_poly()
            pts.insert(e, self._c2i_on(cv, x, y))
            self._drag = (e, None, True)
            self._redraw_polys()
            return

        if closed:
            self._status.configure(
                text="AP is confirmed. Drag a vertex to move it, click an edge to "
                     "add one, right-click a vertex to delete it, Undo to reopen.")
            return

        self._snapshot_poly()
        if self._mode == self._REGION:
            self._add_poly_pt(*self._c2i_on(cv, x, y))
        else:
            self._sreg_add_pt(*self._c2i_on(cv, x, y))

    def _on_left_mv(self, event):
        if self._mode == self._ADD and self._new_mask is not None:
            self._paint(event.widget, event.x, event.y)
        elif self._drag is not None:
            pts, _ = self._active_poly()
            ix, iy = self._c2i_on(event.widget, event.x, event.y)
            pts[self._drag[0]] = (min(max(ix, 0), self._iw - 1),
                                  min(max(iy, 0), self._ih - 1))
            self._redraw_polys()

    def _paint(self, canvas, cx, cy):
        ix, iy = self._c2i_on(canvas, cx, cy)
        br = 2
        for dx in range(-br, br + 1):
            for dy in range(-br, br + 1):
                px, py = ix + dx, iy + dy
                if 0 <= px < self._iw and 0 <= py < self._ih:
                    self._new_mask[py, px] = True
        sx, sy = self._scales_for(canvas)
        r = max(2, int(br * min(sx, sy)))
        col = "#{:02x}{:02x}{:02x}".format(*self._add_col)
        canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                           fill=col, outline=col, tags="paint")

    def _on_left_up(self, event):
        if self._drag is not None:
            k, start, inserted = self._drag
            self._drag = None
            pts, _ = self._active_poly()
            if not inserted and pts[k] == start:
                self._vtx_hist.pop()          # a click on a vertex, not a move
                return
            self._poly_changed("Vertex added." if inserted else "Vertex moved.")
            return
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
            self._ids = np.append(self._ids, self._next_id)
            self._next_id += 1
            self._invalidate_density()
            self._roi_msk[self._new_mask] = np.array(self._add_col, dtype=np.uint8)
            self._status.configure(
                text=f"Added #{self._ids[-1]}. Total: {self._roi_masks.shape[1]}")
        self._new_mask = None
        self._add_col  = None
        self._refresh_canvas()

    # ── region exclusion ──────────────────────────────────────────────────────

    def _add_poly_pt(self, ix, iy):
        """Add an exclusion-polygon vertex, given in image coordinates."""
        self._poly_ids += self._draw_vertex(ix, iy, "yellow", "poly")
        if self._poly_pts:
            self._poly_ids += self._draw_edge(self._poly_pts[-1], (ix, iy),
                                              "yellow", "poly")
        self._poly_pts.append((ix, iy))
        self._status.configure(
            text=f"{len(self._poly_pts)} point(s). Right-click to close.")

    def _close_polygon(self):
        if len(self._poly_pts) < 3:
            self._status.configure(text="Need at least 3 points first.")
            return
        self._poly_ids += self._draw_edge(self._poly_pts[-1], self._poly_pts[0],
                                          "yellow", "poly")

        inside = self._poly_mask(self._poly_pts)

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
        if removed > 0 and not messagebox.askyesno(
                "Exclude region",
                f"Remove {removed} neuron(s) outside the polygon?",
                parent=self):
            # keep the outline so it can be adjusted and applied again
            self._redraw_polys()
            self._status.configure(
                text="Polygon kept. Adjust it, then right-click to apply again "
                     "(Esc cancels).")
            return
        if removed > 0:
            self._push_history()
            for i in np.where(~keep)[0]:
                pxs = self._roi_masks[:, i].reshape((self._ih, self._iw), order='F')
                self._roi_msk[pxs] = 0
                self._removal_reason[int(self._ids[i])] = "region"
            self._roi_masks = self._roi_masks[:, keep]
            self._ids = self._ids[keep]
            # the kept area is the intersection of every applied polygon, NTS
            # area for the density figures is this minus the AP
            self._excl_polys.append(list(self._poly_pts))
            self._keep_mask = self._keep_mask & inside
            self._invalidate_density()
            self._status.configure(
                text=f"Excluded {removed}. Total: {self._roi_masks.shape[1]}")
            self._refresh_canvas()
        elif removed == 0:
            # nothing to remove, but the outline still bounds the analysed area
            self._push_history()
            self._excl_polys.append(list(self._poly_pts))
            self._keep_mask = self._keep_mask & inside
            self._status.configure(
                text="All neurons are inside the polygon: area recorded.")

        self._delete_ids(self._poly_ids)
        self._poly_ids = []
        self._poly_pts = []
        # the exclusion is applied; undoing it now goes through the mask history
        self._vtx_hist = []

    # ── polygon editing (exclusion polygon and AP outline) ────────────────────

    def _active_poly(self):
        """(vertex list, closed) of the polygon the current mode edits.

        The list is the live one, so callers can edit it in place.  (None, False)
        in modes without a polygon.
        """
        if self._mode == self._REGION:
            return self._poly_pts, False
        if self._mode == self._SUBREGION:
            return self._ap_pts, self._ap_mask is not None
        return None, False

    def _snapshot_poly(self):
        """Save the polygon's state before an edit, for one-step undo."""
        pts, closed = self._active_poly()
        if pts is None:
            return
        self._vtx_hist.append((self._mode, list(pts), closed))
        if len(self._vtx_hist) > 200:
            self._vtx_hist.pop(0)

    def _vertex_at(self, canvas, cx, cy):
        """Index of the vertex under a canvas point, or None."""
        pts, _ = self._active_poly()
        best, best_d = None, self._VTX_HIT
        for k, p in enumerate(pts or []):
            x, y = self._i2c_on(canvas, *p)
            d = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = k, d
        return best

    def _edge_at(self, canvas, cx, cy):
        """Insert position (index of the edge's end vertex) for an edge under a
        canvas point, or None.  The closing edge counts once the AP is confirmed."""
        pts, closed = self._active_poly()
        n = len(pts or [])
        if n < 2:
            return None
        edges = list(range(n - 1)) + ([n - 1] if closed and n >= 3 else [])
        best, best_d = None, self._EDGE_HIT
        for k in edges:
            x0, y0 = self._i2c_on(canvas, *pts[k])
            x1, y1 = self._i2c_on(canvas, *pts[(k + 1) % n])
            dx, dy = x1 - x0, y1 - y0
            seg = dx * dx + dy * dy
            t = 0.0 if seg == 0 else max(0.0, min(1.0, ((cx - x0) * dx + (cy - y0) * dy) / seg))
            d = ((x0 + t * dx - cx) ** 2 + (y0 + t * dy - cy) ** 2) ** 0.5
            if d <= best_d:
                best, best_d = k + 1, d
        return best

    def _poly_changed(self, msg=""):
        """Redraw after a vertex edit; a confirmed AP also gets its mask and the
        AP / NTS assignment rebuilt."""
        self._redraw_polys()
        if self._mode == self._SUBREGION and self._ap_mask is not None:
            self._ap_mask = self._poly_mask(self._ap_pts)
            self._outline_cache = {}
            n_ap, n_nts = self._sreg_counts()
            msg = f"{msg}  AP: {n_ap}   NTS: {n_nts}".strip()
            self._refresh_canvas()
        else:
            pts, _ = self._active_poly()
            msg = f"{msg}  {len(pts or [])} point(s).".strip()
        self._status.configure(text=msg)

    def _delete_vertex(self, k):
        pts, closed = self._active_poly()
        if closed and len(pts) <= 3:
            self._status.configure(
                text="An outline needs at least 3 vertices. Undo reopens it instead.")
            return
        self._snapshot_poly()
        del pts[k]
        self._poly_changed("Vertex deleted.")

    def _remove_last_vertex(self):
        pts, closed = self._active_poly()
        if not pts or closed:
            return
        self._snapshot_poly()
        pts.pop()
        self._poly_changed("Last point removed.")

    def _cancel_polygon(self):
        pts, closed = self._active_poly()
        if not pts or closed:
            return
        self._snapshot_poly()
        pts.clear()
        self._poly_changed("Drawing cancelled (Undo brings it back).")

    def _undo_poly(self) -> bool:
        """Step the current polygon back one edit.  False if there is none."""
        while self._vtx_hist and self._vtx_hist[-1][0] != self._mode:
            self._vtx_hist.pop()                  # stale entries from another mode
        if not self._vtx_hist:
            return False
        _, pts, closed = self._vtx_hist.pop()
        if self._mode == self._REGION:
            self._poly_pts[:] = pts
        else:
            self._ap_pts[:] = pts
            was_closed = self._ap_mask is not None
            self._ap_mask = self._poly_mask(pts) if closed and len(pts) >= 3 else None
            self._outline_cache = {}
            if was_closed or closed:
                self._refresh_canvas()
        self._redraw_polys()
        state = "confirmed" if closed else "open"
        self._status.configure(text=f"Undone: {len(pts)} point(s), outline {state}.")
        return True

    # ── sub-region definition (AP outline; NTS = the rest) ────────────────────

    def _poly_mask(self, pts) -> np.ndarray:
        from analysis.roi_curation import polygon_mask
        return polygon_mask(pts, self._ih, self._iw)

    def _sreg_add_pt(self, ix, iy):
        """Add an AP-outline vertex, given in image coordinates."""
        if self._ap_mask is not None:
            self._status.configure(
                text="AP is already outlined. Undo clears it to redraw.")
            return
        self._ap_ids += self._draw_vertex(ix, iy, "yellow", "sreg")
        if self._ap_pts:
            self._ap_ids += self._draw_edge(self._ap_pts[-1], (ix, iy), "yellow", "sreg")
        self._ap_pts.append((ix, iy))
        self._status.configure(
            text=f"AP: {len(self._ap_pts)} point(s). Right-click to confirm.")

    def _sreg_close_region(self):
        if self._ap_mask is not None:
            return
        if len(self._ap_pts) < 3:
            self._status.configure(text="Need at least 3 points first.")
            return
        self._ap_ids += self._draw_edge(self._ap_pts[-1], self._ap_pts[0], "yellow", "sreg")
        self._ap_mask = self._poly_mask(self._ap_pts)
        self._outline_cache = {}
        n_ap, n_nts = self._sreg_counts()
        self._status.configure(
            text=f"AP confirmed.  AP: {n_ap}   NTS: {n_nts} neuron(s).  "
                 "Yellow outlines are AP, cyan are NTS.")
        self._refresh_canvas()

    def _sreg_counts(self):
        """(in AP, NTS) by ROI centre: the same test get_region_labels applies
        downstream, so the numbers shown here are the numbers the analysis uses."""
        cents = self._centroids()
        if self._ap_mask is None or not len(cents):
            return 0, len(cents)
        rr = np.clip(cents[:, 0].astype(int), 0, self._ih - 1)
        cc = np.clip(cents[:, 1].astype(int), 0, self._iw - 1)
        n_ap = int(self._ap_mask[rr, cc].sum())
        return n_ap, len(cents) - n_ap

    # ── undo / finish ─────────────────────────────────────────────────────────

    def _push_history(self):
        self._history.append((self._roi_masks.copy(), self._roi_msk.copy(),
                              self._ids.copy(), dict(self._removal_reason),
                              list(self._excl_polys), self._keep_mask.copy()))
        if len(self._history) > 20:
            self._history.pop(0)

    def _undo(self):
        if self._drag is not None:
            return
        # while a polygon is being edited, Undo steps back one vertex edit;
        # only once there are none left does it undo neuron edits
        if self._mode in (self._REGION, self._SUBREGION) and self._undo_poly():
            return
        if self._mode == self._SUBREGION:
            self._sreg_undo()
            return
        if not self._history:
            self._status.configure(text="Nothing to undo.")
            return
        (self._roi_masks, self._roi_msk, self._ids, self._removal_reason,
         self._excl_polys, self._keep_mask) = self._history.pop()
        self._invalidate_density()
        self._status.configure(text=f"Undone. Total: {self._roi_masks.shape[1]}")
        self._refresh_canvas()

    def _sreg_undo(self):
        """No step history left (e.g. an outline restored from a previous
        curation): clear the whole outline."""
        if not self._ap_pts and self._ap_mask is None:
            self._status.configure(text="Nothing to undo.")
            return
        confirmed = self._ap_mask is not None
        self._delete_ids(self._ap_ids)
        self._ap_ids  = []
        self._ap_pts  = []
        self._ap_mask = None
        self._outline_cache = {}
        self._status.configure(
            text="AP outline removed: redraw it." if confirmed
                 else "AP drawing cleared.")
        self._refresh_canvas()

    def _confirm_subregion_before_finish(self) -> bool:
        """True if finishing may proceed.  Catches a forgotten AP outline, which
        would otherwise only surface later as a plane of unclassified neurons."""
        if self._ap_mask is None and len(self._ap_pts) >= 3:
            ans = messagebox.askyesnocancel(
                "AP outline not confirmed",
                f"The AP outline on {self._z} was drawn but not confirmed "
                "(right-click).\n\nYes: confirm it and finish\n"
                "No: discard it\nCancel: keep editing",
                parent=self)
            if ans is None:
                return False
            if ans:
                self._sreg_close_region()

        if self._ap_mask is None:
            ok = messagebox.askyesno(
                "No sub-region defined",
                f"No AP sub-region was defined for {self._z}.\n\n"
                "Its neurons will be unclassified in sub-region analysis "
                "(it can be added later with 'Sub-region setup').\n\n"
                "Finish without defining it?",
                icon="warning", parent=self)
            if not ok:
                self._set_mode(self._SUBREGION)
                self._status.configure(
                    text="Outline the AP, right-click to confirm, then Finish.")
                return False
        return True

    def _curation_record(self, nonempty) -> dict:
        """Everything the curation record needs, handed to the pipeline."""
        from analysis.roi_curation import label_image
        ids = self._ids[nonempty]
        final_labels = label_image(self._roi_masks[:, nonempty],
                                   self._ih, self._iw, ids)[0]

        # one screenshot per reference LUT, rendered exactly as the Red, Green and
        # Merge buttons show them (same contrast settings and channel choice)
        struct = self._struct_bright() if self._struct_ch is not None else None
        if struct is None and self._mc_bkg is not None and self._ch_cur == self._struct_ch:
            struct = self._ref_stretch(self._mc_bkg)
        func_ref = self._func_ref_bright()
        func_roi = self._bright_bkg()

        green_view = np.zeros_like(func_ref)
        green_view[..., 1] = func_ref[..., 0]
        merge_view = np.zeros_like(func_roi)
        merge_view[..., 1] = func_roi[..., 0]
        views = []
        red_lbl = f"tdTomato · {self._struct_ch}" if self._struct_ch else "tdTomato"
        if struct is not None:
            red_view = np.zeros_like(struct)
            red_view[..., 0] = struct[..., 0]
            merge_view[..., 0] = struct[..., 0]
            views.append(("red", f"{red_lbl} (red)", red_view))
        green_lbl = f"GCaMP · {self._func_ch}" if self._func_ch else "GCaMP"
        views.append(("green", f"{green_lbl} (green)", green_view))
        views.append(("merge", "Merge (tdTomato red + GCaMP green)" if struct is not None
                      else "Merge (GCaMP only, no tdTomato loaded)", merge_view))

        return dict(
            n_detected=self._n_detected,
            next_id=self._next_id,
            initial_labels=self._initial_labels,
            final_labels=final_labels,
            final_ids=[int(i) for i in ids],
            final_masks=self._roi_masks[:, nonempty],
            removal_reason=dict(self._removal_reason),
            exclusion_polygons=[list(p) for p in self._excl_polys],
            keep_mask=self._keep_mask.copy(),
            ap_polygon=list(self._ap_pts) if self._ap_mask is not None else None,
            ap_mask=self._ap_mask,
            views=views,
        )

    def _do_finish(self):
        if not self._confirm_subregion_before_finish():
            return

        nonempty = ~(self._roi_masks.sum(axis=0) == 0)
        clean = self._roi_masks[:, nonempty]
        # A display_settings.yaml written before the panels were split has no
        # ref_* keys; __init__ then falls back to the halo-friendly defaults
        # rather than to the functional settings, which would hide it again.
        settings = {
            "gamma":      round(self._gamma_var.get(),     3),
            "lo_pct":     round(self._lo_var.get(),        3),
            "hi_pct":     round(self._hi_var.get(),        3),
            "ref_gamma":  round(self._ref_gamma_var.get(), 3),
            "ref_lo_pct": round(self._ref_lo_var.get(),    3),
            "ref_hi_pct": round(self._ref_hi_var.get(),    3),
            "dens_sigma": round(self._dens_sigma_var.get(), 3),
            "dens_level": round(self._dens_level_var.get(), 3),
            "roi_style":  self._roi_style_var.get(),
            "ref_outlines": bool(self._ref_outline_var.get()),
        }
        # (AP, NTS): NTS is every pixel outside the AP: excluded neurons are
        # already gone, so whatever is left there is NTS
        sreg = ([self._ap_mask, ~self._ap_mask] if self._ap_mask is not None
                else None)
        record = self._curation_record(nonempty)
        self._on_finish(clean, self._roi_bkg, self._roi_msk, settings, sreg, record)
        self.destroy()
