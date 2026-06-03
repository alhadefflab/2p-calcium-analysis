"""Post-CNMF neuron browser.

Opens after source extraction to let the user inspect each neuron's
fluorescence trace and animated spatial footprint, then accept or reject
components before downstream analysis.
"""
from __future__ import annotations

import tkinter as tk
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from gui.neuron import Neuron


_COL_ACCEPTED = (50,  200, 50)   # RGB green  — accepted outline
_COL_REJECTED = (200, 50,  50)   # RGB red    — rejected outline
_COL_SELECTED = (255, 220, 50)   # RGB yellow — selected outline
_ANIM_FPS     = 20               # mini-video playback rate
_ANIM_FRAMES  = 300              # max frames to loop over

# One colour per session for stimulus windows; cycles if > 4 sessions
_SESSION_STIM_COLS = ['#4a9fff', '#ff7a30', '#50d850', '#e050ff']
_PRE_COLOR  = '#888888'   # grey  — pre-discard region
_BASE_COLOR = '#50c8c8'   # teal  — baseline region


# ── shared annotation helper ──────────────────────────────────────────────────

def _annotate_trace_ax(
    ax,
    timing_info: dict | None,
    fontsize: int = 7,
    add_legend: bool = False,
) -> None:
    """Draw shaded regions and session labels on a matplotlib Axes.

    timing_info keys
    ----------------
    fp               : float  — seconds per frame
    pre_f            : int    — pre-discard frames per session
    base_f           : int    — baseline frames per session
    stim_f           : int    — stimulus frames per session
    session_lengths  : list[int]  — total frames per session (actual, from mmap)
    """
    if timing_info is None:
        return

    fp              = timing_info['fp']
    pre_f           = timing_info['pre_f']
    base_f          = timing_info['base_f']
    stim_f          = timing_info['stim_f']
    session_lengths = timing_info['session_lengths']

    # x-axis transform: data coords for x, axes-fraction for y — avoids needing
    # to know ylim at draw time and text stays inside the axes regardless of zoom.
    xform = ax.get_xaxis_transform()

    offset_f = 0
    for j, ses_len in enumerate(session_lengths):
        t_ses_start = offset_f * fp
        t_pre_end   = (offset_f + pre_f)                    * fp
        t_base_end  = (offset_f + pre_f + base_f)           * fp
        t_stim_end  = (offset_f + pre_f + base_f + stim_f)  * fp
        t_ses_end   = (offset_f + ses_len)                  * fp
        stim_col    = _SESSION_STIM_COLS[j % len(_SESSION_STIM_COLS)]

        # Shaded regions
        ax.axvspan(t_ses_start, t_pre_end,  alpha=0.14, color=_PRE_COLOR,  zorder=0, lw=0)
        ax.axvspan(t_pre_end,   t_base_end, alpha=0.20, color=_BASE_COLOR,  zorder=0, lw=0)
        ax.axvspan(t_base_end,  t_stim_end, alpha=0.22, color=stim_col,    zorder=0, lw=0)

        # Stim label at 90 % height inside the stimulus window
        t_stim_mid = (t_base_end + t_stim_end) / 2
        ax.text(
            t_stim_mid, 0.90, f'Stim {j + 1}',
            transform=xform, ha='center', va='top',
            fontsize=fontsize, color=stim_col, alpha=0.95,
        )

        # Session boundary (dashed vertical, except after the last session)
        if j < len(session_lengths) - 1:
            ax.axvline(
                t_ses_end, color='#aaaaaa', lw=0.8, ls='--', alpha=0.6, zorder=1)

        offset_f += ses_len

    if add_legend:
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        handles = [
            Patch(facecolor=_PRE_COLOR,  alpha=0.6, label='Pre-discard',
                  edgecolor='none'),
            Patch(facecolor=_BASE_COLOR, alpha=0.6, label='Baseline',
                  edgecolor='none'),
        ]
        for j in range(len(session_lengths)):
            col = _SESSION_STIM_COLS[j % len(_SESSION_STIM_COLS)]
            handles.append(
                Patch(facecolor=col, alpha=0.6, label=f'Stim {j + 1}',
                      edgecolor='none'))
        if len(session_lengths) > 1:
            handles.append(
                Line2D([0], [0], color='#aaaaaa', ls='--', lw=1,
                       label='Session boundary'))

        ax.legend(
            handles=handles,
            fontsize=fontsize,
            loc='upper right',
            framealpha=0.35,
            facecolor='#1a1a1a',
            edgecolor='#555555',
            labelcolor='#dddddd',
        )


# ── detachable pop-out trace window ──────────────────────────────────────────

class _TracePopout(ctk.CTkToplevel):
    """Larger, detachable trace window that mirrors the selected neuron.

    Can be moved to a second monitor. Updates every time the user selects
    a different neuron in the main viewer. Has a full matplotlib toolbar
    for zoom, pan, and export.
    """

    def __init__(self, parent_viewer: 'NeuronViewerWindow'):
        super().__init__(parent_viewer)
        self.title("Neuron Trace — Expanded View")
        self.geometry("1200x620")
        self.resizable(True, True)
        self._viewer = parent_viewer
        self._cursor_raw = None
        self._cursor_den = None
        self._build()

    def _build(self):
        self.rowconfigure(1, weight=1)
        self.columnconfigure(0, weight=1)

        self._title_lbl = ctk.CTkLabel(
            self, text='', font=ctk.CTkFont(size=13, weight='bold'))
        self._title_lbl.grid(row=0, column=0, pady=(8, 2), padx=12, sticky='w')

        fig_frame = ctk.CTkFrame(self)
        fig_frame.grid(row=1, column=0, sticky='nsew', padx=6, pady=(0, 2))
        fig_frame.rowconfigure(0, weight=1)
        fig_frame.columnconfigure(0, weight=1)

        self._fig = Figure(figsize=(12, 5), facecolor='#1a1a1a')
        self._ax_raw = self._fig.add_subplot(211)
        self._ax_den = self._fig.add_subplot(212)
        self._fig.subplots_adjust(
            hspace=0.50, left=0.05, right=0.98, top=0.92, bottom=0.10)

        self._canvas = FigureCanvasTkAgg(self._fig, master=fig_frame)
        self._canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        # matplotlib navigation toolbar for zoom / pan / home / save
        from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
        tb_frame = tk.Frame(self, bg='#1a1a1a')
        tb_frame.grid(row=2, column=0, sticky='ew', padx=6)
        self._toolbar = NavigationToolbar2Tk(self._canvas, tb_frame)
        self._toolbar.update()

    def _style_axes(self):
        for ax in (self._ax_raw, self._ax_den):
            ax.set_facecolor('#111111')
            ax.tick_params(colors='#888888', labelsize=9)
            for sp in ax.spines.values():
                sp.set_color('#444444')
        self._ax_raw.set_title(
            'Raw  (C + residuals)', color='#cccccc', fontsize=10, pad=4)
        self._ax_den.set_title(
            'Denoised  (C)', color='#cccccc', fontsize=10, pad=4)
        self._ax_den.set_xlabel('Time (s)', color='#888888', fontsize=9)

    def refresh(
        self,
        neuron_idx: int,
        n: Neuron,
        fp: float,
        timing_info: dict | None,
    ) -> None:
        """Redraw both axes for neuron n."""
        status = 'Accepted ✓' if n.accepted else 'Rejected ✗'
        self._title_lbl.configure(
            text=f'Neuron {neuron_idx + 1}  —  {status}')

        T = len(n.trace_raw)
        t = np.arange(T) * fp

        self._ax_raw.cla()
        self._ax_den.cla()
        self._style_axes()

        self._ax_raw.plot(t, n.trace_raw,      color='#5ab4ff', lw=0.8)
        self._ax_den.plot(t, n.trace_denoised, color='#ff9a50', lw=0.9)
        for ax in (self._ax_raw, self._ax_den):
            ax.set_xlim(t[0], t[-1])

        # Draw region annotations — legend on raw trace only
        _annotate_trace_ax(self._ax_raw, timing_info, fontsize=9, add_legend=True)
        _annotate_trace_ax(self._ax_den, timing_info, fontsize=9, add_legend=False)

        # Add cursor (starts at t=0)
        self._cursor_raw = self._ax_raw.axvline(
            0, color='white', lw=1.0, alpha=0.7, zorder=10)
        self._cursor_den = self._ax_den.axvline(
            0, color='white', lw=1.0, alpha=0.7, zorder=10)

        self._canvas.draw()

    def update_cursor(self, t_sec: float) -> None:
        """Move the time cursor without redrawing the full figure."""
        if self._cursor_raw is not None:
            self._cursor_raw.set_xdata([t_sec, t_sec])
        if self._cursor_den is not None:
            self._cursor_den.set_xdata([t_sec, t_sec])
        self._canvas.draw_idle()


# ── main viewer window ────────────────────────────────────────────────────────

class NeuronViewerWindow(ctk.CTkToplevel):
    """Interactive post-CNMF neuron inspector.

    Left panel  : mean image with coloured ROI outlines; click to select.
    Right panel : fluorescence traces + reconstructed calcium mini-video.

    Parameters
    ----------
    parent       : parent tkinter widget (PipelineGUI)
    neurons      : list of Neuron objects built from CNMF estimates
    mean_image   : (h, w) float array — background image for the canvas
    frame_period : seconds per acquired frame (used for time axis)
    on_close     : callback(is_cell: np.ndarray) called when the window closes
    timing_info  : optional dict with keys fp, pre_f, base_f, stim_f,
                   session_lengths — enables trace region annotations
    """

    def __init__(
        self,
        parent,
        neurons: list[Neuron],
        mean_image: np.ndarray,
        frame_period: float = 0.033,
        on_close=None,
        timing_info: dict | None = None,
    ):
        super().__init__(parent)
        self.title("Neuron Viewer — Post-CNMF Curation")
        self.resizable(True, True)

        self._neurons     = neurons
        self._fp          = frame_period
        self._on_close_cb = on_close
        self._timing_info    = timing_info
        self._selected       = 0
        self._legend_visible = False
        self._popout: _TracePopout | None = None

        # Ensure mean_image is 2-D float
        img = np.asarray(mean_image, dtype=np.float32)
        if img.ndim == 3:
            img = img.mean(axis=2)
        self._mean_image = img
        self._ih, self._iw = img.shape[:2]

        # Canvas display dimensions (updated on resize)
        self._dh, self._dw = 500, 500
        self._scale_x = self._dw / self._iw
        self._scale_y = self._dh / self._ih

        # Pre-computed lookup structures
        self._label_map = self._build_label_map()
        self._outlines  = self._compute_outlines()

        # Normalised background image for compositing (0-200 uint8 grayscale)
        lo, hi = np.percentile(img, 1), np.percentile(img, 99)
        bg = np.clip((img - lo) / (hi - lo + 1e-9), 0, 1) if hi > lo else np.zeros_like(img)
        self._bg_norm = (bg * 200).astype(np.uint8)

        # Animation state
        self._anim_neuron:    Neuron | None = None
        self._anim_t_indices: np.ndarray    = np.array([], dtype=int)
        self._anim_t_vals:    np.ndarray    = np.array([])
        self._anim_frame_idx: int           = 0
        self._anim_playing:   bool          = False
        self._anim_job:       str | None    = None

        # Tkinter image references (must be kept alive to prevent GC)
        self._tk_img:  ImageTk.PhotoImage | None = None
        self._tk_mini: ImageTk.PhotoImage | None = None
        self._img_id:  int | None = None
        self._mini_id: int | None = None

        # Matplotlib cursor line handles (embedded figure)
        self._cursor_raw: object | None = None
        self._cursor_den: object | None = None

        self._build_ui()
        self._show_neuron(0)

        self.bind('<Left>',  lambda e: self._prev())
        self.bind('<Right>', lambda e: self._next())
        self.bind('<space>', lambda e: self._toggle_accept())
        self.protocol('WM_DELETE_WINDOW', self._on_close)

        self.after(50, lambda: self.state('zoomed'))
        self.grab_set()

    # ── pre-compute lookup structures ─────────────────────────────────────────

    def _build_label_map(self) -> np.ndarray:
        """(h, w) int32 array: pixel → neuron index, or -1 for background."""
        label = np.full((self._ih, self._iw), -1, dtype=np.int32)
        for k, n in enumerate(self._neurons):
            mask = n.spatial > n.spatial.max() * 0.05
            label[mask] = k
        return label

    def _compute_outlines(self) -> list[np.ndarray]:
        """One boolean (h, w) boundary mask per neuron via erosion."""
        from scipy.ndimage import binary_erosion
        outlines = []
        for n in self._neurons:
            mask = n.spatial > n.spatial.max() * 0.05
            if mask.any():
                outlines.append(mask & ~binary_erosion(mask))
            else:
                outlines.append(np.zeros_like(mask, dtype=bool))
        return outlines

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        outer = ctk.CTkFrame(self)
        outer.pack(fill='both', expand=True, padx=8, pady=8)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=3)   # image canvas stretches
        outer.columnconfigure(1, weight=0)   # panel fixed

        # ── left: image canvas ────────────────────────────────────────────────
        left = ctk.CTkFrame(outer)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            left,
            text="Click a neuron to select  •  ← / → navigate  •  Space = toggle accept / reject",
            font=ctk.CTkFont(size=10), text_color='gray',
        ).grid(row=0, column=0, pady=(4, 2))

        self._canvas = tk.Canvas(left, bg='black', highlightthickness=0)
        self._canvas.grid(row=1, column=0, sticky='nsew')
        self._canvas.bind('<Configure>', self._on_canvas_resize)
        self._canvas.bind('<Button-1>',  self._on_click)

        leg = ctk.CTkFrame(left, fg_color='transparent')
        leg.grid(row=2, column=0, pady=4)
        for hex_col, lbl in [
            ('#32c832', 'accepted'), ('#c83232', 'rejected'), ('#ffdc00', 'selected'),
        ]:
            ctk.CTkLabel(leg, text=f'● {lbl}', text_color=hex_col,
                         font=ctk.CTkFont(size=10)).pack(side='left', padx=10)

        # ── right: scrollable controls panel ─────────────────────────────────
        right = ctk.CTkScrollableFrame(outer, width=390)
        right.grid(row=0, column=1, sticky='nsew')

        self._cell_label = ctk.CTkLabel(
            right, text='', font=ctk.CTkFont(size=13, weight='bold'))
        self._cell_label.pack(pady=(10, 4), padx=12)

        # Navigation
        nav = ctk.CTkFrame(right, fg_color='transparent')
        nav.pack(fill='x', padx=12, pady=4)
        ctk.CTkButton(nav, text='◄  Prev', width=120,
                      command=self._prev).pack(side='left')
        ctk.CTkButton(nav, text='Next  ►', width=120,
                      command=self._next).pack(side='right')

        # Accept / Reject
        ar = ctk.CTkFrame(right, fg_color='transparent')
        ar.pack(fill='x', padx=12, pady=4)
        self._accept_btn = ctk.CTkButton(
            ar, text='Accept ✓', width=140,
            fg_color='#2d6a2d', hover_color='#1e4d1e',
            command=self._accept)
        self._accept_btn.pack(side='left', padx=(0, 4))
        self._reject_btn = ctk.CTkButton(
            ar, text='Reject ✗', width=140,
            fg_color='#7a1a1a', hover_color='#4d0f0f',
            command=self._reject)
        self._reject_btn.pack(side='right', padx=(4, 0))

        # Batch
        batch = ctk.CTkFrame(right, fg_color='transparent')
        batch.pack(fill='x', padx=12, pady=2)
        ctk.CTkButton(batch, text='Accept All', width=140,
                      fg_color='#1a4a1a', hover_color='#103010',
                      command=self._accept_all).pack(side='left')
        ctk.CTkButton(batch, text='Reject All', width=140,
                      fg_color='#4a1a1a', hover_color='#301010',
                      command=self._reject_all).pack(side='right')

        ctk.CTkFrame(right, height=2, fg_color='gray40').pack(fill='x', padx=12, pady=8)

        # ── embedded matplotlib traces ────────────────────────────────────────
        trace_hdr = ctk.CTkFrame(right, fg_color='transparent')
        trace_hdr.pack(fill='x', padx=12, pady=(0, 2))
        ctk.CTkLabel(trace_hdr, text='Fluorescence traces',
                     font=ctk.CTkFont(size=11, weight='bold')).pack(side='left')
        ctk.CTkButton(
            trace_hdr, text='⤢  Pop out', width=96,
            command=self._open_popout,
        ).pack(side='right')
        self._legend_btn = ctk.CTkButton(
            trace_hdr, text='Legend', width=70,
            command=self._toggle_legend,
        )
        self._legend_btn.pack(side='right', padx=(0, 4))

        self._fig = Figure(figsize=(4.6, 3.6), facecolor='#1a1a1a')
        self._ax_raw = self._fig.add_subplot(211)
        self._ax_den = self._fig.add_subplot(212)
        self._fig.subplots_adjust(
            hspace=0.60, left=0.09, right=0.97, top=0.91, bottom=0.13)
        self._style_axes()

        self._mpl_canvas = FigureCanvasTkAgg(self._fig, master=right)
        self._mpl_canvas.get_tk_widget().pack(fill='x', padx=6, pady=2)

        ctk.CTkFrame(right, height=2, fg_color='gray40').pack(fill='x', padx=12, pady=8)

        # ── mini-video ────────────────────────────────────────────────────────
        ctk.CTkLabel(right, text='Calcium signal  (reconstructed)',
                     font=ctk.CTkFont(size=11, weight='bold')).pack(anchor='w', padx=12)

        self._mini_canvas = tk.Canvas(
            right, bg='black', width=200, height=200, highlightthickness=0)
        self._mini_canvas.pack(padx=12, pady=4)

        anim_row = ctk.CTkFrame(right, fg_color='transparent')
        anim_row.pack(fill='x', padx=12, pady=2)
        self._play_btn = ctk.CTkButton(
            anim_row, text='▶  Play', width=100, command=self._toggle_play)
        self._play_btn.pack(side='left')
        self._frame_lbl = ctk.CTkLabel(
            anim_row, text='', text_color='gray', font=ctk.CTkFont(size=10))
        self._frame_lbl.pack(side='left', padx=10)

        self._info_label = ctk.CTkLabel(
            right, text='', text_color='gray', font=ctk.CTkFont(size=10),
            justify='left', anchor='w')
        self._info_label.pack(anchor='w', padx=12, pady=4)

        ctk.CTkFrame(right, height=2, fg_color='gray40').pack(fill='x', padx=12, pady=8)

        ctk.CTkButton(
            right, text='Save & Close',
            height=42, font=ctk.CTkFont(size=13, weight='bold'),
            fg_color='#1a5276', hover_color='#154360',
            command=self._on_close,
        ).pack(padx=12, pady=(4, 14), fill='x')

    def _style_axes(self):
        for ax in (self._ax_raw, self._ax_den):
            ax.set_facecolor('#111111')
            ax.tick_params(colors='#888888', labelsize=7)
            for sp in ax.spines.values():
                sp.set_color('#444444')
        self._ax_raw.set_title('Raw  (C + residuals)', color='#cccccc', fontsize=8, pad=3)
        self._ax_den.set_title('Denoised  (C)',         color='#cccccc', fontsize=8, pad=3)
        self._ax_den.set_xlabel('Time (s)',             color='#888888', fontsize=7)

    # ── canvas resize ─────────────────────────────────────────────────────────

    def _on_canvas_resize(self, event):
        if hasattr(self, '_resize_job'):
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(80, self._apply_resize)

    def _apply_resize(self):
        w = self._canvas.winfo_width()
        h = self._canvas.winfo_height()
        if w > 1 and h > 1:
            self._dw, self._dh = w, h
            self._scale_x = w / self._iw
            self._scale_y = h / self._ih
            self._refresh_image()

    # ── main canvas rendering ─────────────────────────────────────────────────

    def _refresh_image(self):
        """Composite background + coloured ROI outlines and display on canvas."""
        bg_rgb = np.stack([self._bg_norm] * 3, axis=2).copy()
        for k, (n, outline) in enumerate(zip(self._neurons, self._outlines)):
            if not outline.any():
                continue
            col = (_COL_SELECTED if k == self._selected
                   else _COL_ACCEPTED if n.accepted
                   else _COL_REJECTED)
            bg_rgb[outline] = col

        pil = Image.fromarray(bg_rgb.astype(np.uint8)).resize(
            (self._dw, self._dh), Image.BILINEAR)
        self._tk_img = ImageTk.PhotoImage(pil)
        if self._img_id is None:
            self._img_id = self._canvas.create_image(
                0, 0, anchor='nw', image=self._tk_img)
        else:
            self._canvas.itemconfig(self._img_id, image=self._tk_img)
        self._canvas.tag_lower(self._img_id)

    def _on_click(self, event):
        ix = int(max(0, min(event.x / self._scale_x, self._iw - 1)))
        iy = int(max(0, min(event.y / self._scale_y, self._ih - 1)))
        k  = int(self._label_map[iy, ix])
        if k >= 0:
            self._show_neuron(k)

    # ── neuron selection and panel update ─────────────────────────────────────

    def _show_neuron(self, k: int):
        self._stop_anim()
        self._selected = k
        n     = self._neurons[k]
        K     = len(self._neurons)
        n_acc = sum(1 for nn in self._neurons if nn.accepted)

        self._cell_label.configure(
            text=f'Neuron  {k + 1}  /  {K}     ({n_acc} accepted)')

        if n.accepted:
            self._accept_btn.configure(fg_color='#1a7a1a', text='✓  Accepted')
            self._reject_btn.configure(fg_color='#7a1a1a', text='Reject ✗')
        else:
            self._accept_btn.configure(fg_color='#2d6a2d', text='Accept ✓')
            self._reject_btn.configure(fg_color='#aa1a1a', text='✗  Rejected')

        # Traces
        T = len(n.trace_raw)
        t = np.arange(T) * self._fp
        self._ax_raw.cla()
        self._ax_den.cla()
        self._style_axes()
        self._ax_raw.plot(t, n.trace_raw,      color='#5ab4ff', lw=0.7)
        self._ax_den.plot(t, n.trace_denoised, color='#ff9a50', lw=0.9)
        for ax in (self._ax_raw, self._ax_den):
            ax.set_xlim(t[0], t[-1])

        # Draw region annotations; legend only when toggled on
        _annotate_trace_ax(self._ax_raw, self._timing_info,
                           fontsize=6, add_legend=self._legend_visible)
        _annotate_trace_ax(self._ax_den, self._timing_info,
                           fontsize=6, add_legend=False)

        # Cursor line (high zorder so it sits above shaded regions)
        self._cursor_raw = self._ax_raw.axvline(
            0, color='white', lw=0.8, alpha=0.7, zorder=10)
        self._cursor_den = self._ax_den.axvline(
            0, color='white', lw=0.8, alpha=0.7, zorder=10)
        self._mpl_canvas.draw()

        # Static spatial footprint thumbnail
        self._show_mini_frame(n, t_idx=None)

        # Prepare animation frame indices
        self._anim_neuron    = n
        n_frames             = min(T, _ANIM_FRAMES)
        self._anim_t_indices = np.linspace(0, T - 1, n_frames, dtype=int)
        self._anim_t_vals    = self._anim_t_indices * self._fp
        self._anim_frame_idx = 0

        # Neuron statistics
        area = int((n.spatial > n.spatial.max() * 0.05).sum())
        cy, cx = n.centroid
        snr  = float(n.trace_raw.max() / (n.trace_raw.std() + 1e-9))
        peak = float(n.trace_raw.max())
        self._info_label.configure(
            text=f'Centroid: ({cx}, {cy})   Area: {area} px   Peak: {peak:.1f}   SNR: {snr:.1f}')

        self._refresh_image()

        # Mirror update to pop-out if open
        if self._popout is not None and self._popout.winfo_exists():
            self._popout.refresh(k, n, self._fp, self._timing_info)

    # ── legend toggle ─────────────────────────────────────────────────────────

    def _toggle_legend(self):
        self._legend_visible = not self._legend_visible
        self._legend_btn.configure(
            fg_color='#1a5276' if self._legend_visible else ('#3b8ed0', '#1f6aa5'))
        self._show_neuron(self._selected)

    # ── pop-out ───────────────────────────────────────────────────────────────

    def _open_popout(self):
        """Open (or raise) the detachable trace window."""
        if self._popout is None or not self._popout.winfo_exists():
            self._popout = _TracePopout(self)
        n = self._neurons[self._selected]
        self._popout.refresh(self._selected, n, self._fp, self._timing_info)
        self._popout.lift()
        self._popout.focus()

    # ── mini-video frame rendering ────────────────────────────────────────────

    def _make_footprint_rgb(self, n: Neuron, t_idx=None) -> np.ndarray:
        """Return (crop_h, crop_w, 3) uint8 showing neuron activity at t_idx."""
        spatial = n.spatial
        mask    = spatial > spatial.max() * 0.05
        ys, xs  = np.where(mask)
        if len(ys) == 0:
            return np.zeros((200, 200, 3), dtype=np.uint8)

        pad = 14
        y0 = max(0,         int(ys.min()) - pad)
        y1 = min(self._ih,  int(ys.max()) + pad + 1)
        x0 = max(0,         int(xs.min()) - pad)
        x1 = min(self._iw,  int(xs.max()) + pad + 1)

        bg      = self._bg_norm[y0:y1, x0:x1].astype(np.float32) / 255.0 * 0.45
        sp      = spatial[y0:y1, x0:x1]
        s_mx    = sp.max()
        sp_norm = sp / s_mx if s_mx > 0 else np.zeros_like(sp)

        if t_idx is None:
            amp = 0.7
        else:
            t_mx = float(n.trace_denoised.max())
            amp  = max(0.0, float(n.trace_denoised[t_idx]) / t_mx) if t_mx > 1e-9 else 0.0

        overlay = sp_norm * amp
        r = np.clip(bg,                0, 1)
        g = np.clip(bg + overlay,      0, 1)
        b = np.clip(bg + overlay * 0.4, 0, 1)

        return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)

    def _show_mini_frame(self, n: Neuron, t_idx=None):
        rgb    = self._make_footprint_rgb(n, t_idx=t_idx)
        h, w   = rgb.shape[:2]
        scale  = min(200 / max(h, 1), 200 / max(w, 1))
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        pil    = Image.fromarray(rgb).resize((nw, nh), Image.NEAREST)
        canvas_img = Image.new('RGB', (200, 200), (0, 0, 0))
        canvas_img.paste(pil, ((200 - nw) // 2, (200 - nh) // 2))
        self._tk_mini = ImageTk.PhotoImage(canvas_img)
        if self._mini_id is None:
            self._mini_id = self._mini_canvas.create_image(
                0, 0, anchor='nw', image=self._tk_mini)
        else:
            self._mini_canvas.itemconfig(self._mini_id, image=self._tk_mini)

    # ── animation ─────────────────────────────────────────────────────────────

    def _anim_step(self):
        if not self._anim_playing or self._anim_neuron is None:
            return
        n   = self._anim_neuron
        idx = int(self._anim_t_indices[self._anim_frame_idx])
        self._show_mini_frame(n, t_idx=idx)

        # Sync cursors (embedded + pop-out)
        t_sec = float(self._anim_t_vals[self._anim_frame_idx])
        if self._cursor_raw is not None:
            self._cursor_raw.set_xdata([t_sec, t_sec])
        if self._cursor_den is not None:
            self._cursor_den.set_xdata([t_sec, t_sec])
        self._mpl_canvas.draw_idle()
        if self._popout is not None and self._popout.winfo_exists():
            self._popout.update_cursor(t_sec)

        # Progress indicator
        pct = int(self._anim_frame_idx / max(1, len(self._anim_t_indices) - 1) * 100)
        self._frame_lbl.configure(text=f'{pct}%')

        self._anim_frame_idx = (self._anim_frame_idx + 1) % len(self._anim_t_indices)
        self._anim_job = self.after(1000 // _ANIM_FPS, self._anim_step)

    def _toggle_play(self):
        if self._anim_playing:
            self._stop_anim()
        else:
            self._start_anim()

    def _start_anim(self):
        if not len(self._anim_t_indices):
            return
        self._anim_playing = True
        self._play_btn.configure(text='■  Pause')
        self._anim_step()

    def _stop_anim(self):
        self._anim_playing = False
        self._play_btn.configure(text='▶  Play')
        if self._anim_job is not None:
            self.after_cancel(self._anim_job)
            self._anim_job = None

    # ── accept / reject ───────────────────────────────────────────────────────

    def _accept(self):
        self._neurons[self._selected].accepted = True
        self._show_neuron(self._selected)

    def _reject(self):
        self._neurons[self._selected].accepted = False
        self._show_neuron(self._selected)

    def _toggle_accept(self):
        n = self._neurons[self._selected]
        n.accepted = not n.accepted
        self._show_neuron(self._selected)

    def _accept_all(self):
        for n in self._neurons:
            n.accepted = True
        self._show_neuron(self._selected)

    def _reject_all(self):
        for n in self._neurons:
            n.accepted = False
        self._show_neuron(self._selected)

    # ── navigation ────────────────────────────────────────────────────────────

    def _prev(self):
        if self._selected > 0:
            self._show_neuron(self._selected - 1)

    def _next(self):
        if self._selected < len(self._neurons) - 1:
            self._show_neuron(self._selected + 1)

    # ── close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_anim()
        if self._popout is not None and self._popout.winfo_exists():
            self._popout.destroy()
        is_cell = np.array([n.accepted for n in self._neurons], dtype=bool)
        if self._on_close_cb is not None:
            self._on_close_cb(is_cell)
        self.destroy()
