"""Cross-z-plane neuron map and duplicate detection.

Duplicate detection:
  1. Centroid distance as a fast pre-filter (not exposed to user).
  2. Jaccard IoU of binary spatial masks as the primary spatial criterion.
  3. Temporal trace correlation as confirmation.

Left panel  : 2-D spatial overlay — all neuron outlines on mean image, coloured
              by z-plane, with ID labels on duplicate candidates.
Browser     : plotly 3-D reconstruction — actual spatial footprints + ID labels
              on all accepted neurons, white dashed lines between duplicates.
Right panel : IoU + corr threshold sliders, neuron search box, duplicate cards.
"""
from __future__ import annotations

import tkinter as tk
import customtkinter as ctk
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from gui.neuron import Neuron


_PLANE_COLS = ['#4a9fff', '#ff7a30', '#50d850', '#e050ff']

# Centroid pre-filter — only pairs within this many pixels are IoU-tested.
# Any pair further apart than this cannot have overlapping footprints for
# typical 2P soma sizes (~10-20 px diameter).
_DIST_PREFILTER_PX = 50

# Default thresholds (user-adjustable via sliders)
# Default max_overlap = 0.75 (one-sided coverage).
# Jaccard IoU equivalent for equal-sized neurons ≈ 0.60; cross-plane same neuron
# seen at different depths typically 0.50–0.60.  Use 0.60 as the standard default.
_DEFAULT_IOU_MIN  = 0.60
_DEFAULT_CORR_MIN = 0.85


# ── spatial mask helpers ──────────────────────────────────────────────────────

def _binary_mask(n: Neuron, thr: float = 0.20) -> np.ndarray:
    return n.spatial > n.spatial.max() * thr


def _jaccard_iou(n1: Neuron, n2: Neuron) -> float:
    """Jaccard index (intersection / union) of two binary spatial masks.
    """
    m1 = _binary_mask(n1)
    m2 = _binary_mask(n2)
    inter = int((m1 & m2).sum())
    union = int((m1 | m2).sum())
    return inter / union if union > 0 else 0.0


def _neuron_boundary(n: Neuron) -> tuple[np.ndarray, np.ndarray]:
    """Boundary pixel (ys, xs) of a neuron's spatial mask via erosion."""
    from scipy.ndimage import binary_erosion
    mask = _binary_mask(n)
    if not mask.any():
        return np.array([], int), np.array([], int)
    return np.where(mask & ~binary_erosion(mask))


def _neuron_interior_sparse(n: Neuron, max_pts: int = 30
                             ) -> tuple[np.ndarray, np.ndarray]:
    """Sparse interior sample for 3-D volume fill."""
    from scipy.ndimage import binary_erosion
    mask = _binary_mask(n)
    if not mask.any():
        return np.array([], int), np.array([], int)
    interior = mask & binary_erosion(mask)
    ys, xs = np.where(interior)
    if len(xs) > max_pts:
        idx = np.round(np.linspace(0, len(xs) - 1, max_pts)).astype(int)
        ys, xs = ys[idx], xs[idx]
    return ys, xs


def _neuron_id(z: str, k: int) -> str:
    """Canonical neuron ID string, e.g. 'z3 #5'."""
    return f'{z} #{k + 1}'


# ── duplicate detection ───────────────────────────────────────────────────────

def _find_duplicates(
    plane_data: list[dict],
    decisions: dict[tuple[str, int], bool],
    iou_threshold: float,
    corr_threshold: float,
) -> list[dict]:
    """Return cross-z-plane duplicate-candidate pairs using IoU + correlation.

    Algorithm mirrors CaImAn register_multisession:
      - centroid distance pre-filter for efficiency
      - Jaccard IoU of binary masks as primary spatial criterion
      - trace correlation as temporal confirmation
    Sorted by IoU descending (highest spatial overlap first).
    Each entry: z1, k1, n1, z2, k2, n2, dist, iou, corr, id1, id2.
    """
    dupes: list[dict] = []
    for i in range(len(plane_data)):
        for j in range(i + 1, len(plane_data)):
            p1, p2 = plane_data[i], plane_data[j]
            for k1, n1 in enumerate(p1['neurons']):
                if not decisions.get((p1['z'], k1), True):
                    continue
                cy1, cx1 = n1.centroid
                for k2, n2 in enumerate(p2['neurons']):
                    if not decisions.get((p2['z'], k2), True):
                        continue
                    cy2, cx2 = n2.centroid
                    # Fast centroid pre-filter
                    dist = float(np.hypot(cx1 - cx2, cy1 - cy2))
                    if dist > _DIST_PREFILTER_PX:
                        continue
                    # Spatial: Jaccard IoU
                    iou = _jaccard_iou(n1, n2)
                    if iou < iou_threshold:
                        continue
                    # Temporal: trace correlation
                    T = min(len(n1.trace_denoised), len(n2.trace_denoised))
                    if T < 10:
                        continue
                    corr = float(np.corrcoef(
                        n1.trace_denoised[:T], n2.trace_denoised[:T])[0, 1])
                    if corr < corr_threshold:
                        continue
                    dupes.append(dict(
                        z1=p1['z'], k1=k1, n1=n1,
                        z2=p2['z'], k2=k2, n2=n2,
                        id1=_neuron_id(p1['z'], k1),
                        id2=_neuron_id(p2['z'], k2),
                        dist=dist, iou=iou, corr=corr,
                    ))
    return sorted(dupes, key=lambda d: -d['iou'])


# ── main viewer window ────────────────────────────────────────────────────────

class ZPlaneViewerWindow(ctk.CTkToplevel):
    """Multi-z-plane neuron map with IoU-based duplicate detection.

    Parameters
    ----------
    parent     : parent tkinter widget
    plane_data : list of dicts (one per z-plane):
                     z          str           — label ('z1', 'z2', …)
                     neurons    list[Neuron]  — all K components
                     is_cell    np.ndarray    — bool (K,) current accept state
                     mean_image np.ndarray    — (h, w) float background
    on_close   : callback({z: updated_is_cell_array, …})
    """

    def __init__(self, parent, plane_data: list[dict], on_close=None):
        super().__init__(parent)
        self.title("Multi-plane Neuron Map — Duplicate Review")
        self.resizable(True, True)

        self._plane_data  = plane_data
        self._on_close_cb = on_close

        self._decisions: dict[tuple[str, int], bool] = {}
        for pd in plane_data:
            for k in range(len(pd['neurons'])):
                self._decisions[(pd['z'], k)] = bool(pd['is_cell'][k])

        self._iou_thr  = _DEFAULT_IOU_MIN
        self._corr_thr = _DEFAULT_CORR_MIN
        self._search   = ''
        self._z_step_um: float | None = None   # set by caller if known

        self._duplicates = _find_duplicates(
            plane_data, self._decisions, self._iou_thr, self._corr_thr)

        self._build_ui()
        self._refresh_overlay()
        self._refresh_dup_list()

        # Auto-open 3-D reconstruction in browser
        self.after(400, self._open_plotly)
        self.after(50, lambda: self.state('zoomed'))
        self.grab_set()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = ctk.CTkFrame(self)
        outer.pack(fill='both', expand=True, padx=8, pady=8)
        outer.rowconfigure(0, weight=1)
        outer.columnconfigure(0, weight=3)
        outer.columnconfigure(1, weight=1)

        # ── left: 2-D overlay ─────────────────────────────────────────────────
        left = ctk.CTkFrame(outer)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)

        top_bar = ctk.CTkFrame(left, fg_color='transparent')
        top_bar.grid(row=0, column=0, sticky='ew', pady=(4, 2), padx=8)
        ctk.CTkLabel(
            top_bar,
            text="2-D overlay — all z-planes combined  •  white ring + ID = duplicate candidate",
            font=ctk.CTkFont(size=10), text_color='gray',
        ).pack(side='left')
        ctk.CTkButton(
            top_bar, text='3-D in browser', width=130,
            command=self._open_plotly,
        ).pack(side='right')

        fig_frame = ctk.CTkFrame(left)
        fig_frame.grid(row=1, column=0, sticky='nsew')
        fig_frame.rowconfigure(0, weight=1)
        fig_frame.columnconfigure(0, weight=1)

        self._fig2d = Figure(figsize=(7, 6), facecolor='#111111')
        self._ax2d  = self._fig2d.add_subplot(111)
        self._fig2d.subplots_adjust(left=0, right=1, top=1, bottom=0)

        self._canvas2d = FigureCanvasTkAgg(self._fig2d, master=fig_frame)
        self._canvas2d.get_tk_widget().grid(row=0, column=0, sticky='nsew')

        tb_frame = tk.Frame(left, bg='#1a1a1a')
        tb_frame.grid(row=2, column=0, sticky='ew')
        NavigationToolbar2Tk(self._canvas2d, tb_frame).update()

        leg_row = ctk.CTkFrame(left, fg_color='transparent')
        leg_row.grid(row=3, column=0, pady=4)
        for i, pd in enumerate(self._plane_data):
            col = _PLANE_COLS[i % len(_PLANE_COLS)]
            ctk.CTkLabel(leg_row, text=f'● {pd["z"]}', text_color=col,
                         font=ctk.CTkFont(size=10)).pack(side='left', padx=10)
        ctk.CTkLabel(leg_row, text='○ + ID = dup candidate', text_color='white',
                     font=ctk.CTkFont(size=10)).pack(side='left', padx=10)
        ctk.CTkLabel(leg_row, text='· rejected', text_color='#555555',
                     font=ctk.CTkFont(size=10)).pack(side='left', padx=10)

        # ── right: controls ───────────────────────────────────────────────────
        right = ctk.CTkFrame(outer, width=400)
        right.grid(row=0, column=1, sticky='nsew')
        right.rowconfigure(5, weight=1)   # dup_frame expands
        right.columnconfigure(0, weight=1)
        right.grid_propagate(False)

        self._stats_lbl = ctk.CTkLabel(
            right, text='',
            font=ctk.CTkFont(size=12, weight='bold'),
            wraplength=380, justify='left')
        self._stats_lbl.grid(row=0, column=0, pady=(12, 4), padx=12, sticky='w')

        # Threshold sliders
        thr_frame = ctk.CTkFrame(right, fg_color='#282828', corner_radius=6)
        thr_frame.grid(row=1, column=0, padx=8, pady=4, sticky='ew')
        thr_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(thr_frame,
                     text='Detection  (IoU = spatial mask overlap)',
                     font=ctk.CTkFont(size=9, weight='bold'),
                     text_color='#aaaaaa').grid(
            row=0, column=0, columnspan=3, padx=10, pady=(8, 3), sticky='w')

        self._iou_var  = tk.DoubleVar(value=_DEFAULT_IOU_MIN)
        self._corr_var = tk.DoubleVar(value=_DEFAULT_CORR_MIN)

        def _slider(row, label, var, lo, hi, steps, fmt):
            ctk.CTkLabel(thr_frame, text=label, width=110, anchor='w',
                         font=ctk.CTkFont(size=9),
                         text_color='#cccccc').grid(
                row=row, column=0, padx=(10, 4), pady=3, sticky='w')
            val_lbl = ctk.CTkLabel(thr_frame, text=fmt.format(var.get()),
                                   width=40, anchor='e', font=ctk.CTkFont(size=9),
                                   text_color='#cccccc')
            def _cb(v, lbl=val_lbl, variable=var, f=fmt):
                variable.set(float(v)); lbl.configure(text=f.format(float(v)))
            ctk.CTkSlider(thr_frame, from_=lo, to=hi, number_of_steps=steps,
                          variable=var, command=_cb, width=140).grid(
                row=row, column=1, padx=4, pady=3)
            val_lbl.grid(row=row, column=2, padx=(0, 10), pady=3)

        _slider(1, 'Min IoU:',  self._iou_var,  0.02, 0.80, 39, '{:.2f}')
        _slider(2, 'Min corr:', self._corr_var, 0.70, 0.99, 29, '{:.2f}')

        ctk.CTkButton(
            thr_frame, text='Re-detect', width=110,
            fg_color='#3b8ed0', hover_color='#1f6aa5',
            command=self._redetect,
        ).grid(row=3, column=0, columnspan=3, padx=10, pady=(3, 10))

        ctk.CTkLabel(
            right,
            text='Run per-plane Neuron Curation first to remove noise before duplicate review.',
            font=ctk.CTkFont(size=9), text_color='gray',
            wraplength=380, justify='left',
        ).grid(row=2, column=0, padx=12, pady=(2, 4), sticky='w')

        # Search box
        search_frame = ctk.CTkFrame(right, fg_color='transparent')
        search_frame.grid(row=3, column=0, padx=8, pady=(2, 0), sticky='ew')
        search_frame.columnconfigure(1, weight=1)
        ctk.CTkLabel(search_frame, text='Search:', width=52,
                     font=ctk.CTkFont(size=10)).grid(row=0, column=0, padx=(4, 4))
        self._search_var = ctk.StringVar()
        self._search_var.trace_add('write', self._on_search_change)
        ctk.CTkEntry(
            search_frame, textvariable=self._search_var, width=220,
            placeholder_text="neuron ID e.g.  z3 #5  or just  z3",
        ).grid(row=0, column=1, sticky='ew', padx=(0, 4), pady=4)
        ctk.CTkButton(
            search_frame, text='✕', width=28,
            command=lambda: self._search_var.set(''),
        ).grid(row=0, column=2, padx=(0, 4))

        # Duplicate list header + help button
        list_hdr = ctk.CTkFrame(right, fg_color='transparent')
        list_hdr.grid(row=4, column=0, padx=8, pady=(4, 0), sticky='ew')
        list_hdr.columnconfigure(0, weight=1)
        ctk.CTkLabel(list_hdr, text='Duplicate candidates',
                     font=ctk.CTkFont(size=10, weight='bold'),
                     text_color='#cccccc').pack(side='left')
        ctk.CTkButton(
            list_hdr, text='Which plane to keep?  ℹ', width=195,
            fg_color='#2a3a4a', hover_color='#1a2a3a',
            font=ctk.CTkFont(size=10),
            command=self._show_plane_help,
        ).pack(side='right')

        # Duplicate list
        self._dup_frame = ctk.CTkScrollableFrame(right)
        self._dup_frame.grid(row=5, column=0, sticky='nsew', padx=6, pady=4)
        self._dup_frame.columnconfigure(0, weight=1)

        ctk.CTkButton(
            right, text='Save & Close',
            height=44, font=ctk.CTkFont(size=13, weight='bold'),
            fg_color='#1a5276', hover_color='#154360',
            command=self._on_close,
        ).grid(row=6, column=0, padx=12, pady=(4, 14), sticky='ew')

    # ── 2-D overlay ───────────────────────────────────────────────────────────

    def _refresh_overlay(self):
        self._ax2d.cla()
        self._ax2d.set_facecolor('#111111')
        self._ax2d.axis('off')

        all_imgs = [pd['mean_image'] for pd in self._plane_data]
        bg = np.stack(all_imgs, axis=0).max(axis=0)
        lo, hi = np.percentile(bg, 1), np.percentile(bg, 99)
        bg_norm = np.clip((bg - lo) / (hi - lo + 1e-9), 0, 1)
        self._ax2d.imshow(bg_norm, cmap='gray', vmin=0, vmax=1, origin='upper')

        dup_set: set[tuple[str, int]] = set()
        for d in self._duplicates:
            dup_set.add((d['z1'], d['k1']))
            dup_set.add((d['z2'], d['k2']))

        for i, pd in enumerate(self._plane_data):
            col = _PLANE_COLS[i % len(_PLANE_COLS)]
            for k, n in enumerate(pd['neurons']):
                accepted = self._decisions.get((pd['z'], k), True)
                bys, bxs = _neuron_boundary(n)
                if not len(bxs):
                    continue
                pt_col = col if accepted else '#444444'
                sz     = 1.5 if accepted else 0.8
                alpha  = 0.85 if accepted else 0.35
                self._ax2d.scatter(bxs, bys, c=pt_col, s=sz,
                                   linewidths=0, alpha=alpha)
                if accepted and (pd['z'], k) in dup_set:
                    cy, cx = n.centroid
                    self._ax2d.scatter([cx], [cy], c='none',
                                       edgecolors='white', s=130,
                                       linewidths=1.1, alpha=0.9, zorder=10)
                    self._ax2d.text(cx + 3, cy - 3, _neuron_id(pd['z'], k),
                                   color='white', fontsize=5.5,
                                   va='bottom', zorder=11,
                                   bbox=dict(boxstyle='round,pad=0.15',
                                             fc='#00000099', ec='none'))
        self._canvas2d.draw()

    # ── duplicate list ────────────────────────────────────────────────────────

    def _on_search_change(self, *_):
        self._search = self._search_var.get().strip().lower()
        self._refresh_dup_list()

    def _matches_search(self, dup: dict) -> bool:
        if not self._search:
            return True
        term = self._search.strip().lower()

        for raw_id in (dup['id1'], dup['id2']):
            # raw_id is always "zN #M", e.g. "z3 #5"
            parts = raw_id.lower().split(' #')
            if len(parts) != 2:
                continue
            z_part, num_part = parts  # e.g. "z3", "5"

            if term.isdigit():
                # Pure number → exact match on neuron number only
                # "55" matches z3 #55 but NOT z5 #5
                if num_part == term:
                    return True

            elif term.startswith('z') and term[1:].isdigit():
                # "z3" pattern → match z-plane exactly
                if z_part == term:
                    return True

            elif term.startswith('z') and '#' in term:
                # "z3 #5" or "z3#5" → match both parts exactly
                t_norm  = term.replace(' ', '')          # "z3#5"
                id_norm = raw_id.lower().replace(' ', '') # "z3#5"
                if id_norm == t_norm or id_norm.startswith(t_norm):
                    return True

            else:
                # General substring on the full ID (spaces preserved)
                if term in raw_id.lower():
                    return True

        return False

    def _refresh_dup_list(self):
        for w in self._dup_frame.winfo_children():
            w.destroy()

        n_acc   = sum(1 for v in self._decisions.values() if v)
        n_total = sum(len(pd['neurons']) for pd in self._plane_data)
        visible = [d for d in self._duplicates if self._matches_search(d)]
        self._stats_lbl.configure(
            text=(f'{n_acc} / {n_total} accepted  across '
                  f'{len(self._plane_data)} z-planes\n'
                  f'{len(self._duplicates)} duplicate candidate(s)'
                  + (f'  —  {len(visible)} shown' if self._search else '')))

        if not self._duplicates:
            ctk.CTkLabel(
                self._dup_frame,
                text='No duplicates at current thresholds.',
                text_color='#50d850', font=ctk.CTkFont(size=11),
            ).pack(pady=30, padx=12)
            return

        if not visible:
            ctk.CTkLabel(
                self._dup_frame,
                text=f'No matches for "{self._search_var.get()}".',
                text_color='gray', font=ctk.CTkFont(size=11),
            ).pack(pady=20, padx=12)
            return

        for dup in visible:
            self._add_dup_card(dup)

    def _add_dup_card(self, dup: dict):
        dec1     = self._decisions.get((dup['z1'], dup['k1']), True)
        dec2     = self._decisions.get((dup['z2'], dup['k2']), True)
        resolved = not (dec1 and dec2)
        bg       = '#1a2e1a' if resolved else '#242424'

        # Highlight if search matches this card
        if self._search and self._matches_search(dup):
            bg = '#1a2a3a' if not resolved else bg

        card = ctk.CTkFrame(self._dup_frame, fg_color=bg, corner_radius=6)
        card.pack(fill='x', padx=4, pady=3)
        card.columnconfigure(0, weight=1)

        col1 = _PLANE_COLS[next(
            (ii for ii, p in enumerate(self._plane_data) if p['z'] == dup['z1']), 0
        ) % len(_PLANE_COLS)]
        col2 = _PLANE_COLS[next(
            (ii for ii, p in enumerate(self._plane_data) if p['z'] == dup['z2']), 1
        ) % len(_PLANE_COLS)]

        hdr = ctk.CTkFrame(card, fg_color='transparent')
        hdr.pack(fill='x', padx=8, pady=(6, 1))
        ctk.CTkLabel(hdr, text=dup['id1'],
                     text_color=col1, font=ctk.CTkFont(size=11, weight='bold')
                     ).pack(side='left')
        ctk.CTkLabel(hdr, text='  ↔  ', text_color='gray',
                     font=ctk.CTkFont(size=11)).pack(side='left')
        ctk.CTkLabel(hdr, text=dup['id2'],
                     text_color=col2, font=ctk.CTkFont(size=11, weight='bold')
                     ).pack(side='left')

        status_parts = []
        if not dec1: status_parts.append(f'{dup["z1"]} rejected')
        if not dec2: status_parts.append(f'{dup["z2"]} rejected')
        status_str = f'  →  {", ".join(status_parts)}' if status_parts else ''
        n1, n2 = dup['n1'], dup['n2']
        cy1, cx1 = n1.centroid
        cy2, cx2 = n2.centroid
        info = (f'IoU {dup["iou"]:.3f}   r = {dup["corr"]:.3f}'
                f'   dist {dup["dist"]:.0f} px{status_str}\n'
                f'{dup["id1"]} ({cx1},{cy1})  peak {n1.trace_raw.max():.1f}  |  '
                f'{dup["id2"]} ({cx2},{cy2})  peak {n2.trace_raw.max():.1f}')
        ctk.CTkLabel(card, text=info, text_color='gray',
                     font=ctk.CTkFont(size=9), justify='left'
                     ).pack(anchor='w', padx=10, pady=1)

        btns = ctk.CTkFrame(card, fg_color='transparent')
        btns.pack(fill='x', padx=8, pady=(3, 7))
        ctk.CTkButton(btns, text=f'Keep {dup["z1"]}', width=88,
                      fg_color='#1a4a1a', hover_color='#103010',
                      command=lambda d=dup: self._resolve(d, 'first')
                      ).pack(side='left', padx=(0, 3))
        ctk.CTkButton(btns, text=f'Keep {dup["z2"]}', width=88,
                      fg_color='#1a4a1a', hover_color='#103010',
                      command=lambda d=dup: self._resolve(d, 'second')
                      ).pack(side='left', padx=3)
        ctk.CTkButton(btns, text='Keep both', width=78,
                      fg_color='#2a3a4a', hover_color='#1a2838',
                      command=lambda d=dup: self._resolve(d, 'both')
                      ).pack(side='left', padx=3)
        ctk.CTkButton(btns, text='Reject both', width=84,
                      fg_color='#4a1a1a', hover_color='#301010',
                      command=lambda d=dup: self._resolve(d, 'none')
                      ).pack(side='left', padx=(3, 0))

    # ── help popup ────────────────────────────────────────────────────────────

    def _show_plane_help(self):
        """Popup explaining how to choose which z-plane to keep for a duplicate."""
        content = """\
Which plane to keep?

Use the Peak value shown in each card.

Peak is the maximum raw fluorescence across the entire recording
(C + YrA, CNMF arbitrary units). The plane with the higher peak
is recording from where the neuron soma sits closest to the focal
plane. The 2-photon signal drops steeply above and below focus
(a quadratic nonlinear process), so the brightest copy is always
the most accurate representation of that cell.

Your acquisition: 25x 0.95 NA objective, z-step = 24 micrometers,
1.21 micrometers per pixel lateral.

At 24 micrometer steps a typical cortical neuron (10 to 20
micrometers soma) should appear in at most 1 to 2 planes.

Note on brainstem neurons (DVC, NTS, DMV):
Neurons in these regions have large soma diameters of 25 to 50
micrometers. At a 24 micrometer z-step, a large vagal motor neuron
will appear across 2 or 3 consecutive planes. This is not an
artefact. High IoU and high r across 3 planes simply confirms a
genuinely large cell recorded three times. Keep the middle plane.

Rules of thumb
--------------
2 planes: keep the higher peak, reject the lower.

3 planes: keep the middle plane, which almost always has the
highest peak and the most complete circular footprint. The flanking
planes are imaging the top and bottom caps of the soma.

Peaks very similar (less than 10 percent difference): the soma
centre sits near the boundary between two focal planes. Either copy
is acceptable. Prefer the one with the tighter, more circular
footprint in the 3-D browser.

About the other metrics
-----------------------
r (correlation): how synchronously the two traces fire across the
entire recording. Values above 0.97 with high IoU are nearly
certain duplicates. Values between 0.85 and 0.95 can still be real
duplicates where one plane has weaker signal and therefore noisier
fluctuations.

IoU: fraction of shared pixels out of all pixels belonging to
either neuron. IoU above 0.60 means the
footprints substantially overlap in x-y space, which is strong
spatial evidence for the same cell.

dist: centroid distance in pixels. Used only as a fast pre-filter;
pairs more than 50 pixels apart are never tested. Not a decision
criterion by itself. Two adjacent cells can be 8 pixels apart and
be genuinely different neurons with IoU near zero.
"""
        win = ctk.CTkToplevel(self)
        win.title("Which plane to keep? - Guide")
        win.geometry("560x500")
        win.resizable(True, True)
        win.grab_set()

        box = ctk.CTkTextbox(
            win, wrap='word',
            font=ctk.CTkFont(family='Helvetica', size=11))
        box.pack(fill='both', expand=True, padx=14, pady=(12, 4))
        box.insert('0.0', content)
        box.configure(state='disabled')

        ctk.CTkButton(win, text='Close', width=100,
                      command=win.destroy).pack(pady=(4, 12))

    # ── actions ───────────────────────────────────────────────────────────────

    def _redetect(self):
        self._iou_thr  = float(self._iou_var.get())
        self._corr_thr = float(self._corr_var.get())
        self._duplicates = _find_duplicates(
            self._plane_data, self._decisions, self._iou_thr, self._corr_thr)
        self._refresh_overlay()
        self._refresh_dup_list()

    def _resolve(self, dup: dict, keep: str):
        self._decisions[(dup['z1'], dup['k1'])] = keep in ('first', 'both')
        self._decisions[(dup['z2'], dup['k2'])] = keep in ('second', 'both')
        self._refresh_overlay()
        self._refresh_dup_list()

    # ── plotly 3-D reconstruction ─────────────────────────────────────────────

    def _open_plotly(self):
        """Render actual neuron footprints in 3-D with neuron ID labels.

        Boundary pixels: solid coloured dots (one per accepted neuron).
        Interior pixels: semi-transparent fill for volume feel.
        Rejected neurons: dim grey ghost outlines.
        Duplicate pairs: white dashed line connecting centroids.
        ID labels: shown at every accepted neuron's centroid.
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            from tkinter import messagebox
            messagebox.showwarning(
                'plotly not installed',
                'Install plotly for the 3-D view:\n\n  pip install plotly',
                parent=self)
            return

        import tempfile, webbrowser

        dup_set: set[tuple[str, int]] = set()
        for d in self._duplicates:
            dup_set.add((d['z1'], d['k1']))
            dup_set.add((d['z2'], d['k2']))

        fig = go.Figure()

        for i, pd in enumerate(self._plane_data):
            col = _PLANE_COLS[i % len(_PLANE_COLS)]

            bnd_xs, bnd_ys, bnd_zs, bnd_txt = [], [], [], []
            int_xs, int_ys, int_zs          = [], [], []
            rej_xs, rej_ys, rej_zs          = [], [], []

            # Centroid label data (one point per accepted neuron)
            lbl_xs, lbl_ys, lbl_zs, lbl_txt, lbl_sym = [], [], [], [], []

            for k, n in enumerate(pd['neurons']):
                accepted = self._decisions.get((pd['z'], k), True)
                nid      = _neuron_id(pd['z'], k)
                is_dup   = (pd['z'], k) in dup_set
                hover    = (f'{nid}<br>'
                            f'centroid ({n.centroid[1]}, {n.centroid[0]})<br>'
                            f'peak {n.trace_raw.max():.1f}'
                            + ('<br>⚠ duplicate candidate' if is_dup else ''))

                bys, bxs = _neuron_boundary(n)
                iys, ixs = _neuron_interior_sparse(n)

                if accepted:
                    bnd_xs.extend(bxs.tolist()); bnd_ys.extend(bys.tolist())
                    bnd_zs.extend([float(i)] * len(bxs))
                    bnd_txt.extend([hover] * len(bxs))
                    int_xs.extend(ixs.tolist()); int_ys.extend(iys.tolist())
                    int_zs.extend([float(i)] * len(ixs))
                    # Centroid label point
                    lbl_xs.append(float(n.centroid[1]))
                    lbl_ys.append(float(n.centroid[0]))
                    lbl_zs.append(float(i))
                    lbl_txt.append(nid)
                    lbl_sym.append('diamond' if is_dup else 'circle')
                else:
                    rej_xs.extend(bxs.tolist()); rej_ys.extend(bys.tolist())
                    rej_zs.extend([float(i)] * len(bxs))

            # Footprint outlines
            if bnd_xs:
                fig.add_trace(go.Scatter3d(
                    x=bnd_xs, y=bnd_ys, z=bnd_zs, mode='markers',
                    name=pd['z'],
                    marker=dict(size=3, color=col, opacity=0.90),
                    text=bnd_txt, hoverinfo='text'))
            # Sparse interior fill
            if int_xs:
                fig.add_trace(go.Scatter3d(
                    x=int_xs, y=int_ys, z=int_zs, mode='markers',
                    name=f'{pd["z"]} fill', showlegend=False,
                    marker=dict(size=2, color=col, opacity=0.18),
                    hoverinfo='none'))
            # Rejected ghosts
            if rej_xs:
                fig.add_trace(go.Scatter3d(
                    x=rej_xs, y=rej_ys, z=rej_zs, mode='markers',
                    name=f'{pd["z"]} rejected', showlegend=False,
                    marker=dict(size=2, color='#555555', opacity=0.28),
                    hoverinfo='none'))
            # Neuron ID labels at centroids
            if lbl_xs:
                fig.add_trace(go.Scatter3d(
                    x=lbl_xs, y=lbl_ys, z=lbl_zs,
                    mode='markers+text',
                    name=f'{pd["z"]} IDs', showlegend=False,
                    marker=dict(size=4, color=col, symbol=lbl_sym,
                                opacity=0.95,
                                line=dict(width=1, color='white')),
                    text=lbl_txt,
                    textposition='top center',
                    textfont=dict(size=9, color='white'),
                    hoverinfo='none'))

        # Dashed lines connecting duplicate centroid pairs
        for dup in self._duplicates:
            z1i = next((ii for ii, p in enumerate(self._plane_data)
                        if p['z'] == dup['z1']), 0)
            z2i = next((ii for ii, p in enumerate(self._plane_data)
                        if p['z'] == dup['z2']), 1)
            cx1 = float(dup['n1'].centroid[1])
            cy1 = float(dup['n1'].centroid[0])
            cx2 = float(dup['n2'].centroid[1])
            cy2 = float(dup['n2'].centroid[0])
            fig.add_trace(go.Scatter3d(
                x=[cx1, cx2], y=[cy1, cy2],
                z=[float(z1i), float(z2i)],
                mode='lines', showlegend=False,
                line=dict(color='white', width=2, dash='dash'),
                hoverinfo='none'))

        fig.update_layout(
            title=(f'3-D neuron reconstruction — '
                   f'{sum(len(pd["neurons"]) for pd in self._plane_data)} components, '
                   f'{len(self._plane_data)} z-planes  '
                   f'(◆ = duplicate candidate  --- = duplicate pair)'),
            paper_bgcolor='#1a1a1a',
            font=dict(color='#cccccc'),
            scene=dict(
                xaxis=dict(title='X (px)', backgroundcolor='#111111',
                           gridcolor='#2a2a2a', color='#888888'),
                yaxis=dict(title='Y (px)', backgroundcolor='#111111',
                           gridcolor='#2a2a2a', color='#888888'),
                zaxis=dict(
                    title='', backgroundcolor='#111111',
                    gridcolor='#2a2a2a', color='#888888',
                    tickvals=list(range(len(self._plane_data))),
                    ticktext=[pd['z'] for pd in self._plane_data]),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.15 * len(self._plane_data)),
            ),
            legend=dict(bgcolor='#222222', bordercolor='#444444'),
        )

        with tempfile.NamedTemporaryFile(
                suffix='.html', delete=False, mode='w', encoding='utf-8') as f:
            fig.write_html(f.name)
            webbrowser.open(f.name)

    # ── close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        result: dict[str, np.ndarray] = {}
        for pd in self._plane_data:
            K = len(pd['neurons'])
            result[pd['z']] = np.array(
                [self._decisions.get((pd['z'], k), True) for k in range(K)],
                dtype=bool)
        if self._on_close_cb is not None:
            self._on_close_cb(result)
        self.destroy()
