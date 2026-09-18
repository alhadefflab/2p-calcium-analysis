"""ROI curation record: what the ROI editor changed, and whether it paid off.

The editor hands over a record per z-plane; this module writes it to
``<plane dir>/roi_curation/`` as

    roi_curation_<z>.json      ids added / removed, polygons, counts, densities
    roi_curation_<z>_maps.npz  label images (stable ids) before and after, AP mask
    roi_curation_<z>_red.png   screenshots, one per reference LUT (red = tdTomato,
    roi_curation_<z>_green.png green = GCaMP, merge = both), each showing the
    roi_curation_<z>_merge.png edits (left) and the AP / NTS split (right)

and later fills the record's ``yield`` section as the pipeline proceeds:
how many of the ROIs, split by origin (Cellpose-detected vs hand-added) and by
region, got a CNMF trace, were accepted in the neuron viewer, and responded.
``write_summary`` collects every plane of an animal into one CSV + figure.

Neurons carry a *stable id* through the editor.  Detected ROIs are numbered
``0 .. n_detected-1`` in Cellpose order; hand-added ones continue from there.
So ``id >= n_detected`` means added, regardless of later deletions.  CNMF
components are matched back to ids by spatial overlap with the final label
image, because CNMF may drop seeds and so column order cannot be trusted.

Region convention: 0 = AP (the polygon drawn in the editor), 1 = NTS (every
kept neuron outside it).  Pure numpy / matplotlib-Agg, safe on a worker thread.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

REGIONS  = ("AP", "NTS")
ORIGINS  = ("detected", "added")
STAGES   = ("rois", "traces", "qc_passed", "accepted", "responsive")
STAGE_LABELS = {
    "rois":       "ROIs after curation",
    "traces":     "CNMF traces",
    "qc_passed":  "Passed CaImAn check",
    "accepted":   "Accepted (neuron viewer)",
    "responsive": "Responsive",
}
# densities are reported per 10 000 px² (a 100 × 100 px square) and, when the
# pixel size is known (read from the Prairie View XML into provenance), also per
# mm².  Both are kept side by side: µm is the physical figure, px is what the
# images and the paper's Cellpose settings are expressed in.
DENSITY_UNIT_PX = 10_000
DENSITY_UNIT_UM2 = 1_000_000          # 1 mm² in µm²

_SUBDIR = "roi_curation"

# ── label images ──────────────────────────────────────────────────────────────

def label_image(roi_masks: np.ndarray, h: int, w: int, ids=None):
    """(labels, counts) for a (h*w, N) Fortran-ordered mask matrix.

    labels : (h, w) int32: the id of the ROI covering each pixel, -1 if none.
             Where ROIs overlap, the smallest one wins, so the label under the
             cursor is the neuron you can actually see on top.
    counts : (h, w) int16: how many ROIs cover each pixel.
    """
    n = roi_masks.shape[1]
    ids = np.arange(n) if ids is None else np.asarray(ids)
    counts = roi_masks.sum(axis=1, dtype=np.int16)
    labels = np.full(roi_masks.shape[0], -1, dtype=np.int32)
    if n:
        # paint largest first so smaller ROIs overwrite them where they overlap
        sizes = roi_masks.sum(axis=0)
        for col in np.argsort(-sizes, kind="stable"):
            labels[roi_masks[:, col]] = ids[col]
    return (labels.reshape((h, w), order="F"),
            counts.reshape((h, w), order="F"))

def label_centroids(labels: np.ndarray) -> dict:
    """{id: (row, col)} centre of mass of every labelled region."""
    ys, xs = np.nonzero(labels >= 0)
    if not len(ys):
        return {}
    lab = labels[ys, xs]
    uniq, inv = np.unique(lab, return_inverse=True)
    cnt = np.bincount(inv)
    cy = np.bincount(inv, weights=ys) / cnt
    cx = np.bincount(inv, weights=xs) / cnt
    return {int(u): (float(r), float(c)) for u, r, c in zip(uniq, cy, cx)}

def polygon_mask(pts, h: int, w: int) -> np.ndarray:
    """Bool (h, w) mask of a polygon given as image (x, y) vertices."""
    from PIL import Image, ImageDraw
    img = Image.new("L", (w, h), 0)
    if pts is not None and len(pts) >= 3:
        ImageDraw.Draw(img).polygon([tuple(map(float, p)) for p in pts], fill=255)
    return np.array(img, dtype=bool)

def region_of(points, ap_mask, keep_mask=None) -> np.ndarray:
    """Region per (row, col) point: 0 = AP, 1 = NTS, -1 = excluded / no AP.

    Without an AP mask nothing is classified.  Points outside `keep_mask` (the
    area left after region exclusion) are -1 when a keep mask is given.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    out = np.full(len(pts), -1, dtype=int)
    if ap_mask is None or not len(pts):
        return out
    h, w = ap_mask.shape
    rr = np.clip(np.round(pts[:, 0]).astype(int), 0, h - 1)
    cc = np.clip(np.round(pts[:, 1]).astype(int), 0, w - 1)
    out[:] = 1
    out[ap_mask[rr, cc]] = 0
    if keep_mask is not None:
        out[(out == 1) & ~keep_mask[rr, cc]] = -1
    return out

# ── counts / densities ────────────────────────────────────────────────────────

def _areas(ap_mask, keep_mask) -> dict:
    keep = keep_mask if keep_mask is not None else np.ones_like(ap_mask, bool)
    if ap_mask is None:
        return {"AP": None, "NTS": None, "total": int(keep.sum())}
    ap = int((ap_mask & keep).sum())
    return {"AP": ap, "NTS": int(keep.sum()) - ap, "total": int(keep.sum())}

def _tally(regions) -> dict:
    """{'AP': n, 'NTS': n, 'total': n} from an array of region codes."""
    regions = np.asarray(regions, dtype=int)
    return {"AP": int((regions == 0).sum()),
            "NTS": int((regions == 1).sum()),
            "total": int((regions >= 0).sum())}

def _density(counts: dict, areas: dict) -> dict:
    out = {}
    for k, v in counts.items():
        a = areas.get(k)
        out[k] = round(v * DENSITY_UNIT_PX / a, 3) if a else None
    return out

def areas_um2(areas_px: dict, um_per_px) -> dict | None:
    """Areas in µm², or None when the pixel size is unknown."""
    if not um_per_px:
        return None
    f = float(um_per_px) ** 2
    return {k: (round(v * f, 1) if v else None) for k, v in areas_px.items()}

def _density_mm2(counts: dict, areas_px: dict, um_per_px) -> dict | None:
    """Neurons per mm², or None when the pixel size is unknown."""
    if not um_per_px:
        return None
    f = float(um_per_px) ** 2
    out = {}
    for k, v in counts.items():
        a = areas_px.get(k)
        out[k] = round(v * DENSITY_UNIT_UM2 / (a * f), 2) if a else None
    return out

def _with_densities(count: dict, areas: dict, um_per_px) -> dict:
    """{'count', 'density' (per 10k px²), 'density_mm2' (when µm/px is known)}."""
    counts_valid = {k: v for k, v in count.items() if v is not None}
    block = {"count": count, "density": _density(counts_valid, areas)}
    d_mm2 = _density_mm2(counts_valid, areas, um_per_px)
    if d_mm2 is not None:
        block["density_mm2"] = d_mm2
    return block

def _count_block(ids, regions, n_detected, areas, um_per_px=None) -> dict:
    """Counts + densities split by origin (detected / added / all)."""
    ids = np.asarray(ids, dtype=int)
    regions = np.asarray(regions, dtype=int)
    block = {}
    for origin, sel in (("detected", ids < n_detected),
                        ("added",    ids >= n_detected),
                        ("all",      np.ones(len(ids), bool))):
        c = _tally(regions[sel])
        if areas.get("AP") is None:            # no AP: only the total is meaningful
            c = {"AP": None, "NTS": None, "total": int(sel.sum())}
        block[origin] = _with_densities(c, areas, um_per_px)
    return block

# ── save / load ───────────────────────────────────────────────────────────────

def curation_dir(plane_dir) -> Path:
    return Path(plane_dir) / _SUBDIR

def load_curation(plane_dir, z):
    """(record dict, maps dict) from a previous save, or (None, None)."""
    d = curation_dir(plane_dir)
    jp, mp = d / f"roi_curation_{z}.json", d / f"roi_curation_{z}_maps.npz"
    if not jp.exists():
        return None, None
    with open(jp, "r") as f:
        rec = json.load(f)
    maps = None
    if mp.exists():
        with np.load(mp) as npz:
            maps = {k: npz[k] for k in npz.files}
    return rec, maps

def load_curated_masks(plane_dir, z, n_pixels=None):
    """ROIs as they stood when this plane's curation was finished.

    Returns (roi_masks (h*w, N) bool Fortran order, record) or None when there is
    no usable record.  Uses the exact saved masks when present; records written
    before those were stored are rebuilt from the final label image, which is
    exact except where ROIs overlapped (the shared pixels go to the smaller one).
    `n_pixels`, if given, must match or None is returned (different FOV).
    """
    rec, maps = load_curation(plane_dir, z)
    if rec is None or maps is None or not rec.get("final_ids"):
        return None
    ids = [int(i) for i in rec["final_ids"]]
    if "final_masks" in maps and maps["final_masks"].shape[1] == len(ids):
        masks = maps["final_masks"].astype(bool)
    else:
        lab_f = maps["final_labels"].ravel(order="F")
        masks = np.stack([lab_f == i for i in ids], axis=1)
        keep = masks.any(axis=0)            # an id fully covered by another ROI
        if not keep.all():
            masks = masks[:, keep]
            rec = dict(rec, final_ids=[i for i, k in zip(ids, keep) if k])
    if n_pixels is not None and masks.shape[0] != n_pixels:
        return None
    return masks, rec


def save_curation(plane_dir, z, rec: dict, previous=None, um_per_px=None) -> dict:
    """Write the editor's record.  Returns the JSON summary that was written.

    `rec` keys (from ROIEditorWindow):
        n_detected, next_id, initial_labels, final_labels, final_ids,
        removal_reason {id: 'manual'|'region'}, exclusion_polygons, keep_mask,
        ap_polygon, ap_mask,
        views [(key, title, (h, w, 3) uint8 image)], one screenshot file each
        (older callers: func_view, red_view, red_label)
    `previous`, (record, maps) of an earlier save of this plane.  Given when
    the editor was reopened on already-curated ROIs (sub-region setup), so the
    original detected set and earlier edits are kept rather than overwritten.
    `um_per_px`, pixel size from the Prairie View XML (provenance).  When given,
    areas and densities are recorded in µm as well as px.
    """
    d = curation_dir(plane_dir)
    d.mkdir(parents=True, exist_ok=True)

    n_detected = int(rec["n_detected"])
    initial    = rec["initial_labels"]
    final      = rec["final_labels"]
    reasons    = {int(k): v for k, v in rec["removal_reason"].items()}
    excl       = [np.asarray(p, float).tolist() for p in rec["exclusion_polygons"]]
    keep       = rec["keep_mask"]

    prev_rec, prev_maps = previous if previous else (None, None)
    if prev_rec is not None and prev_maps is not None:
        initial    = prev_maps["initial_labels"]
        n_detected = int(prev_rec["n_detected"])
        for k, v in prev_rec.get("removal_reason", {}).items():
            reasons.setdefault(int(k), v)
        excl = prev_rec.get("exclusion_polygons", []) + excl
        if "keep_mask" in prev_maps:
            keep = prev_maps["keep_mask"] & keep

    ap_mask = rec["ap_mask"]
    areas   = _areas(ap_mask, keep)

    init_c  = label_centroids(initial)
    final_c = label_centroids(final)
    final_ids = [int(i) for i in rec["final_ids"]]
    final_set = set(final_ids)

    init_ids = np.array(sorted(init_c), dtype=int)
    fin_ids  = np.array([i for i in final_ids if i in final_c], dtype=int)
    fin_reg  = region_of([final_c[i] for i in fin_ids], ap_mask)
    # detected neurons inside the kept area, AP / NTS when an AP exists,
    # otherwise just inside (0) vs excluded (-1)
    init_reg = region_of([init_c[i] for i in init_ids],
                         ap_mask if ap_mask is not None else np.zeros_like(keep),
                         keep)
    if ap_mask is None:
        init_reg[init_reg == 1] = 0

    removed = sorted(set(init_ids.tolist()) - final_set)
    added   = sorted(i for i in final_ids if i >= n_detected)
    summary = {
        "z": z,
        "saved": datetime.now().isoformat(timespec="seconds"),
        "region_convention": "0 = AP (drawn polygon), 1 = NTS (kept neurons outside AP)",
        "n_detected": n_detected,
        "n_final": len(final_ids),
        "next_id": int(rec.get("next_id", max(final_ids + [n_detected - 1]) + 1)),
        "added_ids": added,
        "removed_manual_ids": [i for i in removed if reasons.get(i) != "region"],
        "removed_region_ids": [i for i in removed if reasons.get(i) == "region"],
        "removal_reason": {str(k): v for k, v in reasons.items()},
        "final_ids": final_ids,
        "exclusion_polygons": excl,
        "ap_polygon": (np.asarray(rec["ap_polygon"], float).tolist()
                       if rec.get("ap_polygon") is not None else None),
        "area_px": areas,
        "um_per_px": (float(um_per_px) if um_per_px else None),
        "area_um2": areas_um2(areas, um_per_px),
        "density_unit": f"neurons per {DENSITY_UNIT_PX} px²",
        "density_unit_mm2": "neurons per mm²" if um_per_px else None,
        # only detected neurons inside the kept (non-excluded) area count here,
        # so the before/after densities are over the same area
        "detected_before_curation": {
            "count": (_tally(init_reg) if ap_mask is not None
                      else {"AP": None, "NTS": None, "total": int((init_reg >= 0).sum())}),
        },
        "after_curation": _count_block(fin_ids, fin_reg, n_detected, areas, um_per_px),
    }
    before = summary["detected_before_curation"]
    before.update(_with_densities(before["count"], areas, um_per_px))
    # keep yield figures from an earlier save only if the ROI set is unchanged
    # otherwise they describe a CNMF run on different seeds
    if prev_rec is not None and prev_rec.get("final_ids") == final_ids:
        summary["yield"] = prev_rec.get("yield", {})

    with open(d / f"roi_curation_{z}.json", "w") as f:
        json.dump(summary, f, indent=2)

    extra = {}
    if rec.get("final_masks") is not None:
        # exact (h*w, N) masks in final_ids order, overlaps included: what a
        # resumed curation starts from.  Mostly zeros, so it compresses to little.
        extra["final_masks"] = np.asarray(rec["final_masks"], dtype=bool)
    np.savez_compressed(
        d / f"roi_curation_{z}_maps.npz",
        **extra,
        initial_labels=initial.astype(np.int32),
        final_labels=final.astype(np.int32),
        keep_mask=keep.astype(bool),
        ap_mask=(ap_mask if ap_mask is not None else np.zeros_like(keep)).astype(bool),
        has_ap=np.array(ap_mask is not None))

    views = rec.get("views")
    if views is None:
        views = [("functional", "Functional (GCaMP)", rec.get("func_view"))]
        if rec.get("red_view") is not None:
            views.append(("red", f"{rec.get('red_label') or 'tdTomato'} (red)",
                          rec["red_view"]))
    errors = {}
    for key, title, img in views:
        try:
            _save_screenshots(d / f"roi_curation_{z}_{key}.png", z, summary, initial,
                              final, n_detected, ap_mask, title, img)
        except Exception as e:                   # a figure must never cost the record
            errors[key] = repr(e)
    if errors:
        summary["screenshot_error"] = errors
    return summary

# ── screenshots ───────────────────────────────────────────────────────────────

_COL = {
    "detected": "#f2f2f2",
    "added":    "#39ff14",
    "manual":   "#ff2bd6",
    "region":   "#ff9a1f",
    "AP":       "#ffe23d",
    "NTS":      "#3de0ff",
}

def _outline_segments(labels, ids):
    """List of (N, 2) x/y contour arrays for the given label ids."""
    from skimage.measure import find_contours
    from scipy.ndimage import find_objects

    segs = []
    if not len(ids) or labels.max() < 0:
        return segs
    # find_objects indexes by label value, so shift ids up by one
    slices = find_objects(labels + 1)
    for i in ids:
        if i + 1 > len(slices) or slices[i] is None:
            continue
        sy, sx = slices[i]
        crop = np.pad(labels[sy, sx] == i, 1).astype(float)
        for c in find_contours(crop, 0.5):
            segs.append(np.column_stack([c[:, 1] + sx.start - 1,
                                         c[:, 0] + sy.start - 1]))
    return segs

def _poly_xy(pts):
    p = np.asarray(pts, float)
    return np.vstack([p, p[:1]])

def _save_screenshots(path, z, summary, initial, final, n_detected, ap_mask, title, img):
    """One view (e.g. the red LUT): edits on the left, AP / NTS on the right."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    h, w = final.shape
    if img is None:
        img = np.zeros((h, w, 3), np.uint8)

    fin_ids = summary["final_ids"]
    groups_edit = [
        ("detected", [i for i in fin_ids if i < n_detected], "-", 0.8),
        ("added",    summary["added_ids"], "-", 1.6),
        ("manual",   summary["removed_manual_ids"], "-", 1.4),
        ("region",   summary["removed_region_ids"], ":", 1.0),
    ]
    segs_edit = []
    for key, ids, ls, lw in groups_edit:
        src = final if key in ("detected", "added") else initial
        segs_edit.append((key, _outline_segments(src, ids), ls, lw))

    segs_reg = []
    if ap_mask is not None:
        fc = label_centroids(final)
        ids = [i for i in fin_ids if i in fc]
        reg = region_of([fc[i] for i in ids], ap_mask)
        for code, name in enumerate(REGIONS):
            segs_reg.append((name, _outline_segments(final, [i for i, r in zip(ids, reg)
                                                              if r == code]), "-", 1.1))

    fig = Figure(figsize=(12.4, 6.9), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, squeeze=False)

    for col, segs in enumerate((segs_edit, segs_reg)):
        ax = axes[0, col]
        ax.imshow(img, interpolation="nearest")
        for key, s, ls, lw in segs:
            if s:
                ax.add_collection(LineCollection(s, colors=_COL[key],
                                                 linestyles=ls, linewidths=lw))
        for p in summary["exclusion_polygons"]:
            xy = _poly_xy(p)
            ax.plot(xy[:, 0], xy[:, 1], color=_COL["region"], lw=1.2, ls="--")
        if summary["ap_polygon"]:
            xy = _poly_xy(summary["ap_polygon"])
            ax.plot(xy[:, 0], xy[:, 1], color="black", lw=3.2)
            ax.plot(xy[:, 0], xy[:, 1], color=_COL["AP"], lw=1.8)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title}: {'edits' if col == 0 else 'AP / NTS'}", fontsize=10)

    ac = summary["after_curation"]
    edit_handles = [
        Line2D([], [], color=_COL["detected"], label=f"kept detected ({ac['detected']['count']['total']})"),
        Line2D([], [], color=_COL["added"], lw=2, label=f"added ({len(summary['added_ids'])})"),
        Line2D([], [], color=_COL["manual"], lw=2,
               label=f"removed by hand ({len(summary['removed_manual_ids'])})"),
        Line2D([], [], color=_COL["region"], ls=":", lw=2,
               label=f"excluded by region ({len(summary['removed_region_ids'])})"),
    ]
    axes[0, 0].legend(handles=edit_handles, loc="lower left", fontsize=8,
                      facecolor="black", labelcolor="white", framealpha=0.7)

    if ap_mask is not None:
        a = summary["area_px"]
        a_um = summary.get("area_um2") or {}
        d_mm2 = ac["all"].get("density_mm2") or {}
        reg_handles = []
        for name in REGIONS:
            c = ac["all"]["count"][name]
            dn = ac["all"]["density"].get(name)
            label = f"{name}: {c} neurons, {a[name]} px², {dn} / 10k px²"
            if a_um.get(name) and d_mm2.get(name):
                label += f"  ({a_um[name]:.0f} µm², {d_mm2[name]:.0f} / mm²)"
            reg_handles.append(Line2D([], [], color=_COL[name], lw=2, label=label))
        axes[0, 1].legend(handles=reg_handles, loc="lower left", fontsize=8,
                          facecolor="black", labelcolor="white", framealpha=0.7)
    else:
        axes[0, 1].text(0.5, 0.5, "No AP sub-region defined", color="white",
                        ha="center", va="center", transform=axes[0, 1].transAxes,
                        fontsize=12, bbox=dict(facecolor="black", alpha=0.7))

    fig.suptitle(f"ROI curation: {z}   ·   detected {summary['n_detected']}  →  "
                 f"final {summary['n_final']}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)

# ── CNMF yield ────────────────────────────────────────────────────────────────

def match_components(A, final_labels: np.ndarray, min_frac: float = 0.3):
    """Stable ROI id and (row, col) centre for each CNMF component.

    A : (h*w, K) spatial footprints, Fortran pixel order (scipy sparse or dense).
    Each component takes the id carrying most of its footprint weight; if less
    than `min_frac` of the weight sits on any ROI, the id is -1.
    """
    import scipy.sparse as sp

    h, w = final_labels.shape
    A = sp.csc_matrix(A)
    flat_lab = final_labels.ravel(order="F")
    K = A.shape[1]
    ids = np.full(K, -1, dtype=int)
    centres = np.zeros((K, 2), dtype=float)
    for k in range(K):
        s, e = A.indptr[k], A.indptr[k + 1]
        idx, val = A.indices[s:e], np.abs(A.data[s:e])
        tot = val.sum()
        if tot <= 0:
            continue
        centres[k] = ((idx % h) * val).sum() / tot, ((idx // h) * val).sum() / tot
        lab = flat_lab[idx]
        on = lab >= 0
        if not on.any():
            continue
        u, inv = np.unique(lab[on], return_inverse=True)
        wsum = np.bincount(inv, weights=val[on])
        best = int(np.argmax(wsum))
        if wsum[best] >= min_frac * tot:
            ids[k] = int(u[best])
    return ids, centres

def _stage_block(comp_ids, comp_regions, n_detected, areas, has_ap, um_per_px=None):
    block = {}
    for origin, sel in (("detected", (comp_ids >= 0) & (comp_ids < n_detected)),
                        ("added",    comp_ids >= n_detected),
                        ("all",      np.ones(len(comp_ids), bool))):
        if has_ap:
            c = {"AP": int((sel & (comp_regions == 0)).sum()),
                 "NTS": int((sel & (comp_regions == 1)).sum()),
                 "total": int(sel.sum())}
        else:
            c = {"AP": None, "NTS": None, "total": int(sel.sum())}
        block[origin] = _with_densities(c, areas, um_per_px)
    return block

def update_yield(plane_dir, z, A, is_cell=None, responsive=None, qc_pass=None) -> dict | None:
    """Fill the record's `yield` section from a CNMF result.

    A          : CNMF spatial footprints (all components, before is_cell).
    is_cell    : (K,) bool, neuron-viewer acceptance, or None (all accepted).
    responsive : (K_accepted,) bool in is_cell order, or None to leave the
                 responsive stage as it was.
    qc_pass    : (K,) bool, CaImAn's quality-check verdict, or None when the run
                 saved none.  Gives the added-vs-dropped split after CNMF.
    Returns the updated yield dict, or None when the plane has no record.
    """
    rec, maps = load_curation(plane_dir, z)
    if rec is None or maps is None:
        return None

    n_detected = int(rec["n_detected"])
    has_ap = bool(maps.get("has_ap", np.array(False)))
    ap_mask = maps["ap_mask"] if has_ap else None
    areas = rec["area_px"]

    comp_ids, centres = match_components(A, maps["final_labels"])
    comp_reg = region_of(centres, ap_mask) if has_ap else np.full(len(comp_ids), -1)
    K = len(comp_ids)
    is_cell = np.ones(K, bool) if is_cell is None else np.asarray(is_cell, bool)
    if len(is_cell) != K:                       # stale is_cell from another CNMF run
        is_cell = np.ones(K, bool)

    um_per_px = rec.get("um_per_px")
    y = dict(rec.get("yield") or {})
    y["rois"] = rec["after_curation"]
    y["traces"] = _stage_block(comp_ids, comp_reg, n_detected, areas, has_ap, um_per_px)
    if qc_pass is not None:
        qc_pass = np.asarray(qc_pass, bool)
        if len(qc_pass) == K:
            y["qc_passed"] = _stage_block(comp_ids[qc_pass], comp_reg[qc_pass],
                                          n_detected, areas, has_ap, um_per_px)
            y["qc_failed_ids"] = sorted({int(i) for i in comp_ids[~qc_pass] if i >= 0})
    y["accepted"] = _stage_block(comp_ids[is_cell], comp_reg[is_cell],
                                 n_detected, areas, has_ap, um_per_px)
    if responsive is not None:
        responsive = np.asarray(responsive, bool)
        acc_ids, acc_reg = comp_ids[is_cell], comp_reg[is_cell]
        if len(responsive) == len(acc_ids):
            y["responsive"] = _stage_block(acc_ids[responsive], acc_reg[responsive],
                                           n_detected, areas, has_ap, um_per_px)
    y["unmatched_components"] = int((comp_ids < 0).sum())
    y["updated"] = datetime.now().isoformat(timespec="seconds")
    rec["yield"] = y

    with open(curation_dir(plane_dir) / f"roi_curation_{z}.json", "w") as f:
        json.dump(rec, f, indent=2)
    return y

def component_origins(plane_dir, z, A, is_cell=None) -> np.ndarray | None:
    """Per accepted component: 0 = detected, 1 = added, -1 = unmatched.

    Aligned with get_stims_n / get_region_labels row order for this plane.
    """
    rec, maps = load_curation(plane_dir, z)
    if rec is None or maps is None:
        return None
    ids, _ = match_components(A, maps["final_labels"])
    if is_cell is not None and len(is_cell) == len(ids):
        ids = ids[np.asarray(is_cell, bool)]
    return np.where(ids < 0, -1, (ids >= int(rec["n_detected"])).astype(int))

def format_yield(z, y) -> list[str]:
    """Human-readable log lines: detected vs added through each stage."""
    lines = [f"    {z} curation yield (detected | added):"]
    base = {o: (y.get("rois", {}).get(o, {}).get("count", {}).get("total") or 0)
            for o in ORIGINS}
    for st in STAGES:
        if st not in y:
            continue
        parts = []
        for o in ORIGINS:
            c = y[st][o]["count"]
            pct = f" ({100 * c['total'] / base[o]:.0f}%)" if base[o] and st != "rois" else ""
            reg = (f" [AP {c['AP']} · NTS {c['NTS']}]" if c.get("AP") is not None else "")
            parts.append(f"{c['total']}{pct}{reg}")
        lines.append(f"      {STAGE_LABELS[st]:<26} {parts[0]:<28} | {parts[1]}")
    return lines

# ── animal-level summary ──────────────────────────────────────────────────────

def _summary_row(rec, stage, origin, block, base) -> dict:
    """One CSV row: counts, then densities in px and (when known) in µm."""
    c = block["count"]
    d = block.get("density") or {}
    dm = block.get("density_mm2") or {}
    area_px = rec.get("area_px") or {}
    area_um = rec.get("area_um2") or {}
    return {
        "z": rec["z"], "stage": stage, "origin": origin,
        "AP": c.get("AP"), "NTS": c.get("NTS"), "total": c["total"],
        "pct_of_curated_rois": round(100 * c["total"] / base, 1) if base else None,
        "AP_density_per_10k_px2": d.get("AP"),
        "NTS_density_per_10k_px2": d.get("NTS"),
        "total_density_per_10k_px2": d.get("total"),
        "AP_density_per_mm2": dm.get("AP"),
        "NTS_density_per_mm2": dm.get("NTS"),
        "total_density_per_mm2": dm.get("total"),
        "um_per_px": rec.get("um_per_px"),
        "AP_area_px2": area_px.get("AP"), "NTS_area_px2": area_px.get("NTS"),
        "AP_area_um2": (area_um or {}).get("AP"), "NTS_area_um2": (area_um or {}).get("NTS"),
    }

def write_summary(animal_dir) -> Path | None:
    """Collect every plane's record into roi_curation_summary.csv + .png."""
    animal_dir = Path(animal_dir)
    recs = []
    for jp in sorted(animal_dir.glob(f"*/{_SUBDIR}/roi_curation_*.json")):
        try:
            with open(jp, "r") as f:
                recs.append(json.load(f))
        except Exception:
            continue
    if not recs:
        return None

    rows = []
    for r in recs:
        y = r.get("yield") or {"rois": r["after_curation"]}
        for st in STAGES:
            if st not in y:
                continue
            for o in ORIGINS + ("all",):
                base = y["rois"][o]["count"]["total"]
                rows.append(_summary_row(r, st, o, y[st][o], base))
        b = r["detected_before_curation"]
        rows.append(_summary_row(r, "detected_before_curation", "detected", b, None))

    csv_path = animal_dir / "roi_curation_summary.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)

    try:
        _save_yield_figure(animal_dir / "roi_curation_yield.png", recs)
    except Exception:
        pass
    return csv_path

def _save_yield_figure(path, recs):
    """Grouped bars: detected vs added through each stage, summed over planes."""
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    stages = [s for s in STAGES if any(s in (r.get("yield") or {}) for r in recs)]
    stages = stages or ["rois"]
    tot = {o: np.zeros(len(stages)) for o in ORIGINS}
    for r in recs:
        y = r.get("yield") or {"rois": r["after_curation"]}
        for j, st in enumerate(stages):
            for o in ORIGINS:
                tot[o][j] += (y.get(st, {}).get(o, {}).get("count", {}).get("total") or 0)

    fig = Figure(figsize=(7.5, 4.2), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
    x = np.arange(len(stages))
    bw = 0.38
    colors = {"detected": "#4a78c2", "added": "#2ca05a"}
    ax = axes[0]
    for k, o in enumerate(ORIGINS):
        bars = ax.bar(x + (k - 0.5) * bw, tot[o], bw, color=colors[o], label=o)
        for b, v in zip(bars, tot[o]):
            ax.annotate(f"{int(v)}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7, xytext=(0, 1),
                        textcoords="offset points")
    ax.set_xticks(x, [STAGE_LABELS[s].replace(" (", "\n(") for s in stages], fontsize=8)
    ax.set_ylabel("neurons (all planes)")
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for k, o in enumerate(ORIGINS):
        base = tot[o][0]
        pct = 100 * tot[o] / base if base else np.zeros(len(stages))
        ax.plot(x, pct, "o-", color=colors[o], label=o)
    ax.set_xticks(x, [s for s in stages], fontsize=8, rotation=30)
    ax.set_ylabel("% of curated ROIs retained")
    ax.set_ylim(0, 105)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Does manual curation pay off?  Detected vs hand-added neurons",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, dpi=150)