"""ROI curation record: what the ROI editor changed, and whether it paid off.

The editor hands over a record per z-plane; this module writes it to
``<plane dir>/roi_curation/`` as

    roi_curation_<z>.json      ids added / removed, polygons, counts, densities
    roi_curation_<z>_maps.npz  label images (stable ids) before and after, AP mask
    roi_curation_<z>.png       screenshots — functional and tdTomato views, showing
                               the edits (top row) and the AP / NTS split (bottom)

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
STAGES   = ("rois", "traces", "accepted", "responsive")
STAGE_LABELS = {
    "rois":       "ROIs after curation",
    "traces":     "CNMF traces",
    "accepted":   "Accepted (neuron viewer)",
    "responsive": "Responsive",
}
# densities are reported per 10 000 px² (a 100 × 100 px square); the pixel size
# is not recorded in provenance, so there is no µm conversion here
DENSITY_UNIT_PX = 10_000

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

def _count_block(ids, regions, n_detected, areas) -> dict:
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
        block[origin] = {"count": c,
                         "density": _density({k: v for k, v in c.items() if v is not None},
                                             areas)}
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

def save_curation(plane_dir, z, rec: dict, previous=None) -> dict:
    """Write the editor's record.  Returns the JSON summary that was written.

    `rec` keys (from ROIEditorWindow):
        n_detected, next_id, initial_labels, final_labels, final_ids,
        removal_reason {id: 'manual'|'region'}, exclusion_polygons, keep_mask,
        ap_polygon, ap_mask, func_view, red_view, red_label
    `previous`, (record, maps) of an earlier save of this plane.  Given when
    the editor was reopened on already-curated ROIs (sub-region setup), so the
    original detected set and earlier edits are kept rather than overwritten.
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
        "density_unit": f"neurons per {DENSITY_UNIT_PX} px²",
        # only detected neurons inside the kept (non-excluded) area count here,
        # so the before/after densities are over the same area
        "detected_before_curation": {
            "count": (_tally(init_reg) if ap_mask is not None
                      else {"AP": None, "NTS": None, "total": int((init_reg >= 0).sum())}),
        },
        "after_curation": _count_block(fin_ids, fin_reg, n_detected, areas),
    }
    before = summary["detected_before_curation"]
    before["density"] = _density({k: v for k, v in before["count"].items() if v is not None},
                                 areas)
    # keep yield figures from an earlier save only if the ROI set is unchanged
    # otherwise they describe a CNMF run on different seeds
    if prev_rec is not None and prev_rec.get("final_ids") == final_ids:
        summary["yield"] = prev_rec.get("yield", {})

    with open(d / f"roi_curation_{z}.json", "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        d / f"roi_curation_{z}_maps.npz",
        initial_labels=initial.astype(np.int32),
        final_labels=final.astype(np.int32),
        keep_mask=keep.astype(bool),
        ap_mask=(ap_mask if ap_mask is not None else np.zeros_like(keep)).astype(bool),
        has_ap=np.array(ap_mask is not None))

    try:
        _save_screenshots(d / f"roi_curation_{z}.png", z, summary, initial, final,
                          n_detected, keep, ap_mask,
                          rec.get("func_view"), rec.get("red_view"),
                          rec.get("red_label") or "tdTomato")
    except Exception as e:                       # a figure must never cost the record
        summary["screenshot_error"] = repr(e)
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

def _save_screenshots(path, z, summary, initial, final, n_detected, keep, ap_mask,
                      func_view, red_view, red_label):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    h, w = final.shape
    views = [("Functional (GCaMP)", func_view)]
    if red_view is not None:
        views.append((f"{red_label} (red)", red_view))
    views = [(t, v if v is not None else np.zeros((h, w, 3), np.uint8)) for t, v in views]

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

    ncols = len(views)
    fig = Figure(figsize=(6.2 * ncols, 12.8), dpi=150)
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, ncols, squeeze=False)

    for col, (title, img) in enumerate(views):
        for row, segs in enumerate((segs_edit, segs_reg)):
            ax = axes[row, col]
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
            ax.set_title(f"{title} — {'edits' if row == 0 else 'AP / NTS'}", fontsize=10)

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
        reg_handles = []
        for name in REGIONS:
            c = ac["all"]["count"][name]
            dn = ac["all"]["density"].get(name)
            reg_handles.append(Line2D([], [], color=_COL[name], lw=2,
                                      label=f"{name}: {c} neurons, {a[name]} px², "
                                            f"{dn} / 10k px²"))
        axes[1, 0].legend(handles=reg_handles, loc="lower left", fontsize=8,
                          facecolor="black", labelcolor="white", framealpha=0.7)
    else:
        axes[1, 0].text(0.5, 0.5, "No AP sub-region defined", color="white",
                        ha="center", va="center", transform=axes[1, 0].transAxes,
                        fontsize=12, bbox=dict(facecolor="black", alpha=0.7))

    fig.suptitle(f"ROI curation — {z}   ·   detected {summary['n_detected']}  →  "
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

def _stage_block(comp_ids, comp_regions, n_detected, areas, has_ap):
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
        block[origin] = {"count": c,
                         "density": _density({k: v for k, v in c.items() if v is not None},
                                             areas)}
    return block

def update_yield(plane_dir, z, A, is_cell=None, responsive=None) -> dict | None:
    """Fill the record's `yield` section from a CNMF result.

    A          : CNMF spatial footprints (all components, before is_cell).
    is_cell    : (K,) bool, neuron-viewer acceptance, or None (all accepted).
    responsive : (K_accepted,) bool in is_cell order, or None to leave the
                 responsive stage as it was.
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

    y = dict(rec.get("yield") or {})
    y["rois"] = rec["after_curation"]
    y["traces"] = _stage_block(comp_ids, comp_reg, n_detected, areas, has_ap)
    y["accepted"] = _stage_block(comp_ids[is_cell], comp_reg[is_cell],
                                 n_detected, areas, has_ap)
    if responsive is not None:
        responsive = np.asarray(responsive, bool)
        acc_ids, acc_reg = comp_ids[is_cell], comp_reg[is_cell]
        if len(responsive) == len(acc_ids):
            y["responsive"] = _stage_block(acc_ids[responsive], acc_reg[responsive],
                                           n_detected, areas, has_ap)
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
                c, d = y[st][o]["count"], y[st][o]["density"]
                base = y["rois"][o]["count"]["total"]
                rows.append({
                    "z": r["z"], "stage": st, "origin": o,
                    "AP": c.get("AP"), "NTS": c.get("NTS"), "total": c["total"],
                    "pct_of_curated_rois": round(100 * c["total"] / base, 1) if base else None,
                    "AP_density_per_10k_px2": d.get("AP"),
                    "NTS_density_per_10k_px2": d.get("NTS"),
                    "total_density_per_10k_px2": d.get("total"),
                })
        b = r["detected_before_curation"]
        rows.append({
            "z": r["z"], "stage": "detected_before_curation", "origin": "detected",
            "AP": b["count"].get("AP"), "NTS": b["count"].get("NTS"),
            "total": b["count"]["total"], "pct_of_curated_rois": None,
            "AP_density_per_10k_px2": b["density"].get("AP"),
            "NTS_density_per_10k_px2": b["density"].get("NTS"),
            "total_density_per_10k_px2": b["density"].get("total"),
        })

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