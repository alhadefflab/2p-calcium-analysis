"""
Unit tests for analysis.roi_curation, the ROI editor's curation record.

Synthetic 60×80 field: square ROIs on a grid, masks in the editor's
(h*w, N) Fortran-ordered layout.
"""
import json

import numpy as np
import pytest
import scipy.sparse as sp

from analysis.roi_curation import (
    component_origins, label_centroids, label_image, match_components,
    polygon_mask, region_of, save_curation, update_yield, write_summary,
)

H, W = 60, 80

def _square(r, c, size=6):
    m = np.zeros((H, W), dtype=bool)
    m[r:r + size, c:c + size] = True
    return m

def _masks(squares):
    return np.stack([s.flatten("F") for s in squares], axis=1)

class TestLabelImage:

    def test_ids_and_background(self):
        masks = _masks([_square(5, 5), _square(30, 40)])
        lab, cnt = label_image(masks, H, W, ids=[7, 12])
        assert lab[7, 7] == 7
        assert lab[32, 42] == 12
        assert lab[0, 0] == -1
        assert cnt.max() == 1

    def test_fortran_order_matches_editor_reshape(self):
        sq = _square(10, 50)
        lab, _ = label_image(_masks([sq]), H, W)
        np.testing.assert_array_equal(lab >= 0, sq)

    def test_overlap_smallest_wins(self):
        big, small = _square(10, 10, 12), _square(14, 14, 3)
        lab, cnt = label_image(_masks([big, small]), H, W)
        assert lab[15, 15] == 1
        assert cnt[15, 15] == 2
        assert lab[11, 11] == 0

    def test_centroids(self):
        lab, _ = label_image(_masks([_square(10, 20, 4)]), H, W, ids=[3])
        (r, c), = label_centroids(lab).values()
        assert (r, c) == (11.5, 21.5)


class TestRegions:

    def test_ap_inside_nts_outside(self):
        ap = polygon_mask([(0, 0), (40, 0), (40, 30), (0, 30)], H, W)
        reg = region_of([(10, 10), (50, 70)], ap)
        assert reg.tolist() == [0, 1]

    def test_excluded_when_outside_keep(self):
        ap = polygon_mask([(0, 0), (20, 0), (20, 20), (0, 20)], H, W)
        keep = polygon_mask([(0, 0), (50, 0), (50, 50), (0, 50)], H, W)
        reg = region_of([(5, 5), (30, 30), (55, 75)], ap, keep)
        assert reg.tolist() == [0, 1, -1]

    def test_no_ap_is_unclassified(self):
        assert region_of([(5, 5)], None).tolist() == [-1]


class TestMatchComponents:

    def test_matches_by_overlap_not_order(self):
        sqs = [_square(5, 5), _square(30, 40), _square(40, 10)]
        lab, _ = label_image(_masks(sqs), H, W, ids=[0, 1, 25])
        # CNMF dropped the middle seed and reordered the rest
        A = sp.csc_matrix(_masks([sqs[2], sqs[0]]).astype(float))
        ids, centres = match_components(A, lab)
        assert ids.tolist() == [25, 0]
        np.testing.assert_allclose(centres[1], (7.5, 7.5))

    def test_unmatched_component(self):
        lab, _ = label_image(_masks([_square(5, 5)]), H, W)
        A = sp.csc_matrix(_masks([_square(40, 60)]).astype(float))
        assert match_components(A, lab)[0].tolist() == [-1]


def _record(tmp_path):
    """Detected 0,1,2 · removed 1 by hand, 2 by region · added id 3."""
    det = [_square(5, 5), _square(30, 40), _square(50, 70, 5)]
    added = _square(20, 60)
    keep = polygon_mask([(0, 0), (65, 0), (65, 59), (0, 59)], H, W)
    ap = polygon_mask([(0, 0), (20, 0), (20, 20), (0, 20)], H, W)
    rec = dict(
        n_detected=3, next_id=4,
        initial_labels=label_image(_masks(det), H, W)[0],
        final_labels=label_image(_masks([det[0], added]), H, W, ids=[0, 3])[0],
        final_ids=[0, 3],
        removal_reason={1: "manual", 2: "region"},
        exclusion_polygons=[[(0, 0), (65, 0), (65, 59), (0, 59)]],
        keep_mask=keep, ap_polygon=[(0, 0), (20, 0), (20, 20), (0, 20)], ap_mask=ap,
        func_view=np.zeros((H, W, 3), np.uint8), red_view=np.zeros((H, W, 3), np.uint8),
        red_label="tdTomato",
    )
    return rec, det, added


class TestSaveAndYield:

    def test_record_contents(self, tmp_path):
        rec, _, _ = _record(tmp_path)
        s = save_curation(tmp_path / "z1", "z1", rec)
        assert s["added_ids"] == [3]
        assert s["removed_manual_ids"] == [1]
        assert s["removed_region_ids"] == [2]
        ac = s["after_curation"]
        assert ac["detected"]["count"] == {"AP": 1, "NTS": 0, "total": 1}
        assert ac["added"]["count"] == {"AP": 0, "NTS": 1, "total": 1}
        # detected neuron 2 sits outside the kept area — not in the before count
        assert s["detected_before_curation"]["count"]["total"] == 2
        assert s["area_px"]["AP"] == int(rec["ap_mask"].sum())
        d = tmp_path / "z1" / "roi_curation"
        assert (d / "roi_curation_z1.png").exists()
        assert "screenshot_error" not in s
        assert json.loads((d / "roi_curation_z1.json").read_text())["n_final"] == 2

    def test_yield_splits_detected_and_added(self, tmp_path):
        rec, det, added = _record(tmp_path)
        save_curation(tmp_path / "z1", "z1", rec)
        A = sp.csc_matrix(_masks([added, det[0]]).astype(float))
        y = update_yield(tmp_path / "z1", "z1", A, is_cell=np.array([True, True]),
                         responsive=np.array([True, False]))
        assert y["traces"]["added"]["count"]["total"] == 1
        assert y["responsive"]["added"]["count"] == {"AP": 0, "NTS": 1, "total": 1}
        assert y["responsive"]["detected"]["count"]["total"] == 0
        assert component_origins(tmp_path / "z1", "z1", A).tolist() == [1, 0]

    def test_yield_respects_is_cell(self, tmp_path):
        rec, det, added = _record(tmp_path)
        save_curation(tmp_path / "z1", "z1", rec)
        A = sp.csc_matrix(_masks([added, det[0]]).astype(float))
        y = update_yield(tmp_path / "z1", "z1", A, is_cell=np.array([False, True]))
        assert y["traces"]["all"]["count"]["total"] == 2
        assert y["accepted"]["added"]["count"]["total"] == 0
        assert y["accepted"]["detected"]["count"]["total"] == 1

    def test_reopen_keeps_history(self, tmp_path):
        from analysis.roi_curation import load_curation
        rec, _, _ = _record(tmp_path)
        save_curation(tmp_path / "z1", "z1", rec)
        prev = load_curation(tmp_path / "z1", "z1")
        # reopened on the curated set: its "initial" is the final set, no edits
        rec2 = dict(rec, initial_labels=rec["final_labels"], removal_reason={},
                    exclusion_polygons=[], keep_mask=np.ones((H, W), bool),
                    n_detected=2)
        s = save_curation(tmp_path / "z1", "z1", rec2, previous=prev)
        assert s["n_detected"] == 3
        assert s["removed_manual_ids"] == [1]
        assert s["added_ids"] == [3]

    def test_summary_csv(self, tmp_path):
        rec, det, added = _record(tmp_path)
        save_curation(tmp_path / "z1", "z1", rec)
        update_yield(tmp_path / "z1", "z1",
                     sp.csc_matrix(_masks([added, det[0]]).astype(float)))
        path = write_summary(tmp_path)
        text = path.read_text()
        assert "traces,added" in text
        assert (tmp_path / "roi_curation_yield.png").exists()

    def test_no_ap(self, tmp_path):
        rec, _, _ = _record(tmp_path)
        rec.update(ap_polygon=None, ap_mask=None)
        s = save_curation(tmp_path / "z1", "z1", rec)
        assert s["ap_polygon"] is None
        assert s["after_curation"]["all"]["count"]["total"] == 2
        assert s["after_curation"]["all"]["count"]["AP"] is None