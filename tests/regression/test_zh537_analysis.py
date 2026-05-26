"""
Regression tests against the ZH537 ground-truth run.

These tests verify that the analysis code (get_resp1_resp2) still produces
exactly the same outputs when fed the same stims1/stims2 arrays that were
used to generate the stored ground truth.

WHY this matters for the CaImAn TF→PyTorch upgrade
---------------------------------------------------
CaImAn CNMF internals will change (different optimisation paths), so the
CNMF HDF5 outputs will differ.  That is expected.  But the analysis layer
(z-score normalisation, responder classification) is pure numpy and must
remain correct.

Workflow after a CaImAn upgrade:
  1. Re-run the full pipeline on ZH537 to get new CNMF files.
  2. Run:  python tests/regression/update_snapshots.py
     This computes new stims1/stims2 from the new CNMF files and saves them.
  3. Inspect the new analysis/params.yaml counts.  If they look reasonable,
     run this test suite — it should pass (same analysis logic, new data).
  4. Commit the updated snapshots as the new ground truth.

Run with:  pytest tests/regression/ -m regression
"""
import numpy as np
import pytest

pytestmark = pytest.mark.regression

from pipeline_funcs import get_resp1_resp2


class TestZH537GroundTruth:
    """Sanity checks on the stored ground-truth outputs (no re-computation)."""

    def test_total_neuron_count(self, zh537_ground_truth):
        gt = zh537_ground_truth
        total = int(gt["nums"].sum())
        assert total == 177, f"expected 177 responsive neurons, got {total}"

    def test_nums_breakdown(self, zh537_ground_truth):
        nums = zh537_ground_truth["nums"]
        assert int(nums[0]) == 5,   "stim1-only count changed"
        assert int(nums[1]) == 2,   "both count changed"
        assert int(nums[2]) == 170, "stim2-only count changed"

    def test_resp_array_shapes_consistent(self, zh537_ground_truth):
        gt = zh537_ground_truth
        n_total = int(gt["nums"].sum())
        assert gt["resp1"].shape[0] == n_total
        assert gt["resp2"].shape[0] == n_total
        assert gt["resp1"].shape == gt["resp2"].shape

    def test_z_ids_lengths_match_nums(self, zh537_ground_truth):
        gt = zh537_ground_truth
        assert len(gt["z_ids_stim1"]) == int(gt["nums"][0])
        assert len(gt["z_ids_both"])  == int(gt["nums"][1])
        assert len(gt["z_ids_stim2"]) == int(gt["nums"][2])

    def test_z_ids_are_valid_plane_numbers(self, zh537_ground_truth):
        valid = {1, 2, 3, 4, 5}
        for key in ("z_ids_stim1", "z_ids_both", "z_ids_stim2"):
            arr = zh537_ground_truth[key]
            if len(arr) > 0:
                assert set(arr.tolist()).issubset(valid), (
                    f"{key} contains unexpected z-plane IDs: {set(arr.tolist()) - valid}")

    def test_resp_values_are_finite(self, zh537_ground_truth):
        assert np.all(np.isfinite(zh537_ground_truth["resp1"])), "resp1 contains NaN/Inf"
        assert np.all(np.isfinite(zh537_ground_truth["resp2"])), "resp2 contains NaN/Inf"


class TestZH537AnalysisRegression:
    """
    Re-run get_resp1_resp2 on stored stims snapshots and verify exact match
    with the ground-truth outputs.  These tests catch any logic change in the
    classification or sorting code.
    """

    def _run(self, zh537_snapshots):
        p = zh537_snapshots["params"]
        return get_resp1_resp2(
            zh537_snapshots["stims1"],
            zh537_snapshots["stims2"],
            zh537_snapshots["z_ids"],
            stim_onset_idx=p["stim_onset_idx"],
            threshold=p["threshold"],
        )

    def test_nums_exact_match(self, zh537_snapshots):
        _, _, nums, _ = self._run(zh537_snapshots)
        np.testing.assert_array_equal(
            np.array(nums), zh537_snapshots["expected_nums"],
            err_msg="nums changed — classification logic may have regressed"
        )

    def test_resp1_exact_match(self, zh537_snapshots):
        resp1, _, _, _ = self._run(zh537_snapshots)
        np.testing.assert_array_equal(
            resp1, zh537_snapshots["expected_resp1"],
            err_msg="resp1 changed — z-score normalisation or sorting may have regressed"
        )

    def test_resp2_exact_match(self, zh537_snapshots):
        _, resp2, _, _ = self._run(zh537_snapshots)
        np.testing.assert_array_equal(
            resp2, zh537_snapshots["expected_resp2"],
            err_msg="resp2 changed — z-score normalisation or sorting may have regressed"
        )

    def test_z_ids_exact_match(self, zh537_snapshots):
        _, _, _, z_ids_sel = self._run(zh537_snapshots)
        np.testing.assert_array_equal(z_ids_sel[0], zh537_snapshots["expected_z_ids_stim1"])
        np.testing.assert_array_equal(z_ids_sel[1], zh537_snapshots["expected_z_ids_both"])
        np.testing.assert_array_equal(z_ids_sel[2], zh537_snapshots["expected_z_ids_stim2"])
