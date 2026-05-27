"""
Unit tests for pipeline_funcs.py — pure analysis functions.
These tests require no data files and no GPU.
Run with:  pytest tests/unit/test_analysis_funcs.py
"""
import numpy as np
import scipy.sparse as sp
import pytest

from pipeline_funcs import custom_df_f_startend, get_resp1_resp2, _compute_session_windows


# ── mock CNMF object ──────────────────────────────────────────────────────────

class _Estimates:
    def __init__(self, A, C, b, f, YrA):
        self.A, self.C, self.b, self.f, self.YrA = A, C, b, f, YrA


class _MockCNMF:
    """
    Minimal stand-in for a CaImAn CNMF object.

    Uses A = identity(K) so A.T @ A = I, which makes the spatial scaling
    factor exactly 1 for every neuron — the formula reduces to pure
    mean/std z-scoring of C over the baseline window.
    """
    def __init__(self, C: np.ndarray, YrA: np.ndarray | None = None):
        K, T = C.shape
        A   = sp.eye(K, format='csc')          # (K, K) identity
        b   = np.zeros((K, 1), dtype=np.float32)
        f   = np.zeros((1, T), dtype=np.float32)
        YrA = np.zeros_like(C) if YrA is None else YrA
        self.estimates = _Estimates(A, C, b, f, YrA)


# ── custom_df_f_startend  ─────────────────────────────────────────────────────

class TestCustomDfFStartend:

    def _make_step(self, K, T, bl_s, bl_e, baseline_val, step_val, seed=0):
        """K neurons with constant baseline then a step response."""
        rng = np.random.default_rng(seed)
        C = np.full((K, T), baseline_val, dtype=np.float64)
        noise = rng.normal(0, 0.05, C.shape)
        C += noise
        C[:, bl_e:] += step_val
        return C

    def test_zscore_baseline_is_near_zero(self):
        K, T, bl_s, bl_e = 4, 200, 20, 60
        C = self._make_step(K, T, bl_s, bl_e, baseline_val=5.0, step_val=3.0)
        cnm = _MockCNMF(C)
        df = custom_df_f_startend(cnm, bl_s, bl_e, method='zscore')
        baseline_mean = df[:, bl_s:bl_e].mean(axis=1)
        np.testing.assert_allclose(baseline_mean, 0.0, atol=0.05)

    def test_zscore_response_is_positive(self):
        K, T, bl_s, bl_e = 4, 200, 20, 60
        C = self._make_step(K, T, bl_s, bl_e, baseline_val=5.0, step_val=8.0)
        cnm = _MockCNMF(C)
        df = custom_df_f_startend(cnm, bl_s, bl_e, method='zscore')
        response_median = np.median(df[:, bl_e:], axis=1)
        assert np.all(response_median > 5.0), "strong step should yield high z-scores"

    def test_zscore_math_exact(self):
        # Single neuron, T=10, known values
        K, T = 1, 10
        bl_s, bl_e = 0, 5
        C = np.array([[1., 2., 1., 2., 1., 5., 5., 5., 5., 5.]])

        cnm = _MockCNMF(C)
        df = custom_df_f_startend(cnm, bl_s, bl_e, method='zscore')

        # With A=I_1, b=0, f=0: F=C, f0_full=C
        # fb = mean(C[:,0:5]) = mean([1,2,1,2,1]) = 1.4
        # sigma = std(f0[:,0:5]) = std([1,2,1,2,1])
        baseline = C[0, bl_s:bl_e]
        expected_fb    = baseline.mean()
        expected_sigma = baseline.std()
        expected_df    = (C - expected_fb) / expected_sigma

        np.testing.assert_allclose(df, expected_df, rtol=1e-6)

    def test_norm_to_median_math_exact(self):
        K, T = 1, 10
        bl_s, bl_e = 0, 5
        # Odd-length baseline so median is an actual element
        C = np.array([[1., 3., 2., 3., 1., 6., 6., 6., 6., 6.]])

        cnm = _MockCNMF(C)
        df = custom_df_f_startend(cnm, bl_s, bl_e, method='norm_to_median')

        # median of [1,3,2,3,1] = 2.0; f0 and fb both come from C so
        # fb = median(C[:,0:5]) = 2.0
        # f0 = median(f0[:,0:5]) = same = 2.0 (because f0=C when A=I, b=0)
        # df = (C - 2.0) / 2.0
        expected = (C - 2.0) / 2.0
        np.testing.assert_allclose(df, expected, rtol=1e-6)

    def test_residuals_flag(self):
        K, T = 3, 100
        bl_s, bl_e = 10, 40
        C   = self._make_step(K, T, bl_s, bl_e, baseline_val=2.0, step_val=1.0)
        YrA = np.zeros_like(C)
        YrA[:, bl_e:] = 0.5  # residuals add a small offset in response window

        cnm_no_res = _MockCNMF(C, YrA=np.zeros_like(C))
        cnm_res    = _MockCNMF(C, YrA=YrA)

        df_no  = custom_df_f_startend(cnm_no_res, bl_s, bl_e, method='zscore', use_residuals=False)
        df_yes = custom_df_f_startend(cnm_res,    bl_s, bl_e, method='zscore', use_residuals=True)

        # With residuals the response window should be larger
        assert df_yes[:, bl_e:].mean() > df_no[:, bl_e:].mean()

    def test_output_shape(self):
        K, T = 7, 150
        C = np.random.default_rng(1).normal(2, 0.3, (K, T))
        cnm = _MockCNMF(C)
        df = custom_df_f_startend(cnm, 10, 50, method='zscore')
        assert df.shape == (K, T)


# ── get_resp1_resp2 ────────────────────────────────────────────────────────────

class TestGetResp1Resp2:

    def _make_stims(self, K, T, stim_onset,
                    stim1_only_idx, both_idx, stim2_only_idx,
                    response_val=3.0, noise_val=0.3):
        """Build synthetic stimulus arrays with known responders."""
        rng = np.random.default_rng(42)
        stims1 = rng.normal(0, noise_val, (K, T))
        stims2 = rng.normal(0, noise_val, (K, T))

        for idx in stim1_only_idx:
            stims1[idx, stim_onset:] += response_val
        for idx in stim2_only_idx:
            stims2[idx, stim_onset:] += response_val
        for idx in both_idx:
            stims1[idx, stim_onset:] += response_val
            stims2[idx, stim_onset:] += response_val

        return stims1, stims2

    def test_counts_match_known_responders(self):
        K, T, onset = 10, 200, 50
        s1_only = [0, 1, 2]
        both    = [3, 4]
        s2_only = [5, 6, 7]

        stims1, stims2 = self._make_stims(K, T, onset, s1_only, both, s2_only)
        z_ids = np.arange(K)

        _, _, nums, _ = get_resp1_resp2(stims1, stims2, z_ids,
                                        stim_onset_idx=onset, threshold=1.64)
        assert nums[0] == len(s1_only), "stim1-only count wrong"
        assert nums[1] == len(both),    "both count wrong"
        assert nums[2] == len(s2_only), "stim2-only count wrong"

    def test_resp_arrays_have_correct_row_count(self):
        K, T, onset = 10, 200, 50
        stims1, stims2 = self._make_stims(K, T, onset,
                                          stim1_only_idx=[0, 1],
                                          both_idx=[2],
                                          stim2_only_idx=[3, 4, 5])
        z_ids = np.arange(K)
        resp1, resp2, nums, _ = get_resp1_resp2(stims1, stims2, z_ids,
                                                stim_onset_idx=onset, threshold=1.64)
        expected_rows = sum(nums)
        assert resp1.shape[0] == expected_rows
        assert resp2.shape[0] == expected_rows

    def test_resp_arrays_column_count_matches_input(self):
        K, T, onset = 8, 180, 40
        stims1, stims2 = self._make_stims(K, T, onset, [0], [1], [2])
        z_ids = np.arange(K)
        resp1, resp2, _, _ = get_resp1_resp2(stims1, stims2, z_ids,
                                             stim_onset_idx=onset, threshold=1.64)
        assert resp1.shape[1] == T
        assert resp2.shape[1] == T

    def test_no_responders(self):
        K, T, onset = 5, 100, 30
        stims = np.zeros((K, T))  # all z-scores = 0
        z_ids = np.arange(K)
        resp1, resp2, nums, z_ids_sel = get_resp1_resp2(
            stims, stims, z_ids, stim_onset_idx=onset, threshold=1.64)
        assert nums == [0, 0, 0]
        assert resp1.shape[0] == 0
        assert resp2.shape[0] == 0
        for arr in z_ids_sel:
            assert len(arr) == 0

    def test_all_respond_to_both(self):
        K, T, onset = 4, 100, 20
        stims = np.full((K, T), 5.0)  # all neurons, all frames above threshold
        z_ids = np.arange(K)
        _, _, nums, _ = get_resp1_resp2(stims, stims, z_ids,
                                        stim_onset_idx=onset, threshold=1.64)
        assert nums == [0, K, 0]

    def test_threshold_boundary_exactly_at_threshold(self):
        # A neuron whose median exactly equals the threshold should NOT be classified
        K, T, onset = 1, 100, 0
        threshold = 1.64
        stims = np.full((1, T), threshold)  # median == threshold exactly
        z_ids = np.array([1])
        _, _, nums, _ = get_resp1_resp2(stims, stims, z_ids,
                                        stim_onset_idx=onset, threshold=threshold)
        # `> threshold` (strict), so exactly-at-threshold → not a responder
        assert sum(nums) == 0

    def test_threshold_just_above(self):
        K, T, onset = 1, 100, 0
        threshold = 1.64
        stims1 = np.full((1, T), threshold + 0.01)
        stims2 = np.zeros((1, T))
        z_ids = np.array([1])
        _, _, nums, _ = get_resp1_resp2(stims1, stims2, z_ids,
                                        stim_onset_idx=onset, threshold=threshold)
        assert nums[0] == 1   # stim1-only
        assert nums[1] == 0
        assert nums[2] == 0

    def test_resp1_sorted_stim1only_first(self):
        # Stim1-only responses should appear first in resp1, sorted descending by stim1 median
        K, T, onset = 6, 200, 50
        stims1, stims2 = self._make_stims(
            K, T, onset,
            stim1_only_idx=[0, 1, 2],  # respond only to stim1
            both_idx=[],
            stim2_only_idx=[3, 4, 5],  # respond only to stim2
        )
        z_ids = np.arange(K)
        resp1, _, nums, _ = get_resp1_resp2(stims1, stims2, z_ids,
                                            stim_onset_idx=onset, threshold=1.64)
        # First nums[0] rows come from stim1-only → should have high stim1 median
        n1 = nums[0]
        n2 = nums[2]
        assert np.all(np.median(resp1[:n1, onset:], axis=1) > 1.64)
        # Last nums[2] rows come from stim2-only → low stim1 median
        assert np.all(np.median(resp1[n1:n1+n2, onset:], axis=1) < 1.64)

    def test_z_ids_sel_lengths_match_nums(self):
        K, T, onset = 9, 150, 40
        stims1, stims2 = self._make_stims(K, T, onset, [0, 1], [2, 3], [4, 5, 6])
        z_ids = np.arange(K)
        _, _, nums, z_ids_sel = get_resp1_resp2(stims1, stims2, z_ids,
                                                stim_onset_idx=onset, threshold=1.64)
        assert len(z_ids_sel[0]) == nums[0]
        assert len(z_ids_sel[1]) == nums[1]
        assert len(z_ids_sel[2]) == nums[2]


# ── _compute_session_windows ──────────────────────────────────────────────────

class TestComputeSessionWindows:

    def test_session2_boundary_independent_of_stim_s(self):
        """Changing stim_s must not move where session 2 starts."""
        pre_f = base_f = 51
        ses_f = 410  # real 180 s recording (51+51+308)

        _, _, _, bline2_start_180, _, _ = _compute_session_windows(
            pre_f, base_f, 308, ses_f, ses_f)
        _, _, _, bline2_start_240, _, _ = _compute_session_windows(
            pre_f, base_f, 410, ses_f, ses_f)

        assert bline2_start_180 == bline2_start_240, (
            "Session 2 boundary must not shift when stim_s changes")

    def test_stim_window_clipped_when_stim_s_exceeds_data(self):
        """stim_f larger than available stim data is silently clipped."""
        pre_f, base_f = 51, 51
        ses_f = 410   # only 308 frames of stim available

        _, _, stim1_end, _, _, stim2_end = _compute_session_windows(
            pre_f, base_f, 9999, ses_f, ses_f)

        assert stim1_end == ses_f,      "stim1_end must not exceed session 1"
        assert stim2_end == 2 * ses_f,  "stim2_end must not exceed session 2"

    def test_shorter_stim_s_analyzes_fewer_frames_when_data_is_long(self):
        """With a long recording, a smaller stim_s analyses fewer stim frames."""
        pre_f, base_f = 51, 51
        ses_f = 512   # data long enough for 240 s stim

        _, bline1_end, stim1_end_180, _, _, _ = _compute_session_windows(
            pre_f, base_f, 308, ses_f, ses_f)
        _, _, stim1_end_240, _, _, _ = _compute_session_windows(
            pre_f, base_f, 410, ses_f, ses_f)

        assert (stim1_end_180 - bline1_end) < (stim1_end_240 - bline1_end)

    def test_asymmetric_sessions_handled_correctly(self):
        """Sessions of different lengths (edge case) are still split correctly."""
        pre_f, base_f, stim_f = 51, 51, 308
        ses1_f, ses2_f = 410, 420   # session 2 slightly longer

        b1s, b1e, s1e, b2s, b2e, s2e = _compute_session_windows(
            pre_f, base_f, stim_f, ses1_f, ses2_f)

        assert b2s == ses1_f + pre_f,   "session 2 baseline must start at ses1_f + pre_f"
        assert s2e <= ses1_f + ses2_f,  "session 2 stim window must not overflow"
