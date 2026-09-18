"""
Unit tests for the corrected spatial correlation.

A synthetic 24x24 movie: two overlapping neurons that respond together on a
bright, drifting neuropil background — the case where CaImAn's raw-movie
correlation collapses.  No files, no CaImAn.
"""
import numpy as np
import scipy.sparse as sp

from analysis.spatial_quality import (
    active_frames, corrected_spatial_correlation, quality_verdict,
)

H = W = 24
T = 120
FR = 1.709


def _gauss(cy, cx, sigma=2.5):
    y, x = np.mgrid[0:H, 0:W]
    g = np.exp(-(((y - cy) ** 2 + (x - cx) ** 2) / (2 * sigma ** 2)))
    g[g < 0.05] = 0.0
    return g.ravel(order="F")


def _movie(background_gain=40.0):
    """Two co-active neurons, sustained responses, strong drifting background."""
    a1, a2 = _gauss(10, 10), _gauss(12, 13)      # overlapping footprints
    A = np.column_stack([a1, a2])

    c1 = np.zeros(T); c1[40:] = 6.0              # step response, stays up
    c2 = np.zeros(T); c2[40:] = 5.0              # neighbour responds at the same time
    C = np.vstack([c1, c2])

    y, x = np.mgrid[0:H, 0:W]
    b = (1.0 + 0.5 * y / H + 0.3 * x / W).ravel(order="F")[:, None] * background_gain
    f = (1.0 + 0.2 * np.sin(np.linspace(0, 3, T)))[None, :]

    rng = np.random.default_rng(0)
    Yr = A @ C + b @ f + rng.normal(0, 0.05, (H * W, T))
    return Yr, sp.csc_matrix(A), C, b, f


def _raw_style_r(Yr, A, C):
    """CaImAn 1.13's version: raw movie average over active frames, no subtraction."""
    A = sp.csc_matrix(A)
    out = np.zeros(A.shape[1])
    for k in range(A.shape[1]):
        s, e = A.indptr[k], A.indptr[k + 1]
        px, w = A.indices[s:e], A.data[s:e]
        fr_idx = active_frames(C[k])
        if px.size < 3 or fr_idx.size == 0:
            continue
        img = np.asarray(Yr[px][:, fr_idx]).mean(axis=1)
        out[k] = np.corrcoef(img, w)[0, 1]
    return out


class TestActiveFrames:

    def test_finds_sustained_step_with_no_local_maximum(self):
        # the case that defeats peak-based selection: rises and never comes down
        c = np.zeros(T); c[40:] = 5.0
        fr_idx = active_frames(c)
        assert fr_idx.size == T - 40
        assert fr_idx.min() == 40 and fr_idx.max() == T - 1

    def test_finds_sharp_transient(self):
        c = np.zeros(T); c[50] = 4.0; c[51] = 2.0
        assert active_frames(c).tolist() == [50, 51]

    def test_flat_trace_has_none(self):
        assert active_frames(np.zeros(T)).size == 0

    def test_caps_frame_count_keeping_the_highest(self):
        c = np.linspace(0, 10, T)
        fr_idx = active_frames(c, max_frames=5)
        assert fr_idx.tolist() == list(range(T - 5, T))


class TestCorrectedCorrelation:

    def test_recovers_neurons_that_raw_version_misses(self):
        Yr, A, C, b, f = _movie()
        raw = _raw_style_r(Yr, A, C)
        corrected = corrected_spatial_correlation(Yr, A, C, b, f)
        # the raw metric is dragged down by background and the co-active neighbour
        assert raw.max() < 0.8
        # the corrected one identifies both footprints almost perfectly
        assert corrected.min() > 0.95
        assert (corrected > raw).all()

    def test_pure_background_component_scores_low(self):
        Yr, A, C, b, f = _movie()
        # a footprint sitting on background only, with an invented trace
        bogus = sp.csc_matrix(np.column_stack([_gauss(3, 20)]))
        C_bogus = np.vstack([np.where(np.arange(T) >= 40, 4.0, 0.0)])
        r = corrected_spatial_correlation(Yr, bogus, C_bogus, b, f)
        assert abs(r[0]) < 0.5

    def test_no_activity_scores_zero(self):
        Yr, A, C, b, f = _movie()
        flat = np.zeros_like(C)
        assert corrected_spatial_correlation(Yr, A, flat, b, f).tolist() == [0.0, 0.0]

    def test_works_without_background_model(self):
        Yr, A, C, _, _ = _movie(background_gain=0.0)
        r = corrected_spatial_correlation(Yr, A, C, None, None)
        assert r.min() > 0.95

    def test_empty_input(self):
        Yr, A, C, b, f = _movie()
        empty = sp.csc_matrix((H * W, 0))
        assert corrected_spatial_correlation(Yr, empty, C[:0], b, f).size == 0


class TestVerdict:

    def test_pass_on_shape_alone(self):
        passed, good, bad = quality_verdict([0.85, 0.2], [0.7, 0.7])
        assert passed.tolist() == [True, False]
        assert good.tolist() == [0] and bad.tolist() == [1]

    def test_pass_on_snr_alone(self):
        passed, _, _ = quality_verdict([0.3], [3.0])
        assert passed.tolist() == [True]

    def test_snr_floor_overrides_good_shape(self):
        passed, _, _ = quality_verdict([0.95], [0.4])
        assert passed.tolist() == [False]
