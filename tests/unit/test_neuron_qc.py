"""
Unit tests for CaImAn quality-check flags on neurons (flag, don't delete).

Synthetic estimates only: no CNMF run, no data files.
"""
from types import SimpleNamespace

import numpy as np
from scipy.sparse import csc_matrix

from gui.neuron import Neuron

DIMS = (8, 8)
THR = dict(min_SNR=2.0, rval_thr=0.8, SNR_lowest=0.5)


def _estimates(K=3, T=20, **extra):
    """K square footprints on an 8x8 field with random traces."""
    rng = np.random.default_rng(0)
    A = np.zeros((DIMS[0] * DIMS[1], K))
    for k in range(K):
        img = np.zeros(DIMS)
        img[k:k + 2, k:k + 2] = 1.0
        A[:, k] = img.ravel(order='F')
    base = dict(A=csc_matrix(A), C=rng.random((K, T)), YrA=rng.random((K, T)) * 0.1,
                SNR_comp=None, r_values=None, idx_components=None, idx_components_bad=None)
    base.update(extra)
    return SimpleNamespace(**base)


class TestBuildAllVerdict:

    def test_failures_start_rejected(self):
        est = _estimates(SNR_comp=[3.0, 0.4, 1.5], r_values=[0.5, 0.9, 0.6],
                         idx_components=np.array([0]), idx_components_bad=np.array([1, 2]))
        neurons = Neuron.build_all(est, dims=DIMS)
        assert [n.passed_qc for n in neurons] == [True, False, False]
        assert [n.accepted for n in neurons] == [True, False, False]

    def test_scores_attached(self):
        est = _estimates(SNR_comp=[3.0, 0.4, 1.5], r_values=[0.5, 0.9, 0.6],
                         idx_components=np.array([0]), idx_components_bad=np.array([1, 2]))
        n = Neuron.build_all(est, dims=DIMS)[1]
        assert n.snr == 0.4 and n.r_value == 0.9

    def test_old_file_without_verdict_starts_all_accepted(self):
        # runs that used select_components save the lists as None
        est = _estimates(SNR_comp=[3.0, 0.7, 1.5], r_values=[0.9, 0.9, 0.85])
        neurons = Neuron.build_all(est, dims=DIMS)
        assert all(n.accepted for n in neurons)
        assert all(n.passed_qc is None for n in neurons)
        assert neurons[1].snr == 0.7

    def test_partial_verdict_is_ignored(self):
        # lists that do not cover every component are not trusted
        est = _estimates(idx_components=np.array([0]), idx_components_bad=np.array([1]))
        neurons = Neuron.build_all(est, dims=DIMS)
        assert all(n.accepted and n.passed_qc is None for n in neurons)

    def test_score_length_mismatch_is_ignored(self):
        est = _estimates(SNR_comp=[3.0, 0.4])
        assert all(n.snr is None for n in Neuron.build_all(est, dims=DIMS))


class TestQcSummary:

    def _neurons(self):
        est = _estimates(SNR_comp=[3.0, 0.4, 1.5], r_values=[0.5, 0.9, 0.6],
                         idx_components=np.array([0]), idx_components_bad=np.array([1, 2]))
        return Neuron.build_all(est, dims=DIMS)

    def test_failed_below_floor(self):
        from gui.neuron_viewer import _qc_summary, _QC_FAIL_COLOR
        neurons = self._neurons()
        text, color = _qc_summary(neurons[1], neurons, THR)
        assert color == _QC_FAIL_COLOR
        assert 'FAILED' in text and 'floor' in text
        assert '2 / 3 failed on this plane' in text

    def test_failed_on_both_tests(self):
        from gui.neuron_viewer import _qc_summary
        neurons = self._neurons()
        text, _ = _qc_summary(neurons[2], neurons, THR)
        assert 'SNR not above 2 and r below 0.8' in text

    def test_passed(self):
        from gui.neuron_viewer import _qc_summary, _QC_PASS_COLOR
        neurons = self._neurons()
        text, color = _qc_summary(neurons[0], neurons, THR)
        assert color == _QC_PASS_COLOR and 'passed' in text

    def test_undefined_r_is_not_blamed_on_snr(self):
        from gui.neuron_viewer import _qc_summary
        est = _estimates(K=2, SNR_comp=[5.0, 3.0], r_values=[-1.0, 0.9],
                         idx_components=np.array([1]), idx_components_bad=np.array([0]))
        neurons = Neuron.build_all(est, dims=DIMS)
        text, _ = _qc_summary(neurons[0], neurons, {**THR, 'rval_lowest': -1})
        assert 'undefined' in text and 'SNR not above' not in text

    def test_unexplained_failure(self):
        # e.g. rejected by the CNN classifier while both scores pass
        from gui.neuron_viewer import _qc_summary
        est = _estimates(K=2, SNR_comp=[5.0, 3.0], r_values=[0.9, 0.9],
                         idx_components=np.array([1]), idx_components_bad=np.array([0]))
        neurons = Neuron.build_all(est, dims=DIMS)
        text, _ = _qc_summary(neurons[0], neurons, {**THR, 'rval_lowest': -1})
        assert 'do not explain' in text

    def test_no_verdict(self):
        from gui.neuron_viewer import _qc_summary
        est = _estimates()
        neurons = Neuron.build_all(est, dims=DIMS)
        text, color = _qc_summary(neurons[0], neurons, {})
        assert 'not available' in text and color == 'gray'


class TestLoadIsCellFallback:
    """Without an is_cell file, analysis uses CaImAn's saved verdict, not all components."""

    @staticmethod
    def _cnmf_file(path, K, good=None, bad=None):
        import h5py
        with h5py.File(path, 'w') as h:
            h['estimates/A/shape'] = np.array([64, K], dtype=np.int32)
            for name, idx in (('idx_components', good), ('idx_components_bad', bad)):
                h[f'estimates/{name}'] = np.bytes_('NoneType') if idx is None else np.asarray(idx)
        return path

    def test_verdict_used_when_no_is_cell_file(self, tmp_path):
        from pipeline_funcs import _load_is_cell
        f = self._cnmf_file(tmp_path / 'concat_z1_cnmf-out.hdf5', 4, good=[0, 2], bad=[1, 3])
        assert _load_is_cell(f, 'z1').tolist() == [True, False, True, False]

    def test_is_cell_file_wins(self, tmp_path):
        from pipeline_funcs import _load_is_cell
        f = self._cnmf_file(tmp_path / 'concat_z1_cnmf-out.hdf5', 4, good=[0, 2], bad=[1, 3])
        np.save(tmp_path / 'concat_z1_is_cell.npy', np.array([True, True, False, False]))
        assert _load_is_cell(f, 'z1').tolist() == [True, True, False, False]

    def test_old_file_without_verdict_returns_none(self, tmp_path):
        from pipeline_funcs import _load_is_cell
        f = self._cnmf_file(tmp_path / 'concat_z1_cnmf-out.hdf5', 4)
        assert _load_is_cell(f, 'z1') is None

    def test_incomplete_verdict_returns_none(self, tmp_path):
        from pipeline_funcs import _load_is_cell
        f = self._cnmf_file(tmp_path / 'concat_z1_cnmf-out.hdf5', 4, good=[0], bad=[1])
        assert _load_is_cell(f, 'z1') is None
