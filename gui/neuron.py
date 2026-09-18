from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.sparse import issparse


@dataclass
class Neuron:
    """Single CNMF-extracted neuron, wrapping spatial footprint and temporal traces."""
    idx: int
    spatial: np.ndarray          # (h, w) float — A column reshaped in Fortran order
    trace_raw: np.ndarray        # (T,) float — C + YrA
    trace_denoised: np.ndarray   # (T,) float — C only
    centroid: tuple[int, int]    # (row, col) weighted by spatial footprint
    accepted: bool = True
    snr: float | None = None       # CaImAn transient SNR (estimates.SNR_comp)
    r_value: float | None = None   # CaImAn spatial correlation (estimates.r_values)
    passed_qc: bool | None = None  # CaImAn quality-check verdict; None = no verdict saved

    @classmethod
    def from_cnmf(cls, estimates, k: int, dims=None) -> 'Neuron':
        """Build a Neuron from column k of a CaImAn CNMF estimates object.

        ``dims`` is the 2-D ``(height, width)`` field-of-view shape. Pass
        ``cnm.dims`` explicitly: a freshly fit CNMF object leaves
        ``estimates.dims`` as ``None`` (CaImAn only back-fills it in
        ``load_CNMF``), which would reshape the footprint to 1-D and break
        ``np.where`` below.
        """
        if dims is None:
            dims = estimates.dims   # fallback (set on objects loaded via load_CNMF)
        col = estimates.A[:, k]
        if issparse(col):
            col = np.asarray(col.todense()).ravel()
        else:
            col = np.asarray(col).ravel()
        spatial = col.reshape(dims, order='F')

        C   = np.asarray(estimates.C[k],   dtype=float)
        YrA = np.asarray(estimates.YrA[k], dtype=float)

        thr = spatial.max() * 0.1
        ys, xs = np.where(spatial > thr)
        if len(ys) > 0:
            w = spatial[ys, xs]
            cy = int(np.average(ys, weights=w))
            cx = int(np.average(xs, weights=w))
        else:
            cy, cx = dims[0] // 2, dims[1] // 2

        return cls(
            idx=k,
            spatial=spatial,
            trace_raw=C + YrA,
            trace_denoised=C.copy(),
            centroid=(cy, cx),
        )

    @classmethod
    def build_all(cls, estimates, dims=None) -> list['Neuron']:
        """Build one Neuron per component in a CNMF estimates object.

        CaImAn's quality scores are attached when present.  If the pass / fail
        lists from ``evaluate_components`` are saved (``idx_components`` and
        ``idx_components_bad`` covering every component), each neuron starts
        accepted only if it passed; failures start rejected for review.  Files
        without a verdict (e.g. older runs that deleted failures) start all
        accepted, as before.

        Pass ``cnm.dims`` as ``dims`` for objects straight from ``cnm.fit()``;
        ``estimates.dims`` alone is unreliable there (see ``from_cnmf``).
        """
        K = estimates.A.shape[1]
        neurons = [cls.from_cnmf(estimates, k, dims=dims) for k in range(K)]

        snr = _per_component(getattr(estimates, 'SNR_comp', None), K)
        rval = _per_component(getattr(estimates, 'r_values', None), K)
        good = getattr(estimates, 'idx_components', None)
        bad = getattr(estimates, 'idx_components_bad', None)
        has_verdict = (good is not None and bad is not None
                       and len(good) + len(bad) == K)
        good_set = {int(i) for i in good} if has_verdict else set()

        for k, n in enumerate(neurons):
            if snr is not None:
                n.snr = float(snr[k])
            if rval is not None:
                n.r_value = float(rval[k])
            if has_verdict:
                n.passed_qc = k in good_set
                n.accepted = n.passed_qc
        return neurons


def _per_component(values, K):
    """values as a (K,) array, or None when missing or not one value per component."""
    if values is None:
        return None
    arr = np.asarray(values, dtype=float).ravel()
    return arr if arr.size == K else None
