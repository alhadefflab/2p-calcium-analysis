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

    @classmethod
    def from_cnmf(cls, estimates, k: int) -> 'Neuron':
        """Build a Neuron from column k of a CaImAn CNMF estimates object."""
        dims = estimates.dims   # (d1, d2) == (height, width)
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
    def build_all(cls, estimates) -> list['Neuron']:
        """Build one Neuron per accepted component in a CNMF estimates object."""
        K = estimates.A.shape[1]
        return [cls.from_cnmf(estimates, k) for k in range(K)]
