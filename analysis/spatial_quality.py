"""Corrected spatial correlation for CNMF components.

CaImAn's `r_values` answers "does the footprint match the movie while this
component is active".  Its documentation describes subtracting the other
components before correlating, but the implementation (1.13) only drops frames
in which an overlapping neighbour was active and then correlates the *raw*
movie.  In dense, dim tissue, where neighbours respond together and the resting
fluorescence is dominated by neuropil, that raw average is mostly background, so
real neurons score low and the metric stops being a usable test of "is this a
neuron".

This module computes the documented version: for each component, average the
movie over its active frames after removing the modelled background (b·f) and
every other component's contribution, then correlate that residual image with
the component's own footprint over its own pixels.

Pure numpy / scipy.  Each component touches only its own pixels and a few dozen
frames, so memory stays in the low MB regardless of movie length.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

# CaImAn's peak threshold, kept so the two metrics stay comparable
ACTIVE_THRESHOLD = 0.3
MAX_ACTIVE_FRAMES = 200


def active_frames(c: np.ndarray, thres: float = ACTIVE_THRESHOLD,
                  max_frames: int = MAX_ACTIVE_FRAMES) -> np.ndarray:
    """Frames where this component is active: the top part of its own trace.

    CaImAn picks local maxima (`peakutils`) and widens them by a window.  That
    fails exactly where this data lives: a response that rises at stimulus onset
    and stays up has no local maximum, so no frames are found and the
    correlation is reported as 0.  Selecting by amplitude instead handles both
    sharp transients and sustained plateaus.

    Frames at or above `thres` of the trace's range; when more than
    `max_frames` qualify, the highest ones are used (bounds the cost on long
    plateaus).  Empty when the trace is flat.
    """
    c = np.asarray(c, dtype=float).ravel()
    lo, hi = c.min(), c.max()
    if c.size == 0 or hi <= lo:
        return np.empty(0, dtype=int)

    frames = np.flatnonzero(c >= lo + thres * (hi - lo))
    if frames.size > max_frames:
        frames = frames[np.argsort(c[frames])[::-1][:max_frames]]
    return np.sort(frames)


def corrected_spatial_correlation(Yr, A, C, b, f,
                                  thres: float = ACTIVE_THRESHOLD,
                                  max_frames: int = MAX_ACTIVE_FRAMES) -> np.ndarray:
    """(K,) spatial correlation per component, background and neighbours removed.

    Yr : (pixels, T) movie, memory-mapped is fine; pixel order must match A.
    A  : (pixels, K) footprints (sparse or dense).  C, b, f: CNMF estimates.

    A component with no detectable activity, an empty footprint or no variance
    to correlate scores 0, as CaImAn does.
    """
    A = sp.csc_matrix(A)
    C = np.asarray(C, dtype=float)
    K = A.shape[1]
    out = np.zeros(K, dtype=float)
    if K == 0:
        return out

    b = None if b is None else np.asarray(b, dtype=float).reshape(A.shape[0], -1)
    f = None if f is None else np.asarray(f, dtype=float).reshape(-1, C.shape[1])

    for k in range(K):
        s, e = A.indptr[k], A.indptr[k + 1]
        px = A.indices[s:e]
        w = A.data[s:e]
        frames = active_frames(C[k], thres=thres, max_frames=max_frames)
        if px.size < 3 or frames.size == 0 or np.ptp(w) == 0:
            continue

        # movie over this component's pixels and frames, background removed
        resid = np.asarray(Yr[px][:, frames], dtype=float)
        if b is not None and f is not None and b.size and f.size:
            resid -= b[px] @ f[:, frames]
        # remove every other component's contribution on these pixels
        A_px = A[px].toarray()
        A_px[:, k] = 0.0
        resid -= A_px @ C[:, frames]

        img = resid.mean(axis=1)
        if np.ptp(img) == 0:
            continue
        out[k] = float(np.corrcoef(img, w)[0, 1])

    return np.nan_to_num(out, nan=0.0)


def quality_verdict(r_values, snr, min_SNR=2.0, rval_thr=0.8,
                    SNR_lowest=0.5, rval_lowest=-1.0):
    """(passed, idx_good, idx_bad) using CaImAn's rule on the given metrics.

    Pass one high threshold (r or SNR) and clear both low ones.
    """
    r = np.asarray(r_values, dtype=float)
    s = np.asarray(snr, dtype=float)
    passed = ((r >= rval_thr) | (s > min_SNR)) & (s > SNR_lowest) & (r > rval_lowest)
    idx = np.arange(r.size)
    return passed, idx[passed], idx[~passed]
