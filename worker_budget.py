"""Size CaImAn worker pools to the memory that is actually free.

CaImAn's default is one worker per core minus one, whatever the RAM.  On Windows
each worker is a fresh process that re-imports CaImAn, so on a 20-core machine
that is 19 × (import cost + data chunk): enough to exhaust RAM and freeze it.

The plan here:
  * measure one real worker's import cost (one process, then it exits),
  * give every worker a fixed-size data chunk instead of 1/n of the movie,
  * run as many workers as fit in free memory minus a safety reserve,
    capped at cores − 1 unless a higher cap is asked for.

Deliberately light on imports: the probe worker should cost what a real one does.
"""
import os

GB = 1024 ** 3

# peak bytes per (pixel, frame) inside a worker: float32 chunk copy plus the
# float64/complex128 FFT buffers of CaImAn's noise estimate, rounded up
BYTES_PER_SAMPLE = 32
CHUNK_TARGET_BYTES = 128 * 1024 ** 2
FALLBACK_WORKER_BYTES = int(1.0 * GB)


def default_max_workers() -> int:
    """CaImAn's own default: one per core, minus one for the main process."""
    return max(1, (os.cpu_count() or 2) - 1)


def chunk_pixels(n_frames: int, target_bytes: int = CHUNK_TARGET_BYTES) -> int:
    """Pixels per worker chunk so one chunk's peak stays near `target_bytes`."""
    return max(500, int(target_bytes // (max(1, n_frames) * BYTES_PER_SAMPLE)))


def reserve_bytes(total: int) -> int:
    """Memory left untouched for the GUI, the OS and everything else."""
    return int(max(4 * GB, 0.15 * total))


def plan_workers(available: int, total: int, per_worker: int,
                 max_workers: int | None = None) -> int:
    """How many workers fit: (available − reserve) / per-worker cost, clamped
    to [1, max_workers]."""
    max_workers = default_max_workers() if max_workers is None else max(1, int(max_workers))
    budget = available - reserve_bytes(total)
    n = int(budget // per_worker) if per_worker > 0 else max_workers
    return max(1, min(max_workers, n))


def _probe_task(_):
    # the module a CNMF worker has to import to run its first task
    import caiman.source_extraction.cnmf.pre_processing  # noqa: F401
    import psutil
    mi = psutil.Process().memory_info()
    return getattr(mi, "private", 0) or mi.rss


def probe_worker_bytes() -> int:
    """Memory one spawned worker holds after importing CaImAn.

    Starts a single-process pool with the same start method CaImAn uses, so on
    Windows it also re-imports the launching script exactly like a real worker.
    Falls back to a conservative constant if the probe fails.
    """
    import multiprocessing as mp
    try:
        with mp.Pool(1) as pool:
            return int(pool.map(_probe_task, [0])[0])
    except Exception:
        return FALLBACK_WORKER_BYTES


def plan_cnmf_workers(n_frames: int, max_workers: int | None = None):
    """(n_workers, pixels_per_chunk, description) for a CNMF run right now."""
    import psutil
    npx = chunk_pixels(n_frames)
    base = probe_worker_bytes()
    per_worker = base + npx * n_frames * BYTES_PER_SAMPLE
    vm = psutil.virtual_memory()
    n = plan_workers(vm.available, vm.total, per_worker, max_workers)
    desc = (f"CNMF workers: {n}  (RAM free {vm.available / GB:.1f} of {vm.total / GB:.1f} GB, "
            f"reserve {reserve_bytes(vm.total) / GB:.1f} GB, "
            f"~{per_worker / GB:.2f} GB per worker, {npx} px per chunk, "
            f"cap {max_workers or default_max_workers()})")
    return n, npx, desc