"""
Generate (or regenerate) the regression snapshot files used by test_zh537_analysis.py.

Run this script:
  - Once after first setting up the test suite (no snapshots exist yet).
  - After a CaImAn update when CNMF outputs have intentionally changed and you
    have verified the new results look correct visually.

What it does:
  1. Loads the ZH537 provenance.yaml to find all CNMF HDF5 and mmap file paths.
  2. Calls get_stims1_stims2() to compute per-neuron z-score traces.
  3. Saves stims1.npy, stims2.npy, z_ids.npy, and params.npy into
     tests/regression/snapshots/ (creating it if needed).

The snapshot files are small (~300 KB total) and should be committed to git so
that the regression tests can run on any machine without needing the full
~5 GB ZH537 data directory.

Usage:
  cd "c:\\Users\\juare\\Alhadeff Lab\\2p-calcium-analysis"
  python tests/regression/update_snapshots.py
"""
import sys
from pathlib import Path

# Ensure project root is importable
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import os
import numpy as np
import yaml

from pipeline_funcs import get_stims1_stims2
from pipeline import _get_provenance

_zh537_env = os.environ.get("ZH537_DIR", "")
ZH537_DIR     = Path(_zh537_env) if _zh537_env else None
SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"


def main():
    if ZH537_DIR is None:
        print("ERROR: ZH537_DIR environment variable is not set.")
        print("  Set it to the path of your ZH537 data folder, e.g.:")
        print(r'  set ZH537_DIR=C:\path\to\your\ZH537')
        sys.exit(1)
    if not ZH537_DIR.exists():
        print(f"ERROR: ZH537 data directory not found:\n  {ZH537_DIR}")
        sys.exit(1)

    PARAMS_YAML = ZH537_DIR / "analysis" / "params.yaml"

    print(f"Loading provenance from {ZH537_DIR} ...")
    provenance = _get_provenance(str(ZH537_DIR))

    print(f"Loading analysis params from {PARAMS_YAML} ...")
    with open(PARAMS_YAML) as f:
        ap = yaml.safe_load(f)

    frame_period  = float(ap["frame_period"])
    pre_discard_s = float(ap["pre_discard_s"])
    baseline_s    = float(ap["baseline_s"])
    stim_s        = float(ap["stim_s"])
    stim_onset    = int(ap["stim_onset_idx"])
    threshold     = float(ap["threshold"])

    print("Computing stims1 / stims2 (this loads CNMF files — may take a minute) ...")
    stims1, stims2, z_ids = get_stims1_stims2(
        provenance,
        frame_period=frame_period,
        pre_discard_s=pre_discard_s,
        baseline_s=baseline_s,
        stim_s=stim_s,
    )

    from pipeline_funcs import get_resp1_resp2
    print("Computing expected resp1 / resp2 from the same stims ...")
    resp1, resp2, nums, z_ids_sel = get_resp1_resp2(
        stims1, stims2, z_ids,
        stim_onset_idx=stim_onset,
        threshold=threshold,
    )

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(SNAPSHOTS_DIR / "stims1.npy", stims1)
    np.save(SNAPSHOTS_DIR / "stims2.npy", stims2)
    np.save(SNAPSHOTS_DIR / "z_ids.npy",  z_ids)
    np.save(SNAPSHOTS_DIR / "params.npy", {
        "stim_onset_idx": stim_onset,
        "threshold":      threshold,
        "frame_period":   frame_period,
        "pre_discard_s":  pre_discard_s,
        "baseline_s":     baseline_s,
        "stim_s":         stim_s,
    })
    np.save(SNAPSHOTS_DIR / "expected_resp1.npy",      resp1)
    np.save(SNAPSHOTS_DIR / "expected_resp2.npy",      resp2)
    np.save(SNAPSHOTS_DIR / "expected_nums.npy",       np.array(nums))
    np.save(SNAPSHOTS_DIR / "expected_z_ids_stim1.npy", z_ids_sel[0])
    np.save(SNAPSHOTS_DIR / "expected_z_ids_both.npy",  z_ids_sel[1])
    np.save(SNAPSHOTS_DIR / "expected_z_ids_stim2.npy", z_ids_sel[2])

    print(f"\nSnapshots saved to {SNAPSHOTS_DIR}")
    print(f"  stims1.npy  shape={stims1.shape}  ({stims1.nbytes // 1024} KB)")
    print(f"  stims2.npy  shape={stims2.shape}  ({stims2.nbytes // 1024} KB)")
    print(f"  z_ids.npy   shape={z_ids.shape}")
    print(f"  Responders — stim1-only: {nums[0]}  both: {nums[1]}  stim2-only: {nums[2]}")
    print(f"  params.npy  stim_onset={stim_onset}, threshold={threshold}")
    print("\nDone. You can now run:  pytest tests/regression/"  )
    print("  (or 'pytest' to run all tests)")


if __name__ == "__main__":
    main()
