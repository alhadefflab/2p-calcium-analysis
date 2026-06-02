"""
Shared fixtures for regression tests.

Requires ZH539_DIR to be set to a directory containing a completed ZH539
pipeline run (provenance.yaml, CNMF .hdf5 files, mmap files, and
analysis/params.yaml must all be present).

If ZH539_DIR is not set all regression tests skip automatically.
"""
import os
import json
import pickle
import numpy as np
import yaml
import pytest
from pathlib import Path

BASELINE_FILE = Path(__file__).parent / "baseline_zh539.json"

_zh539_env = os.environ.get("ZH539_DIR", "")
ZH539_DIR = Path(_zh539_env) if _zh539_env else None


@pytest.fixture(scope="session")
def zh539_dir():
    if ZH539_DIR is None:
        pytest.skip("ZH539_DIR environment variable not set")
    if not ZH539_DIR.exists():
        pytest.skip(f"ZH539 directory not found: {ZH539_DIR}")
    return ZH539_DIR


@pytest.fixture(scope="session")
def baseline():
    if not BASELINE_FILE.exists():
        pytest.skip(
            "baseline_zh539.json not found. "
            "Run: python tests/regression/save_baseline.py"
        )
    with open(BASELINE_FILE) as f:
        data = json.load(f)
    if not data.get("z_planes"):
        pytest.skip(
            "baseline_zh539.json is a placeholder. "
            "Run: python tests/regression/save_baseline.py"
        )
    return data


@pytest.fixture(scope="session")
def zh539_analysis_results(zh539_dir):
    """
    Re-runs the full analysis pipeline from existing CNMF files and returns
    key metrics in the same structure as baseline_zh539.json.

    This is the core of the regression test: it exercises every step from
    CNMF outputs through z-scoring, stimulus window extraction, and responder
    classification — no GUI, no human intervention.
    """
    import sys
    ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(ROOT))

    from pipeline import _get_provenance
    from pipeline_funcs import get_stims1_stims2, get_resp1_resp2
    from caiman.source_extraction.cnmf import cnmf as cnmf_module

    params_yaml = zh539_dir / "analysis" / "params.yaml"
    if not params_yaml.exists():
        pytest.skip(f"analysis/params.yaml not found at {params_yaml}")

    provenance = _get_provenance(str(zh539_dir))

    with open(params_yaml) as f:
        ap = yaml.safe_load(f)

    # -- Per z-plane CNMF metrics ---------------------------------------------
    z_metrics = {}
    for z in sorted(provenance.get("source_extraction", {}).keys()):
        cnm_file = provenance["source_extraction"][z]["filenames"]["cnm_file"]
        cnm = cnmf_module.load_CNMF(cnm_file)
        m = {"neuron_count": int(cnm.estimates.A.shape[1])}
        if getattr(cnm.estimates, "SNR_comp", None) is not None:
            m["mean_snr"] = float(np.mean(cnm.estimates.SNR_comp))
        if getattr(cnm.estimates, "r_values", None) is not None:
            m["mean_r_value"] = float(np.mean(cnm.estimates.r_values))
        z_metrics[z] = m

    # -- Motion correction shift stats ----------------------------------------
    mc_metrics = {}
    for z in sorted(provenance.get("rigid_motion_correction", {}).keys()):
        mc_file = provenance["rigid_motion_correction"][z].get("motion_correct_obj")
        if mc_file and Path(mc_file).exists():
            with open(mc_file, "rb") as f:
                mc = pickle.load(f)
            shifts = np.array(mc.shifts_rig)
            mc_metrics[z] = {
                "mean_shift_px": float(np.mean(np.abs(shifts))),
                "max_shift_px":  float(np.max(np.abs(shifts))),
            }

    # -- Full analysis: z-score → stim windows → responder classification ----
    stims1, stims2, z_ids = get_stims1_stims2(
        provenance,
        frame_period=float(ap["frame_period"]),
        pre_discard_s=float(ap["pre_discard_s"]),
        baseline_s=float(ap["baseline_s"]),
        stim_s=float(ap["stim_s"]),
    )
    _, _, nums, _ = get_resp1_resp2(
        stims1, stims2, z_ids,
        stim_onset_idx=int(ap["stim_onset_idx"]),
        threshold=float(ap["threshold"]),
    )

    return {
        "z_planes": z_metrics,
        "motion_correction": mc_metrics,
        "analysis": {
            "stim1_only":       int(nums[0]),
            "both":             int(nums[1]),
            "stim2_only":       int(nums[2]),
            "total_responders": int(sum(nums)),
        },
    }
