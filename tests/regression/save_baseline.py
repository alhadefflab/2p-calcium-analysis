"""
Generate the regression baseline for ZH539.

Run this once after a verified ZH539 pipeline run. Reads the existing CNMF
outputs and motion correction results from the pipeline, extracts key
biological metrics, and saves them to baseline_zh539.json which is committed
to git as the frozen ground truth.

The regression tests re-run the analysis from those same CNMF files and
compare the new results to this baseline. Only re-run this script when you
have deliberately decided to update the ground truth (e.g. after a CaImAn
upgrade where you have visually verified the new results look correct).

Usage (Anaconda Prompt):
    set ZH539_DIR=C:\\path\\to\\your\\ZH539
    python tests/regression/save_baseline.py
"""
import os
import sys
import json
import pickle
import numpy as np
import yaml
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import _get_provenance
from pipeline_funcs import get_stims1_stims2, get_resp1_resp2

BASELINE_FILE = Path(__file__).parent / "baseline_zh539.json"


def main():
    zh539_env = os.environ.get("ZH539_DIR", "")
    if not zh539_env:
        print("ERROR: ZH539_DIR environment variable is not set.")
        print(r"  set ZH539_DIR=C:\path\to\your\ZH539")
        sys.exit(1)

    zh539_dir = Path(zh539_env)
    if not zh539_dir.exists():
        print(f"ERROR: ZH539 directory not found: {zh539_dir}")
        sys.exit(1)

    params_yaml = zh539_dir / "analysis" / "params.yaml"
    if not params_yaml.exists():
        print(f"ERROR: analysis/params.yaml not found at {params_yaml}")
        print("  The pipeline analysis must have been run and saved first.")
        sys.exit(1)

    print(f"Loading provenance from {zh539_dir} ...")
    provenance = _get_provenance(str(zh539_dir))

    with open(params_yaml) as f:
        ap = yaml.safe_load(f)

    from caiman.source_extraction.cnmf import cnmf as cnmf_module

    # -- Per z-plane metrics from CNMF files ----------------------------------
    z_metrics = {}
    for z in sorted(provenance.get("source_extraction", {}).keys()):
        cnm_file = provenance["source_extraction"][z]["filenames"]["cnm_file"]
        print(f"  Loading CNMF for {z} ...")
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

    # -- Analysis: z-scores + responder classification -----------------------
    print("Running analysis (get_stims1_stims2) ...")
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

    baseline = {
        "dataset": "ZH539",
        "z_planes": z_metrics,
        "motion_correction": mc_metrics,
        "analysis": {
            "stim1_only":        int(nums[0]),
            "both":              int(nums[1]),
            "stim2_only":        int(nums[2]),
            "total_responders":  int(sum(nums)),
        },
        "tolerances": {
            "neuron_count_pct":    15,
            "responder_count_pct": 20,
            "mean_snr_abs":        0.5,
            "mean_shift_px_abs":   1.0,
        },
    }

    with open(BASELINE_FILE, "w") as f:
        json.dump(baseline, f, indent=2)

    print(f"\nBaseline saved to {BASELINE_FILE}")
    print("  Z-plane neuron counts:")
    for z, m in z_metrics.items():
        snr_str = f"  mean_snr={m['mean_snr']:.2f}" if "mean_snr" in m else ""
        print(f"    {z}: {m['neuron_count']} neurons{snr_str}")
    print(f"  Responders — stim1-only: {nums[0]}  both: {nums[1]}  stim2-only: {nums[2]}")
    print("\nCommit baseline_zh539.json to git as the frozen ground truth.")
    print("Only re-run this script when you have verified new results are correct.")


if __name__ == "__main__":
    main()
