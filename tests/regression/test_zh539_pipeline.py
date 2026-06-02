"""
Regression tests for the ZH539 analysis pipeline.

These tests re-run the full analysis from existing CNMF files (no GUI, no
human intervention) and compare the results to the committed baseline in
baseline_zh539.json. They catch regressions in:

  - Neuron counts per z-plane (Cellpose + CNMF)
  - Motion correction shift statistics
  - Responder classification (z-scoring + threshold logic)

Tolerances are defined in baseline_zh539.json and allow for the natural
run-to-run variability of CNMF (which uses stochastic initialisation).

If the baseline doesn't exist yet:
    python tests/regression/save_baseline.py

Run with:
    set ZH539_DIR=C:\\path\\to\\your\\ZH539
    pytest tests/regression/ -v
"""
import pytest
import numpy as np

pytestmark = pytest.mark.regression


class TestNeuronCounts:
    """CNMF finds roughly the same number of neurons per z-plane."""

    def test_all_zplanes_present(self, zh539_analysis_results, baseline):
        expected_zplanes = set(baseline["z_planes"].keys())
        got_zplanes = set(zh539_analysis_results["z_planes"].keys())
        assert expected_zplanes == got_zplanes, (
            f"z-plane mismatch — expected {sorted(expected_zplanes)}, "
            f"got {sorted(got_zplanes)}"
        )

    def test_neuron_count_per_zplane(self, zh539_analysis_results, baseline):
        tol = baseline["tolerances"]["neuron_count_pct"] / 100
        for z, expected in baseline["z_planes"].items():
            got = zh539_analysis_results["z_planes"][z]["neuron_count"]
            exp = expected["neuron_count"]
            pct_diff = abs(got - exp) / max(exp, 1)
            assert pct_diff <= tol, (
                f"{z}: neuron count {got} differs from baseline {exp} "
                f"by {pct_diff*100:.1f}% (tolerance {tol*100:.0f}%)"
            )

    def test_no_zplane_is_empty(self, zh539_analysis_results):
        for z, m in zh539_analysis_results["z_planes"].items():
            assert m["neuron_count"] > 0, f"{z}: no neurons found"

    def test_mean_snr_per_zplane(self, zh539_analysis_results, baseline):
        tol = baseline["tolerances"]["mean_snr_abs"]
        for z, expected in baseline["z_planes"].items():
            if "mean_snr" not in expected:
                continue
            got_snr = zh539_analysis_results["z_planes"][z].get("mean_snr")
            if got_snr is None:
                continue
            diff = abs(got_snr - expected["mean_snr"])
            assert diff <= tol, (
                f"{z}: mean SNR {got_snr:.2f} differs from baseline "
                f"{expected['mean_snr']:.2f} by {diff:.2f} (tolerance {tol})"
            )


class TestMotionCorrection:
    """Motion correction produces similar shift distributions."""

    def test_mean_shift_per_zplane(self, zh539_analysis_results, baseline):
        tol = baseline["tolerances"]["mean_shift_px_abs"]
        for z, expected in baseline["motion_correction"].items():
            got = zh539_analysis_results["motion_correction"].get(z, {})
            if "mean_shift_px" not in got:
                pytest.skip(f"MC object not found for {z}")
            diff = abs(got["mean_shift_px"] - expected["mean_shift_px"])
            assert diff <= tol, (
                f"{z}: mean shift {got['mean_shift_px']:.2f}px differs from "
                f"baseline {expected['mean_shift_px']:.2f}px "
                f"by {diff:.2f}px (tolerance {tol}px)"
            )


class TestResponderClassification:
    """Responder classification produces consistent counts."""

    def test_total_responders(self, zh539_analysis_results, baseline):
        tol = baseline["tolerances"]["responder_count_pct"] / 100
        got = zh539_analysis_results["analysis"]["total_responders"]
        exp = baseline["analysis"]["total_responders"]
        pct_diff = abs(got - exp) / max(exp, 1)
        assert pct_diff <= tol, (
            f"total responders {got} differs from baseline {exp} "
            f"by {pct_diff*100:.1f}% (tolerance {tol*100:.0f}%)"
        )

    def test_at_least_some_responders(self, zh539_analysis_results):
        total = zh539_analysis_results["analysis"]["total_responders"]
        assert total > 0, "no responders found — analysis may have failed silently"

    def test_responder_categories(self, zh539_analysis_results, baseline):
        tol = baseline["tolerances"]["responder_count_pct"] / 100
        for key in ("stim1_only", "both", "stim2_only"):
            got = zh539_analysis_results["analysis"][key]
            exp = baseline["analysis"][key]
            if exp == 0:
                assert got == 0, f"{key}: expected 0 from baseline, got {got}"
            else:
                pct_diff = abs(got - exp) / exp
                assert pct_diff <= tol, (
                    f"{key}: {got} differs from baseline {exp} "
                    f"by {pct_diff*100:.1f}% (tolerance {tol*100:.0f}%)"
                )

    def test_counts_sum_to_total(self, zh539_analysis_results):
        a = zh539_analysis_results["analysis"]
        assert a["stim1_only"] + a["both"] + a["stim2_only"] == a["total_responders"]
