"""Per-neuron result files (analysis.results_io) and pooling across mice (combine_mice).

Small synthetic arrays only; no CaImAn import.
"""
import numpy as np
import pandas as pd
import pytest
import yaml

from analysis.responders import classify_from_medians, responder_order, sort_key
from analysis.results_io import (neuron_table, save_neuron_results,
                                 load_neuron_results, NEURONS_CSV, trace_file)
import combine_mice as cm


def _stims(medians_per_stim, T=40, onset=10):
    """(K, T) arrays whose post-onset median equals the given values, baseline 0."""
    out = []
    for med in medians_per_stim:
        s = np.zeros((len(med), T))
        s[:, onset:] = np.asarray(med, float)[:, None]
        out.append(s)
    return out


def _write_mouse(root, name, medians_per_stim, z_ids, regions, T=40, onset=10,
                 threshold=1.64, fp=0.5, stim_s=15):
    d = root / name / "analysis"
    stims = _stims(medians_per_stim, T, onset)
    save_neuron_results(d, stims, z_ids, regions, name, onset, threshold)
    with open(d / "params.yaml", "w") as f:
        yaml.safe_dump(dict(frame_period=fp, stim_onset_idx=onset, threshold=threshold,
                            stim_s=stim_s, n_stims=len(stims)), f)
    return root / name


# ── responders ────────────────────────────────────────────────────────────────

class TestClassify:

    def test_two_stim_groups(self):
        med = np.array([[3, 0], [3, 3], [0, 3], [0, 0]], float)
        responds, group = classify_from_medians(med, 1.64)
        assert group.tolist() == [0, 1, 2, -1]
        assert responds.tolist() == [[True, False], [True, True], [False, True], [False, False]]

    def test_three_stim_primary(self):
        med = np.array([[2, 5, 3], [0, 0, 0], [9, 2, 2]], float)
        _, group = classify_from_medians(med, 1.64)
        assert group.tolist() == [1, -1, 0]

    def test_order_sorts_stim2_only_by_stim2(self):
        med = np.array([[0, 2], [0, 7], [4, 0], [6, 0]], float)
        _, group = classify_from_medians(med, 1.64)
        idx, sizes = responder_order(med, group, 2)
        assert idx.tolist() == [3, 2, 1, 0]
        assert sizes == [2, 0, 2]

    def test_nonresponders_keyed_by_largest_median(self):
        med = np.array([[0.1, 1.0], [1.5, 0.2]], float)
        _, group = classify_from_medians(med, 1.64)
        assert sort_key(med, group).tolist() == [1.0, 1.5]


# ── neurons.csv ───────────────────────────────────────────────────────────────

class TestNeuronTable:

    def test_columns_and_values(self):
        stims = _stims([[3, 3, 0, 0], [0, 3, 3, 0]])
        df = neuron_table(stims, [1, 1, 2, 2], [0, 1, -1, 1], "M1", 10, 1.64)
        assert list(df.columns) == [
            "mouse", "z_plane", "plane_index", "region",
            "median_z_stim1", "median_z_stim2",
            "responds_stim1", "responds_stim2", "responder", "group"]
        assert df["z_plane"].tolist() == ["z1", "z1", "z2", "z2"]
        assert df["plane_index"].tolist() == [0, 1, 0, 1]
        assert df["region"].tolist() == ["AP", "NTS", "unclassified", "NTS"]
        assert df["group"].tolist() == ["stim1_only", "both", "stim2_only", "none"]
        assert df["responder"].tolist() == [1, 1, 1, 0]

    def test_plane_index_restarts_per_mouse(self):
        stims = _stims([[0, 0, 0, 0]])
        df = neuron_table(stims, [1, 1, 1, 1], None, ["A", "A", "B", "B"], 10, 1.64)
        assert df["plane_index"].tolist() == [0, 1, 0, 1]
        assert (df["region"] == "unclassified").all()

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            neuron_table(_stims([[1, 2]]), [1], None, "M", 10, 1.64)

    def test_empty(self):
        df = neuron_table([np.zeros((0, 40))], np.array([], int), None, "M", 10, 1.64)
        assert len(df) == 0

    def test_round_trip(self, tmp_path):
        folder = _write_mouse(tmp_path, "NA", [[3, 0, 2], [0, 0, 5]], [1, 1, 2], [0, 1, 1])
        run = load_neuron_results(folder)            # mouse folder, not analysis/
        assert run["dir"].name == "analysis"
        assert run["table"]["mouse"].tolist() == ["NA"] * 3   # not parsed as missing
        assert len(run["traces"]) == 2 and run["traces"][0].shape == (3, 40)
        assert run["traces"][0].dtype == np.float32
        assert run["params"]["stim_onset_idx"] == 10

    def test_missing_files_message(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Re-run the Analysis stage"):
            load_neuron_results(tmp_path)

    def test_stale_traces_detected(self, tmp_path):
        folder = _write_mouse(tmp_path, "M", [[3, 0]], [1, 1], None)
        np.save(folder / "analysis" / trace_file(0), np.zeros((5, 40), np.float32))
        with pytest.raises(ValueError, match="different runs"):
            load_neuron_results(folder)


# ── combine_mice ──────────────────────────────────────────────────────────────

class TestPool:

    def test_crops_to_shared_window_and_keeps_alignment(self, tmp_path):
        a = _write_mouse(tmp_path, "A", [[3, 0]], [1, 1], None, T=40, onset=10)
        b = _write_mouse(tmp_path, "B", [[0, 4, 4]], [1, 1, 2], None, T=50, onset=15)
        table, traces, info = cm.pool(cm.load_mice([a, b]))
        assert (info["pre"], info["post"]) == (10, 30)
        assert traces[0].shape == (5, 40)
        # onset column is the first post-onset frame for every mouse
        assert traces[0][:, info["pre"]].tolist() == [3, 0, 0, 4, 4]
        assert traces[0][:, info["pre"] - 1].tolist() == [0] * 5
        assert any("cropped" in n for n in info["notes"])

    def test_duplicate_mouse_labels_renamed(self, tmp_path):
        a = _write_mouse(tmp_path / "run1", "M", [[3]], [1], None)
        b = _write_mouse(tmp_path / "run2", "M", [[3]], [1], None)
        table, _, _ = cm.pool(cm.load_mice([a, b]))
        assert table["mouse"].nunique() == 2

    def test_frame_period_mismatch_raises(self, tmp_path):
        a = _write_mouse(tmp_path, "A", [[3]], [1], None, fp=0.5)
        b = _write_mouse(tmp_path, "B", [[3]], [1], None, fp=0.6)
        with pytest.raises(ValueError, match="Frame periods differ"):
            cm.pool(cm.load_mice([a, b]))

    def test_threshold_override(self, tmp_path):
        a = _write_mouse(tmp_path, "A", [[3, 1]], [1, 1], None, threshold=1.64)
        table, _, info = cm.pool(cm.load_mice([a]), threshold=0.5)
        assert table["responder"].tolist() == [1, 1]
        assert info["thresholds"] == [0.5]


class TestSummaryAndTraces:

    @pytest.fixture
    def pooled(self, tmp_path):
        a = _write_mouse(tmp_path, "A", [[3, 3, 0, 0], [0, 3, 3, 0]],
                         [1, 1, 1, 1], [0, 0, 1, 1])
        b = _write_mouse(tmp_path, "B", [[5, 0], [0, 0]], [1, 1], [1, 1])
        return cm.pool(cm.load_mice([a, b]))

    def test_mouse_summary(self, pooled):
        table, _, info = pooled
        s = cm.mouse_summary(table, info["n_stims"]).set_index("mouse")
        assert s.loc["A", "n_neurons"] == 4
        assert s.loc["A", "n_responders"] == 3
        assert s.loc["A", "pct_responders"] == 75.0
        assert s.loc["A", "n_both"] == 1
        assert s.loc["A", "pct_of_responders_both"] == pytest.approx(33.33)
        assert s.loc["A", "n_resp_stim1"] == 2
        assert s.loc["A", "n_neurons_AP"] == 2 and s.loc["A", "n_responders_AP"] == 2
        assert s.loc["B", "n_neurons_AP"] == 0
        assert np.isnan(s.loc["B", "pct_responders_AP"])
        assert s.loc["B", "n_responders_NTS"] == 1

    def test_mean_and_sem_across_mice(self, pooled):
        table, traces, info = pooled
        rows = np.ones(len(table), bool)
        per_mouse = cm.mean_traces(table, traces, 0, rows)
        assert per_mouse["A"][1] == 2 and per_mouse["B"][1] == 1
        onset = info["pre"]
        assert per_mouse["A"][0][onset] == pytest.approx(3.0)
        assert per_mouse["B"][0][onset] == pytest.approx(5.0)
        mean, sem = cm.across_mice(per_mouse, traces[0].shape[1])
        assert mean[onset] == pytest.approx(4.0)
        assert sem[onset] == pytest.approx(np.std([3, 5], ddof=1) / np.sqrt(2))

    def test_single_mouse_sem_is_nan(self, pooled):
        table, traces, _ = pooled
        rows = (table["mouse"] == "B").to_numpy()
        _, sem = cm.across_mice(cm.mean_traces(table, traces, 0, rows), traces[0].shape[1])
        assert np.isnan(sem).all()


def test_main_writes_outputs(tmp_path):
    a = _write_mouse(tmp_path, "A", [[3, 3, 0, 0], [0, 3, 3, 0]], [1, 1, 2, 2], [0, 0, 1, 1])
    b = _write_mouse(tmp_path, "B", [[5, 0, 2], [0, 4, 2]], [1, 1, 1], [0, 1, 1])
    out = tmp_path / "combined"
    cm.main([str(a), str(b), "--out", str(out), "--split-region",
             "--stim-names", "Saline", "CCK", "--include-nonresponders"])
    for name in ["pooled_neurons.csv", "mouse_summary.csv", "mean_traces_All.csv",
                 "mean_traces_AP.csv", "mean_traces_NTS.csv", "pooled_heatmap.png",
                 "pooled_heatmap_AP.png", "pooled_heatmap_NTS.png",
                 "mean_sem_traces.png", "responders_by_mouse.png"]:
        assert (out / name).exists(), name
    pooled = pd.read_csv(out / "pooled_neurons.csv")
    assert len(pooled) == 7
    traces = pd.read_csv(out / "mean_traces_All.csv")
    assert {"time_s", "stim1_A", "stim1_B", "stim1_mean", "stim1_sem"} <= set(traces.columns)
    assert traces["time_s"].iloc[10] == 0.0
