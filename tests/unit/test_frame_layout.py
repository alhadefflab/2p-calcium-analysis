"""
Unit tests for _frame_layout in gui.py.
Run with:  pytest tests/unit/test_frame_layout.py
"""
import pytest

pytest.importorskip("customtkinter")  # skip if GUI deps not installed

from gui import _frame_layout


class TestFrameLayout:

    def test_clean_integer_seconds(self):
        # fp=0.5 → each second is exactly 2 frames
        r = _frame_layout(0.5, 10.0, 30.0, 60.0)
        assert r['pre_f']  == 20
        assert r['base_f'] == 60
        assert r['stim_f'] == 120

    def test_ses_f_identity(self):
        # ses_f must always equal the sum of the three windows
        cases = [
            (0.585, 30, 30, 180),
            (0.5,   10, 60, 360),
            (1.0,   30, 30, 180),
            (0.1,    5, 10,  60),
        ]
        for fp, pre_s, bl_s, st_s in cases:
            r = _frame_layout(fp, pre_s, bl_s, st_s)
            assert r['ses_f'] == r['pre_f'] + r['base_f'] + r['stim_f'], (
                f"identity failed for fp={fp}, pre={pre_s}, bl={bl_s}, st={st_s}")

    def test_zh537_params(self):
        # Verify against values in ZH537/analysis/params.yaml
        # frame_period=0.585, pre_discard_s=30, baseline_s=30, stim_s=180
        r = _frame_layout(0.585, 30.0, 30.0, 180.0)
        # round(30 / 0.585) = round(51.28) = 51
        assert r['pre_f']  == 51
        assert r['base_f'] == 51
        # round(180 / 0.585) = round(307.69) = 308
        assert r['stim_f'] == 308
        assert r['ses_f']  == 410  # 51 + 51 + 308

    def test_fractional_period_uses_round_not_floor(self):
        # round(30 / 0.585) = 51, not 50 (floor would give 51 too, but
        # for a case that differs: round(9.5) = 10, floor(9.5) = 9)
        r = _frame_layout(1.0, 9.5, 9.5, 9.5)
        assert r['pre_f']  == round(9.5 / 1.0)   # 10, not 9
        assert r['base_f'] == round(9.5 / 1.0)
        assert r['stim_f'] == round(9.5 / 1.0)

    def test_zero_pre_discard(self):
        r = _frame_layout(0.5, 0.0, 30.0, 60.0)
        assert r['pre_f']  == 0
        assert r['ses_f']  == r['base_f'] + r['stim_f']

    def test_return_keys(self):
        r = _frame_layout(0.5, 1.0, 2.0, 3.0)
        assert set(r.keys()) == {'pre_f', 'base_f', 'stim_f', 'ses_f'}

    def test_all_values_are_ints(self):
        r = _frame_layout(0.585, 30.0, 30.0, 180.0)
        for key, val in r.items():
            assert isinstance(val, int), f"{key} should be int, got {type(val)}"
