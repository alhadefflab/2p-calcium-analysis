"""
Unit tests for ROI editor coordinate conversion (_canvas_to_image).

These tests guard against the bug introduced in commit 8f8c80f where _apply_resize
set a single self._scale = min(w/iw, h/ih) but the PIL image was stretched to fill
the full canvas (w, h).  For non-square canvases the Y-axis mapping was wrong,
causing right-click removal to target the wrong neuron.

"""
import pytest

pytest.importorskip("customtkinter")

from gui import _canvas_to_image, _frame_layout


class TestCanvasToImage:

    def test_square_image_square_canvas(self):
        # Trivial case: uniform scale, both axes identical.
        sx, sy = 512 / 512, 512 / 512   # scale = 1.0 each
        assert _canvas_to_image(100, 200, sx, sy) == (100, 200)

    def test_square_image_square_canvas_scaled(self):
        # Canvas is 2× the image — scale_x == scale_y == 2.
        sx, sy = 1024 / 512, 1024 / 512
        ix, iy = _canvas_to_image(400, 600, sx, sy)
        assert ix == 200
        assert iy == 300

    def test_square_image_non_square_canvas_x_axis(self):
        # Square 512×512 image on a 835×1020 canvas (typical maximised window).
        # scale_x = 835/512 ≈ 1.631, scale_y = 1020/512 ≈ 1.992
        # The old single-scale bug used min(scale_x, scale_y) = scale_x for BOTH axes.
        iw, ih = 512, 512
        cw, ch = 835, 1020
        scale_x = cw / iw
        scale_y = ch / ih

        cx, cy = 300.0, 700.0
        ix, iy = _canvas_to_image(cx, cy, scale_x, scale_y)

        assert ix == int(cx / scale_x)   # 184
        assert iy == int(cy / scale_y)   # 351

        # Verify the old approach (single min scale) would give the WRONG y.
        old_scale = min(scale_x, scale_y)   # = scale_x ≈ 1.631
        old_iy = int(cy / old_scale)        # ≈ 429  (wrong)
        assert old_iy != iy, (
            "Old single-scale formula should give a different (incorrect) y value")

    def test_non_square_image_non_square_canvas(self):
        # Wide image (iw > ih) on a tall canvas.
        iw, ih = 800, 400
        cw, ch = 500, 600
        scale_x = cw / iw   # 0.625
        scale_y = ch / ih   # 1.5

        ix, iy = _canvas_to_image(250, 300, scale_x, scale_y)
        assert ix == int(250 / 0.625)   # 400
        assert iy == int(300 / 1.5)     # 200

    def test_origin_always_maps_to_origin(self):
        for sx, sy in [(1.0, 1.0), (2.0, 3.0), (0.5, 0.8)]:
            assert _canvas_to_image(0, 0, sx, sy) == (0, 0)

    def test_result_is_truncated_not_rounded(self):
        # int() truncates toward zero — verify this is the behaviour.
        sx, sy = 1.5, 1.5
        ix, iy = _canvas_to_image(4, 5, sx, sy)
        assert ix == 2   # 4/1.5 = 2.666… → 2
        assert iy == 3   # 5/1.5 = 3.333… → 3

    def test_axes_are_independent(self):
        # Changing cx must not affect iy; changing cy must not affect ix.
        sx, sy = 2.0, 3.0
        ix1, iy1 = _canvas_to_image(100, 150, sx, sy)
        ix2, iy2 = _canvas_to_image(200, 150, sx, sy)
        ix3, iy3 = _canvas_to_image(100, 300, sx, sy)
        assert iy1 == iy2, "iy should not depend on cx"
        assert ix1 == ix3, "ix should not depend on cy"