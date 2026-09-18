"""
Unit tests for the µm scale (read from the Prairie View XML) and for the
post-CNMF qc_passed stage of the curation record.

Synthetic XML and masks only: no recordings, no CaImAn.
"""
import numpy as np
import scipy.sparse as sp

from analysis.roi_curation import (
    areas_um2, label_image, polygon_mask, save_curation, update_yield, write_summary,
)
from pipeline_utils import read_microns_per_pixel, scale_of, um_per_px_of

H, W = 60, 80
UM = 1.2109375

XML_HEADER = """<?xml version="1.0" encoding="utf-8"?>
<PVScan version="5.7.64.300">
  <PVStateShard>
    <PVStateValue key="linesPerFrame" value="512" />
    <PVStateValue key="micronsPerPixel">
      <IndexedValue index="XAxis" value="{x}" />
      <IndexedValue index="YAxis" value="{y}" />
      <IndexedValue index="ZAxis" value="{z}" />
    </PVStateValue>
  </PVStateShard>
  <Sequence type="TSeries ZSeries Element" cycle="1">
    <Frame relativeTime="0" index="1" />
  </Sequence>
</PVScan>
"""


def _session(tmp_path, name="ZH999_veh-000", x=UM, y=UM, z=24):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.xml").write_text(XML_HEADER.format(x=x, y=y, z=z), encoding="utf-8")
    return d


class TestReadScale:

    def test_reads_axes(self, tmp_path):
        got = read_microns_per_pixel(_session(tmp_path))
        assert got == {"x": UM, "y": UM, "z": 24.0}

    def test_missing_xml_returns_none(self, tmp_path):
        (tmp_path / "empty").mkdir()
        assert read_microns_per_pixel(tmp_path / "empty") is None

    def test_per_cycle_files_are_skipped(self, tmp_path):
        d = _session(tmp_path)
        (d / "ZH999_veh-000_Cycle00001.xml").write_text("<broken>", encoding="utf-8")
        assert read_microns_per_pixel(d)["x"] == UM

    def test_scale_recorded_with_warning_when_sessions_differ(self, tmp_path):
        from pipeline import _read_scale
        a = _session(tmp_path, "s0")
        b = _session(tmp_path, "s1", x=1.5, y=1.5)
        sc = _read_scale([a, b])
        assert sc["um_per_px"] == UM and sc["z_step_um"] == 24.0
        assert any("disagree" in w for w in sc["warnings"])

    def test_warns_when_x_and_y_differ(self, tmp_path):
        from pipeline import _read_scale
        sc = _read_scale([_session(tmp_path, "s0", x=1.2, y=1.4)])
        assert sc["um_per_px"] == 1.2
        assert any("x and y" in w for w in sc["warnings"])

    def test_accessors(self):
        prov = {"load_data": {"scale": {"um_per_px": UM, "z_step_um": 24.0}}}
        assert um_per_px_of(prov) == UM
        assert scale_of(prov)["z_step_um"] == 24.0
        assert um_per_px_of({"load_data": {}}) is None
        assert um_per_px_of({}) is None


class TestAreaConversion:

    def test_areas_um2(self):
        assert areas_um2({"AP": 100, "NTS": None}, 2.0) == {"AP": 400.0, "NTS": None}

    def test_none_without_scale(self):
        assert areas_um2({"AP": 100}, None) is None


def _square(r, c, size=6):
    m = np.zeros((H, W), dtype=bool)
    m[r:r + size, c:c + size] = True
    return m


def _masks(squares):
    return np.stack([s.flatten("F") for s in squares], axis=1)


def _record(ap_inside=(5, 5), added_at=(30, 40)):
    det = [_square(*ap_inside), _square(40, 60)]
    added = _square(*added_at)
    ap = polygon_mask([(0, 0), (20, 0), (20, 20), (0, 20)], H, W)
    return dict(
        n_detected=2, next_id=3,
        initial_labels=label_image(_masks(det), H, W)[0],
        final_labels=label_image(_masks([det[0], added]), H, W, ids=[0, 2])[0],
        final_ids=[0, 2],
        removal_reason={1: "manual"},
        exclusion_polygons=[],
        keep_mask=np.ones((H, W), bool),
        ap_polygon=[(0, 0), (20, 0), (20, 20), (0, 20)], ap_mask=ap,
        views=[],
    ), det, added


class TestCurationInMicrons:

    def test_record_carries_both_units(self, tmp_path):
        rec, _, _ = _record()
        s = save_curation(tmp_path / "z1", "z1", rec, um_per_px=UM)
        assert s["um_per_px"] == UM
        assert s["area_um2"]["AP"] == round(s["area_px"]["AP"] * UM * UM, 1)
        assert s["density_unit_mm2"] == "neurons per mm²"
        ac = s["after_curation"]["all"]
        assert ac["density"]["AP"] is not None          # px density kept
        assert ac["density_mm2"]["AP"] is not None      # µm density added

    def test_px_only_without_scale(self, tmp_path):
        rec, _, _ = _record()
        s = save_curation(tmp_path / "z1", "z1", rec)
        assert s["um_per_px"] is None and s["area_um2"] is None
        assert "density_mm2" not in s["after_curation"]["all"]

    def test_summary_csv_has_both_units(self, tmp_path):
        rec, det, added = _record()
        save_curation(tmp_path / "z1", "z1", rec, um_per_px=UM)
        update_yield(tmp_path / "z1", "z1",
                     sp.csc_matrix(_masks([added, det[0]]).astype(float)))
        text = write_summary(tmp_path).read_text(encoding="utf-8")
        assert "total_density_per_10k_px2" in text and "total_density_per_mm2" in text
        assert "AP_area_um2" in text and "um_per_px" in text


class TestQcPassedStage:

    def test_added_split_into_kept_and_dropped(self, tmp_path):
        rec, det, added = _record()
        save_curation(tmp_path / "z1", "z1", rec, um_per_px=UM)
        # two components: the hand-added one (fails), the detected one (passes)
        A = sp.csc_matrix(_masks([added, det[0]]).astype(float))
        y = update_yield(tmp_path / "z1", "z1", A,
                         is_cell=np.array([False, True]),
                         qc_pass=np.array([False, True]))
        assert y["traces"]["all"]["count"]["total"] == 2
        assert y["qc_passed"]["added"]["count"]["total"] == 0
        assert y["qc_passed"]["detected"]["count"]["total"] == 1
        assert y["qc_failed_ids"] == [2]               # the added ROI's stable id
        assert y["qc_passed"]["all"].get("density_mm2") is not None

    def test_stage_absent_when_no_verdict(self, tmp_path):
        rec, det, added = _record()
        save_curation(tmp_path / "z1", "z1", rec)
        A = sp.csc_matrix(_masks([added, det[0]]).astype(float))
        y = update_yield(tmp_path / "z1", "z1", A)
        assert "qc_passed" not in y

    def test_wrong_length_verdict_ignored(self, tmp_path):
        rec, det, added = _record()
        save_curation(tmp_path / "z1", "z1", rec)
        A = sp.csc_matrix(_masks([added, det[0]]).astype(float))
        y = update_yield(tmp_path / "z1", "z1", A, qc_pass=np.array([True]))
        assert "qc_passed" not in y
