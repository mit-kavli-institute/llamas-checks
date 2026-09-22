"""Tests for llamas_checks.validate and its wiring into check_image.

All frames are small synthetic MEFs built in ``tmp_path``; no external FITS
fixtures are required.
"""
import copy

import numpy as np
import yaml
from astropy.io import fits

from llamas_checks import validate as V
from llamas_checks.llamasQATests import check_image
from llamas_checks.qa_engine import QAEngine

SHAPE = (40, 40)
RNG = np.random.default_rng(42)


def make_mef(tmp_path, name, skip=(), placeholder=(), swap=None):
    """Write a 24-extension MEF in IDX_LOOKUP order.

    ``skip`` drops those HDU indices entirely, ``placeholder`` writes all-ones
    int16 at those indices, ``swap`` exchanges the BENCH values between two HDUs.
    """
    ph = fits.PrimaryHDU()
    ph.header["PRODCATG"] = "CAL.R-BIA"
    ph.header["READ-MDE"] = "SLOW"
    ph.header["REXPTIME"] = 0.001
    ph.header["SEXPTIME"] = 0.001
    hdus = {}
    for (channel, bench, side), idx in V.IDX_LOOKUP.items():
        if idx in skip:
            continue
        if idx in placeholder:
            data = np.ones(SHAPE, dtype=np.int16)
        else:
            data = (700.0 + RNG.normal(0.0, 3.0, SHAPE)).astype(np.float32)
        ext = fits.ImageHDU(data=data)
        ext.header["BENCH"] = int(bench)
        ext.header["SIDE"] = side
        ext.header["COLOR"] = channel
        hdus[idx] = ext
    if swap is not None:
        a, b = swap
        hdus[a].header["BENCH"], hdus[b].header["BENCH"] = (
            hdus[b].header["BENCH"], hdus[a].header["BENCH"])
    hdul = fits.HDUList([ph] + [hdus[idx] for idx in sorted(hdus)])
    path = tmp_path / name
    hdul.writeto(path, overwrite=True)
    return str(path)


def full_extensions():
    return [{"name": V.camera_label(cfg), "hdu_index": idx, "color": cfg[0].capitalize(),
             "bench": int(cfg[1]), "side": cfg[2]} for cfg, idx in V.IDX_LOOKUP.items()]


# ------- inspect_structure -------

def test_complete_file_has_no_missing_cameras(tmp_path):
    rep = V.inspect_structure(make_mef(tmp_path, "full.fits"))
    assert rep["n_extensions"] == 24
    assert rep["expected_extensions"] == 24
    assert rep["missing_cameras"] == []
    assert rep["identity_mismatches"] == []


def test_skipped_hdus_are_reported_in_idx_order(tmp_path):
    rep = V.inspect_structure(make_mef(tmp_path, "gaps.fits", skip=(3, 15, 24)))
    assert rep["n_extensions"] == 21
    assert rep["missing_cameras"] == ["1.A.Blue", "3.A.Blue", "4.B.Blue"]


def test_swapped_bench_is_an_identity_mismatch(tmp_path):
    path = make_mef(tmp_path, "swap.fits", swap=(1, 7))  # 1.A.Red <-> 2.A.Red
    rep = V.inspect_structure(path, full_extensions())
    got = sorted(rep["identity_mismatches"], key=lambda m: m["hdu_index"])
    assert got == [
        {"extension": "1.A.Red", "hdu_index": 1, "header_identity": "2.A.Red"},
        {"extension": "2.A.Red", "hdu_index": 7, "header_identity": "1.A.Red"},
    ]
    # the swap only relabels; every detector is still present
    assert rep["missing_cameras"] == []


def test_colour_only_extensions_skip_identity_check(tmp_path):
    path = make_mef(tmp_path, "swap2.fits", swap=(1, 7))
    exts = [{"name": e["name"], "hdu_index": e["hdu_index"], "color": e["color"]}
            for e in full_extensions()]
    assert V.inspect_structure(path, exts)["identity_mismatches"] == []


def test_inspect_structure_out_of_range_hdu_index_is_skipped(tmp_path):
    path = make_mef(tmp_path, "short.fits", skip=tuple(range(2, 25)))
    rep = V.inspect_structure(path, full_extensions())
    assert rep["n_extensions"] == 1
    assert rep["identity_mismatches"] == []
    assert len(rep["missing_cameras"]) == 23


# ------- is_placeholder_extension -------

def test_placeholder_all_ones_int16():
    assert V.is_placeholder_extension(fits.ImageHDU(data=np.ones(SHAPE, dtype=np.int16)))


def test_placeholder_all_zeros_uint16():
    assert V.is_placeholder_extension(fits.ImageHDU(data=np.zeros(SHAPE, dtype=np.uint16)))


def test_placeholder_comment_marker_on_noisy_data():
    hdu = fits.ImageHDU(data=RNG.normal(700.0, 3.0, SHAPE).astype(np.float32))
    hdu.header["COMMENT"] = "Placeholder extension created for missing camera"
    assert V.is_placeholder_extension(hdu)


def test_noisy_data_is_not_placeholder():
    hdu = fits.ImageHDU(data=RNG.normal(700.0, 3.0, SHAPE).astype(np.float32))
    assert not V.is_placeholder_extension(hdu)


def test_railed_constant_frame_is_not_placeholder():
    # A saturated (ADC ceiling) or pedestal-level constant frame is a real fault,
    # not a placeholder: it must stay eligible for saturation/structure rules.
    railed = np.full(SHAPE, 65535, dtype=np.uint16)
    assert not V.is_placeholder_extension(fits.ImageHDU(data=railed))
    assert not QAEngine._is_placeholder_data(railed)
    assert not QAEngine._is_placeholder_data(np.full(SHAPE, 700.0, dtype=np.float32))
    assert QAEngine._is_placeholder_data(np.ones(SHAPE, dtype=np.int16))
    assert QAEngine._is_placeholder_data(np.zeros(SHAPE, dtype=np.uint16))


def test_cam_name_only_extensions_are_counted_present(tmp_path):
    path = make_mef(tmp_path, "camname.fits")
    with fits.open(path, mode="update") as hdul:
        for hdu in hdul[1:]:
            cam = f"{hdu.header['BENCH']}{hdu.header['SIDE']}_{hdu.header['COLOR']}"
            for key in ("BENCH", "SIDE", "COLOR"):
                del hdu.header[key]
            hdu.header["CAM_NAME"] = cam
    assert V.inspect_structure(path)["missing_cameras"] == []


def test_all_nan_is_not_placeholder():
    hdu = fits.ImageHDU(data=np.full(SHAPE, np.nan, dtype=np.float32))
    assert not V.is_placeholder_extension(hdu)


# ------- detector_label -------

def test_detector_label_from_bench_side_color():
    hdr = fits.Header()
    hdr["BENCH"] = 1
    hdr["SIDE"] = "A"
    hdr["COLOR"] = "red"
    assert V.detector_label(hdr) == "1.A.Red"


def test_detector_label_from_cam_name():
    hdr = fits.Header()
    hdr["CAM_NAME"] = "2B_green"
    assert V.detector_label(hdr) == "2.B.Green"


def test_detector_label_none_without_identity():
    hdr = fits.Header()
    hdr["EXTNAME"] = "IMAGE"
    assert V.detector_label(hdr) is None


# ------- placeholder round-trip -------

def test_fix_extensions_round_trip(tmp_path):
    path = make_mef(tmp_path, "gaps.fits", skip=(3, 15, 24))
    out = str(tmp_path / "fixed.fits")
    assert V.validate_and_fix_extensions(path, output_file=out) == out
    with fits.open(out, memmap=False) as hdul:
        assert len(hdul) - 1 == 24
        assert V.is_placeholder_extension(hdul[3])
        assert V.detector_label(hdul[3].header) == "1.A.Blue"
        assert not V.is_placeholder_extension(hdul[1])
    info = V.validate_fits_structure(out)
    assert info["missing_cameras"] == []
    assert info["needs_validation"] is False
    assert V.get_placeholder_extension_indices(out) == [3, 15, 24]
    # the original was not modified
    assert V.validate_fits_structure(path)["missing_count"] == 3


# ------- check_image wiring -------

CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [{"name": "1.A.Red", "hdu_index": 1, "color": "Red", "bench": 1, "side": "A"}],
    "regions": {"full_frame": {"type": "full"}},
    "metrics": {"mean": {"type": "mean"}},
    "lookup_tables": {"noop": {"a": 1}},
    "rule_sets": {
        "BIAS": {
            "applies_when": {"exposure_type": "CAL.R-BIA"},
            "rules": [
                {"name": "level", "region": "full_frame", "metric": "mean",
                 "per_extension": True, "severity": "FAIL",
                 "limits": {"min": 600, "max": 800}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def write_config(tmp_path, extensions=None):
    cfg = copy.deepcopy(CONFIG)
    if extensions is not None:
        cfg["extensions"] = extensions
    path = tmp_path / "qa.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def test_check_image_reports_structure_on_single_extension(tmp_path):
    path = make_mef(tmp_path, "one.fits", skip=tuple(range(2, 25)))
    result = check_image(path, qa_yaml=write_config(tmp_path))
    structure = result["structure"]
    assert structure["n_extensions"] == 1
    assert len(structure["missing_cameras"]) == 23
    assert "1.A.Red" not in structure["missing_cameras"]
    assert structure["placeholder_extensions"] == []
    assert structure["identity_mismatches"] == []
    assert result["status"] == {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}[result["overall_verdict"]]
    assert result["status"] == "pass"


def test_check_image_reports_placeholder_extensions(tmp_path):
    path = make_mef(tmp_path, "ph.fits", placeholder=(3,))
    exts = [e for e in full_extensions() if e["name"] in ("1.A.Red", "1.A.Blue")]
    result = check_image(path, qa_yaml=write_config(tmp_path, exts))
    assert result["structure"]["placeholder_extensions"] == ["1.A.Blue"]
    assert result["structure"]["missing_cameras"] == []
    assert result["status"] == "pass"


def test_check_image_structure_present_with_report_and_fail(tmp_path):
    # unmatched PRODCATG -> early "fail" return; structure must still be attached
    path = make_mef(tmp_path, "sci.fits", skip=(3,))
    with fits.open(path, mode="update") as h:
        h[0].header["PRODCATG"] = "SCI.R-XX"
    report = tmp_path / "report.json"
    result = check_image(path, qa_yaml=write_config(tmp_path), report=str(report))
    assert result["status"] == "fail"
    assert result["structure"]["missing_cameras"] == ["1.A.Blue"]
    payload = yaml.safe_load(report.read_text())
    assert payload["structure"]["missing_cameras"] == ["1.A.Blue"]
