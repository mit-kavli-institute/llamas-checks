"""Vertical halo / smear metric on lamp frames (``vertical_smear``).

A curved, narrow arc line occupies a small fraction of every detector column, so
the per-column median stays at the background; a vertical halo fills whole columns
around the bright lines and lifts the median of just those columns. The metric is
std(per-column median) / lamp signal, read off the same sorted array as the column
structure metric. Frames are small synthetic MEFs.
"""
import copy
import json

import numpy as np
import pytest
import yaml
from astropy.io import fits

from llamas_checks import qa_engine as E
from llamas_checks.llamasQATests import check_image
from llamas_checks.qa_config_validator import QAConfigValidator
from llamas_checks.qa_engine import QAEngine, line_profiles, trimmed_mean_profile

SHAPE = (64, 64)
BOTTOM = {"type": "rectangle", "x_start": 2, "x_end": 62, "y_start": 1, "y_end": 5}

CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [{"name": "1.A.Red", "hdu_index": 1, "color": "Red", "bench": 1, "side": "A"},
                   {"name": "1.B.Red", "hdu_index": 2, "color": "Red", "bench": 1, "side": "B"}],
    "regions": {"full_frame": {"type": "full"}, "bottom_stripe": BOTTOM},
    "metrics": {
        "column_banding": {"type": "column_structure"},
        "column_banding_norm": {"type": "column_structure_norm",
                                "background_region": "bottom_stripe", "min_signal": 2.0},
        "vertical_smear": {"type": "vertical_smear",
                           "background_region": "bottom_stripe", "min_signal": 20.0},
    },
    "lookup_tables": {"noop": {"a": 1}},
    "rule_sets": {
        "ARC": {
            "applies_when": {"exposure_type": "CAL.R-ARC"},
            "rules": [
                {"name": "vertical_smear", "region": "full_frame", "metric": "vertical_smear",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 0.08}},
                {"name": "column_structure", "region": "full_frame", "metric": "column_banding_norm",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 1e9}},
                {"name": "column_abs", "region": "full_frame", "metric": "column_banding",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 1e9}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def arc_frame(amplitude, halo=0.0, pedestal=700.0, seed=0):
    """Pedestal + curved narrow 'emission lines' (each line drifts 10 columns over the
    frame height, so every column holds a line for only a few rows) + checkerboard
    noise. ``halo`` adds a smooth full-height vertical band of ``halo * amplitude``
    around each line: the vertical smear signature."""
    rng = np.random.default_rng(seed)
    frame = np.full(SHAPE, pedestal, dtype=np.float64)
    ny, nx = SHAPE
    rows = np.arange(ny)
    for x0 in range(8, nx - 8, 12):
        xs = (x0 + 10 * (rows - 6) / (ny - 6)).round().astype(int)
        for y, x in zip(rows, xs):
            if y >= 6 and 0 <= x < nx:
                frame[y, x] += amplitude
        if halo:
            for dx in range(-3, 4):
                if 0 <= x0 + dx < nx:
                    frame[6:, x0 + dx] += halo * amplitude * (1 - abs(dx) / 4)
    frame += np.where(np.indices(SHAPE).sum(axis=0) % 2, 0.5, -0.5)
    frame += rng.normal(0.0, 0.3, SHAPE)
    return frame


def write_mef(tmp_path, name, frames, dtype=np.uint16):
    ph = fits.PrimaryHDU()
    ph.header["PRODCATG"] = "CAL.R-ARC"
    ph.header["READ-MDE"] = "FAST"
    hdus = [ph]
    for data in frames:
        hdus.append(fits.ImageHDU(data=np.asarray(np.round(data) if dtype == np.uint16 else data,
                                                  dtype=dtype)))
    path = tmp_path / name
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def by_rule(report, ext):
    return {r["rule"]: r for r in report["results"] if r["extension"] == ext}


def expected_smear(data):
    signal = float(np.mean(data)) - float(np.median(data[1:5, 2:62]))
    return float(np.std(np.median(np.asarray(data, float), axis=0))) / signal


# ---------------------------------------------------------------- line_profiles / wrapper
@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float32])
@pytest.mark.parametrize("axis", [-1, -2])
def test_line_profiles_match_numpy(dtype, axis):
    rng = np.random.default_rng(1)
    data = (700 + rng.normal(0, 30, (37, 52))).astype(dtype)
    trimmed, median = line_profiles(data, axis)
    np_axis = 1 if axis == -1 else 0
    assert np.allclose(median, np.median(np.asarray(data, float), axis=np_axis))
    ordered = np.sort(np.asarray(data, float), axis=np_axis)
    n = ordered.shape[np_axis]; k = max(1, int(n * E.STRUCTURE_TRIM_FRACTION))
    sl = [slice(None)] * 2; sl[np_axis] = slice(k, n - k)
    assert np.allclose(trimmed, ordered[tuple(sl)].mean(axis=np_axis))
    assert np.array_equal(trimmed_mean_profile(data, axis), trimmed)


def test_line_profiles_float_with_nans():
    data = np.full((20, 20), 5.0, dtype=np.float32)
    data[3, :] = np.nan              # a whole row of NaN
    data[:, 4] = np.nan              # a whole column of NaN
    data[7, 9] = np.nan
    trimmed, median = line_profiles(data, -1)
    assert np.isnan(trimmed[3]) and np.isnan(median[3])
    assert np.allclose(trimmed[[0, 1, 2]], 5.0) and np.allclose(median[[0, 1, 2]], 5.0)
    assert np.array_equal(trimmed_mean_profile(data, -1), trimmed, equal_nan=True)


# ---------------------------------------------------------------- the metric
def test_config_is_valid():
    assert QAConfigValidator(CONFIG, filename="smear").validate() == []


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_smear_matches_independent_computation(tmp_path, dtype):
    frame = arc_frame(2000.0, halo=0.2)
    path = write_mef(tmp_path, "arc.fits", [frame, frame], dtype=dtype)
    stored = fits.getdata(path, 1)
    report = QAEngine(CONFIG, cameras_down=[]).run(path)
    value = by_rule(report, "1.A.Red")["vertical_smear"]["measured_value"]
    assert value == pytest.approx(expected_smear(stored), rel=1e-5)


def test_curved_lines_pass_and_halo_fails_at_any_brightness(tmp_path):
    cfg = copy.deepcopy(CONFIG)
    cfg["extensions"] = cfg["extensions"][:1]
    clean_values = []
    for amp in (400.0, 2000.0, 8000.0):
        clean = QAEngine(cfg, cameras_down=[]).run(write_mef(tmp_path, f"ok{int(amp)}.fits", [arc_frame(amp)]))
        halo = QAEngine(cfg, cameras_down=[]).run(write_mef(tmp_path, f"halo{int(amp)}.fits",
                                                            [arc_frame(amp, halo=0.15)]))
        c = by_rule(clean, "1.A.Red")["vertical_smear"]
        h = by_rule(halo, "1.A.Red")["vertical_smear"]
        assert c["passed"] is True and clean["overall_verdict"] == "PASS", c
        assert h["passed"] is False and halo["overall_verdict"] == "FAIL", h
        assert h["measured_value"] > 3 * c["measured_value"]
        clean_values.append(c["measured_value"])
    # the clean value does not scale with the lamp (20x brightness range)
    assert max(clean_values) < 1.5 * min(clean_values) + 0.01


def test_faint_frame_skips_the_smear_rule(tmp_path):
    # signal ~6 ADU: below min_signal 20 -> SKIPPED; the structure_norm rule (min 2) still runs
    path = write_mef(tmp_path, "faint.fits", [arc_frame(60.0), arc_frame(2000.0)])
    report = QAEngine(CONFIG, cameras_down=[]).run(path)
    faint = by_rule(report, "1.A.Red")
    assert faint["vertical_smear"]["status"] == "SKIPPED"
    assert "min_signal 20" in faint["vertical_smear"]["message"]
    assert faint["column_structure"]["status"] == "EVALUATED"
    assert by_rule(report, "1.B.Red")["vertical_smear"]["status"] == "EVALUATED"
    assert report["overall_verdict"] == "PASS"


def test_smear_default_min_signal_is_20(tmp_path):
    cfg = copy.deepcopy(CONFIG)
    cfg["metrics"]["vertical_smear"] = {"type": "vertical_smear"}
    assert QAConfigValidator(cfg, filename="smear").validate() == []
    path = write_mef(tmp_path, "faint.fits", [arc_frame(60.0), arc_frame(2000.0)])
    report = QAEngine(cfg, cameras_down=[]).run(path)
    assert by_rule(report, "1.A.Red")["vertical_smear"]["status"] == "SKIPPED"
    assert by_rule(report, "1.B.Red")["vertical_smear"]["status"] == "EVALUATED"


def test_one_sort_per_axis_per_detector(tmp_path, monkeypatch):
    """column_structure, column_banding_norm and vertical_smear share one sorted
    array: line_profiles runs once per (detector, axis) and the cache is per run."""
    calls = []
    original = E.line_profiles

    def counting(values, axis, trim=E.STRUCTURE_TRIM_FRACTION):
        calls.append(axis)
        return original(values, axis, trim)

    monkeypatch.setattr(E, "line_profiles", counting)
    engine = QAEngine(CONFIG, cameras_down=[])
    engine.run(write_mef(tmp_path, "a.fits", [arc_frame(2000.0), arc_frame(900.0)]))
    assert calls == [-2, -2]                      # one column sort per detector, 3 metrics
    assert len(engine._profile_cache) == 2
    assert len(engine._signal_cache) == 2
    engine.run(write_mef(tmp_path, "b.fits", [arc_frame(2000.0), arc_frame(900.0)]))
    assert len(calls) == 4 and len(engine._profile_cache) == 2


@pytest.mark.parametrize("field, value, message", [
    ("background_region", "nowhere", "must name a region"),
    ("min_signal", -1, "non-negative"),
    ("threshold", 5, "unknown field"),
])
def test_validator_checks_smear_options(field, value, message):
    cfg = copy.deepcopy(CONFIG)
    cfg["metrics"]["vertical_smear"][field] = value
    errors = [f"{e.path}: {e.message}" for e in QAConfigValidator(cfg, filename="smear").validate()]
    assert any("vertical_smear" in e and message in e for e in errors), errors


def test_check_image_end_to_end(tmp_path):
    path = write_mef(tmp_path, "arc.fits", [arc_frame(2000.0), arc_frame(2000.0, halo=0.2)])
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(CONFIG))
    out = tmp_path / "rep.json"
    result = check_image(str(path), qa_yaml=str(cfg_path), report=str(out), cameras_down="none")
    assert result["status"] == "fail"
    assert result["message"] == "FAIL: 1 fail check(s): vertical_smear@1.B.Red"
    payload = json.loads(out.read_text())
    rows = {r["extension"]: r for r in payload["report"]["results"] if r["rule"] == "vertical_smear"}
    assert rows["1.A.Red"]["passed"] is True and rows["1.B.Red"]["passed"] is False
