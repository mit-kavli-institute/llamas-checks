"""Bias row-stripe metric (``row_stripes``).

A faulty readout (2.B.Green SLOW, 2026-10-14) adds random bright or dark rows to a
bias, each offset by a few to ~30 ADU over its whole length. The metric is the
largest |P - running_median(P, 15 rows)| over rows edge..n-edge-1, where P is the
per-row interquartile mean (25 % trimmed from each end): a smooth gradient follows
its running median, hot pixels filling < 25 % of a row are trimmed away, and the
bias ramp in the first rows is ignored. P is a second slice of the same sorted rows
as the row structure metric, so the rule costs no extra sort.
Frames are small synthetic MEFs; two real-frame regressions skip without Box.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from llamas_checks import qa_engine as E
from llamas_checks.llamasQATests import check_image
from llamas_checks.qa_config_validator import QAConfigValidator
from llamas_checks.qa_engine import QAEngine, line_profiles, stripe_deviation

SHAPE = (96, 200)   # rows x columns

CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [{"name": "2.B.Green", "hdu_index": 1, "color": "Green", "bench": 2, "side": "B"},
                   {"name": "1.A.Red", "hdu_index": 2, "color": "Red", "bench": 1, "side": "A"}],
    "regions": {"full_frame": {"type": "full"}},
    "metrics": {
        "row_banding": {"type": "row_structure"},
        "row_stripes": {"type": "row_stripes", "trim": 0.25, "smooth_rows": 15, "edge_rows": 8},
    },
    "lookup_tables": {"noop": {"a": 1}},
    "rule_sets": {
        "BIAS": {
            "applies_when": {"exposure_type": "CAL.R-BIA"},
            "rules": [
                {"name": "row_structure", "region": "full_frame", "metric": "row_banding",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 1e9}},
                {"name": "row_stripes", "region": "full_frame", "metric": "row_stripes",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 2.0}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def bias_frame(seed=0, level=650.0, sigma=4.0):
    rng = np.random.default_rng(seed)
    return rng.normal(level, sigma, SHAPE)


def write_mef(tmp_path, name, frames, dtype=np.uint16):
    ph = fits.PrimaryHDU()
    ph.header["PRODCATG"] = "CAL.R-BIA"
    ph.header["READ-MDE"] = "SLOW"
    hdus = [ph]
    for data in frames:
        data = np.clip(np.round(data), 0, 65535) if dtype == np.uint16 else data
        hdus.append(fits.ImageHDU(data=np.asarray(data, dtype=dtype)))
    path = tmp_path / name
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def stripe_result(tmp_path, frame, dtype=np.uint16):
    path = write_mef(tmp_path, "bias.fits", [frame, bias_frame(seed=99)], dtype=dtype)
    report = QAEngine(CONFIG, cameras_down=[]).run(path)
    return {r["extension"]: r for r in report["results"] if r["rule"] == "row_stripes"}


# ---------------------------------------------------------------- synthetic frames
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_clean_noisy_bias_passes(tmp_path, dtype):
    res = stripe_result(tmp_path, bias_frame(), dtype)
    for r in res.values():
        assert r["status"] == "EVALUATED" and r["passed"], r
        assert r["measured_value"] < 1.5


@pytest.mark.parametrize("offset", [5.0, -5.0])
def test_one_offset_row_fails(tmp_path, offset):
    frame = bias_frame()
    frame[40] += offset
    res = stripe_result(tmp_path, frame)
    assert not res["2.B.Green"]["passed"] and res["2.B.Green"]["severity"] == "FAIL"
    assert res["2.B.Green"]["measured_value"] == pytest.approx(abs(offset), abs=1.0)
    assert res["1.A.Red"]["passed"]


def test_smooth_vertical_gradient_does_not_trip(tmp_path):
    frame = bias_frame() + np.linspace(0.0, 60.0, SHAPE[0])[:, None]
    res = stripe_result(tmp_path, frame)
    assert res["2.B.Green"]["passed"], res["2.B.Green"]


def test_hot_pixels_in_many_rows_do_not_trip(tmp_path):
    rng = np.random.default_rng(3)
    frame = bias_frame()
    for row in range(SHAPE[0]):           # up to 20 % of every row railed
        cols = rng.choice(SHAPE[1], size=int(rng.integers(0, SHAPE[1] // 5)), replace=False)
        frame[row, cols] = 65535
    res = stripe_result(tmp_path, frame)
    assert res["2.B.Green"]["passed"], res["2.B.Green"]


def test_ramp_in_edge_rows_is_ignored(tmp_path):
    frame = bias_frame()
    frame[:6] -= np.array([40.0, 25.0, 15.0, 8.0, 4.0, 2.0])[:, None]
    frame[-3:] += 30.0
    res = stripe_result(tmp_path, frame)
    assert res["2.B.Green"]["passed"], res["2.B.Green"]


def test_stripe_deviation_direct():
    profile = np.full(100, 650.0)
    profile[50] = 657.0
    profile[2] = 600.0                    # inside the edge margin
    assert stripe_deviation(profile, 15, 8) == pytest.approx(7.0)
    profile[60] = np.nan                  # non-finite rows are dropped
    assert stripe_deviation(profile, 15, 8) == pytest.approx(7.0)
    with pytest.raises(E.QAEngineError):
        stripe_deviation(np.full(10, 650.0), 15, 5)


# ---------------------------------------------------------------- shared sort
def sorted_slice_mean(values, axis, trim):
    a = np.sort(np.asarray(values, dtype=float), axis=axis)
    n = a.shape[axis]
    k = max(1, int(n * trim))
    return np.take(a, np.arange(k, n - k), axis=axis).mean(axis=axis)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(32, 2048), (2048, 33), (10, 10), (7, 3)])
def test_line_profiles_inner_trim_equals_sorted_slice(dtype, shape):
    rng = np.random.default_rng(17)
    values = rng.integers(0, 65536, size=shape).astype(dtype)
    for axis in (-1, -2):
        trimmed, median = line_profiles(values, axis)
        for want_median in (True, False):
            t, m, inner = line_profiles(values, axis, median=want_median, inner_trim=0.25)
            np.testing.assert_array_equal(t, trimmed)
            if want_median:
                np.testing.assert_array_equal(m, median)
            else:
                assert m is None
            np.testing.assert_allclose(inner, sorted_slice_mean(values, axis, 0.25),
                                       rtol=1e-12, atol=1e-9)


def test_row_sort_shared_with_row_structure(tmp_path, monkeypatch):
    """row_structure and row_stripes share one row sort per detector; the
    interquartile profile is cached only where a row_stripes rule needs it."""
    calls = []
    original = E.line_profiles

    def counting(values, axis, *args, **kwargs):
        calls.append(axis)
        return original(values, axis, *args, **kwargs)

    monkeypatch.setattr(E, "line_profiles", counting)
    path = write_mef(tmp_path, "a.fits", [bias_frame(1), bias_frame(2)])
    for order in (CONFIG, {**CONFIG, "rule_sets": {"BIAS": {
            **CONFIG["rule_sets"]["BIAS"],
            "rules": CONFIG["rule_sets"]["BIAS"]["rules"][::-1]}}}):
        calls.clear()
        engine = QAEngine(order, cameras_down=[])
        engine.run(path, workers=1)
        assert calls == [-1, -1]                 # one row sort per detector, 2 metrics
        for entry in engine.last_run.extensions.values():
            assert entry.profiles[("full_frame", -1)][2] is not None
    calls.clear()
    no_stripes = copy.deepcopy(CONFIG)
    no_stripes["rule_sets"]["BIAS"]["rules"] = no_stripes["rule_sets"]["BIAS"]["rules"][:1]
    engine = QAEngine(no_stripes, cameras_down=[])
    engine.run(path, workers=1)
    assert calls == [-1, -1]
    for entry in engine.last_run.extensions.values():
        assert entry.profiles[("full_frame", -1)][2] is None    # no row_stripes rule


# ---------------------------------------------------------------- validator
@pytest.mark.parametrize("field, value, message", [
    ("trim", 0.5, "0 < trim < 0.5"),
    ("trim", 0, "0 < trim < 0.5"),
    ("trim", "a", "0 < trim < 0.5"),
    ("smooth_rows", 14, "odd integer"),
    ("smooth_rows", 1, "odd integer"),
    ("smooth_rows", 15.0, "odd integer"),
    ("edge_rows", -1, "non-negative integer"),
    ("smooth_columns", 15, "unknown field"),
])
def test_validator_checks_stripe_options(field, value, message):
    cfg = copy.deepcopy(CONFIG)
    cfg["metrics"]["row_stripes"][field] = value
    errors = [f"{e.path}: {e.message}" for e in QAConfigValidator(cfg, filename="stripes").validate()]
    assert any("row_stripes" in e and message in e for e in errors), errors


def test_validator_accepts_defaults():
    cfg = copy.deepcopy(CONFIG)
    cfg["metrics"]["row_stripes"] = {"type": "row_stripes"}
    assert QAConfigValidator(cfg, filename="stripes").validate() == []


# ---------------------------------------------------------------- real frames
OCT_DIR = Path("/Users/slh/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/20261014_15")


def _real(name):
    path = OCT_DIR / f"LLAMAS_2026-10-14_{name}_CAL24_mef.fits"
    if not path.exists():
        pytest.skip(f"real frame not available: {path}")
    return path


def test_real_striped_bias_fails_on_row_stripes(tmp_path):
    path = _real("18-09-47.2")
    assert check_image(path, report_dir=tmp_path)["status"] == "fail"
    report = QAEngine(E.load_yaml(E.CONFIG_DIR / "qa_config_cal.yaml"), cameras_down=[]).run(path)
    stripe = {r["extension"]: r for r in report["results"] if r["rule"] == "row_stripes"}
    assert not stripe["2.B.Green"]["passed"] and stripe["2.B.Green"]["severity"] == "FAIL"
    assert stripe["2.B.Green"]["measured_value"] > 8
    assert all(r["passed"] for name, r in stripe.items() if name != "2.B.Green")
    failed = {(r["rule"], r["extension"]) for r in report["results"]
              if not r["passed"] and r["severity"] == "FAIL" and r["status"] == "EVALUATED"}
    assert failed == {("row_stripes", "2.B.Green")}


def test_real_clean_bias_passes(tmp_path):
    result = check_image(_real("18-23-36.0"), report_dir=tmp_path)
    assert result["status"] == "pass", result.get("message")
