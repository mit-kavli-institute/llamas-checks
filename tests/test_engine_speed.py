"""Speed paths added for the observing-GUI timeout (2026-10).

A raw frame is 24 x 2048x2048 uint16; the engine used to sort/partition a float64
copy of every detector once per *rule*, so the WARN/FAIL tiers of one check
doubled the work. These tests pin the three guarantees of the fix:

  * integer frames take cheaper paths (radix sort, histogram median) that give
    exactly the values the float64 path gives;
  * one measurement per (HDU, region, metric) per run, shared by tiered rules,
    including a shared ERROR when the measurement cannot be made;
  * ``check_image`` reports how long it took (``elapsed_s``; a stage line on
    stderr with ``LLAMAS_CHECKS_TIMING=1``).
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest
import yaml
from astropy.io import fits

from llamas_checks.llamasQATests import check_image
from llamas_checks.qa_config_validator import QAConfigValidator
from llamas_checks.qa_engine import (QAEngine, integer_mad, integer_median,
                                     trimmed_mean_profile)

RNG = np.random.default_rng(20261008)


def noisy_uint16(shape=(120, 260), level=700, sigma=4):
    """Read-noise-like integer frame plus a few hot pixels and one railed column fragment."""
    frame = RNG.normal(level, sigma, size=shape).round().clip(0, 65535).astype(np.uint16)
    frame[5, 7] = 65535
    frame[40:44, 100] = 65535
    return frame


# ---------------------------------------------------------------- integer fast paths
@pytest.mark.parametrize("axis", [-1, -2])
def test_integer_trimmed_profile_matches_float_path(axis):
    frame = noisy_uint16()
    fast = trimmed_mean_profile(frame, axis)
    slow = trimmed_mean_profile(frame.astype(np.float64), axis)
    assert fast.shape == slow.shape
    np.testing.assert_allclose(fast, slow, rtol=0, atol=1e-9)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 7, 10, 1001, 4096])
@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.uint8])
def test_integer_median_and_mad_match_numpy(n, dtype):
    info = np.iinfo(dtype)
    values = RNG.integers(max(info.min, -300), min(info.max, 300) + 1, size=n).astype(dtype)
    as_float = values.astype(np.float64)
    assert integer_median(values) == np.median(as_float)
    assert integer_mad(values) == np.median(np.abs(as_float - np.median(as_float)))


def test_integer_median_with_many_duplicates():
    values = np.array([3, 3, 3, 4, 4, 4], dtype=np.uint16)   # even count, split across two values
    assert integer_median(values) == 3.5
    assert integer_mad(values) == 0.5


@pytest.mark.parametrize("metric", [
    {"type": "median"}, {"type": "robust_std"}, {"type": "mean"}, {"type": "std"},
    {"type": "min"}, {"type": "max"}, {"type": "sum"},
    {"type": "percentile", "percentile": 99.5},
    {"type": "fraction_above", "threshold": 63000}, {"type": "count_above", "threshold": 63000},
    {"type": "row_structure"}, {"type": "column_structure"},
])
def test_compute_metric_integer_input_matches_float_input(metric):
    frame = noisy_uint16()
    fast = QAEngine._compute_metric(frame, metric)
    slow = QAEngine._compute_metric(frame.astype(np.float64), metric)
    assert fast == pytest.approx(slow, rel=1e-12, abs=1e-9)


def test_background_gradient_rate_integer_input():
    frame = noisy_uint16()
    frame[:, :] += np.linspace(0, 40, frame.shape[0]).astype(np.uint16)[:, None]
    metric = {"type": "background_gradient_rate"}
    fast = QAEngine._compute_metric(frame, metric, exptime=10.0)
    slow = QAEngine._compute_metric(frame.astype(np.float64), metric, exptime=10.0)
    assert fast == pytest.approx(slow, rel=1e-12)


@pytest.mark.parametrize("value,expected", [(0, True), (1, True), (65535, False), (700, False)])
def test_placeholder_detection_on_integer_frames(value, expected):
    frame = np.full((30, 30), value, dtype=np.uint16)
    assert QAEngine._is_placeholder_data(frame) is expected
    assert QAEngine._is_placeholder_data(frame.astype(np.float32)) is expected


# ---------------------------------------------------------------- shared measurement per run
CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [{"name": "1.A.Green", "hdu_index": 1, "color": "Green", "bench": 1, "side": "A"},
                   {"name": "1.A.Blue", "hdu_index": 2, "color": "Blue", "bench": 1, "side": "A"}],
    "regions": {
        "full_frame": {"type": "full"},
        "bottom_stripe": {"type": "rectangle", "x_start": 10, "x_end": 250, "y_start": 2, "y_end": 28},
        "off_frame": {"type": "rectangle", "x_start": 0, "x_end": 5000, "y_start": 0, "y_end": 5},
    },
    "metrics": {"median": {"type": "median"}, "row_banding": {"type": "row_structure"}},
    "lookup_tables": {"noop": {"a": 1}},
    "rule_sets": {
        "BIAS": {
            "applies_when": {"exposure_type": "CAL.R-BIA"},
            "rules": [
                {"name": "edge_background_level", "region": "bottom_stripe", "metric": "median",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 650}},
                {"name": "edge_saturated", "region": "bottom_stripe", "metric": "median",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 63000}},
                {"name": "row_structure", "region": "full_frame", "metric": "row_banding",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 0.5}},
                {"name": "row_structure_gross", "region": "full_frame", "metric": "row_banding",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 2.0}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def write_mef(tmp_path, name="frame_mef.fits", n_ext=2):
    ph = fits.PrimaryHDU()
    ph.header["PRODCATG"] = "CAL.R-BIA"
    ph.header["READ-MDE"] = "SLOW"
    hdus = [ph]
    for i in range(n_ext):
        ext = fits.ImageHDU(data=noisy_uint16())
        ext.header["COLOR"] = ["Green", "Blue"][i]
        ext.header["BENCH"] = 1
        ext.header["SIDE"] = "A"
        hdus.append(ext)
    path = tmp_path / name
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_config_is_valid():
    assert QAConfigValidator(CONFIG, filename="speed").validate() == []


def test_tiered_rules_share_one_measurement(tmp_path, monkeypatch):
    calls = []
    original = QAEngine._compute_metric

    def counting(region_data, metric, exptime=None, **kwargs):
        calls.append(metric["type"])
        return original(region_data, metric, exptime=exptime, **kwargs)

    monkeypatch.setattr(QAEngine, "_compute_metric", staticmethod(counting))
    report = QAEngine(CONFIG).run(write_mef(tmp_path))

    # 2 extensions x 2 distinct (region, metric) pairs, although 4 rules were evaluated.
    assert sorted(calls) == ["median", "median", "row_structure", "row_structure"]
    # 4 rules x 2 extensions, plus one camera_present result per extension.
    assert report["summary"]["evaluated_checks"] == 10
    for ext in ("1.A.Green", "1.A.Blue"):
        by_rule = {r["rule"]: r for r in report["results"] if r["extension"] == ext}
        assert by_rule["edge_background_level"]["measured_value"] == by_rule["edge_saturated"]["measured_value"]
        assert by_rule["row_structure"]["measured_value"] == by_rule["row_structure_gross"]["measured_value"]
        # the shared value still gets each rule's own limits
        assert by_rule["edge_background_level"]["passed"] is False      # ~700 > 650
        assert by_rule["edge_saturated"]["passed"] is True              # ~700 <= 63000


def test_measurement_error_is_shared_by_tiered_rules(tmp_path):
    config = json.loads(json.dumps(CONFIG))
    for rule in config["rule_sets"]["BIAS"]["rules"][:2]:
        rule["region"] = "off_frame"
    report = QAEngine(config).run(write_mef(tmp_path))
    errors = [r for r in report["results"] if r["status"] == "ERROR"]
    assert {r["rule"] for r in errors} == {"edge_background_level", "edge_saturated"}
    assert len(errors) == 4                               # both rules on both extensions
    assert len({r["message"] for r in errors if r["extension"] == "1.A.Green"}) == 1
    assert report["overall_verdict"] == "ERROR"


def test_caches_are_reset_between_runs(tmp_path):
    engine = QAEngine(CONFIG)
    first = engine.run(write_mef(tmp_path, "a_mef.fits"))
    second = engine.run(write_mef(tmp_path, "b_mef.fits"))
    assert first["fits_file"] != second["fits_file"]
    assert len(engine._metric_cache) == 4                  # only the second run's entries
    assert len(engine._placeholder_cache) == 2


def test_camera_presence_check_adds_no_pixel_work(tmp_path, monkeypatch):
    """The camera_present check reuses the cached placeholder test: exactly one
    placeholder evaluation per HDU per run, and no extra metric measurements."""
    placeholder_calls = []
    original = QAEngine._is_placeholder_data

    def counting(data):
        placeholder_calls.append(1)
        return original(data)

    monkeypatch.setattr(QAEngine, "_is_placeholder_data", staticmethod(counting))
    engine = QAEngine(CONFIG)
    report = engine.run(write_mef(tmp_path))
    assert len(placeholder_calls) == len(CONFIG["extensions"])
    assert len(engine._metric_cache) == 4
    assert sum(1 for r in report["results"] if r["rule"] == "camera_present") == 2


# ---------------------------------------------------------------- elapsed-time reporting
def test_check_image_reports_elapsed_seconds(tmp_path):
    frame = write_mef(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    result = check_image(str(frame), qa_yaml=str(config_path), report_dir=str(tmp_path / "rep"))
    assert result["status"] == "warn"
    assert isinstance(result["elapsed_s"], float) and 0 <= result["elapsed_s"] < 60
    written = json.loads((tmp_path / "rep" / "frame_mef.qa.json").read_text())
    assert 0 <= written["elapsed_s"] <= result["elapsed_s"]


def test_cli_verbose_shows_elapsed_and_timing_env_adds_stage_line(tmp_path):
    frame = write_mef(tmp_path)
    config_path = tmp_path / "cfg.yaml"
    config_path.write_text(yaml.safe_dump(CONFIG))
    env = dict(os.environ, LLAMAS_CHECKS_TIMING="1")
    proc = subprocess.run([sys.executable, "-m", "llamas_checks", str(frame),
                           "--qa-yaml", str(config_path), "-v"],
                          capture_output=True, text=True, env=env)
    assert proc.returncode == 1, (proc.stdout, proc.stderr)
    verdict_lines = [l for l in proc.stderr.splitlines() if l.startswith("WARN:")]
    assert len(verdict_lines) == 1 and verdict_lines[0].endswith(" s)"), proc.stderr
    timing_lines = [l for l in proc.stderr.splitlines() if l.startswith("timing:")]
    assert len(timing_lines) == 1, proc.stderr
    for stage in ("total=", "config=", "structure=", "engine=", "report="):
        assert stage in timing_lines[0]

    quiet = subprocess.run([sys.executable, "-m", "llamas_checks", str(frame),
                           "--qa-yaml", str(config_path)], capture_output=True, text=True)
    assert quiet.returncode == 1 and quiet.stdout == "" and quiet.stderr == ""
