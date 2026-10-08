"""Signal-normalised structure metrics (``row_structure_norm`` / ``column_structure_norm``).

On a lamp frame the row/column structure is the line pattern itself and scales
with the lamp signal, so the absolute metric encodes the exposure time. The
normalised metric divides by (region mean - background-stripe median) and must be
independent of how bright the lamp pattern is. Frames are small synthetic MEFs.
"""
import copy
import json

import numpy as np
import pytest
import yaml
from astropy.io import fits

from llamas_checks.llamasQATests import check_image
from llamas_checks.qa_config_validator import QAConfigValidator
from llamas_checks.qa_engine import QAEngine, trimmed_mean_profile

SHAPE = (40, 40)
BOTTOM = {"type": "rectangle", "x_start": 2, "x_end": 38, "y_start": 1, "y_end": 5}

CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [{"name": "1.A.Green", "hdu_index": 1, "color": "Green", "bench": 1, "side": "A"},
                   {"name": "1.A.Blue", "hdu_index": 2, "color": "Blue", "bench": 1, "side": "A"}],
    "regions": {"full_frame": {"type": "full"}, "bottom_stripe": BOTTOM},
    "metrics": {
        "row_banding": {"type": "row_structure"},
        "column_banding": {"type": "column_structure"},
        "row_banding_norm": {"type": "row_structure_norm",
                             "background_region": "bottom_stripe", "min_signal": 2.0},
        "column_banding_norm": {"type": "column_structure_norm",
                                "background_region": "bottom_stripe", "min_signal": 2.0},
    },
    "lookup_tables": {"noop": {"a": 1}},
    "rule_sets": {
        "ARC": {
            "applies_when": {"exposure_type": "CAL.R-ARC"},
            "rules": [
                {"name": "row_structure", "region": "full_frame", "metric": "row_banding_norm",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 1.0}},
                {"name": "column_structure", "region": "full_frame", "metric": "column_banding_norm",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 5.0}},
                {"name": "row_abs", "region": "full_frame", "metric": "row_banding",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 1e9}},
                {"name": "column_abs", "region": "full_frame", "metric": "column_banding",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 1e9}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def arc_frame(amplitude, pedestal=700.0, row_offsets=0.0, seed=0):
    """Pedestal + vertical 'emission lines' every 5 columns of height ``amplitude``,
    + 1 ADU checkerboard noise; the bottom stripe (rows 1-5) stays unilluminated.
    ``row_offsets`` adds an alternating whole-row offset (banding)."""
    rng = np.random.default_rng(seed)
    frame = np.full(SHAPE, pedestal, dtype=np.float64)
    lines = np.zeros(SHAPE)
    lines[6:, 3::5] = amplitude
    frame += lines
    frame += np.where(np.indices(SHAPE).sum(axis=0) % 2, 0.5, -0.5)
    frame += rng.normal(0.0, 0.3, SHAPE)
    if row_offsets:
        frame[::2, :] += row_offsets
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


def expected_norm(data, axis):
    structure = float(np.std(trimmed_mean_profile(np.asarray(data, float), axis)))
    signal = float(np.mean(data)) - float(np.median(data[1:5, 2:38]))
    return structure / signal, structure


def by_rule(report, ext):
    return {r["rule"]: r for r in report["results"] if r["extension"] == ext}


def test_config_is_valid():
    assert QAConfigValidator(CONFIG, filename="norm").validate() == []


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_norm_metric_matches_independent_computation(tmp_path, dtype):
    frame = arc_frame(400.0)
    path = write_mef(tmp_path, "arc.fits", [frame, frame], dtype=dtype)
    stored = fits.getdata(path, 1)
    report = QAEngine(CONFIG, cameras_down=[]).run(path)
    rules = by_rule(report, "1.A.Green")
    for rule, axis in (("row_structure", -1), ("column_structure", -2)):
        want_norm, want_abs = expected_norm(stored, axis)
        assert rules[rule]["measured_value"] == pytest.approx(want_norm, rel=1e-5)
        abs_rule = "row_abs" if rule == "row_structure" else "column_abs"
        assert rules[abs_rule]["measured_value"] == pytest.approx(want_abs, rel=1e-5)
    assert report["overall_verdict"] == "PASS"


def test_norm_metric_is_independent_of_lamp_brightness(tmp_path):
    faint = write_mef(tmp_path, "faint.fits", [arc_frame(100.0)])
    bright = write_mef(tmp_path, "bright.fits", [arc_frame(400.0)])
    cfg = copy.deepcopy(CONFIG)
    cfg["extensions"] = cfg["extensions"][:1]
    r_faint = by_rule(QAEngine(cfg, cameras_down=[]).run(faint), "1.A.Green")
    r_bright = by_rule(QAEngine(cfg, cameras_down=[]).run(bright), "1.A.Green")
    # the absolute column structure scales with the lamp (x4), the normalised one does not
    ratio_abs = r_bright["column_abs"]["measured_value"] / r_faint["column_abs"]["measured_value"]
    ratio_norm = r_bright["column_structure"]["measured_value"] / r_faint["column_structure"]["measured_value"]
    assert ratio_abs == pytest.approx(4.0, rel=0.05)
    assert ratio_norm == pytest.approx(1.0, rel=0.05)


def test_banding_is_caught_at_any_brightness(tmp_path):
    cfg = copy.deepcopy(CONFIG)
    cfg["extensions"] = cfg["extensions"][:1]
    clean = by_rule(QAEngine(cfg, cameras_down=[]).run(
        write_mef(tmp_path, "clean.fits", [arc_frame(400.0)])), "1.A.Green")
    cap = 1.5 * clean["row_structure"]["measured_value"]
    cfg["rule_sets"]["ARC"]["rules"][0]["limits"] = {"max": cap}
    for amp in (100.0, 400.0):
        banded = write_mef(tmp_path, f"banded{int(amp)}.fits",
                           [arc_frame(amp, row_offsets=0.4 * amp)])
        report = QAEngine(cfg, cameras_down=[]).run(banded)
        assert by_rule(report, "1.A.Green")["row_structure"]["passed"] is False
        assert report["overall_verdict"] == "FAIL"
        ok = QAEngine(cfg, cameras_down=[]).run(write_mef(tmp_path, f"ok{int(amp)}.fits", [arc_frame(amp)]))
        assert ok["overall_verdict"] == "PASS"


def test_no_signal_skips_the_normalised_rules(tmp_path):
    path = write_mef(tmp_path, "dark.fits", [arc_frame(0.0), arc_frame(400.0)])
    report = QAEngine(CONFIG, cameras_down=[]).run(path)
    dark = by_rule(report, "1.A.Green")
    for rule in ("row_structure", "column_structure"):
        assert dark[rule]["status"] == "SKIPPED"
        assert "min_signal" in dark[rule]["message"]
        assert dark[rule]["verdict_effect"] == "SKIPPED"
    assert dark["row_abs"]["status"] == "EVALUATED"       # the absolute metric still runs
    assert by_rule(report, "1.A.Blue")["row_structure"]["status"] == "EVALUATED"
    assert report["overall_verdict"] == "PASS"
    assert report["summary"]["skipped_checks"] == 2


def test_signal_computed_once_per_detector(tmp_path, monkeypatch):
    engine = QAEngine(CONFIG, cameras_down=[])
    calls = []
    original = QAEngine._is_placeholder_data
    monkeypatch.setattr(QAEngine, "_is_placeholder_data",
                        staticmethod(lambda d: (calls.append(1), original(d))[1]))
    engine.run(write_mef(tmp_path, "arc.fits", [arc_frame(400.0), arc_frame(200.0)]))
    assert len(calls) == 2
    assert len(engine._signal_cache) == 2          # one signal per detector, shared by row/col
    assert len(engine._metric_cache) == 8          # 4 metrics x 2 detectors, nothing extra


@pytest.mark.parametrize("field, value, message", [
    ("background_region", "nowhere", "must name a region"),
    ("min_signal", -1, "non-negative"),
    ("min_signal", "two", "non-negative"),
    ("percentile", 50, "unknown field"),
])
def test_validator_checks_norm_metric_options(field, value, message):
    cfg = copy.deepcopy(CONFIG)
    cfg["metrics"]["row_banding_norm"][field] = value
    errors = [f"{e.path}: {e.message}" for e in QAConfigValidator(cfg, filename="norm").validate()]
    assert any("row_banding_norm" in e and message in e for e in errors), errors


def test_norm_metric_defaults_to_bottom_stripe(tmp_path):
    cfg = copy.deepcopy(CONFIG)
    cfg["metrics"]["row_banding_norm"] = {"type": "row_structure_norm"}
    cfg["metrics"]["column_banding_norm"] = {"type": "column_structure_norm"}
    assert QAConfigValidator(cfg, filename="norm").validate() == []
    path = write_mef(tmp_path, "arc.fits", [arc_frame(400.0), arc_frame(400.0)])
    full = by_rule(QAEngine(CONFIG, cameras_down=[]).run(path), "1.A.Green")
    dflt = by_rule(QAEngine(cfg, cameras_down=[]).run(path), "1.A.Green")
    assert dflt["row_structure"]["measured_value"] == full["row_structure"]["measured_value"]


def test_check_image_end_to_end(tmp_path):
    path = write_mef(tmp_path, "arc.fits", [arc_frame(400.0), arc_frame(100.0)])
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(CONFIG))
    out = tmp_path / "rep.json"
    result = check_image(str(path), qa_yaml=str(cfg_path), report=str(out), report_all=True,
                         cameras_down="none")
    assert result["status"] == "pass"
    payload = json.loads(out.read_text())
    norm = [r for r in payload["report"]["results"] if r["metric"] == "row_banding_norm"]
    assert len(norm) == 2 and all(r["status"] == "EVALUATED" for r in norm)
