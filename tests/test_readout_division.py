"""Science readout division (``row_gradient_division`` / rule ``readout_division``).

A camera whose digitiser has a stuck bit for part of the readout (2.B.Green,
2026-11-01: bit 5 = 32 ADU stuck high for the first 897 rows) shows a solid level
step across one row boundary, the same in every column. The metric is the
significance z of the strongest such step against the detector's own row-to-row
gradient noise, keeping only boundaries whose jump has one sign in every column
block, is coherent along the dispersion (smallest block jump >= half the largest)
and persists for 32 rows. Healthy frames of every type read z <= 4.4; the faulty
frames 44-97. Frames are synthetic; four real-frame regressions skip without Box.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from llamas_checks import qa_engine as E
from llamas_checks.qa_config_validator import QAConfigValidator
from llamas_checks.qa_engine import QAEngine, row_gradient_division

SHAPE = (1200, 1024)   # rows x columns
LEVEL, SIGMA = 667.0, 2.5


def healthy_frame(seed=0, fibres=True, sky_lines=False, stars=False, flat_block=False):
    """A science-like float frame: bias + read noise, optionally a 7-row (tilted)
    fibre modulation scaled by a column-varying spectrum, sky-line columns, bright 5-row
    star traces, and a 5000 ADU illuminated block whose edge amplitude varies
    +-50 % along the columns."""
    rng = np.random.default_rng(seed)
    frame = rng.normal(LEVEL, SIGMA, SHAPE)
    rows, cols = np.arange(SHAPE[0]), np.arange(SHAPE[1])
    spectrum = 5.0 + 40.0 * np.exp(-((cols - 400.0) / 250.0) ** 2) + 10.0 * rng.random(SHAPE[1])
    # fibre traces 7 rows apart, tilted by one pitch across the detector as real
    # traces are (so the column blocks do not all see the same fibre phase)
    fibre = 0.5 * (1.0 + np.cos(2.0 * np.pi * (rows[:, None] + 7.0 * cols[None, :] / SHAPE[1]) / 7.0))
    if fibres:
        frame += fibre * spectrum[None, :]
    if sky_lines:
        for c in (130, 131, 132, 515, 516, 870):
            frame[:, c] += 400.0 * fibre[:, c]
    if stars:
        for r0, amp in ((300, 3000.0), (700, 8000.0), (1010, 1500.0)):
            frame[r0:r0 + 5] += amp * (spectrum / spectrum.max())[None, :]
    if flat_block:
        amplitude = 5000.0 * (1.0 + 0.5 * np.sin(2.0 * np.pi * cols / SHAPE[1]))
        frame[400:800] += amplitude[None, :]
    return frame


def as_uint16(frame):
    return np.clip(np.round(frame), 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------- synthetic frames
@pytest.mark.parametrize("kind", [
    {}, {"fibres": False}, {"sky_lines": True}, {"stars": True}, {"flat_block": True},
    {"sky_lines": True, "stars": True, "flat_block": True},
])
def test_healthy_frames_stay_below_threshold(kind):
    z, row, jump, label = row_gradient_division(as_uint16(healthy_frame(**kind)))
    assert z < 20.0, (kind, z, row, jump, label)


def test_whole_bottom_offset_is_a_division():
    # a sharp step gives the full jump on the 2*gap+1 boundaries around it, so the
    # reported row is within +-gap of the step
    frame = healthy_frame()
    frame[:900] += 32.0
    z, row, jump, label = row_gradient_division(as_uint16(frame))
    assert z > 50.0
    assert abs(row - 900) <= 2
    assert jump == pytest.approx(-32.0, abs=2.0)
    assert label in ("row floor", "row median")


def test_jump_starting_mid_row_is_found():
    """The 2026-11-01 jump happens inside row 897: columns read before it are high."""
    frame = healthy_frame(seed=1)
    frame[:897] += 32.0
    frame[897, :1000] += 32.0
    z, row, jump, _ = row_gradient_division(as_uint16(frame))
    assert z > 20.0
    assert abs(row - 897) <= 2
    assert jump == pytest.approx(-32.0, abs=3.0)


def test_band_is_a_division():
    frame = healthy_frame(seed=2, stars=True)
    frame[500:640] += 32.0
    z, row, jump, _ = row_gradient_division(as_uint16(frame))
    assert z > 20.0
    assert abs(row - 500) <= 2 or abs(row - 640) <= 2


def test_saturated_block_is_ignored():
    frame = healthy_frame(seed=3)
    frame[600:] = 65535.0
    z, *_ = row_gradient_division(as_uint16(frame))
    assert z < 20.0


def test_float_input_with_nan():
    frame = healthy_frame(seed=4)
    frame[:900] += 32.0
    rng = np.random.default_rng(5)
    frame[rng.random(SHAPE) < 0.01] = np.nan
    frame[100:103] = np.nan                      # whole rows missing
    z, row, jump, _ = row_gradient_division(frame)
    assert z > 20.0 and abs(row - 900) <= 2
    assert jump == pytest.approx(-32.0, abs=2.0)
    healthy = healthy_frame(seed=4)
    healthy[rng.random(SHAPE) < 0.01] = np.nan
    assert row_gradient_division(healthy)[0] < 20.0


def test_too_few_rows_raises():
    minimum = 2 * (32 + 2 + 4) + 1
    row_gradient_division(as_uint16(healthy_frame()[:minimum]))
    with pytest.raises(E.QAEngineError):
        row_gradient_division(as_uint16(healthy_frame()[:minimum - 1]))


# ---------------------------------------------------------------- engine
CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [{"name": "2.B.Green", "hdu_index": 1, "color": "Green", "bench": 2, "side": "B"},
                   {"name": "1.A.Red", "hdu_index": 2, "color": "Red", "bench": 1, "side": "A"}],
    "regions": {"full_frame": {"type": "full"},
                "fibre_core": {"type": "rectangle", "x_start": 12, "x_end": 1012,
                               "y_start": 50, "y_end": 1150}},
    "metrics": {"row_division": {"type": "row_gradient_division"}},
    "lookup_tables": {"noop": {"a": 1}},
    "rule_sets": {
        "SCIENCE": {
            "applies_when": {"exposure_type": "SCI.R-*"},
            "rules": [
                {"name": "readout_division", "region": "fibre_core", "metric": "row_division",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 20.0}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def write_sci_mef(tmp_path, frames):
    ph = fits.PrimaryHDU()
    ph.header["PRODCATG"] = "SCI.R-SL"
    ph.header["READ-MDE"] = "SLOW"
    hdus = [ph] + [fits.ImageHDU(data=as_uint16(f)) for f in frames]
    path = tmp_path / "sci.fits"
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def division_results(report):
    return {r["extension"]: r for r in report["results"] if r["rule"] == "readout_division"}


def test_engine_fails_division_and_names_row(tmp_path):
    bad = healthy_frame(seed=6)
    bad[:897] += 32.0
    path = write_sci_mef(tmp_path, [bad, healthy_frame(seed=7)])
    report = QAEngine(CONFIG, cameras_down=[]).run(path)
    res = division_results(report)
    assert report["overall_verdict"] == "FAIL"
    assert not res["2.B.Green"]["passed"] and res["2.B.Green"]["verdict_effect"] == "FAIL"
    message = res["2.B.Green"]["message"]
    assert "solid division at row 89" in message and "ADU" in message, message
    row = int(message.split("solid division at row ")[1].split(":")[0])
    assert abs(row - 897) <= 2                         # detector row, region offset added
    assert res["1.A.Red"]["passed"]
    assert "solid division" not in res["1.A.Red"]["message"]


def test_engine_passes_healthy(tmp_path):
    path = write_sci_mef(tmp_path, [healthy_frame(seed=8, stars=True),
                                    healthy_frame(seed=9, sky_lines=True)])
    report = QAEngine(CONFIG, cameras_down=[]).run(path)
    assert report["overall_verdict"] == "PASS"
    assert all(r["passed"] and r["status"] == "EVALUATED"
               for r in division_results(report).values())


# ---------------------------------------------------------------- validator
def test_validator_accepts_defaults_and_full_params():
    cfg = copy.deepcopy(CONFIG)
    assert QAConfigValidator(cfg, filename="division").validate() == []
    cfg["metrics"]["row_division"].update({
        "blocks": 8, "column_stride": 4, "quantiles": [0.1, 0.5], "rows": 4, "gap": 0,
        "persist_rows": 32, "min_uniformity": 1.0, "persist_min": 0.6, "persist_max": 1.6,
        "min_mad": 0.1, "min_z": 5.0, "saturation": 63000})
    assert QAConfigValidator(cfg, filename="division").validate() == []


@pytest.mark.parametrize("field, value, message", [
    ("blocks", 0, "positive integer"),
    ("column_stride", 2.0, "positive integer"),
    ("rows", True, "positive integer"),
    ("persist_rows", -3, "positive integer"),
    ("gap", -1, "non-negative integer"),
    ("quantiles", [], "0 < q < 1"),
    ("quantiles", [0.1, 1.0], "0 < q < 1"),
    ("quantiles", 0.5, "0 < q < 1"),
    ("min_uniformity", 0, "0 < value <= 1"),
    ("min_uniformity", 1.5, "0 < value <= 1"),
    ("persist_min", 2.0, "greater than persist_min"),
    ("persist_max", 0.5, "greater than persist_min"),
    ("persist_min", -1, "positive number"),
    ("min_mad", -0.1, "non-negative number"),
    ("min_z", "x", "non-negative number"),
    ("saturation", -1, "non-negative number"),
    ("trim", 0.25, "unknown field"),
])
def test_validator_rejects_bad_params(field, value, message):
    cfg = copy.deepcopy(CONFIG)
    cfg["metrics"]["row_division"][field] = value
    errors = [f"{e.path}: {e.message}" for e in QAConfigValidator(cfg, filename="division").validate()]
    assert any("row_division" in e and message in e for e in errors), errors


def test_shipped_science_config_has_readout_division():
    cfg = E.load_yaml(E.CONFIG_DIR / "qa_config_science.yaml")
    rules = {r["name"]: r for r in cfg["rule_sets"]["SCIENCE"]["rules"]}
    rule = rules["readout_division"]
    assert rule["severity"] == "FAIL" and rule["limits"] == {"max": 20.0}
    assert cfg["metrics"][rule["metric"]]["type"] == "row_gradient_division"
    assert cfg["regions"][rule["region"]] == {"type": "rectangle", "x_start": 100, "x_end": 1948,
                                              "y_start": 100, "y_end": 1950}
    cal = E.load_yaml(E.CONFIG_DIR / "qa_config_cal.yaml")
    assert not any(r["name"] == "readout_division"
                   for rs in cal["rule_sets"].values() for r in rs["rules"])


# ---------------------------------------------------------------- real frames
BOX = Path("/Users/slh/Library/CloudStorage/Box-Box/slhughes")
NIGHT = BOX / "Llamas_Commissioning_Data"


def _real(path):
    if not path.exists():
        pytest.skip(f"real frame not available: {path}")
    return path


def _science_division(path):
    report = QAEngine(E.load_yaml(E.CONFIG_DIR / "qa_config_science.yaml"),
                      cameras_down=[]).run(path)
    return report, division_results(report)


def test_real_nov01_fails_at_row_896():
    path = _real(BOX / "LLAMAS_analysis" / "LLAMAS_2026-11-01_05-54-08.1_SCI24_mef.fits")
    report, res = _science_division(path)
    assert report["overall_verdict"] == "FAIL"
    green = res["2.B.Green"]
    assert not green["passed"] and green["measured_value"] > 20
    row = int(green["message"].split("solid division at row ")[1].split(":")[0])
    assert abs(row - 896) <= 2
    assert all(r["passed"] for name, r in res.items() if name != "2.B.Green")


@pytest.mark.parametrize("night, stamp", [
    ("20261109_10", "2026-11-10_02-57-14.7"),     # Feige110, bit 5 stuck on 99.7 %
    ("20261015_16", "2026-10-16_03-50-21.7"),     # EG21, bit-flip band
])
def test_real_faulty_frames_fail(night, stamp):
    path = _real(NIGHT / night / f"LLAMAS_{stamp}_SCI24_mef.fits")
    report, res = _science_division(path)
    assert report["overall_verdict"] == "FAIL"
    assert not res["2.B.Green"]["passed"] and res["2.B.Green"]["measured_value"] > 20


def test_real_long_exposure_passes_division():
    """J023, 1800 s: bright, long, healthy -- the rule passes on every detector."""
    path = _real(NIGHT / "20261109_10" / "LLAMAS_2026-11-10_05-08-01.9_SCI24_mef.fits")
    _, res = _science_division(path)
    assert len(res) == 24
    assert all(r["status"] == "EVALUATED" and r["passed"] for r in res.values()), \
        {k: r["measured_value"] for k, r in res.items() if not r["passed"]}
