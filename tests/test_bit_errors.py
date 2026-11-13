"""Pixel bit errors (``code_discontinuity`` / rule ``pixel_bit_errors``, WARN only).

Read noise (sigma ~2.5 ADU) makes the value histogram of any real image smooth from
one value to the next; a jump in log-counts between neighbouring values beyond the
local curvature means the digitiser is corrupting the values (a stuck or flipping
bit). Healthy cameras read <= 0.27, 2.B.Green 0.4-5.0 at a multiple of 16; the rule
WARNs above 0.6 and never FAILs (frames with bit errors are often usable; unusable
ones FAIL readout_division). Frames are synthetic; two real-frame checks skip
without Box.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

from llamas_checks import qa_engine as E
from llamas_checks.llamasQATests import check_image
from llamas_checks.qa_config_validator import QAConfigValidator
from llamas_checks.qa_engine import IntegerHistogram, QAEngine, code_discontinuity

SHAPE = (1000, 1000)
LEVEL, SIGMA = 667.0, 2.5


def gaussian(seed=0, pattern=False):
    rng = np.random.default_rng(seed)
    frame = rng.normal(LEVEL, SIGMA, SHAPE)
    if pattern:   # fibre / spectrum pattern: 7-row traces scaled by a varying spectrum
        rows, cols = np.arange(SHAPE[0]), np.arange(SHAPE[1])
        spectrum = 5.0 + 300.0 * np.exp(-((cols - 400.0) / 200.0) ** 2) + 50.0 * rng.random(SHAPE[1])
        frame += (0.5 * (1 + np.cos(2 * np.pi * rows / 7.0)))[:, None] * spectrum[None, :]
    return np.clip(np.round(frame), 0, 65535).astype(np.uint16)


def sparse_lit(seed=0):
    """Bias + read noise with 25 % of pixels lit by a sparse exponential (mean 160 ADU)
    signal: the faint wing that randomly flipped pixels land on."""
    rng = np.random.default_rng(seed)
    frame = rng.normal(LEVEL, SIGMA, SHAPE)
    lit = rng.random(SHAPE) < 0.25
    frame[lit] += rng.exponential(160.0, lit.sum())
    return np.clip(np.round(frame), 0, 65535).astype(np.uint16)


def anomaly(values):
    histogram = IntegerHistogram(values)
    return code_discontinuity(histogram.counts, histogram.offset)


# ---------------------------------------------------------------- the kernel
@pytest.mark.parametrize("pattern", [False, True])
def test_healthy_histogram_is_smooth(pattern):
    value, _ = anomaly(gaussian(pattern=pattern))
    assert value < 0.3


def test_bit5_stuck_in_40_percent_of_rows():
    frame = gaussian(1, pattern=True)
    frame[:400] |= 32
    value, code = anomaly(frame)
    assert value > 0.6
    assert code % 32 == 0


def test_random_bit5_flips():
    """A flipped pixel moves 32 ADU; it is seen where it lands on a populated part of
    the histogram (here the lit wing), not when it lands on empty values."""
    frame = sparse_lit(2)
    healthy, _ = anomaly(frame)
    assert healthy < 0.3
    flips = np.random.default_rng(3).random(SHAPE) < 0.05
    frame[flips] ^= 32
    value, code = anomaly(frame)
    assert value > 0.6
    assert code % 32 == 0


def test_offset_histogram_reports_the_pixel_value():
    """``counts[i]`` holds value ``i + offset`` (IntegerHistogram of negative data):
    the reported code is a pixel value, not a bin index."""
    frame = gaussian(4, pattern=True)
    frame[:400] |= 32
    counts = np.bincount(frame.ravel())
    value, code = code_discontinuity(counts, 0)
    assert code_discontinuity(counts[100:], 100) == (value, code)
    assert code_discontinuity(counts, -700) == (value, code - 700)
    shifted = IntegerHistogram(frame.astype(np.int32) - 2000)
    assert shifted.offset < 0
    assert code_discontinuity(shifted.counts, shifted.offset) == (value, code - 2000)


@pytest.mark.parametrize("frame", [
    np.full(SHAPE, 667, dtype=np.uint16),     # constant
    np.full(SHAPE, 65535, dtype=np.uint16),   # railed
])
def test_constant_or_railed_frame_is_skipped(frame):
    with pytest.raises(E.LookupMissError):
        anomaly(frame)


def test_float_data():
    metric = {"type": "code_discontinuity"}
    rng = np.random.default_rng(5)
    with pytest.raises(E.LookupMissError, match="integer"):
        QAEngine._compute_metric(rng.normal(LEVEL, SIGMA, SHAPE), metric)
    frame = gaussian(6, pattern=True)
    frame[:400] |= 32
    as_float = frame.astype(np.float64)
    as_float[0, :10] = np.nan
    assert QAEngine._compute_metric(as_float, metric) == pytest.approx(anomaly(frame)[0], rel=1e-3)


# ---------------------------------------------------------------- engine
CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [{"name": "2.B.Green", "hdu_index": 1, "color": "Green", "bench": 2, "side": "B"},
                   {"name": "1.A.Red", "hdu_index": 2, "color": "Red", "bench": 1, "side": "A"}],
    "regions": {"full_frame": {"type": "full"}},
    "metrics": {"median": {"type": "median"},
                "bit_errors": {"type": "code_discontinuity", "min_count": 1000,
                               "max_value": 63000}},
    "lookup_tables": {"noop": {"a": 1}},
    "rule_sets": {
        "SCIENCE": {
            "applies_when": {"exposure_type": "SCI.R-*"},
            "rules": [
                {"name": "frame_median", "region": "full_frame", "metric": "median",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 60000}},
                {"name": "pixel_bit_errors", "region": "full_frame", "metric": "bit_errors",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 0.6}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def write_sci_mef(tmp_path, frames):
    ph = fits.PrimaryHDU()
    ph.header["PRODCATG"] = "SCI.R-SL"
    ph.header["READ-MDE"] = "SLOW"
    path = tmp_path / "sci.fits"
    fits.HDUList([ph] + [fits.ImageHDU(data=f) for f in frames]).writeto(path, overwrite=True)
    return path


def test_engine_warns_and_names_value_and_bit(tmp_path):
    bad = gaussian(7, pattern=True)
    bad[:400] |= 32
    value, code = anomaly(bad)
    path = write_sci_mef(tmp_path, [bad, gaussian(8)])
    engine = QAEngine(CONFIG, cameras_down=[])
    report = engine.run(path)
    res = {r["extension"]: r for r in report["results"] if r["rule"] == "pixel_bit_errors"}
    assert report["overall_verdict"] == "WARN"
    green = res["2.B.Green"]
    assert not green["passed"] and green["verdict_effect"] == "WARN"
    assert green["measured_value"] == pytest.approx(value, abs=1e-6)
    bit = (code & -code).bit_length() - 1
    assert bit >= 5
    assert f"pixel values jump at value {code} (bit-{bit} boundary)" in green["message"]
    assert "digitiser bit errors" in green["message"]
    assert res["1.A.Red"]["passed"]
    # the histogram is shared with the median rule: one per extension
    for entry in engine.last_run.extensions.values():
        assert list(entry.histograms) == ["full_frame"]


def test_shipped_bit_rules_never_fail():
    """pixel_bit_errors is in SCIENCE and every calibration rule set, always WARN."""
    for name, sets in (("qa_config_science.yaml", {"SCIENCE"}),
                       ("qa_config_cal.yaml", {"BIAS", "DARK", "ARC_THAR", "LDLS_FLAT", "SKY_FLAT"})):
        cfg = E.load_yaml(E.CONFIG_DIR / name)
        assert set(cfg["rule_sets"]) == sets
        for rule_set in cfg["rule_sets"].values():
            rules = [r for r in rule_set["rules"] if r["name"] == "pixel_bit_errors"]
            assert len(rules) == 1
            assert rules[0]["severity"] == "WARN" and rules[0]["limits"] == {"max": 0.6}
            assert rules[0]["region"] == "full_frame"
            assert cfg["metrics"][rules[0]["metric"]]["type"] == "code_discontinuity"


# ---------------------------------------------------------------- validator
def test_validator_accepts_bit_metric():
    assert QAConfigValidator(copy.deepcopy(CONFIG), filename="bits").validate() == []
    cfg = copy.deepcopy(CONFIG)
    cfg["metrics"]["bit_errors"] = {"type": "code_discontinuity"}
    assert QAConfigValidator(cfg, filename="bits").validate() == []


@pytest.mark.parametrize("field, value, message", [
    ("min_count", 0, "positive integer"),
    ("min_count", 10.5, "positive integer"),
    ("max_value", 0, "positive number"),
    ("max_value", "x", "positive number"),
    ("threshold", 5, "unknown field"),
])
def test_validator_rejects_bad_bit_params(field, value, message):
    cfg = copy.deepcopy(CONFIG)
    cfg["metrics"]["bit_errors"][field] = value
    errors = [f"{e.path}: {e.message}" for e in QAConfigValidator(cfg, filename="bits").validate()]
    assert any("bit_errors" in e and message in e for e in errors), errors


# ---------------------------------------------------------------- real frames
BOX = Path("/Users/slh/Library/CloudStorage/Box-Box/slhughes")


def _real(path):
    if not path.exists():
        pytest.skip(f"real frame not available: {path}")
    return path


def test_real_nov01_bit_errors_only_on_2BGreen():
    path = _real(BOX / "LLAMAS_analysis" / "LLAMAS_2026-11-01_05-54-08.1_SCI24_mef.fits")
    report = QAEngine(E.load_yaml(E.CONFIG_DIR / "qa_config_science.yaml"), cameras_down=[]).run(path)
    fired = {r["extension"] for r in report["results"]
             if r["rule"] == "pixel_bit_errors" and not r["passed"]}
    assert fired == {"2.B.Green"}


def test_real_usable_gd71_is_warn_not_fail(tmp_path):
    """GD71 200 s (2026-11-10): 6.6 % flipped pairs on 2.B.Green, but no division --
    usable, so the frame WARNs and does not FAIL."""
    path = _real(BOX / "Llamas_Commissioning_Data" / "20261109_10"
                 / "LLAMAS_2026-11-10_08-04-55.3_SCI24_mef.fits")
    result = check_image(path, report_dir=tmp_path)
    assert result["status"] == "warn", result["message"]
    assert "pixel_bit_errors@2.B.Green" in result["message"]
