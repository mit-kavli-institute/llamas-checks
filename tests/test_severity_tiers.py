"""Robust metrics and WARN/FAIL severity tiers (2026-09 revision).

Synthetic single-detector MEFs exercise the engine directly; the last block
guards the *shipped* generated configs so a regenerated YAML cannot silently
revert the tiering.
"""
import numpy as np
import pytest
import yaml
from astropy.io import fits

from llamas_checks.paths import CONFIG_DIR
from llamas_checks.qa_config_validator import QAConfigValidator
from llamas_checks.qa_engine import QAEngine

SHAPE = (200, 2000)         # covers bottom_stripe y[2:28], x[100:1948]; 2 % trim = 4 px per column
CAP = 2.0                   # per-detector structure WARN cap in the test table
GROSS = 4.0 * CAP           # FAIL tier
SAT_CAP = 0.0005
SAT_GROSS = 0.01
EDGE_SAT = 63000.0


NOISE_MAD = 0.3   # median |x| of the 11-level pattern below (levels -0.5 .. +0.5)


def checkerboard(shape=SHAPE, level=700.0):
    """Uniform frame plus a deterministic 11-level zero-median pattern (-0.5 .. +0.5).

    Never a constant frame (constant 0/1 frames are placeholders), and every row and
    column cycles through all levels so its median is ~0. Eleven plateaus keep the
    frame median stable when a few pixels are made hot (a two-level checkerboard is
    degenerate: its median flips between the two levels on a rank change).
    """
    yy, xx = np.indices(shape)
    pattern = (((7 * yy + 13 * xx) % 11) - 5) * 0.1
    return (np.full(shape, level, dtype=np.float32) + pattern.astype(np.float32))


CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [{"name": "1.A.Green", "hdu_index": 1, "color": "Green", "bench": 1, "side": "A"}],
    "regions": {
        "full_frame": {"type": "full"},
        "bottom_stripe": {"type": "rectangle", "x_start": 100, "x_end": 1948, "y_start": 2, "y_end": 28},
    },
    "metrics": {
        "median": {"type": "median"},
        "rms": {"type": "robust_std"},
        "saturated_fraction": {"type": "fraction_above", "threshold": 63000},
        "row_banding": {"type": "row_structure"},
        "column_banding": {"type": "column_structure"},
    },
    "lookup_tables": {
        "struct": {"1.A.Green": {"SLOW": {"row_max": CAP, "col_max": CAP,
                                          "row_fail_max": GROSS, "col_fail_max": GROSS}}},
        "sat": {"1.A.Green": {"SLOW": {"frac_max": SAT_CAP}}},
    },
    "rule_sets": {
        "BIAS": {
            "applies_when": {"exposure_type": "CAL.R-BIA"},
            "rules": [
                {"name": "edge_saturated", "region": "bottom_stripe", "metric": "median",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": EDGE_SAT}},
                {"name": "row_structure", "region": "full_frame", "metric": "row_banding",
                 "per_extension": True, "severity": "WARN",
                 "expected_from_lookup": {"table": "struct", "keys": [{"from": "extension.name"},
                                          {"from": "metadata.readout_mode"}], "max_field": "row_max"}},
                {"name": "column_structure", "region": "full_frame", "metric": "column_banding",
                 "per_extension": True, "severity": "WARN",
                 "expected_from_lookup": {"table": "struct", "keys": [{"from": "extension.name"},
                                          {"from": "metadata.readout_mode"}], "max_field": "col_max"}},
                {"name": "row_structure_gross", "region": "full_frame", "metric": "row_banding",
                 "per_extension": True, "severity": "FAIL",
                 "expected_from_lookup": {"table": "struct", "keys": [{"from": "extension.name"},
                                          {"from": "metadata.readout_mode"}], "max_field": "row_fail_max"}},
                {"name": "column_structure_gross", "region": "full_frame", "metric": "column_banding",
                 "per_extension": True, "severity": "FAIL",
                 "expected_from_lookup": {"table": "struct", "keys": [{"from": "extension.name"},
                                          {"from": "metadata.readout_mode"}], "max_field": "col_fail_max"}},
                {"name": "saturation_fraction", "region": "full_frame", "metric": "saturated_fraction",
                 "per_extension": True, "severity": "WARN",
                 "expected_from_lookup": {"table": "sat", "keys": [{"from": "extension.name"},
                                          {"from": "metadata.readout_mode"}], "max_field": "frac_max"}},
                {"name": "saturation_gross", "region": "full_frame", "metric": "saturated_fraction",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": SAT_GROSS}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def write_mef(tmp_path, data, name="frame_mef.fits"):
    ph = fits.PrimaryHDU()
    ph.header["PRODCATG"] = "CAL.R-BIA"
    ph.header["READ-MDE"] = "SLOW"
    ph.header["REXPTIME"] = 0.001
    ph.header["SEXPTIME"] = 0.002
    ext = fits.ImageHDU(data=np.asarray(data, dtype=np.float32))
    ext.header["COLOR"] = "Green"
    ext.header["BENCH"] = 1
    ext.header["SIDE"] = "A"
    path = tmp_path / name
    fits.HDUList([ph, ext]).writeto(path, overwrite=True)
    return path


def run(tmp_path, data):
    return QAEngine(CONFIG).run(write_mef(tmp_path, data))


def result_for(report, rule):
    return next(r for r in report["results"] if r["rule"] == rule)


def metric(data, mtype):
    return QAEngine._compute_metric(data, {"type": mtype})


# ---------------------------------------------------------------- metrics
def test_config_is_valid():
    assert QAConfigValidator(CONFIG, filename="tiers").validate() == []


def test_validator_accepts_robust_std():
    assert "robust_std" in QAConfigValidator.METRIC_TYPES


def test_trimmed_structure_ignores_hot_column_fragment():
    # 4 railed pixels in one column (2 % of its 200 rows): the old per-column MEAN
    # profile moved by ~1300 ADU in that column (std >> cap); the trimmed mean drops them.
    data = checkerboard()
    data[30:34, 500] = 65535.0
    assert metric(data, "column_structure") < 0.1
    assert metric(data, "row_structure") < 0.1


def test_trimmed_structure_measures_full_column_bar():
    data = checkerboard()
    data[:, 400:600] += 10.0          # a 200-column, full-height bar
    assert metric(data, "column_structure") > 2.0


def test_trimmed_structure_measures_row_banding():
    data = checkerboard()
    data[::2, :] += 3.0               # alternate rows offset by +3 -> row std 1.5
    assert metric(data, "row_structure") == pytest.approx(1.5, abs=0.1)


def test_robust_std_is_read_noise_and_blind_to_hot_pixels():
    clean = checkerboard()
    assert metric(clean, "robust_std") == pytest.approx(1.4826 * NOISE_MAD, rel=0.05)
    hot = clean.copy()
    hot[30:34, 500] = 65535.0
    assert metric(hot, "robust_std") == pytest.approx(metric(clean, "robust_std"), rel=1e-2)
    assert metric(hot, "std") > 50 * metric(clean, "std")   # the plain std is not


# ---------------------------------------------------------------- tiers
def test_structure_above_cap_is_warn_not_fail(tmp_path):
    data = checkerboard()
    data[::2, :] += 2 * 1.5 * CAP     # row std = 1.5 x cap
    rep = run(tmp_path, data)
    assert result_for(rep, "row_structure")["verdict_effect"] == "WARN"
    assert result_for(rep, "row_structure_gross")["verdict_effect"] == "PASS"
    assert rep["overall_verdict"] == "WARN"


def test_gross_structure_fails(tmp_path):
    data = checkerboard()
    data[::2, :] += 2 * 5.0 * CAP     # row std = 5 x cap
    rep = run(tmp_path, data)
    assert result_for(rep, "row_structure")["verdict_effect"] == "WARN"
    assert result_for(rep, "row_structure_gross")["verdict_effect"] == "FAIL"
    assert rep["overall_verdict"] == "FAIL"


def test_hot_pixel_cluster_bias_passes(tmp_path):
    data = checkerboard()
    data[30:34, 500] = 65535.0        # 4 px = 1e-5 of the frame, below SAT_CAP
    rep = run(tmp_path, data)
    assert rep["overall_verdict"] == "PASS"


def test_modest_saturation_is_warn_only(tmp_path):
    data = checkerboard()
    for i in range(400):              # 400 scattered px = 1e-3 > SAT_CAP, < SAT_GROSS;
        data[30 + i % 100, (i * 37) % 2000] = 65535.0   # <= 4 per row, <= 1 per column

    rep = run(tmp_path, data)
    assert result_for(rep, "saturation_fraction")["verdict_effect"] == "WARN"
    assert result_for(rep, "saturation_gross")["verdict_effect"] == "PASS"
    assert rep["overall_verdict"] == "WARN"


def test_gross_saturation_fails(tmp_path):
    data = checkerboard()
    data[30:70, 0:200] = 65535.0      # 8000 px = 2 % of the frame
    rep = run(tmp_path, data)
    assert result_for(rep, "saturation_gross")["verdict_effect"] == "FAIL"
    assert rep["overall_verdict"] == "FAIL"


def test_railed_edge_stripe_fails(tmp_path):
    data = checkerboard()
    data[0:28, :] = 65535.0           # the unilluminated bottom stripe is at the ADC ceiling
    rep = run(tmp_path, data)
    edge = result_for(rep, "edge_saturated")
    assert edge["verdict_effect"] == "FAIL" and edge["measured_value"] == 65535.0
    assert rep["overall_verdict"] == "FAIL"


def test_bright_but_unsaturated_edge_stripe_passes_edge_rule(tmp_path):
    data = checkerboard()
    data[0:28, :] += 4000.0
    rep = run(tmp_path, data)
    assert result_for(rep, "edge_saturated")["verdict_effect"] == "PASS"


# ---------------------------------------------------------------- shipped configs
@pytest.fixture(scope="module")
def cal_cfg():
    return yaml.safe_load((CONFIG_DIR / "qa_config_cal.yaml").read_text())


@pytest.fixture(scope="module")
def sci_cfg():
    return yaml.safe_load((CONFIG_DIR / "qa_config_science.yaml").read_text())


def rules_of(cfg, rule_set):
    return {r["name"]: r for r in cfg["rule_sets"][rule_set]["rules"]}


@pytest.mark.parametrize("rule_set", ["BIAS", "DARK"])
def test_shipped_uniform_frames_have_two_tier_structure(cal_cfg, rule_set):
    rules = rules_of(cal_cfg, rule_set)
    for base in ("row_structure", "column_structure"):
        assert rules[base]["severity"] == "WARN"
        assert rules[base + "_gross"]["severity"] == "FAIL"
        assert rules[base + "_gross"]["expected_from_lookup"]["max_field"].endswith("_fail_max")
    assert rules["saturation_fraction"]["severity"] == "WARN"
    assert rules["saturation_gross"]["severity"] == "FAIL"
    assert rules["saturation_gross"]["limits"] == {"max": 0.01}


@pytest.mark.parametrize("rule_set", ["BIAS", "DARK"])
def test_shipped_gross_caps_are_four_times_warn_caps(cal_cfg, rule_set):
    table = cal_cfg["lookup_tables"][f"struct_{rule_set}"]
    for detector, modes in table.items():
        for mode, leaf in modes.items():
            assert leaf["row_fail_max"] == pytest.approx(4 * leaf["row_max"], rel=1e-3), (detector, mode)
            assert leaf["col_fail_max"] == pytest.approx(4 * leaf["col_max"], rel=1e-3), (detector, mode)


def test_shipped_illuminated_structure_severities(cal_cfg):
    assert rules_of(cal_cfg, "SKY_FLAT")["row_structure"]["severity"] == "WARN"
    assert rules_of(cal_cfg, "LDLS_FLAT")["row_structure"]["severity"] == "FAIL"
    assert rules_of(cal_cfg, "ARC_THAR")["row_structure"]["severity"] == "FAIL"
    for rule_set in ("SKY_FLAT", "LDLS_FLAT", "ARC_THAR"):
        assert "row_structure_gross" not in rules_of(cal_cfg, rule_set)
        assert rules_of(cal_cfg, rule_set)["saturation_fraction"]["severity"] == "WARN"


def test_every_shipped_rule_set_has_edge_saturated_fail(cal_cfg, sci_cfg):
    for cfg in (cal_cfg, sci_cfg):
        for rule_set in cfg["rule_sets"]:
            rule = rules_of(cfg, rule_set)["edge_saturated"]
            assert rule["severity"] == "FAIL"
            assert rule["region"] == "bottom_stripe" and rule["metric"] == "median"
            assert rule["limits"] == {"max": 63000.0}


def test_shipped_rms_metric_is_robust(cal_cfg):
    assert cal_cfg["metrics"]["rms"]["type"] == "robust_std"
    assert cal_cfg["metrics"]["row_banding"]["type"] == "row_structure"
