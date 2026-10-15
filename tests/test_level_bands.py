"""Level / edge-background bands (range + drift margin) and the per-field colour
fallback of gen_configs, plus guards on the shipped config after the 2026-10-14
level-only epoch was added (1.A.Blue / 4.A.Blue bands; 2.B.Green SLOW, frozen at
its legacy band until 2026-10-15, now on the range + margin band like the rest)."""
import json

import numpy as np
import pytest
import yaml

import aggregate_thresholds as agg
import gen_configs as gc
from llamas_checks.paths import BASELINES_DIR, CONFIG_DIR

OCT = "2026-10-14"
SET_FOR = {"CAL.R-BIA": "BIAS", "CAL.R-DRK": "DARK", "CAL.R-ARC": "ARC_THAR",
           "CAL.R-SKY": "SKY_FLAT", "CAL.R-FLT": "LDLS_FLAT"}
BLUE = ["1.A.Blue", "1.B.Blue", "2.A.Blue", "2.B.Blue", "3.A.Blue", "3.B.Blue", "4.A.Blue", "4.B.Blue"]


# ---------------------------------------------------------------- band functions

def test_level_band_uses_drift_margin_on_zero_sigma_sample():
    lo, hi = agg.level_band([700.0] * 5 + [710.0] * 2)
    assert (lo, hi) == (700.0 - agg.DRIFT_MARGIN_ADU, 710.0 + agg.DRIFT_MARGIN_ADU)
    assert agg.DRIFT_MARGIN_ADU == 30.0


def test_level_band_uses_six_sigma_when_larger():
    a = np.array([600.0, 620.0, 640.0, 660.0, 680.0])
    half = agg.K_LEVEL * agg.robust_sigma(a)
    assert half > agg.DRIFT_MARGIN_ADU
    assert agg.level_band(a) == (round(600.0 - half, 2), round(680.0 + half, 2))


def test_level_band_contains_legacy_band_on_bimodal_sample():
    a = np.array([640.0] * 10 + [700.0] * 10 + [705.0])
    lo, hi = agg.level_band(a)
    llo, lhi = agg.legacy_level_band(a)
    assert lo <= llo and hi >= lhi
    assert lo <= a.min() - 30 and hi >= a.max() + 30


def test_band_for_dispatches_legacy_detectors(monkeypatch):
    a = np.array([630.0, 631.0, 632.0])
    assert agg.LEGACY_BAND_DETECTORS == set()          # 2.B.Green SLOW unfrozen 2026-10-15
    assert agg.band_for("2.B.Green", "SLOW", a) == agg.level_band(a)
    # the mechanism is kept for the next detector that needs freezing
    monkeypatch.setattr(agg, "LEGACY_BAND_DETECTORS", {("2.B.Green", "SLOW")})
    assert agg.band_for("2.B.Green", "SLOW", a) == agg.legacy_level_band(a)
    assert agg.band_for("2.B.Green", "FAST", a) == agg.level_band(a)
    assert agg.band_for("1.A.Red", "SLOW", a) == agg.level_band(a)


def test_heavy_tail_cap_empty_sample_is_none():
    assert agg.heavy_tail_cap([], agg.FLOOR_STRUCT) is None
    assert agg.heavy_tail_cap([None, float("nan")], agg.FLOOR_RMS) is None
    assert agg.heavy_tail_cap([1.0, 1.0, 1.0], agg.FLOOR_STRUCT) == agg.FLOOR_STRUCT


# ------------------------------------------------- gen_configs per-field fallback

def _entry(row, col, edge=(600.0, 700.0)):
    return {"edge_bg": {"min": edge[0], "max": edge[1]},
            "structure": {"row_max": row, "col_max": col}}


@pytest.fixture
def fake_der(monkeypatch):
    der = {"per_detector": {"CAL.R-BIA": {"SLOW": {
        "1.A.Red": _entry(2.0, 3.0),
        "2.A.Red": _entry(4.0, 5.0),
        "3.A.Red": _entry(6.0, 7.0),
        # a camera seen only in a level-only epoch: band but no caps
        "1.A.Blue": _entry(None, None, edge=(900.0, 960.0)),
        "2.A.Blue": _entry(8.0, 9.0),
    }}}}
    monkeypatch.setattr(gc, "DER", der)
    return der


def test_build_table_fills_none_fields_with_colour_median(fake_der):
    t = gc.build_table("CAL.R-BIA", gc.struct_leaf)
    leaf = t["1.A.Blue"]["SLOW"]
    assert leaf == {"row_max": 8.0, "col_max": 9.0, "row_fail_max": 32.0, "col_fail_max": 36.0}
    assert gc.build_table("CAL.R-BIA", gc.edge_leaf)["1.A.Blue"]["SLOW"] == {"min": 900.0, "max": 960.0}


def test_build_table_fills_missing_detector_and_keeps_4x(fake_der):
    t = gc.build_table("CAL.R-BIA", gc.struct_leaf)
    leaf = t["4.B.Red"]["SLOW"]          # no entry at all -> colour median of the reds
    assert leaf["row_max"] == 4.0 and leaf["col_max"] == 5.0
    assert leaf["row_fail_max"] == gc.STRUCT_FAIL_FACTOR * leaf["row_max"]
    assert leaf["col_fail_max"] == gc.STRUCT_FAIL_FACTOR * leaf["col_max"]


def test_build_table_omits_unfillable_leaf(fake_der):
    t = gc.build_table("CAL.R-BIA", gc.struct_leaf)
    assert "1.A.Green" not in t            # no green peer has a value
    for per_det in t.values():
        for leaf in per_det.values():
            assert all(v is not None for v in leaf.values())


# ------------------------------------------------------------ shipped config

@pytest.fixture(scope="module")
def cal():
    return yaml.safe_load((CONFIG_DIR / "qa_config_cal.yaml").read_text())


@pytest.fixture(scope="module")
def derived():
    return json.loads((BASELINES_DIR / "qa_thresholds_derived.json").read_text())


@pytest.fixture(scope="module")
def raw():
    return json.loads((BASELINES_DIR / "qa_stats_raw.json").read_text())


@pytest.mark.parametrize("det,mode,oct_med", [
    ("1.A.Blue", "FAST", 962), ("1.A.Blue", "SLOW", 729),
    ("4.A.Blue", "FAST", 1092), ("4.A.Blue", "SLOW", 861),
])
def test_placeholder_blue_bias_bands_contain_october_level(cal, det, mode, oct_med):
    lv = cal["lookup_tables"]["level_BIAS"][det][mode]
    eb = cal["lookup_tables"]["edge_bg_BIAS"][det][mode]
    assert lv["med_min"] < oct_med < lv["med_max"]
    assert eb["min"] < oct_med < eb["max"]


def test_unfrozen_detector_band_covers_october_level(cal, derived):
    """2.B.Green SLOW biases sit at ~667 ADU since 2026-10-14 (the new normal): the
    band must hold that level plus the 30 ADU drift margin, and no frame type keeps
    the legacy band."""
    lt = cal["lookup_tables"]
    lv = lt["level_BIAS"]["2.B.Green"]["SLOW"]
    assert lv["med_max"] >= 667 + 30 and lv["med_min"] <= 629 - 30
    assert lt["edge_bg_BIAS"]["2.B.Green"]["SLOW"]["max"] >= 671 + 30
    for pc, per_mode in derived["per_detector"].items():
        if "2.B.Green" in per_mode.get("SLOW", {}):
            assert per_mode["SLOW"]["2.B.Green"]["band"] == "range+margin", pc
    assert derived["meta"]["legacy_band_detectors"] == []


def _blue_median(derived, pc, mode, getter):
    vals = sorted(v for nm, e in derived["per_detector"][pc][mode].items()
                  if nm.endswith(".Blue") and (v := getter(e)) is not None)
    return round(float(vals[len(vals) // 2]), 4)


@pytest.mark.parametrize("det", ["1.A.Blue", "4.A.Blue"])
def test_unvetted_blue_caps_are_colour_median(cal, derived, det):
    lt = cal["lookup_tables"]
    for mode in ("FAST", "SLOW"):
        e = derived["per_detector"]["CAL.R-BIA"][mode][det]
        assert e["n_vetted"] == 0 and e["structure"]["row_max"] is None
        s = lt["struct_BIAS"][det][mode]
        assert s["row_max"] == _blue_median(derived, "CAL.R-BIA", mode, lambda x: x["structure"]["row_max"])
        assert s["col_max"] == _blue_median(derived, "CAL.R-BIA", mode, lambda x: x["structure"]["col_max"])
        assert s["row_fail_max"] == pytest.approx(4 * s["row_max"])
        assert lt["sat_BIAS"][det][mode]["frac_max"] == _blue_median(
            derived, "CAL.R-BIA", mode, lambda x: x["sat_frac_max"])
        assert lt["level_BIAS"][det][mode]["rms_max"] == _blue_median(
            derived, "CAL.R-BIA", mode, lambda x: x["full_level"]["rms_max"])
        arc = lt["struct_ARC_THAR"][det][mode]
        assert arc["row_max"] == _blue_median(
            derived, "CAL.R-ARC", mode, lambda x: (x["structure_norm"] or {}).get("row_max"))
        assert lt["smear_ARC_THAR"][det][mode]["max"] == _blue_median(
            derived, "CAL.R-ARC", mode, lambda x: (x["smear"] or {}).get("max"))


def _det(r):
    return agg.det_name(r)


def test_october_records_inside_their_bands(cal, raw):
    lt = cal["lookup_tables"]
    checked = 0
    for r in raw:
        if r["date"] != OCT or r.get("placeholder") or "error" in r or r["prodcatg"] == "CAL.R-FLT":
            continue
        det, mode = _det(r), r["mode"]
        if (det, mode) in agg.LEGACY_BAND_DETECTORS:
            continue
        name = SET_FOR[r["prodcatg"]]
        eb = lt[f"edge_bg_{name}"][det][mode]
        assert eb["min"] <= r["edge_med"] <= eb["max"], (r["filename"], det)
        if name in ("BIAS", "DARK"):
            lv = lt[f"level_{name}"][det][mode]
            assert lv["med_min"] <= r["full_med"] <= lv["med_max"], (r["filename"], det)
        checked += 1
    assert checked > 800


def test_unfrozen_october_biases_lie_inside(cal, raw):
    lv = cal["lookup_tables"]["level_BIAS"]["2.B.Green"]["SLOW"]
    oct_recs = [r for r in raw if r["date"] == OCT and r["prodcatg"] == "CAL.R-BIA"
                and r["mode"] == "SLOW" and _det(r) == "2.B.Green" and not r.get("placeholder")]
    assert len(oct_recs) == 11
    assert all(lv["med_min"] <= r["full_med"] <= lv["med_max"] for r in oct_recs)


def test_october_epoch_does_not_move_shutter_or_meta(cal, derived):
    bias = {r["name"]: r for r in cal["rule_sets"]["BIAS"]["rules"]}
    assert bias["shutter_exptime_consistency"]["header_check"]["abs_tol"] == 0.214
    assert derived["meta"]["level_only_epochs"] == [OCT]
    assert derived["meta"]["n_vetted_records"] == 3344
