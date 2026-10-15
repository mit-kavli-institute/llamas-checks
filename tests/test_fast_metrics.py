"""The fast metric kernels must reproduce the generic float implementations exactly.

Raw LLAMAS frames are 16-bit integers; the engine measures them with a value
histogram (median, MAD, threshold counts) and a two-pivot partition (trimmed-mean
structure profiles) instead of sorting float64 copies. These tests pin the two
paths to each other, for integer and float input, serial and threaded.
"""
import numpy as np
import pytest
from astropy.io import fits

from llamas_checks.qa_engine import (DEFAULT_WORKERS, IntegerHistogram, QAEngine,
                                     trimmed_mean_profile)


def float_metric(values, mtype, **extra):
    """The generic float64 path (integer kernels are only used for integer dtypes)."""
    return QAEngine._compute_metric(np.asarray(values, dtype=float), {"type": mtype, **extra})


def int_metric(values, mtype, **extra):
    return QAEngine._compute_metric(values, {"type": mtype, **extra})


def sorted_profile(values, axis, trim=0.02):
    """The original sort-based trimmed-mean profile."""
    ordered = np.sort(np.asarray(values, dtype=float), axis=axis)
    n = ordered.shape[axis]
    k = max(1, int(n * trim))
    index = [slice(None)] * ordered.ndim
    index[axis] = slice(k, n - k)
    return np.nanmean(ordered[tuple(index)], axis=axis)


# ---------------------------------------------------------------- histogram kernels
@pytest.mark.parametrize("dtype,low,high", [
    (np.uint16, 0, 65536),        # full ADC range, as in raw frames
    (np.uint16, 1600, 1800),      # a bias pedestal
    (np.int16, -32768, 32768),    # signed with negatives (offset bins)
    (np.uint8, 0, 256),
])
@pytest.mark.parametrize("shape", [(64, 64), (63, 65), (1, 10), (10, 1), (7, 3, 5)])
def test_histogram_metrics_match_float_path(dtype, low, high, shape):
    rng = np.random.default_rng(12345)
    values = rng.integers(low, high, size=shape, dtype=dtype)
    for mtype in ("median", "robust_std", "min", "max"):
        assert int_metric(values, mtype) == float_metric(values, mtype), mtype
    for threshold in (low, low - 5, high - 1, high + 5, (low + high) / 2, 1234.5, -0.5):
        for mtype in ("fraction_above", "count_above"):
            assert (int_metric(values, mtype, threshold=threshold)
                    == float_metric(values, mtype, threshold=threshold)), (mtype, threshold)


def test_histogram_even_and_odd_counts_follow_numpy_median():
    for n in (1, 2, 3, 4, 5, 10, 11):
        values = np.array([5, 1, 9, 3, 3, 8, 2, 7, 4, 6, 0][:n], dtype=np.uint16)
        assert IntegerHistogram(values).median() == np.median(values)
        assert int_metric(values, "robust_std") == float_metric(values, "robust_std")


def test_constant_and_two_valued_arrays():
    constant = np.full((5, 5), 1700, dtype=np.uint16)
    assert int_metric(constant, "median") == 1700.0
    assert int_metric(constant, "robust_std") == 0.0
    two = np.array([[0, 65535]], dtype=np.uint16)
    assert int_metric(two, "median") == float_metric(two, "median") == 32767.5
    assert int_metric(two, "fraction_above", threshold=63000) == 0.5


def test_histogram_is_shared_between_metrics_of_one_region():
    values = np.random.default_rng(0).integers(0, 65536, size=(50, 50), dtype=np.uint16)
    histograms = {}
    median = QAEngine._compute_metric(values, {"type": "median"},
                                      histograms=histograms, region_name="full_frame")
    rms = QAEngine._compute_metric(values, {"type": "robust_std"},
                                   histograms=histograms, region_name="full_frame")
    assert list(histograms) == ["full_frame"]
    assert median == float_metric(values, "median")
    assert rms == float_metric(values, "robust_std")


def test_gradient_rate_matches_float_path():
    rng = np.random.default_rng(3)
    values = (rng.normal(700, 5, (64, 80)) + np.linspace(0, 40, 64)[:, None]).astype(np.uint16)
    metric = {"type": "background_gradient_rate"}
    assert (QAEngine._compute_metric(values, metric, exptime=10.0)
            == QAEngine._compute_metric(values.astype(float), metric, exptime=10.0))


def test_metric_types_without_integer_kernel_still_work():
    values = np.arange(1, 101, dtype=np.uint16).reshape(10, 10)
    for mtype in ("mean", "std", "sum"):
        assert int_metric(values, mtype) == pytest.approx(float_metric(values, mtype))
    assert int_metric(values, "percentile", percentile=90) == float_metric(values, "percentile",
                                                                           percentile=90)


# ---------------------------------------------------------------- trimmed-mean profiles
@pytest.mark.parametrize("shape", [(32, 2048), (2048, 32), (10, 10), (3, 200), (200, 3)])
def test_partition_profile_equals_sorted_profile_for_integers(shape):
    values = np.random.default_rng(7).integers(0, 65536, size=shape, dtype=np.uint16)
    for axis in (-1, -2, 0, 1):
        np.testing.assert_array_equal(trimmed_mean_profile(values, axis), sorted_profile(values, axis))


def test_partition_profile_matches_sorted_profile_for_floats_with_nans():
    rng = np.random.default_rng(8)
    values = rng.normal(700, 3, (40, 300)).astype(np.float32)
    values[rng.random(values.shape) < 0.01] = np.nan       # fewer NaNs than the trim per line
    values[5, :] = np.nan                                  # one all-NaN row
    for axis in (0, 1):
        with np.errstate(all="ignore"):
            new = trimmed_mean_profile(values, axis)
            ref = sorted_profile(values, axis)
        np.testing.assert_allclose(new, ref, rtol=1e-12, atol=0, equal_nan=True)


def test_profile_is_undefined_when_nothing_survives_the_trim():
    tiny = np.ones((4, 2), dtype=np.uint16)           # 2 pixels per row, 1 trimmed from each end
    assert np.isnan(trimmed_mean_profile(tiny, -1)).all()
    with pytest.raises(Exception):
        int_metric(tiny, "row_structure")              # "no finite rows/columns"


def test_structure_metric_integer_equals_float():
    rng = np.random.default_rng(9)
    values = (rng.normal(1700, 4, (200, 300)) + rng.choice([-20, 0, 20], 200)[:, None]).astype(np.uint16)
    for mtype in ("row_structure", "column_structure"):
        assert int_metric(values, mtype) == float_metric(values, mtype)


# ---------------------------------------------------------------- engine, serial vs threaded
SHAPE = (64, 2000)   # covers bottom_stripe x[100:1948], y[2:28]; 2 % trim = 40 px per row

CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [
        {"name": "1.A.Red", "hdu_index": 1}, {"name": "1.A.Green", "hdu_index": 2},
        {"name": "1.A.Blue", "hdu_index": 3}, {"name": "1.B.Red", "hdu_index": 4},
        {"name": "1.B.Green", "hdu_index": 5}, {"name": "1.B.Blue", "hdu_index": 6},
        {"name": "2.A.Red", "hdu_index": 7},     # placeholder (constant 1)
        {"name": "2.A.Green", "hdu_index": 8},   # missing HDU
    ],
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
        "gradient_rate": {"type": "background_gradient_rate", "min_exptime": 1.0},
    },
    "lookup_tables": {},
    "rule_sets": {
        "BIAS": {
            "applies_when": {"exposure_type": "CAL.R-BIA"},
            "rules": [
                {"name": "edge_level", "region": "bottom_stripe", "metric": "median",
                 "per_extension": True, "severity": "WARN", "limits": {"min": 600, "max": 800}},
                {"name": "edge_saturated", "region": "bottom_stripe", "metric": "median",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 63000}},
                {"name": "level", "region": "full_frame", "metric": "median",
                 "per_extension": True, "severity": "WARN", "limits": {"min": 600, "max": 800}},
                {"name": "noise", "region": "full_frame", "metric": "rms",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 6}},
                {"name": "row_structure", "region": "full_frame", "metric": "row_banding",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 2}},
                {"name": "row_structure_gross", "region": "full_frame", "metric": "row_banding",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 8}},
                {"name": "column_structure", "region": "full_frame", "metric": "column_banding",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 2}},
                {"name": "saturation", "region": "full_frame", "metric": "saturated_fraction",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 0.0005}},
                {"name": "warming", "region": "full_frame", "metric": "gradient_rate",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 1.3}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def write_mef(tmp_path, name, dtype):
    rng = np.random.default_rng(42)
    primary = fits.PrimaryHDU()
    primary.header["PRODCATG"] = "CAL.R-BIA"
    primary.header["READ-MDE"] = "SLOW"
    primary.header["REXPTIME"] = 10.0
    primary.header["SEXPTIME"] = 10.0
    hdus = [primary]
    for i in range(6):
        frame = rng.normal(700, 3, SHAPE)
        if i == 1:
            frame += rng.choice([-10, 0, 10], SHAPE[0])[:, None]    # banded -> structure FAIL
        if i == 2:
            frame[::7, ::11] = 65535                                 # hot pixels -> saturation WARN
        if i == 3:
            frame += np.linspace(0, 100, SHAPE[0])[:, None]          # glow -> warming WARN
        # integer-valued in both dtypes so the two frames hold identical pixels
        hdus.append(fits.ImageHDU(data=np.floor(np.clip(frame, 0, 65535)).astype(dtype)))
    hdus.append(fits.ImageHDU(data=np.ones(SHAPE, dtype=dtype)))      # placeholder
    path = tmp_path / name
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def strip_timings(report):
    return {key: value for key, value in report.items() if key not in ("timings", "workers")}


def test_threaded_run_matches_serial_and_float_path(tmp_path):
    int_frame = write_mef(tmp_path, "int_mef.fits", np.uint16)
    float_frame = write_mef(tmp_path, "float_mef.fits", np.float32)
    serial = QAEngine(CONFIG).run(int_frame, workers=1)
    threaded = QAEngine(CONFIG).run(int_frame, workers=4)
    default = QAEngine(CONFIG).run(int_frame)
    reference = QAEngine(CONFIG).run(float_frame)       # generic float path, same pixels

    assert default["workers"] == DEFAULT_WORKERS
    assert strip_timings(serial) == strip_timings(threaded) == strip_timings(default)
    for key in ("read", "metrics", "rules", "total"):
        assert key in threaded["timings"]

    assert serial["overall_verdict"] == reference["overall_verdict"] == "FAIL"
    serial_results = [(r["rule"], r["extension"], r["status"], r["verdict_effect"], r["measured_value"])
                      for r in serial["results"]]
    reference_results = [(r["rule"], r["extension"], r["status"], r["verdict_effect"], r["measured_value"])
                         for r in reference["results"]]
    assert serial_results == reference_results

    statuses = {(r["rule"], r["extension"]): r["status"] for r in serial["results"]}
    assert statuses[("level", "2.A.Red")] == "PLACEHOLDER"
    assert statuses[("level", "2.A.Green")] == "MISSING"
    fired = {(r["rule"], r["extension"]) for r in serial["results"]
             if r["status"] == "EVALUATED" and not r["passed"]}
    assert ("row_structure_gross", "1.A.Green") in fired
    assert ("saturation", "1.A.Blue") in fired
    assert ("warming", "1.B.Red") in fired
    assert serial["summary"]["placeholder_extensions"] == 1
    assert serial["summary"]["missing_extensions"] == 1


def test_short_exposure_skips_rate_rule_without_affecting_others(tmp_path):
    path = write_mef(tmp_path, "short_mef.fits", np.uint16)
    with fits.open(path, mode="update") as hdul:
        hdul[0].header["REXPTIME"] = 0.001
        hdul[0].header["SEXPTIME"] = 0.002
    report = QAEngine(CONFIG).run(path, workers=2)
    warming = [r for r in report["results"] if r["rule"] == "warming"]
    assert {r["status"] for r in warming if r["extension"] not in ("2.A.Red", "2.A.Green")} == {"SKIPPED"}
    assert report["overall_verdict"] == "FAIL"     # structure/saturation still evaluated


# ---------------------------------------------------------------- shared line profiles
def sorted_median(values, axis):
    return np.nanmedian(np.sort(np.asarray(values, dtype=float), axis=axis), axis=axis)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
@pytest.mark.parametrize("shape", [(32, 2048), (2048, 33), (10, 10), (3, 200), (7, 3)])
def test_line_profiles_with_median_equal_sorted_profiles(dtype, shape):
    from llamas_checks.qa_engine import line_profiles
    rng = np.random.default_rng(11)
    values = rng.integers(0, 65536, size=shape).astype(dtype)
    for axis in (-1, -2, 0, 1):
        trimmed, median = line_profiles(values, axis)
        with np.errstate(all="ignore"):
            np.testing.assert_array_equal(trimmed, sorted_profile(values, axis))
        np.testing.assert_array_equal(median, sorted_median(values, axis))
        only_trimmed, none = line_profiles(values, axis, median=False)
        assert none is None
        np.testing.assert_array_equal(only_trimmed, trimmed)


LAMP_CONFIG = {
    **CONFIG,
    "extensions": CONFIG["extensions"][:3],
    "metrics": {
        "row_banding": {"type": "row_structure"},
        "row_banding_norm": {"type": "row_structure_norm"},
        "smear": {"type": "vertical_smear", "smooth_columns": 5},
        "column_banding_norm": {"type": "column_structure_norm"},
    },
    "rule_sets": {
        "ARC": {
            "applies_when": {"exposure_type": "CAL.R-BIA"},
            "rules": [
                {"name": "row_structure", "region": "full_frame", "metric": "row_banding",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 1e6}},
                {"name": "row_structure_norm", "region": "full_frame", "metric": "row_banding_norm",
                 "per_extension": True, "severity": "WARN", "limits": {"max": 1e6}},
                {"name": "vertical_smear", "region": "full_frame", "metric": "smear",
                 "per_extension": True, "severity": "FAIL", "limits": {"max": 1e6}},
                {"name": "column_structure_norm", "region": "full_frame",
                 "metric": "column_banding_norm", "per_extension": True, "severity": "WARN",
                 "limits": {"max": 1e6}},
            ],
        }
    },
}


def write_lamp_mef(tmp_path):
    rng = np.random.default_rng(5)
    primary = fits.PrimaryHDU()
    primary.header["PRODCATG"] = "CAL.R-BIA"
    primary.header["READ-MDE"] = "SLOW"
    hdus = [primary]
    for i in range(3):
        frame = rng.normal(700, 3, SHAPE)
        frame[30:, :] += 400 + 100 * i                       # lamp signal above the stripe
        frame[:, 500 + 40 * i: 530 + 40 * i] += 150          # a halo some columns wide
        hdus.append(fits.ImageHDU(data=np.floor(frame).astype(np.uint16)))
    path = tmp_path / "lamp_mef.fits"
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def test_profile_metrics_share_one_partition_per_axis_and_ignore_threads(tmp_path, monkeypatch):
    import llamas_checks.qa_engine as E
    path = write_lamp_mef(tmp_path)
    calls = []
    original = E.line_profiles

    def counting(values, axis, *args, **kwargs):
        calls.append(axis)
        return original(values, axis, *args, **kwargs)

    monkeypatch.setattr(E, "line_profiles", counting)
    serial = QAEngine(LAMP_CONFIG, cameras_down=[]).run(path, workers=1)
    assert sorted(calls) == [-2, -2, -2, -1, -1, -1]          # one per (detector, axis)
    calls.clear()
    threaded_engine = QAEngine(LAMP_CONFIG, cameras_down=[])
    threaded = threaded_engine.run(path, workers=4)
    assert sorted(calls) == [-2, -2, -2, -1, -1, -1]
    assert strip_timings(serial) == strip_timings(threaded)
    evaluated = [r for r in serial["results"] if r["rule"] != "camera_present"]
    assert len(evaluated) == 12 and all(r["status"] == "EVALUATED" for r in evaluated)
    for entry in threaded_engine.last_run.extensions.values():
        assert set(entry.profiles) == {("full_frame", -1), ("full_frame", -2)}
        assert entry.profiles[("full_frame", -2)][1] is not None    # median placed for smear
        assert entry.profiles[("full_frame", -1)][1] is None        # no smear rule on rows
        assert len(entry.signals) == 1
