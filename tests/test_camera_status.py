"""Camera presence check and the cameras-down allow-list.

A camera whose HDU is absent or whose image is a constant placeholder must FAIL
the ``camera_present`` check unless it is listed in ``camera_status.yaml`` (or
via ``--cameras-down`` / ``LLAMAS_CHECKS_CAMERAS_DOWN``). Frames are small
synthetic MEFs from ``test_validate.make_mef``.
"""
import copy

import pytest
import yaml

from llamas_checks import qa_engine as E
from llamas_checks.llamasQATests import check_image
from llamas_checks.paths import CONFIG_DIR
from llamas_checks.qa_engine import QAEngine, QAEngineError

from test_validate import CONFIG, full_extensions, make_mef, write_config

TWO = [e for e in full_extensions() if e["name"] in ("1.A.Red", "1.A.Blue")]


def two_camera_config():
    cfg = copy.deepcopy(CONFIG)
    cfg["extensions"] = copy.deepcopy(TWO)
    return cfg


# ---------------------------------------------------------------- list parsing / loading
@pytest.mark.parametrize("text, expected", [
    (None, []), ("", []), ("none", []), ("None", []), ("[]", []), ("  ", []),
    ("1.A.Blue", ["1.A.Blue"]),
    ("1.A.Blue,4.A.Blue", ["1.A.Blue", "4.A.Blue"]),
    ("1.A.Blue, 4.A.Blue ;2.B.Red", ["1.A.Blue", "4.A.Blue", "2.B.Red"]),
])
def test_parse_cameras_down(text, expected):
    assert E.parse_cameras_down(text) == expected


def test_shipped_camera_status_lists_no_cameras_down():
    assert E.DEFAULT_CAMERA_STATUS == CONFIG_DIR / "camera_status.yaml"
    assert E.load_cameras_down() == []


@pytest.mark.parametrize("content, expected", [
    ("cameras_down: []\n", []),
    ("cameras_down:\n", []),
    ("cameras_down: [1.A.Blue, 4.A.Blue]\n", ["1.A.Blue", "4.A.Blue"]),
    ("cameras_down:\n  - 1.A.Blue\n", ["1.A.Blue"]),
    ("cameras_down: 1.A.Blue, 4.A.Blue\n", ["1.A.Blue", "4.A.Blue"]),
    ("# only comments\n", []),
])
def test_load_cameras_down_accepts_the_documented_shapes(tmp_path, content, expected):
    path = tmp_path / "camera_status.yaml"
    path.write_text(content)
    assert E.load_cameras_down(path) == expected


@pytest.mark.parametrize("content", [
    "- 1.A.Blue\n",                       # a list, not a mapping
    "down: [1.A.Blue]\n",                 # wrong key
    "cameras_down: {a: 1}\n",             # not a list
    "cameras_down: [1, 2]\n",             # not strings
])
def test_load_cameras_down_rejects_malformed_files(tmp_path, content):
    path = tmp_path / "camera_status.yaml"
    path.write_text(content)
    with pytest.raises(QAEngineError, match="camera_status.yaml"):
        E.load_cameras_down(path)


def test_load_cameras_down_missing_file(tmp_path):
    with pytest.raises(QAEngineError, match="not found"):
        E.load_cameras_down(tmp_path / "nope.yaml")


def test_canonical_names_are_case_insensitive_and_in_detector_order():
    exts = full_extensions()
    assert E.canonical_cameras_down(["4.a.blue", "1.A.BLUE"], exts) == ["1.A.Blue", "4.A.Blue"]
    with pytest.raises(QAEngineError, match="unknown camera name\\(s\\) 1.A.Bleu"):
        E.canonical_cameras_down(["1.A.Bleu"], exts)


def test_resolve_precedence_cli_env_calib_root_default(tmp_path, monkeypatch):
    monkeypatch.delenv(E.CAMERAS_DOWN_ENV, raising=False)
    assert E.resolve_cameras_down(None, None) == ([], str(E.DEFAULT_CAMERA_STATUS))
    root = tmp_path / "root"
    root.mkdir()
    (root / "camera_status.yaml").write_text("cameras_down: [2.B.Red]\n")
    assert E.resolve_cameras_down(None, root) == (["2.B.Red"], str(root / "camera_status.yaml"))
    monkeypatch.setenv(E.CAMERAS_DOWN_ENV, "3.A.Green")
    assert E.resolve_cameras_down(None, root) == (["3.A.Green"], "env")
    assert E.resolve_cameras_down("none", root) == ([], "cli")
    assert E.resolve_cameras_down("1.A.Blue", root) == (["1.A.Blue"], "cli")
    # a calib root without its own file falls through to the shipped default
    monkeypatch.delenv(E.CAMERAS_DOWN_ENV)
    assert E.resolve_cameras_down(None, tmp_path) == ([], str(E.DEFAULT_CAMERA_STATUS))


# ---------------------------------------------------------------- engine results
def results_for(report, rule="camera_present"):
    return {r["extension"]: r for r in report["results"] if r["rule"] == rule}


def test_engine_fails_placeholder_camera_and_reports_list(tmp_path):
    report = QAEngine(two_camera_config(), cameras_down=[]).run(
        make_mef(tmp_path, "ph.fits", placeholder=(3,)))
    assert report["cameras_down"] == []
    present = results_for(report)
    assert set(present) == {"1.A.Red", "1.A.Blue"}
    assert present["1.A.Red"]["passed"] is True
    assert present["1.A.Red"]["measured_value"] == 1.0
    bad = present["1.A.Blue"]
    assert bad["status"] == "EVALUATED" and bad["passed"] is False
    assert bad["verdict_effect"] == "FAIL" and bad["severity"] == "FAIL"
    assert bad["measured_value"] == 0.0 and bad["limits"] == {"min": 1.0}
    assert "placeholder" in bad["message"] and "not listed as down" in bad["message"]
    assert report["overall_verdict"] == "FAIL"
    # the pixel rule on the placeholder is still skipped, not evaluated
    level = results_for(report, "level")
    assert level["1.A.Blue"]["status"] == "PLACEHOLDER"
    assert report["summary"]["placeholder_extensions"] == 1
    assert report["summary"]["total_checks"] == 4   # 2 level + 2 camera_present


def test_engine_fails_absent_hdu_unless_listed(tmp_path):
    frame = make_mef(tmp_path, "skip.fits", skip=(3,))
    cfg = two_camera_config()
    # With HDU 3 dropped the later HDUs shift down, so point 1.A.Blue past the
    # end of the file to model a camera whose HDU is absent.
    cfg["extensions"][1]["hdu_index"] = 99
    report = QAEngine(cfg, cameras_down=[]).run(frame)
    bad = results_for(report)["1.A.Blue"]
    assert bad["passed"] is False and "absent" in bad["message"]
    assert results_for(report, "level")["1.A.Blue"]["status"] == "MISSING"
    assert report["overall_verdict"] == "FAIL"

    report = QAEngine(cfg, cameras_down=["1.A.Blue"]).run(frame)
    ok = results_for(report)["1.A.Blue"]
    assert ok["passed"] is True and "listed as down" in ok["message"]
    assert report["cameras_down"] == ["1.A.Blue"]
    assert report["overall_verdict"] == "PASS"


def test_listed_camera_that_delivers_data_is_evaluated_without_warning(tmp_path):
    frame = make_mef(tmp_path, "live.fits")
    report = QAEngine(two_camera_config(), cameras_down=["1.A.Blue"]).run(frame)
    present = results_for(report)["1.A.Blue"]
    assert present["passed"] is True and present["verdict_effect"] == "PASS"
    assert "evaluated normally" in present["message"]
    assert results_for(report, "level")["1.A.Blue"]["status"] == "EVALUATED"
    assert not any(r["verdict_effect"] == "WARN" for r in report["results"])
    assert report["overall_verdict"] == "PASS"


def test_camera_check_runs_when_no_rule_set_matches(tmp_path):
    frame = make_mef(tmp_path, "ph.fits", placeholder=(3,))
    cfg = two_camera_config()
    cfg["rule_sets"]["BIAS"]["applies_when"]["exposure_type"] = "CAL.R-XXX"
    report = QAEngine(cfg, cameras_down=[]).run(frame)
    assert report["active_rule_sets"] == []
    assert results_for(report)["1.A.Blue"]["passed"] is False
    assert report["overall_verdict"] == "FAIL"


def test_engine_default_list_is_the_shipped_file(tmp_path):
    engine = QAEngine(two_camera_config())
    assert engine.cameras_down == []


def test_unknown_name_is_a_loud_error():
    with pytest.raises(QAEngineError, match="unknown camera name"):
        QAEngine(two_camera_config(), cameras_down=["1.A.Bleu"])


# ---------------------------------------------------------------- check_image wiring
def test_check_image_env_var_and_calib_root_file(tmp_path, monkeypatch):
    frame = make_mef(tmp_path, "ph.fits", placeholder=(3,))
    cfg_path = write_config(tmp_path, TWO)
    monkeypatch.delenv(E.CAMERAS_DOWN_ENV, raising=False)
    assert check_image(frame, qa_yaml=cfg_path)["status"] == "fail"

    monkeypatch.setenv(E.CAMERAS_DOWN_ENV, "1.A.Blue")
    result = check_image(frame, qa_yaml=cfg_path)
    assert result["status"] == "pass"
    assert result["structure"]["cameras_down_source"] == "env"
    # an explicit argument beats the environment
    assert check_image(frame, qa_yaml=cfg_path, cameras_down="none")["status"] == "fail"
    monkeypatch.delenv(E.CAMERAS_DOWN_ENV)

    root = tmp_path / "root"
    root.mkdir()
    (root / "camera_status.yaml").write_text("cameras_down: [1.A.Blue]\n")
    result = check_image(frame, qa_yaml=cfg_path, calib_root=str(root))
    assert result["status"] == "pass"
    assert result["structure"]["cameras_down_source"] == str(root / "camera_status.yaml")


def test_check_image_bad_list_raises_system_error(tmp_path):
    frame = make_mef(tmp_path, "ok.fits")
    cfg_path = write_config(tmp_path, TWO)
    with pytest.raises(QAEngineError, match="unknown camera name"):
        check_image(frame, qa_yaml=cfg_path, cameras_down="1.A.Bleu")
    root = tmp_path / "root"
    root.mkdir()
    (root / "camera_status.yaml").write_text("- 1.A.Blue\n")
    with pytest.raises(QAEngineError, match="camera_status.yaml"):
        check_image(frame, qa_yaml=cfg_path, calib_root=str(root))


def test_report_file_carries_camera_presence(tmp_path):
    frame = make_mef(tmp_path, "ph.fits", placeholder=(3,))
    out = tmp_path / "rep.json"
    result = check_image(frame, qa_yaml=write_config(tmp_path, TWO), report=str(out))
    payload = yaml.safe_load(out.read_text())
    assert payload["structure"]["cameras_down"] == []
    assert payload["report"]["cameras_down"] == []
    rows = [r for r in payload["report"]["results"] if r["rule"] == "camera_present"]
    assert [r["extension"] for r in rows] == ["1.A.Red", "1.A.Blue"]
    assert result["summary"]["fail_effects"] == 1
