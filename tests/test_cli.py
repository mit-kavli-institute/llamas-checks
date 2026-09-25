"""End-to-end tests for the llamas-checks command-line entry points.

Runs the CLIs as subprocesses (``python -m llamas_checks`` and the config
validator) against a tiny synthetic MEF, so exit codes, stdout/stderr hygiene,
report layout and import hygiene are exercised the way the observing GUI sees
them. No external FITS fixtures are required.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from astropy.io import fits

from llamas_checks.llamasQATests import check_image
from llamas_checks.paths import CONFIG_DIR
from llamas_checks.qa_engine import QAEngineError, report_path_for

REPO_ROOT = Path(__file__).resolve().parents[1]
EXIT_FOR_STATUS = {"pass": 0, "warn": 1, "fail": 2}


def flat_frame(shape, level):
    """Uniform frame at ``level`` with a +/-0.5 ADU checkerboard on top.

    Real detectors always carry read noise, so tests never use a perfectly
    constant frame (a constant 0/1 frame is the engine's placeholder signature).
    The checkerboard is zero-mean in every row, column and block, so means,
    medians and background gradients are exactly those of the constant frame.
    """
    checker = np.indices(shape).sum(axis=0) % 2
    return (np.full(shape, level, dtype=np.float32)
            + np.where(checker, 0.5, -0.5).astype(np.float32))

CONFIG = {
    "config_version": "1.0",
    "instrument": {"name": "LLAMAS"},
    "metadata_keys": {"exposure_type": "PRODCATG", "readout_mode": "READ-MDE"},
    "extensions": [{"name": "1.A.Green", "hdu_index": 1, "color": "Green", "bench": 1, "side": "A"}],
    "regions": {"full_frame": {"type": "full"}},
    "metrics": {"mean": {"type": "mean"}},
    "lookup_tables": {"noop": {"a": 1}},  # unused but block is required + non-empty
    "rule_sets": {
        "BIAS": {
            "applies_when": {"exposure_type": "CAL.R-BIA"},
            "rules": [
                {"name": "shutter", "severity": "FAIL",
                 "header_check": {"op": "abs_or_rel_diff", "source": "SEXPTIME",
                                  "other": "REXPTIME", "abs_tol": 0.3, "rel_tol": 0.10}},
                {"name": "temp", "severity": "WARN", "per_extension": True,
                 "header_check": {"op": "range", "per_extension_key": "CCDTEMP_1",
                                  "limits": {"min": -140, "max": -60}}},
                {"name": "level", "region": "full_frame", "metric": "mean",
                 "per_extension": True, "severity": "FAIL",
                 "limits": {"min": 600, "max": 800}},
            ],
        }
    },
    "verdict_policy": {"fail_if_any_fail": True, "warn_if_any_warn": True},
}


def make_mef(tmp_path, name, *, rexp, sexp, ccdtemp, level=700.0, temp_keyword="CCDTEMP_1"):
    """Write a 1-detector MEF with the given primary/extension headers."""
    ph = fits.PrimaryHDU()
    ph.header["PRODCATG"] = "CAL.R-BIA"
    ph.header["READ-MDE"] = "SLOW"
    ph.header["REXPTIME"] = rexp
    ph.header["SEXPTIME"] = sexp
    ext = fits.ImageHDU(data=flat_frame((40, 40), level))
    ext.header["COLOR"] = "Green"
    ext.header["BENCH"] = 1
    ext.header["SIDE"] = "A"
    if ccdtemp is not None:
        ext.header[temp_keyword] = ccdtemp
    path = tmp_path / name
    fits.HDUList([ph, ext]).writeto(path, overwrite=True)
    return str(path)


def write_config(tmp_path):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(yaml.safe_dump(CONFIG))
    return str(cfg)


def run_cli(args, cwd):
    """Run ``python <args>`` with the repo root on PYTHONPATH and return the process."""
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + existing if existing else "")
    return subprocess.run([sys.executable, *args], cwd=str(cwd), env=env,
                          capture_output=True, text=True)


@pytest.fixture
def good_frame(tmp_path):
    return make_mef(tmp_path, "ok_mef.fits", rexp=0.001, sexp=0.003, ccdtemp="-90.0")


@pytest.fixture
def cfg(tmp_path):
    return write_config(tmp_path)


def test_module_entry_quiet_and_report(tmp_path, good_frame, cfg):
    out = tmp_path / "report.json"
    proc = run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", cfg, "--report", str(out),
                    "--report-all"], cwd=tmp_path)
    assert proc.stdout == "", proc.stdout
    assert proc.stderr == "", proc.stderr
    assert out.exists()
    report = json.loads(out.read_text())
    for key in ("status", "message", "suite", "fits_file", "structure", "report", "report_path"):
        assert key in report, f"missing key {key!r} in report: {sorted(report)}"
    assert report["report_path"] == str(out)
    assert proc.returncode == EXIT_FOR_STATUS[report["status"]], (proc.returncode, report["status"])
    structure = report["structure"]
    for key in ("n_extensions", "missing_cameras", "placeholder_extensions"):
        assert key in structure, f"missing structure key {key!r}: {sorted(structure)}"
    assert structure["n_extensions"] == 1


def test_no_sidecar_written_next_to_input(tmp_path, good_frame, cfg):
    out = tmp_path / "elsewhere" / "report.json"
    out.parent.mkdir()
    run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", cfg, "--report", str(out),
             "--report-all"], cwd=tmp_path)
    warn = make_mef(tmp_path, "warm_mef.fits", rexp=0.001, sexp=0.003, ccdtemp="-50.0")
    run_cli(["-m", "llamas_checks", warn, "--qa-yaml", cfg,
             "--report-dir", str(tmp_path / "elsewhere")], cwd=tmp_path)
    frame_dir = Path(good_frame).parent
    assert list(frame_dir.glob("*.qa.json")) == []
    assert (tmp_path / "elsewhere" / "warm_mef.qa.json").exists()


# ------- report gating: warn/fail only by default, --report-dir names after the frame -------

def test_pass_writes_no_report_by_default(tmp_path, good_frame, cfg):
    out = tmp_path / "pass.json"
    proc = run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", cfg, "--report", str(out)],
                   cwd=tmp_path)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stderr == "", proc.stderr
    assert not out.exists()


def test_pass_writes_report_with_report_all(tmp_path, good_frame, cfg):
    out = tmp_path / "pass.json"
    proc = run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", cfg, "--report", str(out),
                    "--report-all", "-v"], cwd=tmp_path)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert json.loads(out.read_text())["status"] == "pass"
    assert proc.stderr.strip().splitlines()[-1] == f"report: {out}"


def test_warn_report_dir_names_report_after_frame(tmp_path, cfg):
    warn = make_mef(tmp_path, "LLAMAS_2026-09-06_19-14-52.1_CAL22_mef.fits",
                    rexp=0.001, sexp=0.003, ccdtemp="-50.0")
    reports = tmp_path / "reports"
    proc = run_cli(["-m", "llamas_checks", warn, "--qa-yaml", cfg,
                    "--report-dir", str(reports), "-v"], cwd=tmp_path)
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    expected = reports / "LLAMAS_2026-09-06_19-14-52.1_CAL22_mef.qa.json"
    assert expected.exists(), sorted(reports.iterdir()) if reports.exists() else "no dir"
    assert expected.name == Path(warn).with_suffix(".qa.json").name
    payload = json.loads(expected.read_text())
    assert payload["status"] == "warn"
    assert payload["report_path"] == str(expected)
    assert "report" in payload
    assert f"report: {expected}" in proc.stderr


def test_bare_report_names_after_frame_in_cwd(tmp_path, cfg):
    warn = make_mef(tmp_path, "LLAMAS_2026-09-06_19-14-52.1_CAL22_mef.fits",
                    rexp=0.001, sexp=0.003, ccdtemp="-50.0")
    work = tmp_path / "work"
    work.mkdir()
    proc = run_cli(["-m", "llamas_checks", warn, "--qa-yaml", cfg, "--report"], cwd=work)
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert (work / "LLAMAS_2026-09-06_19-14-52.1_CAL22_mef.qa.json").exists()
    assert list(Path(warn).parent.glob("*.qa.json")) == []


def test_report_pointed_at_directory_names_after_frame(tmp_path, cfg):
    warn = make_mef(tmp_path, "warm_mef.fits", rexp=0.001, sexp=0.003, ccdtemp="-50.0")
    reports = tmp_path / "reports"
    reports.mkdir()
    proc = run_cli(["-m", "llamas_checks", warn, "--qa-yaml", cfg, "--report", str(reports)],
                   cwd=tmp_path)
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert (reports / "warm_mef.qa.json").exists()


def test_report_refuses_fits_path(tmp_path, good_frame, cfg):
    other = make_mef(tmp_path, "other_mef.fits", rexp=0.001, sexp=0.003, ccdtemp="-90.0")
    before = Path(other).read_bytes()
    proc = run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", cfg, "--report", other],
                   cwd=tmp_path)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    assert "refusing to overwrite" in proc.stderr
    assert Path(other).read_bytes() == before


def test_error_verdict_written_with_report_dir(tmp_path, good_frame):
    reports = tmp_path / "reports"
    proc = run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", oob_config(tmp_path),
                    "--report-dir", str(reports)], cwd=tmp_path)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert (reports / "ok_mef.qa.json").exists()


def test_pass_with_report_dir_creates_nothing(tmp_path, good_frame, cfg):
    reports = tmp_path / "reports"
    proc = run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", cfg,
                    "--report-dir", str(reports)], cwd=tmp_path)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert not reports.exists()


def test_report_and_report_dir_are_exclusive(tmp_path, good_frame, cfg):
    proc = run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", cfg,
                    "--report", str(tmp_path / "a.json"), "--report-dir", str(tmp_path)],
                   cwd=tmp_path)
    assert proc.returncode == 2
    assert "not allowed with" in proc.stderr
    with pytest.raises(QAEngineError):
        check_image(good_frame, qa_yaml=cfg, report=str(tmp_path / "a.json"),
                    report_dir=str(tmp_path))


def test_check_image_report_path_key(tmp_path, cfg, good_frame):
    reports = tmp_path / "reports"
    passed = check_image(good_frame, qa_yaml=cfg, report_dir=str(reports))
    assert passed["status"] == "pass"
    assert passed["report_path"] is None
    warn = make_mef(tmp_path, "warm_mef.fits", rexp=0.001, sexp=0.003, ccdtemp="-50.0")
    warned = check_image(warn, qa_yaml=cfg, report_dir=str(reports))
    assert warned["status"] == "warn"
    assert warned["report_path"] == str(reports / "warm_mef.qa.json")
    assert Path(warned["report_path"]).exists()


def test_report_path_for_matches_engine_sidecar_name(tmp_path):
    frame = tmp_path / "LLAMAS_2026-09-07_18-17-12.8_CAL22_mef.fits"
    assert report_path_for(frame) == tmp_path / "LLAMAS_2026-09-07_18-17-12.8_CAL22_mef.qa.json"
    assert report_path_for(frame, tmp_path / "out") == (
        tmp_path / "out" / "LLAMAS_2026-09-07_18-17-12.8_CAL22_mef.qa.json")


def test_shutter_fault_exits_fail_and_verbose_reports(tmp_path, cfg):
    bad = make_mef(tmp_path, "bad_mef.fits", rexp=600.0, sexp=181.0, ccdtemp="-90.0")
    proc = run_cli(["-m", "llamas_checks", bad, "--qa-yaml", cfg], cwd=tmp_path)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)

    proc_v = run_cli(["-m", "llamas_checks", bad, "--qa-yaml", cfg, "-v"], cwd=tmp_path)
    assert proc_v.returncode == 2
    assert proc_v.stderr != ""


def test_missing_input_is_system_error(tmp_path, cfg):
    proc = run_cli(["-m", "llamas_checks", str(tmp_path / "does_not_exist.fits"),
                    "--qa-yaml", cfg], cwd=tmp_path)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    lines = proc.stderr.splitlines()
    assert len(lines) == 1, proc.stderr
    assert lines[0].startswith("system error")


@pytest.mark.parametrize("name", ["qa_config_cal.yaml", "qa_config_science.yaml", "qa_config.yaml"])
def test_shipped_configs_validate(tmp_path, name):
    proc = run_cli(["-m", "llamas_checks.qa_config_validator", str(CONFIG_DIR / name)],
                   cwd=tmp_path)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)


def test_import_hygiene_no_pipeline_dependencies(tmp_path):
    # The pipeline package name is split so a repo-wide grep for it stays clean.
    snippet = (
        "import llamas_checks, sys; "
        "bad=[m for m in sys.modules if m.split('.')[0] in "
        "('ray','pypeit','llamas_'+'pyjamas','matplotlib','scipy')]; "
        "print(bad); sys.exit(1 if bad else 0)"
    )
    proc = run_cli(["-c", snippet], cwd=tmp_path)
    assert proc.returncode == 0, proc.stdout


# ------- llamas-checks-engine (python -m llamas_checks.qa_engine) -------

def oob_config(tmp_path):
    """CONFIG with a rectangle region larger than the 40x40 frame -> ERROR verdict."""
    cfg = json.loads(json.dumps(CONFIG))
    cfg["regions"]["oob"] = {"type": "rectangle", "x_start": 0, "x_end": 5000,
                             "y_start": 0, "y_end": 5000}
    cfg["rule_sets"]["BIAS"]["rules"] = [
        {"name": "oob_level", "region": "oob", "metric": "mean", "per_extension": True,
         "severity": "FAIL", "limits": {"min": 0, "max": 1e9}}]
    path = tmp_path / "oob.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return str(path)


def run_engine(args, cwd):
    return run_cli(["-m", "llamas_checks.qa_engine", *args, "--summary-only"], cwd=cwd)


def test_engine_pass_exits_0_and_writes_sidecar(tmp_path, good_frame, cfg):
    proc = run_engine([good_frame, "--config", cfg], cwd=tmp_path)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stderr == "", proc.stderr  # no RuntimeWarning from the package import
    assert proc.stdout.strip() == "ok_mef.fits PASS"
    sidecar = Path(good_frame).with_suffix(".qa.json")
    assert sidecar.exists()
    assert json.loads(sidecar.read_text())["overall_verdict"] == "PASS"


def test_engine_fail_exits_1(tmp_path, cfg):
    bad = make_mef(tmp_path, "bad_mef.fits", rexp=600.0, sexp=181.0, ccdtemp="-90.0")
    proc = run_engine([bad, "--config", cfg], cwd=tmp_path)
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout.strip() == "bad_mef.fits FAIL"


def test_engine_error_verdict_exits_2(tmp_path, good_frame):
    proc = run_engine([good_frame, "--config", oob_config(tmp_path)], cwd=tmp_path)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stdout.strip() == "ok_mef.fits ERROR"
    report = json.loads(Path(good_frame).with_suffix(".qa.json").read_text())
    assert report["overall_verdict"] == "ERROR"


def test_engine_batch_with_jobs_is_clean(tmp_path, cfg):
    frames = tmp_path / "frames"
    frames.mkdir()
    make_mef(frames, "a_mef.fits", rexp=0.001, sexp=0.003, ccdtemp="-90.0")
    make_mef(frames, "b_mef.fits", rexp=600.0, sexp=181.0, ccdtemp="-90.0")
    proc = run_engine([str(frames), "--config", cfg, "--jobs", "2"], cwd=tmp_path)
    assert proc.returncode == 1, (proc.returncode, proc.stdout, proc.stderr)
    assert proc.stderr == "", proc.stderr  # spawn workers must not re-import the package noisily
    assert sorted(p.name for p in frames.glob("*.qa.json")) == ["a_mef.qa.json", "b_mef.qa.json"]


def test_engine_bare_config_name_resolves_to_shipped_copy(tmp_path, good_frame):
    proc = run_engine([good_frame, "--config", "qa_config_cal.yaml"], cwd=tmp_path)
    report = json.loads(Path(good_frame).with_suffix(".qa.json").read_text())
    # The 40x40 frame is not the shipped geometry, so the verdict is not PASS; what
    # matters is that the bare name was found and the run reached the engine.
    assert "configuration file not found" not in json.dumps(report), report
    assert proc.returncode in (0, 1, 2)


def test_engine_wrong_path_to_shipped_name_errors(tmp_path, good_frame):
    missing = str(tmp_path / "no_such_dir" / "qa_config_cal.yaml")
    proc = run_engine([good_frame, "--config", missing], cwd=tmp_path)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    report = json.loads(Path(good_frame).with_suffix(".qa.json").read_text())
    assert "configuration file not found" in report["error"]


# ------- llamas-checks: ERROR verdict, config resolution, -v structure, exit 3 -------

def test_error_verdict_is_fail_with_message_and_report(tmp_path, good_frame):
    out = tmp_path / "err_report.json"
    result = check_image(good_frame, qa_yaml=oob_config(tmp_path), report=str(out))
    assert result["status"] == "fail"
    assert result["message"].startswith("QA could not be completed:")
    assert "oob" in result["message"]
    assert result["overall_verdict"] == "ERROR"
    assert out.exists()
    assert json.loads(out.read_text())["report"]["overall_verdict"] == "ERROR"


def test_error_verdict_through_cli_exits_2_not_3(tmp_path, good_frame):
    out = tmp_path / "err_cli.json"
    proc = run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", oob_config(tmp_path),
                    "--report", str(out), "-v"], cwd=tmp_path)
    assert proc.returncode == 2, (proc.returncode, proc.stdout, proc.stderr)
    assert "QA could not be completed" in proc.stderr
    assert out.exists()


def marked_config(path, marker):
    cfg = json.loads(json.dumps(CONFIG))
    cfg["rule_sets"]["BIAS"]["rules"][-1]["name"] = marker
    Path(path).write_text(yaml.safe_dump(cfg))


def rule_names(report_path):
    return {r["rule"] for r in json.loads(Path(report_path).read_text())["report"]["results"]}


def test_calib_root_default_suite_selects_prodcatg_config(tmp_path, good_frame):
    root = tmp_path / "root"
    root.mkdir()
    marked_config(root / "qa_config_cal.yaml", "from_calib_root")
    out = tmp_path / "root_report.json"
    proc = run_cli(["-m", "llamas_checks", good_frame, "--calib-root", str(root),
                    "--report", str(out), "--report-all"], cwd=tmp_path)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "from_calib_root" in rule_names(out)


def test_suite_name_selects_root_suite_yaml(tmp_path, good_frame):
    root = tmp_path / "root"
    root.mkdir()
    marked_config(root / "qa_config_cal.yaml", "from_calib_root")
    marked_config(root / "mysuite.yaml", "from_suite")
    out = tmp_path / "suite_report.json"
    proc = run_cli(["-m", "llamas_checks", good_frame, "--calib-root", str(root),
                    "--suite", "mysuite", "--report", str(out), "--report-all"], cwd=tmp_path)
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    assert "from_suite" in rule_names(out)
    assert "from_calib_root" not in rule_names(out)


def test_verbose_structure_line_lists_placeholder(tmp_path, good_frame, cfg):
    with fits.open(good_frame, mode="update") as hdul:
        hdul[1].data = np.ones((40, 40), dtype=np.int16)
    proc = run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", cfg, "-v"], cwd=tmp_path)
    structure_lines = [l for l in proc.stderr.splitlines() if l.startswith("structure:")]
    assert len(structure_lines) == 1, proc.stderr
    assert "placeholder: 1.A.Green" in structure_lines[0]


@pytest.mark.parametrize("kind", ["invalid_config", "yaml_syntax"])
def test_multiline_system_error_prints_one_line(tmp_path, good_frame, kind):
    bad = tmp_path / "bad.yaml"
    if kind == "invalid_config":
        bad.write_text(yaml.safe_dump({"config_version": "1.0"}))  # missing required blocks
    else:
        bad.write_text("rule_sets:\n  BIAS: [unclosed\n  other: {a: 1\n")
    proc = run_cli(["-m", "llamas_checks", good_frame, "--qa-yaml", str(bad)], cwd=tmp_path)
    assert proc.returncode == 3, (proc.returncode, proc.stdout, proc.stderr)
    lines = proc.stderr.splitlines()
    assert len(lines) == 1, proc.stderr
    assert lines[0].startswith("system error")


def test_help_shows_console_script_name(tmp_path):
    proc = run_cli(["-m", "llamas_checks", "--help"], cwd=tmp_path)
    assert proc.returncode == 0
    assert proc.stdout.startswith("usage: llamas-checks")
