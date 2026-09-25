#!/usr/bin/env python3
"""LLAMAS calibration QA entry point.

``check_image`` is the single function ``QA_assess.py`` calls. It:

  1. loads and validates the YAML rule definitions,
  2. inspects the MEF structure (header-only: extension count, missing
     cameras, header-identity mismatches) via ``validate.inspect_structure``,
  3. runs a header preflight (can the image type be identified?) before any
     type-specific pixel test is allowed to count,
  4. delegates the type-specific metric rules to ``qa_engine.QAEngine``,
  5. collapses the engine verdict into a simple pass/warn/fail result.

The structure report is attached as ``result["structure"]`` on every return
path (with ``placeholder_extensions`` taken from the engine's PLACEHOLDER
results). It is informational only and never changes ``status``.

Failure model:
- *System* problems (missing file/config, invalid config, astropy absent)
  raise, so the CLI reports a system error.
- *QA* problems (unidentifiable header, triggered rules) are returned as a
  "fail"/"warn" status, never raised.

Report files:
- The report is named after the frame, ``<frame>.qa.json`` (the same name the
  engine gives its sidecars), unless an explicit file is asked for: ``report``
  naming an existing directory (``"."`` for the current one) or ``report_dir``
  put the derived name inside that directory; any other ``report`` value is
  the file to write. ``report`` and ``report_dir`` are mutually exclusive, and
  a report path ending in a FITS suffix is refused so a frame is never overwritten.
- A report is written only when the status is "warn" or "fail" (which
  includes ERROR verdicts and unidentified frames); a passing frame writes
  nothing unless ``report_all`` is set. Nothing is written when a system
  problem raises. ``result["report_path"]`` says what was written (or None).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .paths import CAL_CONFIG_NAME, CONFIG_DIR, DEFAULT_CONFIG_NAME, SCIENCE_CONFIG_NAME
from .qa_config_validator import QAConfigValidator
from .qa_engine import (FITS_SUFFIXES, QAEngine, QAEngineError, fits as _fits, load_yaml,
                        report_path_for)
from .validate import inspect_structure

_VERDICT_TO_STATUS = {"PASS": "pass", "WARN": "warn", "FAIL": "fail"}


def check_image(
    input_path: str,
    suite: str = "basic_cal",
    qa_yaml: str | None = None,
    calib_root: str | None = None,
    report: str | None = None,
    verbose: bool = False,
    report_dir: str | None = None,
    report_all: bool = False,
) -> dict[str, Any]:
    """Run the QA suite on a single calibration or science FITS/MEF image.

    Returns a dict with at least ``status`` ("pass"|"warn"|"fail"),
    ``message`` and ``report_path`` (the JSON file written, or None).
    Raises ``QAEngineError`` on system-level problems.
    """
    image_path = Path(input_path).expanduser()
    if not image_path.is_file():
        raise QAEngineError(f"input is not a file: {image_path}")

    destination = _report_destination(image_path, report, report_dir)

    config_path = _resolve_config_path(qa_yaml, calib_root, suite, image_path)
    config = load_yaml(config_path)
    _validate_config(config, config_path)

    structure = _inspect_structure(image_path, config.get("extensions"))

    engine = QAEngine(config)
    try:
        engine_report = engine.run(image_path)
    except QAEngineError as exc:
        # Config is valid, so a runtime error means this image/header is unfit
        # for the rules it matched -> a QA failure, not a system error.
        if verbose:
            _print_structure(structure)
        return _result("fail", f"QA could not be completed: {exc}",
                       suite, image_path, destination, report_all, structure=structure)

    structure["placeholder_extensions"] = sorted(
        {r["extension"] for r in engine_report["results"] if r["status"] == "PLACEHOLDER"})
    if verbose:
        _print_structure(structure)

    # Header preflight: the image type must be identifiable from the header.
    if not engine_report["active_rule_sets"]:
        return _result("fail", _unidentified_message(engine_report, config),
                       suite, image_path, destination, report_all, engine_report, structure)

    # An unevaluable rule (region out of bounds, no finite pixels, unsupported
    # metric) gives an ERROR verdict: the frame could not be judged, so it is a
    # QA failure with the per-rule reasons, and the report is still written.
    if engine_report["overall_verdict"] == "ERROR":
        errors = [r["message"] for r in engine_report["results"] if r["status"] == "ERROR"]
        extra = "" if len(errors) <= 3 else f" (+{len(errors) - 3} more)"
        return _result("fail", "QA could not be completed: " + "; ".join(errors[:3]) + extra,
                       suite, image_path, destination, report_all, engine_report, structure)

    status = _VERDICT_TO_STATUS[engine_report["overall_verdict"]]
    message = _build_message(engine_report)
    if verbose:
        _print_details(engine_report)

    return _result(status, message, suite, image_path, destination, report_all,
                   engine_report, structure)


def _report_destination(image_path: Path, report: str | None,
                        report_dir: str | None) -> Path | None:
    """Where the JSON report would go, or None.

    ``<frame>.qa.json`` inside ``report_dir``, or inside ``report`` when that names
    an existing directory; otherwise ``report`` is the file itself.
    """
    if report and report_dir:
        raise QAEngineError("give either report or report_dir, not both")
    if report_dir:
        return report_path_for(image_path, Path(report_dir).expanduser())
    if not report:
        return None
    target = Path(report).expanduser()
    if target.is_dir():
        return report_path_for(image_path, target)
    if target.name.lower().endswith(FITS_SUFFIXES):
        raise QAEngineError(f"report path looks like a FITS file, refusing to overwrite it: {target}")
    return target


def _inspect_structure(image_path: Path, extensions: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Header-only structure report; a structural problem never raises here."""
    try:
        structure = inspect_structure(image_path, extensions)
    except Exception as exc:  # noqa: BLE001 - structure is informational only
        structure = {"n_extensions": None, "expected_extensions": None,
                     "missing_cameras": [], "identity_mismatches": [],
                     "error": f"{type(exc).__name__}: {exc}"}
    structure["placeholder_extensions"] = []
    return structure


def _print_structure(structure: dict[str, Any]) -> None:
    def fmt(items: list[Any]) -> str:
        return ", ".join(str(item) for item in items) if items else "none"

    mismatches = [f"{m['extension']}->{m['header_identity']}"
                  for m in structure["identity_mismatches"]]
    line = (f"structure: {structure['n_extensions']}/{structure['expected_extensions']} "
            f"extensions; missing: {fmt(structure['missing_cameras'])}; "
            f"placeholder: {fmt(structure['placeholder_extensions'])}; "
            f"identity mismatches: {fmt(mismatches)}")
    if structure.get("error"):
        line += f"; error: {structure['error']}"
    print(line, file=sys.stderr)


def _resolve_config_path(qa_yaml: str | None, calib_root: str | None, suite: str,
                         image_path: Path) -> Path:
    if qa_yaml:
        return Path(qa_yaml).expanduser()
    base = Path(calib_root).expanduser() if calib_root else CONFIG_DIR
    if suite:
        by_suite = base / f"{suite}.yaml"
        if by_suite.exists():
            return by_suite
    # Default path: auto-select the generated cal/science config from the frame's
    # PRODCATG so QA_assess runs the real per-type rules, not the stale base config.
    auto = _config_for_prodcatg(image_path, base)
    if auto is not None:
        return auto
    return base / DEFAULT_CONFIG_NAME


def _config_for_prodcatg(image_path: Path, base: Path) -> Path | None:
    """Pick the generated per-type config from the frame's primary-header PRODCATG.

    Returns the cal config for ``CAL*`` frames and the science config for ``SCI*``
    frames, or ``None`` when the type is unknown/missing, the config file is absent,
    or the FITS cannot be opened (the caller then falls back to the base config).
    """
    try:
        with _fits.open(image_path, memmap=False) as hdul:
            prodcatg = str(hdul[0].header.get("PRODCATG", "")).strip().upper()
    except Exception:
        return None
    if prodcatg.startswith("CAL"):
        candidate = base / CAL_CONFIG_NAME
    elif prodcatg.startswith("SCI"):
        candidate = base / SCIENCE_CONFIG_NAME
    else:
        return None
    return candidate if candidate.exists() else None


def _validate_config(config: dict[str, Any], config_path: Path) -> None:
    errors = QAConfigValidator(config, filename=str(config_path)).validate()
    if errors:
        details = "\n".join(f"  {error}" for error in errors)
        raise QAEngineError(f"invalid QA configuration: {config_path}\n{details}")


def _unidentified_message(engine_report: dict[str, Any], config: dict[str, Any]) -> str:
    """Explain why no rule set matched: missing keyword vs unknown type."""
    metadata = engine_report["metadata"]
    needed = sorted({key for rule_set in config["rule_sets"].values()
                     for key in rule_set["applies_when"]})
    missing = [key for key in needed if metadata.get(key) is None]
    if missing:
        return f"header missing identification keyword(s): {', '.join(missing)}"
    return f"unrecognised image type for header metadata {metadata}"


def _build_message(engine_report: dict[str, Any]) -> str:
    verdict = engine_report["overall_verdict"]
    summary = engine_report["summary"]
    if verdict == "PASS":
        return f"PASS: {summary['passed_checks']}/{summary['evaluated_checks']} checks passed"
    triggered = [result for result in engine_report["results"]
                 if result["status"] == "EVALUATED" and result["verdict_effect"] == verdict]
    shown = ", ".join(f"{result['rule']}@{result['extension']}" for result in triggered[:5])
    extra = "" if len(triggered) <= 5 else f" (+{len(triggered) - 5} more)"
    return f"{verdict}: {len(triggered)} {verdict.lower()} check(s): {shown}{extra}"


def _print_details(engine_report: dict[str, Any]) -> None:
    for result in engine_report["results"]:
        if result["status"] == "EVALUATED" and not result["passed"]:
            print(f"  {result['verdict_effect']} {result['rule']} @ "
                  f"{result['extension']}: {result['message']}", file=sys.stderr)


def _result(
    status: str,
    message: str,
    suite: str,
    image_path: Path,
    destination: Path | None,
    report_all: bool,
    engine_report: dict[str, Any] | None = None,
    structure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "message": message,
        "suite": suite,
        "fits_file": str(image_path),
        "structure": structure if structure is not None else {
            "n_extensions": None, "expected_extensions": None, "missing_cameras": [],
            "identity_mismatches": [], "placeholder_extensions": []},
        "report_path": None,
    }
    if engine_report is not None:
        result["overall_verdict"] = engine_report["overall_verdict"]
        result["summary"] = engine_report["summary"]

    # A passing frame needs no report: write only for warn/fail unless asked for
    # every frame. The directory is created on write, so a pass leaves no trace.
    if destination is not None and (report_all or status != "pass"):
        result["report_path"] = str(destination)
        payload = dict(result)
        if engine_report is not None:
            payload["report"] = engine_report
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    return result