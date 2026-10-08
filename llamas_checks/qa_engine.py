#!/usr/bin/env python3
"""Minimal FITS/MEF QA engine driven by the YAML configuration.
 
This engine assumes the YAML structure validated by qa_config_validator.py.
It validates the configuration before every run, reads FITS/MEF files,
extracts configured metadata, evaluates matching rule sets, and writes JSON
QA reports.
 
Input handling:
- If the input path is a FITS file, a single-file QA report is produced.
- If the input path is a directory, all FITS files in that directory are
  processed in batch mode using ProcessPoolExecutor.

Running:
- ``python -m llamas_checks.qa_engine <path> --config <yaml>`` or the installed
  ``llamas-checks-engine`` console script. Direct ``python qa_engine.py`` no
  longer works because the module uses package-relative imports.
- ``--config`` accepts a full path, or the bare name of a shipped config
  (e.g. ``qa_config_cal.yaml``) which is resolved against the package
  ``configs/`` directory.

Camera presence:
- Every configured camera must deliver data. A camera whose HDU is absent or
  whose image is a constant placeholder (every pixel 0 or 1 = not read out)
  FAILs the ``camera_present`` check unless it is listed in
  ``configs/camera_status.yaml`` (``cameras_down:``), the file observers edit
  when a camera is out for maintenance. ``--cameras-down`` / the
  ``LLAMAS_CHECKS_CAMERAS_DOWN`` environment variable override that file.

Output:
- Every processed FITS file gets a ``<stem>.qa.json`` sidecar written next to
  it (an error report if the file could not be evaluated).
- Exit code: PASS/WARN -> 0, FAIL -> 1, ERROR (unreadable file, bad config,
  off-mode frame, unevaluable rule) -> 2.
"""
 
from __future__ import annotations
 
import argparse
import concurrent.futures
import fnmatch
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
 
import numpy as np
import yaml
from astropy.io import fits

from .paths import CONFIG_DIR
from .qa_config_validator import QAConfigValidator


FITS_SUFFIXES = (
    ".fits",
    ".fit",
    ".fts",
)

# Row/column structure profiles use a trimmed mean: the lowest and highest
# STRUCTURE_TRIM_FRACTION of each row (column) are dropped before averaging, so a
# hot-pixel cluster or cosmic-ray trail (< 2 % of a 2048-pixel line) cannot move
# the profile, while banding, bars and glow gradients (which fill a line) still do.
STRUCTURE_TRIM_FRACTION = 0.02

# Constant-frame values that mark a software placeholder for a missing camera:
# 1.0 in real frames, 0.0 from the pipeline validator (validate.create_placeholder_hdu).
PLACEHOLDER_VALUES = (0.0, 1.0)

# Signal-normalised structure metrics: the row/column structure of a lamp frame
# (ThAr arc) is the line pattern itself and scales with the lamp signal, so the
# absolute metric encodes exposure time. These divide it by the frame's signal
# above the unilluminated edge stripe (full-region mean - background-region
# median), which makes 0.07 s and 1 s arcs comparable.
STRUCTURE_NORM_TYPES = ("row_structure_norm", "column_structure_norm")
DEFAULT_SIGNAL_REGION = "bottom_stripe"
DEFAULT_MIN_SIGNAL = 2.0   # ADU; below this the ratio is noise and the rule SKIPs
 
 
@dataclass
class RuleResult:
    rule_set: str
    rule: str
    extension: str | None
    hdu_index: int | None
    region: str
    metric: str
    measured_value: float | None
    passed: bool
    severity: str
    verdict_effect: str
    limits: dict[str, float] | None = None
    lookup_path: list[str] | None = None
    status: str = "EVALUATED"
    message: str = ""
 
 
class QAEngineError(RuntimeError):
    """Raised when the QA engine cannot complete evaluation."""


class LookupMissError(QAEngineError):
    """Raised when a lookup table has no entry for a detector/mode combination.

    Treated as a benign SKIP for that one rule+extension (e.g. an off-mode frame
    whose readout mode is not modelled), rather than an ERROR that aborts the file.
    """


# ---------------------------------------------------------------------------
# Cameras allowed to be down (missing HDU or placeholder image without a FAIL).
# The list lives in a small hand-edited YAML next to the generated QA configs so
# it survives regeneration and can be changed without touching code.
# ---------------------------------------------------------------------------
CAMERA_STATUS_NAME = "camera_status.yaml"
DEFAULT_CAMERA_STATUS = CONFIG_DIR / CAMERA_STATUS_NAME
CAMERAS_DOWN_ENV = "LLAMAS_CHECKS_CAMERAS_DOWN"
CAMERA_PRESENT_RULE = "camera_present"
_NO_CAMERAS_DOWN_WORDS = {"", "none", "[]"}


def parse_cameras_down(text: str | None) -> list[str]:
    """Split a ``--cameras-down`` / environment value into detector names.

    Comma, semicolon or whitespace separated; ``""``, ``none`` and ``[]`` mean
    no cameras down. Names are returned as written (validated later against the
    config's extension names).
    """
    if text is None:
        return []
    cleaned = text.strip()
    if cleaned.lower() in _NO_CAMERAS_DOWN_WORDS:
        return []
    for sep in (",", ";"):
        cleaned = cleaned.replace(sep, " ")
    return [part for part in cleaned.split() if part]


def load_cameras_down(path: Path | str | None = None) -> list[str]:
    """Read ``cameras_down:`` from a camera status YAML (the shipped one by default).

    The file must be a mapping whose ``cameras_down`` entry is a list of detector
    names (``null`` / ``[]`` for none). Anything else raises ``QAEngineError`` so
    a typo in the file is a loud system error rather than a silently ignored list.
    """
    status_path = Path(path) if path is not None else DEFAULT_CAMERA_STATUS
    if not status_path.is_file():
        raise QAEngineError(f"camera status file not found: {status_path}")
    try:
        with status_path.open("r", encoding="utf-8") as handle:
            content = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise QAEngineError(f"YAML syntax error in {status_path}: {exc}") from exc
    if content is None:  # comments only: nothing down
        return []
    if not isinstance(content, dict) or "cameras_down" not in content:
        raise QAEngineError(
            f"{status_path}: expected a mapping with a 'cameras_down' list"
        )
    names = content["cameras_down"]
    if names is None:
        return []
    if isinstance(names, str):
        names = parse_cameras_down(names)
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise QAEngineError(
            f"{status_path}: 'cameras_down' must be a list of detector names "
            "(e.g. [1.A.Blue, 4.A.Blue]) or []"
        )
    return [name.strip() for name in names if name.strip()]


def resolve_cameras_down(cli_value: str | None = None,
                         calib_root: str | Path | None = None) -> tuple[list[str], str]:
    """Pick the cameras-down list and say where it came from.

    Precedence: an explicit value (``--cameras-down``), then the
    ``LLAMAS_CHECKS_CAMERAS_DOWN`` environment variable, then
    ``<calib_root>/camera_status.yaml`` when a calib root is given and has one,
    then the shipped ``configs/camera_status.yaml``. Returns ``(names, source)``
    with ``source`` one of ``"cli"``, ``"env"`` or the file path read.
    """
    if cli_value is not None:
        return parse_cameras_down(cli_value), "cli"
    env_value = os.environ.get(CAMERAS_DOWN_ENV)
    if env_value is not None:
        return parse_cameras_down(env_value), "env"
    if calib_root is not None:
        candidate = Path(calib_root).expanduser() / CAMERA_STATUS_NAME
        if candidate.is_file():
            return load_cameras_down(candidate), str(candidate)
    return load_cameras_down(DEFAULT_CAMERA_STATUS), str(DEFAULT_CAMERA_STATUS)


def canonical_cameras_down(names: list[str] | tuple[str, ...] | set[str],
                           extensions: list[dict[str, Any]],
                           source: str = "cameras-down list") -> list[str]:
    """Map down-camera names onto the configured extension names (case-insensitive).

    Returns the canonical names in detector order. Unknown names raise
    ``QAEngineError`` naming them and the valid choices, so a misspelt camera can
    never silently forgive nothing.
    """
    by_key = {QAEngine.normalize(ext["name"]): ext["name"] for ext in extensions}
    unknown = sorted({name for name in names if QAEngine.normalize(name) not in by_key})
    if unknown:
        raise QAEngineError(
            f"{source}: unknown camera name(s) {', '.join(unknown)}; "
            f"valid names are {', '.join(ext['name'] for ext in extensions)}"
        )
    wanted = {QAEngine.normalize(name) for name in names}
    return [ext["name"] for ext in extensions if QAEngine.normalize(ext["name"]) in wanted]


class QAEngine:
    def __init__(self, config: dict[str, Any],
                 cameras_down: list[str] | tuple[str, ...] | set[str] | None = None):
        """``cameras_down``: detector names allowed to be missing or placeholder
        without failing; ``None`` reads the shipped ``configs/camera_status.yaml``.
        """
        self.config = config
        if cameras_down is None:
            cameras_down = load_cameras_down()
        self.cameras_down = canonical_cameras_down(cameras_down, config["extensions"])
        self._cameras_down_keys = {self.normalize(name) for name in self.cameras_down}

    @staticmethod
    def normalize(value: Any) -> str:
        """Normalize metadata values and lookup keys for robust comparison."""
        return str(value).strip().lower()
 
    def run(self, fits_path: Path) -> dict[str, Any]:
        # Per-run caches: one placeholder test per HDU and one measurement per
        # (HDU, region, metric). The WARN and FAIL tiers of a check (row_structure /
        # row_structure_gross, edge_background_level / edge_saturated, ...) differ
        # only in their limits, so they share the value instead of sorting or
        # partitioning the same 4-Mpix frame twice.
        self._placeholder_cache: dict[int, bool] = {}
        self._metric_cache: dict[tuple[int, str, str], Any] = {}
        # (hdu, region, background region) -> signal above the edge stripe, shared by
        # the row and column normalised-structure metrics of one detector.
        self._signal_cache: dict[tuple[int, str, str], float] = {}
        # memmap=False is required for FITS files containing BZERO/BSCALE/BLANK
        # keywords, because Astropy needs to scale the image data in memory.
        with fits.open(fits_path, memmap=False) as hdul:
            metadata = self._extract_metadata(hdul)
            active_rule_sets = self._select_rule_sets(metadata)
            # Camera presence is structural, so it is checked for every frame
            # type (even one no rule set matches). It reuses the per-HDU
            # placeholder test the pixel rules need anyway, so it adds no work.
            results: list[RuleResult] = self._camera_presence_results(hdul)

            if not active_rule_sets:
                results.append(self._no_rules_matched_result(metadata))

            for rule_set_name, rule_set in active_rule_sets.items():
                for rule in rule_set["rules"]:
                    if "header_check" in rule:
                        results.extend(
                            self._evaluate_header_rule(hdul, metadata, rule_set_name, rule)
                        )
                    elif rule.get("per_extension", False):
                        for extension in self.config["extensions"]:
                            results.append(
                                self._evaluate_rule_for_extension(
                                    hdul, metadata, rule_set_name, rule, extension
                                )
                            )
                    else:
                        extension = self._default_extension_for_non_per_extension_rule(hdul)
                        results.append(
                            self._evaluate_rule_for_extension(
                                hdul, metadata, rule_set_name, rule, extension
                            )
                        )
 
        verdict = self._final_verdict(results)
        return {
            "fits_file": str(fits_path),
            "instrument": self.config.get("instrument", {}),
            "metadata": metadata,
            "active_rule_sets": list(active_rule_sets.keys()),
            "cameras_down": list(self.cameras_down),
            "overall_verdict": verdict,
            "summary": self._summary(results),
            "results": [asdict(result) for result in results],
        }

    def _camera_presence_results(self, hdul: Any) -> list[RuleResult]:
        """One ``camera_present`` result per configured camera.

        A camera whose HDU is absent, has no image, or holds a placeholder image
        (constant 0/1 = not read out) FAILs unless it is in ``cameras_down``; a
        listed camera passes with a note, and a listed camera that did deliver
        data is simply evaluated like any other (no warning). ``measured_value``
        is 1 when data is present, 0 otherwise, against ``min: 1``.
        """
        results: list[RuleResult] = []
        for extension in self.config["extensions"]:
            name = extension.get("name")
            hdu_index = extension["hdu_index"]
            listed = self.normalize(name) in self._cameras_down_keys
            absent_reason: str | None = None
            if hdu_index >= len(hdul):
                absent_reason = f"HDU {hdu_index} is absent from the file"
            else:
                data = getattr(hdul[hdu_index], "data", None)
                if data is None:
                    absent_reason = f"HDU {hdu_index} has no image data"
                elif self._placeholder_for_hdu(hdu_index, data):
                    absent_reason = (f"HDU {hdu_index} is a placeholder image "
                                     "(constant frame, camera not read out)")
            if absent_reason is None:
                passed = True
                message = ("camera delivered data"
                           + ("; listed as down but evaluated normally" if listed else ""))
            elif listed:
                passed = True
                message = (f"camera {name} {absent_reason}; listed as down in "
                           f"{CAMERA_STATUS_NAME}, its checks are skipped")
            else:
                passed = False
                message = (f"camera {name} {absent_reason} and is not listed as down "
                           f"in {CAMERA_STATUS_NAME}")
            results.append(RuleResult(
                rule_set="STRUCTURE",
                rule=CAMERA_PRESENT_RULE,
                extension=name,
                hdu_index=hdu_index,
                region="full_frame",
                metric=CAMERA_PRESENT_RULE,
                measured_value=0.0 if absent_reason else 1.0,
                passed=passed,
                severity="FAIL",
                verdict_effect="PASS" if passed else "FAIL",
                limits={"min": 1.0},
                lookup_path=None,
                status="EVALUATED",
                message=message,
            ))
        return results
 
    def _extract_metadata(self, hdul: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        for internal_name, fits_keyword in self.config["metadata_keys"].items():
            raw_value = self._find_header_value(hdul, fits_keyword)
            if raw_value is None:
                metadata[internal_name] = None
            elif isinstance(raw_value, str):
                metadata[internal_name] = self.normalize(raw_value)
            else:
                metadata[internal_name] = raw_value
        return metadata
 
    @staticmethod
    def _find_header_value(hdul: Any, keyword: str) -> Any | None:
        """Find a FITS header value, checking primary first, then extensions.
 
        Header keyword names are intentionally not normalized. Values are
        normalized later only when they are strings.
        """
        for hdu in hdul:
            if keyword in hdu.header:
                return hdu.header[keyword]
        return None
 
    def _select_rule_sets(self, metadata: dict[str, Any]) -> dict[str, Any]:
        selected: dict[str, Any] = {}
        for name, rule_set in self.config["rule_sets"].items():
            if self._applies_when_matches(rule_set["applies_when"], metadata):
                selected[name] = rule_set
        return selected
 
    @staticmethod
    def _match(value: Any, pattern: Any) -> bool:
        value_s = str(value).strip().lower()
        pattern_s = str(pattern).strip().lower()
        return fnmatch.fnmatch(value_s, pattern_s)
 
    def _applies_when_matches(self, applies_when: dict[str, Any], metadata: dict[str, Any]) -> bool:
        for field, expected in applies_when.items():
            actual = metadata.get(field)
            if actual is None:
                return False
            if isinstance(actual, str) or isinstance(expected, str):
                if not self._match(self.normalize(actual), self.normalize(expected)):
                    return False
            elif actual != expected:
                return False
        return True
 
    def _default_extension_for_non_per_extension_rule(self, hdul: Any) -> dict[str, Any]:
        # Current YAML uses per_extension rules. This fallback keeps the engine
        # deterministic if a future rule omits per_extension or sets it false.
        for extension in self.config["extensions"]:
            hdu_index = extension["hdu_index"]
            if hdu_index >= len(hdul):
                continue
            data = getattr(hdul[hdu_index], "data", None)
            if data is not None and not self._placeholder_for_hdu(hdu_index, data):
                return extension
        raise QAEngineError("no configured extension with usable image data is available")

    # ---------------------------------------------------------------------
    # Header-value rules (shutter/EXPTIME, CCD temperature)
    # ---------------------------------------------------------------------
    def _evaluate_header_rule(
        self,
        hdul: Any,
        metadata: dict[str, Any],
        rule_set_name: str,
        rule: dict[str, Any],
    ) -> list[RuleResult]:
        """Evaluate a ``header_check`` rule.

        Header rules check FITS header values rather than image-region metrics.
        A non-``per_extension`` rule (e.g. shutter REXPTIME vs SEXPTIME) is
        evaluated once from the file's global header. A ``per_extension`` rule
        (e.g. per-detector CCD temperature) is evaluated once per configured
        extension, reading the keyword from that extension's own header. Absent
        keywords are reported with status ``SKIPPED`` and never affect the
        verdict, because keywords such as CCDTEMP_1 are only intermittently
        populated. ``metadata`` is threaded through so a ``range`` check can
        resolve per-detector limits from a lookup table (``expected_from_lookup``).
        """
        if rule.get("per_extension", False):
            return [self._eval_header_one(hdul, metadata, rule_set_name, rule, extension)
                    for extension in self.config["extensions"]]
        return [self._eval_header_one(hdul, metadata, rule_set_name, rule, None)]

    def _eval_header_one(
        self,
        hdul: Any,
        metadata: dict[str, Any],
        rule_set_name: str,
        rule: dict[str, Any],
        extension: dict[str, Any] | None,
    ) -> RuleResult:
        hc = rule["header_check"]
        op = hc["op"]
        severity = rule["severity"]
        ext_name = extension.get("name") if extension else None
        hdu_index = extension.get("hdu_index") if extension else None

        def skipped(msg: str) -> RuleResult:
            return RuleResult(
                rule_set=rule_set_name, rule=rule["name"], extension=ext_name,
                hdu_index=hdu_index, region="header", metric=f"header:{op}",
                measured_value=None, passed=True, severity=severity,
                verdict_effect="SKIPPED", limits=None, lookup_path=None,
                status="SKIPPED", message=msg,
            )

        # For per-extension keys (e.g. CCDTEMP_1), resolve this extension's header.
        ext_header = None
        if extension is not None:
            if hdu_index is None or hdu_index >= len(hdul):
                return skipped(f"extension {ext_name!r} references missing HDU index {hdu_index}")
            ext_header = hdul[hdu_index].header

        # --- single-value range check ---
        if op == "range":
            if "per_extension_key" in hc:
                # per_extension_key may be a single keyword or a list of spelling
                # variants (e.g. CCDTEMP_1 / CCDTEMP1 / CCDTEMP-1 differ across
                # file generations); use the first one populated in this header.
                keys = hc["per_extension_key"]
                keys = [keys] if isinstance(keys, str) else list(keys)
                key, raw = keys[0], None
                if ext_header is not None:
                    for candidate in keys:
                        if ext_header.get(candidate) is not None:
                            key, raw = candidate, ext_header.get(candidate)
                            break
            else:
                key = hc["source"]
                raw = self._find_header_value(hdul, key)
            value = self._to_number(raw)
            if value is None:
                return skipped(f"header keyword {key!r} absent or non-numeric")
            # Limits come either from a static block (hc["limits"]) or, for
            # per-detector checks (e.g. per-camera temperature caps), from a
            # lookup table via the rule's expected_from_lookup.
            if "expected_from_lookup" in rule:
                if extension is None:
                    return skipped("expected_from_lookup requires a per_extension rule")
                try:
                    limits, _ = self._resolve_rule_limits(rule, metadata, extension)
                except LookupMissError:
                    return skipped(f"no lookup entry for extension {ext_name!r}")
            else:
                limits = {k: float(v) for k, v in hc["limits"].items()}
            passed, message = self._check_limits(value, limits)
            return self._header_result(rule_set_name, rule, op, ext_name, hdu_index,
                                       f"header:{key}", value, passed, severity,
                                       limits, message)

        # --- difference checks between two global keywords ---
        a = self._to_number(self._find_header_value(hdul, hc["source"]))
        b = self._to_number(self._find_header_value(hdul, hc["other"]))
        if a is None or b is None:
            return skipped(f"header keyword {hc['source']!r} or {hc['other']!r} "
                           "absent or non-numeric")
        absdiff = abs(a - b)
        reldiff = absdiff / abs(b) if b != 0 else float("inf")
        abs_tol = hc.get("abs_tol")
        rel_tol = hc.get("rel_tol")
        if op == "abs_diff":
            passed = absdiff <= float(abs_tol)
            message = (f"|{hc['source']}-{hc['other']}|={absdiff:.4g} "
                       f"{'<=' if passed else '>'} abs_tol {abs_tol}")
        elif op == "rel_diff":
            passed = reldiff <= float(rel_tol)
            message = (f"rel|{hc['source']}-{hc['other']}|={reldiff:.4g} "
                       f"{'<=' if passed else '>'} rel_tol {rel_tol}")
        else:  # abs_or_rel_diff: passes if within EITHER tolerance
            passed = (absdiff <= float(abs_tol)) or (reldiff <= float(rel_tol))
            message = (f"absdiff={absdiff:.4g} (abs_tol {abs_tol}), "
                       f"reldiff={reldiff:.4g} (rel_tol {rel_tol}) -> "
                       f"{'within tolerance' if passed else 'exceeds both'}")
        limits = {k: float(v) for k, v in
                  (("abs_tol", abs_tol), ("rel_tol", rel_tol)) if v is not None}
        return self._header_result(rule_set_name, rule, op, ext_name, hdu_index,
                                   f"header:{hc['source']}-{hc['other']}", absdiff,
                                   passed, severity, limits, message)

    def _header_result(self, rule_set_name, rule, op, ext_name, hdu_index,
                       region_label, value, passed, severity, limits, message) -> RuleResult:
        return RuleResult(
            rule_set=rule_set_name, rule=rule["name"], extension=ext_name,
            hdu_index=hdu_index, region=region_label, metric=f"header:{op}",
            measured_value=round(float(value), 6), passed=passed, severity=severity,
            verdict_effect="PASS" if passed else severity, limits=limits,
            lookup_path=None, status="EVALUATED", message=message,
        )

    @staticmethod
    def _to_number(raw: Any) -> float | None:
        """Coerce a header value (number or numeric string) to float, else None.

        Empty strings, astropy Undefined cards, and non-numeric values return None
        so that intermittently-populated keywords (e.g. CCDTEMP_1) are skipped, not
        treated as failures.
        """
        try:
            if raw is None or (isinstance(raw, str) and not raw.strip()):
                return None
            return float(raw)
        except (TypeError, ValueError):
            return None
 
    def _placeholder_for_hdu(self, hdu_index: int, data: Any) -> bool:
        """``_is_placeholder_data`` evaluated once per HDU per run."""
        cache = getattr(self, "_placeholder_cache", None)
        if cache is None:
            return self._is_placeholder_data(data)
        if hdu_index not in cache:
            cache[hdu_index] = self._is_placeholder_data(data)
        return cache[hdu_index]

    @staticmethod
    def _is_placeholder_data(data: Any) -> bool:
        """Return True for software-generated placeholder images.

        Placeholder convention: a constant finite frame at one of the two
        known placeholder values -- every finite pixel is exactly 1 (real
        frames) or exactly 0 (the pipeline validator's ``create_placeholder_hdu``).
        Any other constant frame (e.g. a railed detector at the ADC ceiling, or
        a dead readout at a non-zero pedestal) is *not* a placeholder and is
        evaluated normally, so saturation/structure rules can still fire on it.
        Placeholder extensions are skipped by the metric rules (status
        PLACEHOLDER); whether they fail the frame is decided once per camera by
        the ``camera_present`` check against ``cameras_down``.
        """
        array = np.asarray(data)
        if array.size == 0:
            return False

        if np.issubdtype(array.dtype, np.integer):
            # Integer pixels (raw frames are uint16) are always finite: no mask copy.
            min_value, max_value = float(array.min()), float(array.max())
        else:
            finite_values = array[np.isfinite(array)]
            if finite_values.size == 0:
                return False
            min_value = float(np.min(finite_values))
            max_value = float(np.max(finite_values))
        return min_value == max_value and min_value in PLACEHOLDER_VALUES
 
    def _skipped_rule_result(
        self,
        rule_set_name: str,
        rule: dict[str, Any],
        extension: dict[str, Any],
        status: str,
        message: str,
    ) -> RuleResult:
        return RuleResult(
            rule_set=rule_set_name,
            rule=rule["name"],
            extension=extension.get("name"),
            hdu_index=extension.get("hdu_index"),
            region=self._resolve_rule_region_name(rule, extension),
            metric=rule["metric"],
            measured_value=None,
            passed=True,
            severity=rule["severity"],
            verdict_effect=status,
            limits=None,
            lookup_path=None,
            status=status,
            message=message,
        )

    def _error_rule_result(
        self,
        rule_set_name: str,
        rule: dict[str, Any],
        extension: dict[str, Any],
        message: str,
    ) -> RuleResult:
        """A rule that could not be evaluated due to an unexpected error.

        verdict_effect='ERROR' propagates to the overall verdict (never silently
        passes), but only this one rule+extension is affected.
        """
        return RuleResult(
            rule_set=rule_set_name,
            rule=rule["name"],
            extension=extension.get("name") if extension else None,
            hdu_index=extension.get("hdu_index") if extension else None,
            region=rule.get("region", "unknown"),
            metric=rule.get("metric", "unknown"),
            measured_value=None,
            passed=False,
            severity=rule.get("severity", "FAIL"),
            verdict_effect="ERROR",
            limits=None,
            lookup_path=None,
            status="ERROR",
            message=message,
        )

    def _no_rules_matched_result(self, metadata: dict[str, Any]) -> RuleResult:
        """Surface a frame that no rule set matched (mis-routed / bad PRODCATG).

        WARN severity so it is visible in the report and verdict rather than a
        silent PASS with zero checks.
        """
        return RuleResult(
            rule_set="(none)", rule="no_rules_matched", extension=None, hdu_index=None,
            region="n/a", metric="n/a", measured_value=None, passed=False,
            severity="WARN", verdict_effect="WARN", limits=None, lookup_path=None,
            status="NO_RULES_MATCHED",
            message=f"no rule set matched this frame's metadata: {metadata}",
        )

    def _resolve_rule_region_name(
        self,
        rule: dict[str, Any],
        extension: dict[str, Any],
    ) -> str:
        """Resolve the region name for a rule and extension."""
        if "region" in rule:
            return rule["region"]
 
        region_by_extension = rule["region_by_extension"]
        key = region_by_extension["key"]
        values = region_by_extension.get("values", {})
        default_region = region_by_extension.get("default")
 
        extension_value = extension.get(key)
        if extension_value is None:
            if default_region is not None:
                return default_region
            raise QAEngineError(
                f"extension {extension.get('name')!r} has no field {key!r} "
                f"required by rule {rule.get('name')!r}"
            )
 
        normalized_extension_value = self.normalize(extension_value)
 
        for configured_value, region_name in values.items():
            if self.normalize(configured_value) == normalized_extension_value:
                return region_name
 
        if default_region is not None:
            return default_region
 
        raise QAEngineError(
            f"rule {rule.get('name')!r} has no region mapping for "
            f"extension {extension.get('name')!r} field {key!r}={extension_value!r}"
        )
 
    def _evaluate_rule_for_extension(
        self,
        hdul: Any,
        metadata: dict[str, Any],
        rule_set_name: str,
        rule: dict[str, Any],
        extension: dict[str, Any],
    ) -> RuleResult:
        hdu_index = extension["hdu_index"]
        extension_name = extension.get("name")
 
        if hdu_index >= len(hdul):
            return self._skipped_rule_result(
                rule_set_name,
                rule,
                extension,
                status="MISSING",
                message=f"extension {extension_name!r} references missing HDU index {hdu_index}",
            )
 
        data = getattr(hdul[hdu_index], "data", None)
        if data is None:
            return self._skipped_rule_result(
                rule_set_name,
                rule,
                extension,
                status="MISSING",
                message=f"extension {extension_name!r} / HDU {hdu_index} has no image data",
            )
 
        if self._placeholder_for_hdu(hdu_index, data):
            return self._skipped_rule_result(
                rule_set_name,
                rule,
                extension,
                status="PLACEHOLDER",
                message=(
                    f"extension {extension_name!r} / HDU {hdu_index} "
                    "appears to be a placeholder image"
                ),
            )
 
        region_name = self._resolve_rule_region_name(rule, extension)
        metric_name = rule["metric"]
        severity = rule["severity"]
        # Isolate per-rule failures: a lookup miss (e.g. an off-mode frame whose
        # readout mode is unmodelled) SKIPs just this rule+extension; any other
        # evaluation error becomes an ERROR result for this rule+extension only —
        # neither aborts QA for the rest of the 24 detectors.
        try:
            measured_value = self._measure(hdul, data, hdu_index, region_name,
                                           metric_name, extension_name)
            limits, lookup_path = self._resolve_rule_limits(rule, metadata, extension)
        except LookupMissError as exc:
            return self._skipped_rule_result(rule_set_name, rule, extension,
                                             status="SKIPPED", message=str(exc))
        except QAEngineError as exc:
            return self._error_rule_result(rule_set_name, rule, extension, str(exc))
        passed, message = self._check_limits(measured_value, limits)
        verdict_effect = "PASS" if passed else severity
 
        return RuleResult(
            rule_set=rule_set_name,
            rule=rule["name"],
            extension=extension_name,
            hdu_index=hdu_index,
            region=region_name,
            metric=metric_name,
            measured_value=round(float(measured_value), 6),
            passed=passed,
            severity=severity,
            verdict_effect=verdict_effect,
            limits=limits,
            lookup_path=lookup_path,
            status="EVALUATED",
            message=message,
        )
 
    def _measure(
        self,
        hdul: Any,
        data: Any,
        hdu_index: int,
        region_name: str,
        metric_name: str,
        extension_name: str | None,
    ) -> float:
        """Metric value for one HDU / region / metric, computed once per run.

        Rules that differ only in severity and limits share the measurement. An
        evaluation failure (LookupMissError / QAEngineError) is cached as well, so
        every rule on that measurement reports the same SKIP or ERROR.
        """
        cache = getattr(self, "_metric_cache", None)
        key = (hdu_index, region_name, metric_name)
        if cache is not None and key in cache:
            outcome = cache[key]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        try:
            region = self.config["regions"][region_name]
            metric = self.config["metrics"][metric_name]
            region_data = self._extract_region(np.asarray(data), region, region_name, extension_name)
            metric_type = metric.get("type")
            if metric_type == "background_gradient_rate":
                # rate metrics need the exposure time; absent/too-short -> SKIP this rule
                exptime = self._resolve_exptime(hdul, metric)
                value = self._compute_metric(region_data, metric, exptime=exptime)
            elif metric_type in STRUCTURE_NORM_TYPES:
                # normalised structure needs the lamp signal; too faint -> SKIP this rule
                signal = self._resolve_signal(data, hdu_index, region_name, region_data,
                                              metric, extension_name)
                value = self._compute_metric(region_data, metric, signal=signal)
            else:
                value = self._compute_metric(region_data, metric)
        except QAEngineError as exc:
            if cache is not None:
                cache[key] = exc
            raise
        if cache is not None:
            cache[key] = value
        return value

    @staticmethod
    def _extract_region(
        data: Any,
        region: dict[str, Any],
        region_name: str,
        extension_name: str | None,
    ) -> Any:
        if data.ndim < 2:
            raise QAEngineError(f"extension {extension_name!r} does not contain a 2D image")
 
        if region["type"] == "full":
            return data
 
        if region["type"] != "rectangle":
            raise QAEngineError(f"unsupported region type {region['type']!r} "
                                f"in region {region_name!r}")
 
        x_start = region["x_start"]
        x_end = region["x_end"]
        y_start = region["y_start"]
        y_end = region["y_end"]
 
        height, width = data.shape[-2], data.shape[-1]
        if not (0 <= x_start < x_end <= width and 0 <= y_start < y_end <= height):
            raise QAEngineError(
                f"region {region_name!r} is outside image bounds for extension {extension_name!r}: "
                f"region x=[{x_start}:{x_end}], y=[{y_start}:{y_end}], "
                f"image width={width}, height={height}"
            )
        return data[..., y_start:y_end, x_start:x_end]
 
    def _resolve_signal(self, data: Any, hdu_index: int, region_name: str, region_data: Any,
                        metric: dict[str, Any], extension_name: str | None) -> float:
        """Lamp signal for a normalised-structure metric: mean of the metric's region
        minus the median of ``metric['background_region']`` (default bottom_stripe),
        computed once per HDU per run. Below ``metric['min_signal']`` (default 2 ADU)
        raises LookupMissError so the rule is SKIPPED, like a too-short exposure for
        the gradient-rate metric: the ratio of two noise terms would be meaningless.
        """
        background_region = metric.get("background_region", DEFAULT_SIGNAL_REGION)
        min_signal = float(metric.get("min_signal", DEFAULT_MIN_SIGNAL))
        cache = getattr(self, "_signal_cache", None)
        key = (hdu_index, region_name, background_region)
        if cache is not None and key in cache:
            signal = cache[key]
        else:
            background = self.config["regions"].get(background_region)
            if background is None:
                raise QAEngineError(f"background_region {background_region!r} of a normalised "
                                    "structure metric is not a configured region")
            background_data = self._extract_region(np.asarray(data), background,
                                                   background_region, extension_name)
            signal = (self._compute_metric(region_data, {"type": "mean"})
                      - self._compute_metric(background_data, {"type": "median"}))
            if cache is not None:
                cache[key] = signal
        if not np.isfinite(signal) or signal < min_signal:
            raise LookupMissError(
                f"signal {signal:.4g} ADU (region mean minus {background_region} median) is "
                f"below min_signal {min_signal:g}; structure not normalised for this detector")
        return signal

    def _resolve_exptime(self, hdul: Any, metric: dict[str, Any]) -> float:
        """Exposure time (s) for a rate metric: first populated positive value among
        ``metric['exptime_keys']`` (default SEXPTIME -> REXPTIME -> EXPTIME; SEXPTIME
        is the actual shutter-open time, the correct integration for dark current).
        Absent or below ``min_exptime`` raises LookupMissError -> the rule is SKIPPED
        (a rate is undefined/noise-dominated for a ~0 s frame), never failed."""
        keys = metric.get("exptime_keys", ["SEXPTIME", "REXPTIME", "EXPTIME"])
        min_exptime = float(metric.get("min_exptime", 1.0))
        for key in keys:
            value = self._to_number(self._find_header_value(hdul, key))
            if value is not None and value >= min_exptime:
                return float(value)
        raise LookupMissError(f"no usable exposure time ({', '.join(keys)}) "
                              f">= {min_exptime}s for gradient-rate metric")

    @staticmethod
    def _compute_metric(region_data: Any, metric: dict[str, Any],
                        exptime: float | None = None, signal: float | None = None) -> float:
        """Evaluate ``metric`` on ``region_data``. ``exptime`` is required by the
        gradient-rate metric, ``signal`` (ADU above the edge stripe, see
        ``_resolve_signal``) by the normalised structure metrics."""
        raw = np.asarray(region_data)
        integer_input = np.issubdtype(raw.dtype, np.integer)
        if integer_input:
            # Raw frames are uint16, so every pixel is finite: reduce the integer
            # array directly instead of making a float64 copy and a finite-mask
            # copy (two 32 MB copies per rule on a 2048x2048 detector). numpy
            # accumulates integer means/medians/std in float64, so values match.
            values = raw
            finite_values = raw.ravel()
        else:
            values = np.asarray(raw, dtype=float)
            finite_values = values[np.isfinite(values)]
        if finite_values.size == 0:
            raise QAEngineError("metric cannot be computed because the selected "
                                "region contains no finite pixels")
 
        metric_type = metric["type"]

        if metric_type == "background_gradient_rate":
            # Camera-warming signal. A warming detector grows a dark-current glow --
            # a smooth large-scale background gradient -- at a rate (ADU/s) far above
            # the slow sky/scattered-light gradient of a normal frame. Measure the
            # gradient with quarter-block MEDIANS (robust to the sparse bright fiber/
            # target flux, so it tracks the background, not the astrophysical signal)
            # and divide by exposure time to isolate the dark-current RATE -- which
            # separates a warming detector (~2 ADU/s) from bright/long science (whose
            # background gradient accrues at <=~0.06 ADU/s). See QA_TESTS_SUMMARY.md.
            if values.ndim < 2:
                raise QAEngineError("background_gradient_rate requires a 2D region")
            if not exptime or exptime <= 0:
                raise QAEngineError("background_gradient_rate requires a positive exposure time")
            ny, nx = values.shape[-2], values.shape[-1]
            qy, qx = max(ny // 4, 1), max(nx // 4, 1)
            v_grad = abs(np.nanmedian(values[..., :qy, :]) - np.nanmedian(values[..., -qy:, :]))
            h_grad = abs(np.nanmedian(values[..., :, :qx]) - np.nanmedian(values[..., :, -qx:]))
            return float(max(v_grad, h_grad) / exptime)

        if metric_type == "mean":
            return float(np.mean(finite_values))
        
        if metric_type == "median":
            if integer_input and raw.dtype.itemsize <= 2:
                return float(integer_median(finite_values))
            return float(np.median(finite_values))
        
        if metric_type == "std":
            return float(np.std(finite_values))

        if metric_type == "robust_std":
            # Read-noise monitor: 1.4826 x MAD. A plain std on a 4-Mpix frame is
            # dominated by a handful of hot/saturated pixels or a cosmic-ray
            # cluster (20 railed pixels alone give std ~150 ADU); the MAD is not.
            if integer_input and raw.dtype.itemsize <= 2:
                return float(1.4826 * integer_mad(finite_values))
            median = np.median(finite_values)
            return float(1.4826 * np.median(np.abs(finite_values - median)))

        if metric_type == "min":
            return float(np.min(finite_values))
        
        if metric_type == "max":
            return float(np.max(finite_values))
        
        if metric_type == "percentile":
            return float(np.percentile(finite_values, metric["percentile"]))
        
        if metric_type == "fraction_above":
            return float(np.count_nonzero(finite_values > metric["threshold"]) / finite_values.size)
        
        if metric_type == "sum":
            return float(np.sum(finite_values))
        
        if metric_type == "count_above":
            return float(np.count_nonzero(finite_values > metric["threshold"]))
        
        if metric_type in ("row_structure", "column_structure") or metric_type in STRUCTURE_NORM_TYPES:
            # Banding metric: std of the per-row (or per-column) TRIMMED means.
            # Whole-row/column offsets, bars and glow gradients move the profile;
            # isolated hot pixels, hot-column fragments and cosmic rays do not
            # (a plain mean profile let a 20-pixel saturated cluster FAIL a bias;
            # a median profile missed real bars confined to part of a column).
            # The *_norm variants divide by the lamp signal so a bright 1 s arc
            # and a faint 0.07 s arc give the same number for the same pattern.
            axis = -1 if metric_type.startswith("row_structure") else -2  # collapse cols / rows
            profile = trimmed_mean_profile(values, axis)
            profile = profile[np.isfinite(profile)]
            if profile.size == 0:
                raise QAEngineError("structure metric has no finite rows/columns")
            structure = float(np.nanstd(profile))
            if metric_type in STRUCTURE_NORM_TYPES:
                if signal is None or not np.isfinite(signal) or signal <= 0:
                    raise QAEngineError(f"{metric_type} requires a positive signal")
                return structure / float(signal)
            return structure
        
        raise QAEngineError(f"unsupported metric type {metric_type!r}")
 
    def _resolve_rule_limits(
        self,
        rule: dict[str, Any],
        metadata: dict[str, Any],
        extension: dict[str, Any],
    ) -> tuple[dict[str, float], list[str] | None]:
        if "limits" in rule:
            return {key: float(value) for key, value in rule["limits"].items()}, None
 
        lookup = rule["expected_from_lookup"]
        table = self.config["lookup_tables"][lookup["table"]]
        path_parts: list[str] = []
 
        for key_spec in lookup["keys"]:
            source = key_spec["from"]
            scope, field = source.split(".", 1)
            if scope == "extension":
                value = extension.get(field)
            elif scope == "metadata":
                value = metadata.get(field)
            else:  # Should be impossible after validation.
                raise QAEngineError(f"unsupported lookup source {source!r}")
            if value is None:
                raise QAEngineError(f"lookup source {source!r} is missing for rule {rule['name']!r}")
            path_parts.append(self.normalize(value))
 
        leaf = self._resolve_lookup_leaf(table, path_parts, lookup["table"])
        limits: dict[str, float] = {}
        if "min_field" in lookup:
            limits["min"] = float(leaf[lookup["min_field"]])
        if "max_field" in lookup:
            limits["max"] = float(leaf[lookup["max_field"]])
        return limits, path_parts
 
    def _resolve_lookup_leaf(self, table: Any, path_parts: list[str], table_name: str) -> dict[str, Any]:
        node = table
        resolved_path: list[str] = []
        for part in path_parts:
            if not isinstance(node, dict):
                raise QAEngineError(f"lookup table {table_name!r} path {resolved_path!r} "
                                    f"is not a mapping")
            matched_key = None
            for candidate in node:
                if self.normalize(candidate) == self.normalize(part):
                    matched_key = candidate
                    break
            if matched_key is None:
                raise LookupMissError(f"lookup table {table_name!r} has no entry for path {path_parts!r}")
            resolved_path.append(str(matched_key))
            node = node[matched_key]
        if not isinstance(node, dict):
            raise QAEngineError(f"lookup table {table_name!r} leaf at {resolved_path!r} "
                                f"is not a mapping")
        return node
 
    @staticmethod
    def _check_limits(value: float, limits: dict[str, float]) -> tuple[bool, str]:
        if "min" in limits and value < limits["min"]:
            return False, f"value {value:.6g} is below min {limits['min']:.6g}"
        if "max" in limits and value > limits["max"]:
            return False, f"value {value:.6g} is above max {limits['max']:.6g}"
        return True, "value is within configured limits"
 
    def _final_verdict(self, results: list[RuleResult]) -> str:
        policy = self.config["verdict_policy"]
        # An unevaluable rule (ERROR) must never be masked as PASS; it outranks FAIL.
        if any(result.verdict_effect == "ERROR" for result in results):
            return "ERROR"
        if policy.get("fail_if_any_fail", True) and any(
            result.verdict_effect == "FAIL" for result in results
        ):
            return "FAIL"
        if policy.get("warn_if_any_warn", True) and any(
            result.verdict_effect == "WARN" for result in results
        ):
            return "WARN"
        return "PASS"
 
    @staticmethod
    def _summary(results: list[RuleResult]) -> dict[str, int]:
        return {
            "total_checks": len(results),
            "evaluated_checks": sum(1 for result in results if result.status == "EVALUATED"),
            "skipped_checks": sum(1 for result in results if result.status != "EVALUATED"),
            "passed_checks": sum(1 for result in results
                                 if result.status == "EVALUATED" and result.passed),
            "failed_checks": sum(1 for result in results
                                 if result.status == "EVALUATED" and not result.passed),
            "fail_effects": sum(1 for result in results if result.verdict_effect == "FAIL"),
            "warn_effects": sum(1 for result in results if result.verdict_effect == "WARN"),
            "missing_extensions": len({
                (result.extension, result.hdu_index)
                for result in results
                if result.status == "MISSING"
            }),
            "placeholder_extensions": len({
                (result.extension, result.hdu_index)
                for result in results
                if result.status == "PLACEHOLDER"
            }),
        }
 
 
def _value_histogram(values: Any) -> tuple[Any, Any]:
    """Counts of every integer value present, as (values, counts), values ascending."""
    flat = np.asarray(values).ravel()
    low = int(flat.min())
    counts = np.bincount((flat.astype(np.int64) - low))
    present = np.flatnonzero(counts)
    return present + low, counts[present]


def _median_from_histogram(values: Any, counts: Any) -> float:
    """numpy's median (mean of the two middle ranks for an even count) from a histogram."""
    total = int(counts.sum())
    cumulative = np.cumsum(counts)
    upper = float(values[np.searchsorted(cumulative, total // 2 + 1)])
    if total % 2:
        return upper
    lower = float(values[np.searchsorted(cumulative, total // 2)])
    return (lower + upper) / 2.0


def integer_median(values: Any) -> float:
    """Exact ``np.median`` of an integer array via a value histogram.

    A 4-Mpix uint16 frame has at most 65536 distinct values, so counting them
    (``np.bincount``) and walking the cumulative counts is several times cheaper
    than numpy's partition-based median, and gives the same value.
    """
    present, counts = _value_histogram(values)
    return _median_from_histogram(present, counts)


def integer_mad(values: Any) -> float:
    """Exact median absolute deviation, ``np.median(np.abs(x - np.median(x)))``, of an
    integer array via its value histogram (no 4-Mpix deviation array is formed)."""
    present, counts = _value_histogram(values)
    median = _median_from_histogram(present, counts)
    deviations = np.abs(present - median)
    order = np.argsort(deviations, kind="stable")
    return _median_from_histogram(deviations[order], counts[order])


def trimmed_mean_profile(values: Any, axis: int, trim: float = STRUCTURE_TRIM_FRACTION) -> Any:
    """Mean along ``axis`` after dropping the lowest/highest ``trim`` fraction of each line.

    At least one pixel is trimmed from each end so the metric is defined for short
    test frames; NaNs sort to the top end and are trimmed or ignored by nanmean.
    """
    array = np.asarray(values)
    if np.issubdtype(array.dtype, np.integer):
        # Integer frames cannot hold NaN, and numpy sorts 16-bit integers with a
        # radix sort, several times faster than sorting a float64 copy. Columns
        # are sorted along the last axis of a contiguous transpose so they cost
        # the same as rows; the profile keeps its orientation either way.
        if axis in (-2, array.ndim - 2):
            array = np.ascontiguousarray(np.swapaxes(array, -1, -2))
            axis = -1
        ordered = np.sort(array, axis=axis)
        n = ordered.shape[axis]
        k = max(1, int(n * trim))
        index = [slice(None)] * ordered.ndim
        index[axis] = slice(k, n - k)
        return ordered[tuple(index)].mean(axis=axis, dtype=np.float64)

    ordered = np.sort(np.asarray(array, dtype=float), axis=axis)
    n = ordered.shape[axis]
    k = max(1, int(n * trim))
    index = [slice(None)] * ordered.ndim
    index[axis] = slice(k, n - k)
    return np.nanmean(ordered[tuple(index)], axis=axis)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError as fnf_exc:
        raise QAEngineError(f"configuration file not found: {path}") from fnf_exc
    except yaml.YAMLError as yaml_exc:
        raise QAEngineError(f"YAML syntax error in {path}: {yaml_exc}") from yaml_exc
 
    if not isinstance(data, dict):
        raise QAEngineError(f"configuration root must be a mapping: {path}")
    return data
 
 
def validate_config(config: dict[str, Any], config_path: Path) -> None:
    validator = QAConfigValidator(config, filename=str(config_path))
    errors = validator.validate()
    if errors:
        details = "\n".join(f"  ERROR: {error}" for error in errors)
        raise QAEngineError(f"invalid QA configuration: {config_path}\n{details}")
 
 
def report_path_for(fits_path: Path, directory: Path | None = None) -> Path:
    """Report file for a frame: ``<frame>.qa.json`` beside it, or inside ``directory``.

    Shared by ``llamas-checks-engine`` (sidecars next to the frame) and
    ``llamas-checks --report-dir`` so one frame has one report name everywhere,
    e.g. ``LLAMAS_2026-09-07_18-17-12.8_CAL22_mef.fits`` ->
    ``LLAMAS_2026-09-07_18-17-12.8_CAL22_mef.qa.json``.
    """
    name = fits_path.with_suffix(".qa.json").name
    return (directory if directory is not None else fits_path.parent) / name


def write_report(report: dict[str, Any], output_path: Path | None) -> None:
    text = json.dumps(report, indent=2, sort_keys=False)
    if output_path is None:
        print(text)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
 
 
def is_fits_file(path: Path) -> bool:
    name = path.name.lower()
    return path.is_file() and any(name.endswith(suffix) for suffix in FITS_SUFFIXES)
 
 
def collect_fits_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise QAEngineError(f"not a directory: {directory}")
    files = [path for path in directory.iterdir() if is_fits_file(path)]
    return sorted(files)
 
 
def run_single_file(fits_file: Path, config_path: Path, no_validate: bool = False,
                    cameras_down: list[str] | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    if not no_validate:
        validate_config(config, config_path)
    engine = QAEngine(config, cameras_down=cameras_down)
    return engine.run(fits_file)


def _batch_worker(args: tuple[str, str, bool, list[str] | None]) -> dict[str, Any]:
    fits_file_s, config_path_s, no_validate, cameras_down = args
    fits_file = Path(fits_file_s)
    config_path = Path(config_path_s)
    try:
        report = run_single_file(fits_file, config_path, no_validate=no_validate,
                                 cameras_down=cameras_down)
        return {
            "fits_file": str(fits_file),
            "status": "OK",
            "overall_verdict": report["overall_verdict"],
            "report": report,
        }
    except Exception as exc:  # The parent process should receive all per-file failures.
        return {
            "fits_file": str(fits_file),
            "status": "ERROR",
            "overall_verdict": "ERROR",
            "error": str(exc),
        }
 
 
def run_batch(input_dir: Path, config_path: Path, jobs: int, no_validate: bool = False,
              cameras_down: list[str] | None = None) -> dict[str, Any]:
    files = collect_fits_files(input_dir)
    if not files:
        raise QAEngineError(f"no FITS files found in directory: {input_dir}")

    # Validate once in the parent process for fast feedback before starting workers.
    if not no_validate:
        config = load_yaml(config_path)
        validate_config(config, config_path)

    worker_args = [(str(path), str(config_path), no_validate, cameras_down) for path in files]
    max_workers = max(1, int(jobs))
 
    results: list[dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_batch_worker, item): item[0]
            for item in worker_args
        }
        for future in concurrent.futures.as_completed(future_to_path):
            path = future_to_path[future]
            try:
                results.append(future.result())
            except Exception as exc:  # pragma: no cover - defensive fallback.
                results.append({
                    "fits_file": path,
                    "status": "ERROR",
                    "overall_verdict": "ERROR",
                    "error": str(exc),
                })
 
    results.sort(key=lambda item: item["fits_file"])
 
    ok_count = sum(1 for item in results if item["status"] == "OK")
    # An ERROR overall_verdict (unevaluable rule) counts as an error even when the
    # worker itself did not throw (status == "OK").
    error_count = sum(1 for item in results
                      if item["status"] == "ERROR" or item["overall_verdict"] == "ERROR")
    fail_count = sum(1 for item in results if item["overall_verdict"] == "FAIL")
    warn_count = sum(1 for item in results if item["overall_verdict"] == "WARN")
    pass_count = sum(1 for item in results if item["overall_verdict"] == "PASS")
 
    if error_count:
        overall_status = "ERROR"
    elif fail_count:
        overall_status = "FAIL"
    elif warn_count:
        overall_status = "WARN"
    else:
        overall_status = "PASS"
 
    return {
        "mode": "batch",
        "input_directory": str(input_dir),
        "config_file": str(config_path),
        "jobs": max_workers,
        "overall_status": overall_status,
        "summary": {
            "total_files": len(results),
            "ok_files": ok_count,
            "error_files": error_count,
            "pass_files": pass_count,
            "warn_files": warn_count,
            "fail_files": fail_count,
        },
        "files": results,
    }
 
 
def main() -> int:
    parser = argparse.ArgumentParser(description="Run YAML-driven FITS/MEF QA checks.")
    parser.add_argument("input_path", type=Path, help="Input FITS/MEF file or directory")
    parser.add_argument("--config", type=Path, required=True,
                        help="QA YAML configuration file (a path, or the bare name "
                             "of a config shipped in the package configs/ directory)")
    parser.add_argument("--jobs", type=int, default=8,
                        help="Number of parallel workers in directory/batch mode")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip config validation before running")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print quick human-readable summary")
    parser.add_argument("--cameras-down", default=None, metavar="NAMES",
                        help="comma-separated detector names allowed to be missing or "
                             "placeholder (e.g. 1.A.Blue,4.A.Blue; 'none' for none); "
                             f"overrides configs/{CAMERA_STATUS_NAME} and the "
                             f"{CAMERAS_DOWN_ENV} environment variable")
    args = parser.parse_args()
    cameras_down = resolve_cameras_down(args.cameras_down)[0]

    # A bare config name (e.g. "qa_config_cal.yaml", no directory part) resolves
    # to the shipped copy; any path with a directory component is used as given,
    # so a mistyped path errors (exit 2) instead of silently running another config.
    if (len(args.config.parts) == 1 and not args.config.exists()
            and (CONFIG_DIR / args.config.name).exists()):
        args.config = CONFIG_DIR / args.config.name

    try:
        input_path = args.input_path

        # --------------------------------------------------
        # SINGLE FILE MODE
        # --------------------------------------------------
        if input_path.is_file():
            try:
                report = run_single_file(
                    input_path,
                    args.config,
                    no_validate=args.no_validate,
                    cameras_down=cameras_down,
                )
                verdict = report["overall_verdict"]
 
                out_path = report_path_for(input_path)
                write_report(report, out_path)
            except Exception as exc:
                verdict = "ERROR"
                out_path = report_path_for(input_path)
 
                error_report = {
                    "fits_file": str(input_path),
                    "overall_verdict": "ERROR",
                    "error": str(exc),
                }
                write_report(error_report, out_path)
 
            if args.summary_only:
                print(f"{input_path.name} {verdict}")

            # PASS/WARN -> 0; FAIL -> 1; ERROR (unreadable file, bad config,
            # off-mode frame, unevaluable rule) -> 2, so a run that never actually
            # executed does not silently pass the CI gate.
            return {"PASS": 0, "WARN": 0, "FAIL": 1}.get(verdict, 2)
 
        # --------------------------------------------------
        # DIRECTORY / BATCH MODE
        # --------------------------------------------------
        if input_path.is_dir():
            report = run_batch(
                input_path,
                args.config,
                jobs=args.jobs,
                no_validate=args.no_validate,
                cameras_down=cameras_down,
            )
 
            any_fail = False
 
            for item in report["files"]:
                fits_path = Path(item["fits_file"])
                out_path = report_path_for(fits_path)
 
                if item["status"] == "OK":
                    write_report(item["report"], out_path)
                    verdict = item["overall_verdict"]
                else:
                    verdict = "ERROR"
                    error_report = {
                        "fits_file": item["fits_file"],
                        "overall_verdict": "ERROR",
                        "error": item.get("error", "unknown error"),
                    }
                    write_report(error_report, out_path)
 
                if verdict == "FAIL":
                    any_fail = True
 
                if args.summary_only:
                    print(f"{fits_path.name} {verdict}")

            # Propagate ERROR (any file errored) as exit 2, FAIL as 1, else 0 —
            # so errored/off-mode files cannot slip through the gate as green.
            status = report.get("overall_status", "PASS")
            return {"PASS": 0, "WARN": 0, "FAIL": 1}.get(status, 2)
 
        # --------------------------------------------------
        # INVALID INPUT
        # --------------------------------------------------
        raise QAEngineError(f"input path does not exist: {input_path}")
 
    except QAEngineError as qa_exc:
        print(f"ERROR: {qa_exc}", file=sys.stderr)
        return 2
 
 
if __name__ == "__main__":
    raise SystemExit(main())