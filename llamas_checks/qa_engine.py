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
import sys
import time
from dataclasses import asdict, dataclass, field
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

# Threads used to measure the extensions of one frame in parallel. The numpy
# kernels (partition, bincount) release the GIL, so four threads bring a 24-detector
# frame from ~1.9 s to ~0.6 s on a laptop; more gives little and competes with the
# observing GUI. Batch mode already runs one process per file and uses one thread.
DEFAULT_WORKERS = 4

# Constant-frame values that mark a software placeholder for a missing camera:
# 1.0 in real frames, 0.0 from the pipeline validator (validate.create_placeholder_hdu).
PLACEHOLDER_VALUES = (0.0, 1.0)
 
 
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


@dataclass
class _ExtensionCache:
    """Pixel data and precomputed metrics for one configured extension."""
    data: Any                      # ndarray, or None when the HDU is absent / has no image
    present: bool                  # hdu_index < len(hdul)
    placeholder: bool = False
    # (region_name, metric_name) -> measured float, or the QAEngineError to re-raise
    metrics: dict[tuple[str, str], Any] = field(default_factory=dict)


@dataclass
class _RunCache:
    """Everything measured once per frame before the rules are evaluated."""
    extensions: dict[int, _ExtensionCache] = field(default_factory=dict)
    # metric name -> exposure time (s), or the LookupMissError to re-raise
    exptimes: dict[str, Any] = field(default_factory=dict)


class LookupMissError(QAEngineError):
    """Raised when a lookup table has no entry for a detector/mode combination.

    Treated as a benign SKIP for that one rule+extension (e.g. an off-mode frame
    whose readout mode is not modelled), rather than an ERROR that aborts the file.
    """
 
 
class QAEngine:
    def __init__(self, config: dict[str, Any]):
        self.config = config
 
    @staticmethod
    def normalize(value: Any) -> str:
        """Normalize metadata values and lookup keys for robust comparison."""
        return str(value).strip().lower()
 
    def run(self, fits_path: Path, workers: int | None = None) -> dict[str, Any]:
        """Evaluate one FITS/MEF file and return the report dict.

        ``workers`` is the number of threads used to measure the extensions
        (default ``DEFAULT_WORKERS``). Every pixel metric is computed once per
        extension, up front and in parallel; the rules are then evaluated in config
        order against those values, so the report is identical for any worker count.
        """
        workers = DEFAULT_WORKERS if workers is None else max(1, int(workers))
        timings: dict[str, float] = {}
        t_start = time.perf_counter()
        # memmap=False is required for FITS files containing BZERO/BSCALE/BLANK
        # keywords, because Astropy needs to scale the image data in memory.
        with fits.open(fits_path, memmap=False) as hdul:
            metadata = self._extract_metadata(hdul)
            active_rule_sets = self._select_rule_sets(metadata)
            results: list[RuleResult] = []

            if not active_rule_sets:
                results.append(self._no_rules_matched_result(metadata))

            cache = self._precompute(hdul, active_rule_sets, workers, timings)

            t_rules = time.perf_counter()
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
                                    metadata, rule_set_name, rule, extension, cache
                                )
                            )
                    else:
                        extension = self._default_extension_for_non_per_extension_rule(cache)
                        results.append(
                            self._evaluate_rule_for_extension(
                                metadata, rule_set_name, rule, extension, cache
                            )
                        )
            timings["rules"] = time.perf_counter() - t_rules
        timings["total"] = time.perf_counter() - t_start

        verdict = self._final_verdict(results)
        return {
            "fits_file": str(fits_path),
            "instrument": self.config.get("instrument", {}),
            "metadata": metadata,
            "active_rule_sets": list(active_rule_sets.keys()),
            "overall_verdict": verdict,
            "summary": self._summary(results),
            "results": [asdict(result) for result in results],
            "workers": workers,
            "timings": {key: round(value, 4) for key, value in timings.items()},
        }

    def _precompute(
        self,
        hdul: Any,
        active_rule_sets: dict[str, Any],
        workers: int,
        timings: dict[str, float],
    ) -> _RunCache:
        """Load each configured extension once and measure, in parallel, every
        (region, metric) pair the active rules will ask for.

        A metric shared by several rules (the WARN and FAIL structure tiers, the
        level and rms rules' median) is computed once. Placeholder detection runs
        once per extension instead of once per rule. Evaluation errors are stored
        and re-raised when the rule is evaluated, so per-rule SKIP/ERROR reporting
        is unchanged. Pixel data is read in the calling thread (astropy HDU access
        is not thread-safe); only numpy work runs in the pool.
        """
        t_read = time.perf_counter()
        cache = _RunCache()
        for extension in self.config["extensions"]:
            hdu_index = extension["hdu_index"]
            if hdu_index in cache.extensions:
                continue
            present = hdu_index < len(hdul)
            data = getattr(hdul[hdu_index], "data", None) if present else None
            if data is not None:
                data = np.asarray(data)
            placeholder = data is not None and self._is_placeholder_data(data)
            cache.extensions[hdu_index] = _ExtensionCache(data, present, placeholder)
        timings["read"] = time.perf_counter() - t_read

        t_metrics = time.perf_counter()
        metric_rules = [rule for rule_set in active_rule_sets.values()
                        for rule in rule_set["rules"] if "header_check" not in rule]
        needed: dict[int, dict[tuple[str, str], dict[str, Any]]] = {}
        default_extension: dict[str, Any] | None = None
        for rule in metric_rules:
            if rule.get("per_extension", False):
                targets = self.config["extensions"]
            else:
                if default_extension is None:
                    try:
                        default_extension = self._default_extension_for_non_per_extension_rule(cache)
                    except QAEngineError:
                        continue  # run() raises the same error when it reaches the rule
                targets = [default_extension]
            for extension in targets:
                entry = cache.extensions[extension["hdu_index"]]
                if entry.data is None or entry.placeholder:
                    continue
                try:
                    region_name = self._resolve_rule_region_name(rule, extension)
                except QAEngineError:
                    continue  # raised again, unchanged, at evaluation time
                needed.setdefault(extension["hdu_index"], {})[(region_name, rule["metric"])] = extension

        for metric_name, metric in self.config["metrics"].items():
            if metric.get("type") == "background_gradient_rate":
                try:
                    cache.exptimes[metric_name] = self._resolve_exptime(hdul, metric)
                except LookupMissError as exc:
                    cache.exptimes[metric_name] = exc

        def measure_extension(hdu_index: int) -> None:
            entry = cache.extensions[hdu_index]
            histograms: dict[str, Any] = {}
            for (region_name, metric_name), extension in needed[hdu_index].items():
                entry.metrics[(region_name, metric_name)] = self._measure(
                    entry.data, region_name, metric_name, extension, cache, histograms)

        jobs = list(needed)
        if workers > 1 and len(jobs) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(measure_extension, jobs))
        else:
            for hdu_index in jobs:
                measure_extension(hdu_index)
        timings["metrics"] = time.perf_counter() - t_metrics
        return cache

    def _measure(
        self,
        data: Any,
        region_name: str,
        metric_name: str,
        extension: dict[str, Any],
        cache: _RunCache,
        histograms: dict[str, Any] | None = None,
    ) -> Any:
        """One metric on one extension: the value, or the QAEngineError to re-raise."""
        try:
            region = self.config["regions"][region_name]
            metric = self.config["metrics"][metric_name]
            region_data = self._extract_region(data, region, region_name, extension.get("name"))
            if metric.get("type") == "background_gradient_rate":
                # rate metrics need the exposure time; absent/too-short -> SKIP this rule
                exptime = cache.exptimes.get(metric_name)
                if isinstance(exptime, Exception):
                    raise exptime
                return self._compute_metric(region_data, metric, exptime=exptime)
            return self._compute_metric(region_data, metric, histograms=histograms,
                                        region_name=region_name)
        except QAEngineError as exc:
            return exc


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
 
    def _default_extension_for_non_per_extension_rule(self, cache: _RunCache) -> dict[str, Any]:
        # Current YAML uses per_extension rules. This fallback keeps the engine
        # deterministic if a future rule omits per_extension or sets it false.
        for extension in self.config["extensions"]:
            entry = cache.extensions.get(extension["hdu_index"])
            if entry is not None and entry.data is not None and not entry.placeholder:
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
 
    @staticmethod
    def _is_placeholder_data(data: Any) -> bool:
        """Return True for software-generated placeholder images.

        Placeholder convention: a constant finite frame at one of the two
        known placeholder values -- every finite pixel is exactly 1 (real
        frames) or exactly 0 (the pipeline validator's ``create_placeholder_hdu``).
        Any other constant frame (e.g. a railed detector at the ADC ceiling, or
        a dead readout at a non-zero pedestal) is *not* a placeholder and is
        evaluated normally, so saturation/structure rules can still fire on it.
        Placeholder extensions are reported but skipped from metric evaluation
        and do not affect the final PASS/WARN/FAIL verdict.
        """
        array = np.asarray(data)
        if array.size == 0:
            return False

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
        metadata: dict[str, Any],
        rule_set_name: str,
        rule: dict[str, Any],
        extension: dict[str, Any],
        cache: _RunCache,
    ) -> RuleResult:
        hdu_index = extension["hdu_index"]
        extension_name = extension.get("name")
        entry = cache.extensions.get(hdu_index)

        if entry is None or not entry.present:
            return self._skipped_rule_result(
                rule_set_name,
                rule,
                extension,
                status="MISSING",
                message=f"extension {extension_name!r} references missing HDU index {hdu_index}",
            )

        if entry.data is None:
            return self._skipped_rule_result(
                rule_set_name,
                rule,
                extension,
                status="MISSING",
                message=f"extension {extension_name!r} / HDU {hdu_index} has no image data",
            )

        if entry.placeholder:
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
            key = (region_name, metric_name)
            if key not in entry.metrics:  # not precomputed: measure it now
                entry.metrics[key] = self._measure(entry.data, region_name, metric_name,
                                                   extension, cache)
            measured_value = entry.metrics[key]
            if isinstance(measured_value, QAEngineError):
                raise measured_value
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
                        exptime: float | None = None,
                        histograms: dict[str, Any] | None = None,
                        region_name: str | None = None) -> float:
        raw = np.asarray(region_data)
        if raw.size and raw.dtype.kind in "iu" and raw.dtype.itemsize <= 2:
            # Raw frames are 16-bit integers: nothing to mask, and medians, MADs and
            # threshold counts come exactly from a value histogram instead of
            # partitions on a float64 copy. Metric types without an integer kernel
            # fall through to the generic float path below.
            fast = _integer_metric(raw, metric, exptime, histograms, region_name)
            if fast is not None:
                return fast
        values = np.asarray(region_data, dtype=float)
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
            return float(np.median(finite_values))
        
        if metric_type == "std":
            return float(np.std(finite_values))

        if metric_type == "robust_std":
            # Read-noise monitor: 1.4826 x MAD. A plain std on a 4-Mpix frame is
            # dominated by a handful of hot/saturated pixels or a cosmic-ray
            # cluster (20 railed pixels alone give std ~150 ADU); the MAD is not.
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
        
        if metric_type in ("row_structure", "column_structure"):
            # Banding metric: std of the per-row (or per-column) TRIMMED means.
            # Whole-row/column offsets, bars and glow gradients move the profile;
            # isolated hot pixels, hot-column fragments and cosmic rays do not
            # (a plain mean profile let a 20-pixel saturated cluster FAIL a bias;
            # a median profile missed real bars confined to part of a column).
            axis = -1 if metric_type == "row_structure" else -2  # collapse cols / rows
            profile = trimmed_mean_profile(values, axis)
            profile = profile[np.isfinite(profile)]
            if profile.size == 0:
                raise QAEngineError("structure metric has no finite rows/columns")
            return float(np.nanstd(profile))
        
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
 
 
class IntegerHistogram:
    """Exact order statistics of a 16-bit integer array from its value histogram.

    ``np.bincount`` over at most 65536 bins costs a few ms for a 4 Mpix frame,
    against tens of ms per ``np.partition`` on a float64 copy, and the median, MAD
    and threshold counts it yields are identical to the float implementations:
    integers are exactly representable and numpy's even-count median convention
    (mean of the two middle values) is reproduced.
    """

    def __init__(self, values: Any):
        flat = np.asarray(values).ravel()
        self.offset = min(int(flat.min()), 0)   # bins stay aligned to value 0
        if self.offset < 0:
            flat = flat.astype(np.int64) - self.offset
        self.counts = np.bincount(flat)
        self.cumulative = np.cumsum(self.counts)
        self.n = int(self.cumulative[-1])

    @staticmethod
    def _median_of(cumulative: Any, n: int, positions: Any) -> float:
        lower = np.searchsorted(cumulative, (n - 1) // 2 + 1)
        upper = np.searchsorted(cumulative, n // 2 + 1)
        return (float(positions[lower]) + float(positions[upper])) / 2.0

    def median(self) -> float:
        return self._median_of(self.cumulative, self.n, np.arange(self.counts.size)) + self.offset

    def mad(self) -> float:
        """Median absolute deviation about the median (not yet scaled by 1.4826)."""
        centre = self.median() - self.offset
        deviations = np.abs(np.arange(self.counts.size) - centre)
        order = np.argsort(deviations, kind="stable")
        return self._median_of(np.cumsum(self.counts[order]), self.n, deviations[order])

    def count_above(self, threshold: float) -> int:
        """Number of values strictly greater than ``threshold``."""
        first = int(np.floor(threshold)) + 1 - self.offset   # smallest integer > threshold
        first = min(max(first, 0), self.counts.size)
        return int(self.counts[first:].sum())


def _integer_metric(values: Any, metric: dict[str, Any], exptime: float | None,
                    histograms: dict[str, Any] | None, region_name: str | None) -> float | None:
    """Fast exact kernels for 16-bit integer regions; None when the metric type has none."""
    metric_type = metric["type"]

    if metric_type in ("median", "robust_std", "fraction_above", "count_above"):
        histogram = None
        if histograms is not None and region_name is not None:
            histogram = histograms.get(region_name)
        if histogram is None:
            histogram = IntegerHistogram(values)
            if histograms is not None and region_name is not None:
                histograms[region_name] = histogram
        if metric_type == "median":
            return float(histogram.median())
        if metric_type == "robust_std":
            return float(1.4826 * histogram.mad())
        if metric_type == "fraction_above":
            return float(histogram.count_above(metric["threshold"]) / histogram.n)
        return float(histogram.count_above(metric["threshold"]))

    if metric_type == "min":
        return float(np.min(values))

    if metric_type == "max":
        return float(np.max(values))

    if metric_type in ("row_structure", "column_structure"):
        axis = -1 if metric_type == "row_structure" else -2  # collapse cols / rows
        profile = trimmed_mean_profile(values, axis)
        profile = profile[np.isfinite(profile)]
        if profile.size == 0:
            raise QAEngineError("structure metric has no finite rows/columns")
        return float(np.nanstd(profile))

    if metric_type == "background_gradient_rate":
        if values.ndim < 2:
            raise QAEngineError("background_gradient_rate requires a 2D region")
        if not exptime or exptime <= 0:
            raise QAEngineError("background_gradient_rate requires a positive exposure time")
        ny, nx = values.shape[-2], values.shape[-1]
        qy, qx = max(ny // 4, 1), max(nx // 4, 1)
        v_grad = abs(IntegerHistogram(values[..., :qy, :]).median()
                     - IntegerHistogram(values[..., -qy:, :]).median())
        h_grad = abs(IntegerHistogram(values[..., :, :qx]).median()
                     - IntegerHistogram(values[..., :, -qx:]).median())
        return float(max(v_grad, h_grad) / exptime)

    return None


def trimmed_mean_profile(values: Any, axis: int, trim: float = STRUCTURE_TRIM_FRACTION) -> Any:
    """Mean along ``axis`` after dropping the lowest/highest ``trim`` fraction of each line.

    At least one pixel is trimmed from each end so the metric is defined for short
    test frames; NaNs sort to the top end and are trimmed or ignored by nanmean.

    Uses a two-pivot ``np.partition`` instead of a full sort: once the k-th and
    (n-k-1)-th order statistics are in place, everything between them is exactly
    the untrimmed set, so the mean is the same as the sorted version at a fraction
    of the cost. Integer input is partitioned natively (its mean is accumulated in
    float64 either way); float input is promoted to float64 as before. A 2-D
    column profile is taken on a contiguous transpose, which is much faster than
    partitioning along a strided axis.
    """
    array = np.asarray(values)
    if array.dtype.kind not in "iu":
        array = np.asarray(array, dtype=float)
    axis = axis % array.ndim
    if array.ndim == 2 and axis == 0:
        array = np.ascontiguousarray(array.T)
        axis = 1
    n = array.shape[axis]
    k = max(1, int(n * trim))
    if n <= 2 * k:   # nothing left after trimming: every line is undefined
        return np.full(array.shape[:axis] + array.shape[axis + 1:], np.nan)
    index = [slice(None)] * array.ndim
    index[axis] = slice(k, n - k)
    partitioned = np.partition(array, (k, n - k - 1), axis=axis)
    return np.nanmean(partitioned[tuple(index)], axis=axis)


_YAML_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            # The C loader parses the 3,500-line generated cal config in ~30 ms
            # against ~350 ms for the pure-Python SafeLoader; same safe semantics.
            data = yaml.load(handle, Loader=_YAML_LOADER)
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
                    workers: int | None = None) -> dict[str, Any]:
    config = load_yaml(config_path)
    if not no_validate:
        validate_config(config, config_path)
    engine = QAEngine(config)
    return engine.run(fits_file, workers=workers)
 
 
def _batch_worker(args: tuple[str, str, bool, int]) -> dict[str, Any]:
    fits_file_s, config_path_s, no_validate, workers = args
    fits_file = Path(fits_file_s)
    config_path = Path(config_path_s)
    try:
        report = run_single_file(fits_file, config_path, no_validate=no_validate, workers=workers)
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
              threads: int = 1) -> dict[str, Any]:
    """Batch mode runs one process per file (``jobs``), each measuring its frame
    with ``threads`` threads (default 1: the files already run in parallel)."""
    files = collect_fits_files(input_dir)
    if not files:
        raise QAEngineError(f"no FITS files found in directory: {input_dir}")
 
    # Validate once in the parent process for fast feedback before starting workers.
    if not no_validate:
        config = load_yaml(config_path)
        validate_config(config, config_path)
 
    worker_args = [(str(path), str(config_path), no_validate, max(1, int(threads)))
                   for path in files]
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
    parser.add_argument("--threads", type=int, default=None,
                        help="Threads used to measure the extensions of one frame "
                             f"(default: {DEFAULT_WORKERS} for a single file, 1 per file in batch mode)")
    parser.add_argument("--no-validate", action="store_true",
                        help="Skip config validation before running")
    parser.add_argument("--summary-only", action="store_true",
                        help="Print quick human-readable summary")
    args = parser.parse_args()

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
                    workers=args.threads,
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
                threads=args.threads or 1,
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