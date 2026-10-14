#!/usr/bin/env python3
"""Compare two sets of QA reports rule by rule.

Use it to confirm that a change to the engine (or a machine, or a numpy version)
leaves every verdict and every measured value unchanged. Both directories hold
JSON reports for the same frames, written either by ``llamas-checks --report DIR
--report-all`` (engine report nested under ``"report"``) or by
``llamas-checks-engine`` (engine report at top level). Files are matched by frame
name, ignoring the ``.qa.json`` / ``.old.json`` / ``.new.json`` suffixes.

    scripts/compare_reports.py reports_main/ reports_fast/

Exit 0 when every overall verdict, per-rule status, verdict effect and measured
value agree (to --tolerance, default 0 = bit-identical after the engine's own
6-decimal rounding); exit 1 otherwise, listing the differences.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_reports(directory: Path) -> dict[str, dict]:
    reports = {}
    for path in sorted(directory.glob("*.json")):
        key = path.name
        for suffix in (".qa.json", ".old.json", ".new.json", ".json"):
            if key.endswith(suffix):
                key = key[: -len(suffix)]
                break
        report = json.loads(path.read_text())
        if "results" not in report and isinstance(report.get("report"), dict):
            report = report["report"]
        reports[key] = report
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("old_dir", type=Path)
    parser.add_argument("new_dir", type=Path)
    parser.add_argument("--tolerance", type=float, default=0.0,
                        help="largest allowed |new - old| for a measured value (default 0)")
    args = parser.parse_args()

    old = load_reports(args.old_dir)
    new = load_reports(args.new_dir)
    common = sorted(set(old) & set(new))
    if not common:
        print("no frames in common", file=sys.stderr)
        return 2
    for missing in sorted(set(old) ^ set(new)):
        print(f"only in one directory, skipped: {missing}")

    differences = []
    per_metric = defaultdict(lambda: [0, 0.0])
    compared = 0
    old_seconds = new_seconds = 0.0
    for key in common:
        a, b = old[key], new[key]
        old_seconds += a.get("timings", {}).get("total", a.get("_engine_seconds", 0.0))
        new_seconds += b.get("timings", {}).get("total", b.get("_engine_seconds", 0.0))
        if a.get("overall_verdict") != b.get("overall_verdict"):
            differences.append(f"{key}: overall {a.get('overall_verdict')} -> {b.get('overall_verdict')}")
        a_results = {(r["rule"], r["extension"]): r for r in a.get("results", [])}
        b_results = {(r["rule"], r["extension"]): r for r in b.get("results", [])}
        for rule_key in sorted(set(a_results) | set(b_results), key=str):
            ra, rb = a_results.get(rule_key), b_results.get(rule_key)
            label = f"{key} {rule_key[0]}@{rule_key[1]}"
            if ra is None or rb is None:
                differences.append(f"{label}: present in only one report")
                continue
            for field in ("status", "verdict_effect"):
                if ra[field] != rb[field]:
                    differences.append(f"{label}: {field} {ra[field]} -> {rb[field]}")
            va, vb = ra.get("measured_value"), rb.get("measured_value")
            if (va is None) != (vb is None):
                differences.append(f"{label}: measured_value {va} -> {vb}")
            elif va is not None:
                compared += 1
                delta = abs(va - vb)
                metric = ra.get("metric", "?")
                per_metric[metric][0] += 1
                per_metric[metric][1] = max(per_metric[metric][1], delta)
                if delta > args.tolerance:
                    differences.append(f"{label}: measured_value {va} -> {vb} (|d|={delta:.3g})")

    print(f"frames compared: {len(common)}   values compared: {compared}   "
          f"differences: {len(differences)}")
    if old_seconds and new_seconds:
        print(f"engine time: old {old_seconds:.1f}s  new {new_seconds:.1f}s  "
              f"({old_seconds / new_seconds:.1f}x)")
    print("per metric (n, max |new - old|):")
    for metric, (n, worst) in sorted(per_metric.items()):
        print(f"  {metric:28s} {n:6d}  {worst:.3g}")
    for line in differences[:200]:
        print("DIFF", line)
    if len(differences) > 200:
        print(f"... {len(differences) - 200} more")
    return 1 if differences else 0


if __name__ == "__main__":
    sys.exit(main())
