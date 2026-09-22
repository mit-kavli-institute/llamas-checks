#!/usr/bin/env python3
"""Aggregate verdict tally over ALL baseline files per type, WITHOUT writing reports.

Uses QAEngine.run() directly (no CLI, no .qa.json side effects) so the Box
baseline folders stay clean. Reports PASS/WARN/FAIL counts per type and lists any
FAIL (which would be a real anomaly among the 'normal' baselines).

The baselines root (containing Bias/ Darks/ Arcs/ lamp_flats/ twilight_flats/)
comes from --baselines-root or the LLAMAS_QA_BASELINES environment variable.
"""
import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

from llamas_checks import qa_engine as E
from llamas_checks.paths import CONFIG_DIR

FOLDERS = ["Bias", "Darks", "Arcs", "lamp_flats", "twilight_flats"]


def main():
    parser = argparse.ArgumentParser(
        description="Tally PASS/WARN/FAIL over every baseline file per type (no .qa.json written).")
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "qa_config_cal.yaml",
                        help="QA YAML config (default: the shipped qa_config_cal.yaml)")
    parser.add_argument("--baselines-root", default=os.environ.get("LLAMAS_QA_BASELINES"),
                        help="QA_baselines root with the sorted type folders (default: $LLAMAS_QA_BASELINES)")
    parser.add_argument("--out", default="batch_tally.json",
                        help="output JSON path (default: ./batch_tally.json)")
    args = parser.parse_args()
    if not args.baselines_root:
        parser.error("--baselines-root is required (or set LLAMAS_QA_BASELINES)")
    base = args.baselines_root

    # validate once, then run with no_validate for speed
    cfg = E.load_yaml(args.config)
    E.validate_config(cfg, args.config)
    eng = E.QAEngine(cfg)

    grand = {}
    fails = []
    for fold in FOLDERS:
        files = sorted(glob.glob(f"{base}/{fold}/*.fits"))
        tally = Counter()
        for f in files:
            try:
                rep = eng.run(Path(f))
                v = rep["overall_verdict"]
            except Exception as exc:
                v = "ERROR"
            tally[v] += 1
            if v == "FAIL":
                fails.append((fold, os.path.basename(f)))
        grand[fold] = dict(tally)
        print(f"{fold:16s} n={len(files):3d}  " +
              "  ".join(f"{k}={tally[k]}" for k in ("PASS", "WARN", "FAIL", "ERROR") if tally[k]))

    print("\nFAIL files (real anomalies among baselines):")
    for fold, name in fails:
        print(f"  {fold}: {name}")
    if not fails:
        print("  (none)")

    out = os.path.abspath(args.out)
    json.dump({"per_type": grand, "fails": fails}, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
