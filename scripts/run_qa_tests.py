#!/usr/bin/env python3
"""Run the YAML QA engine on each test case and collect the numeric result.

For every case: invoke the engine (python -m llamas_checks.qa_engine) as a
subprocess (so the real EXIT CODE is exercised: 0 = PASS/WARN, 1 = FAIL,
2 = ERROR), then read the emitted .qa.json to extract the overall verdict and the
specific rules that fired FAIL / WARN and on which extensions. Prints a table and
writes qa_test_results.json.

NOTE: because it drives the engine CLI, this script writes a <name>.qa.json next
to every input frame -- on purpose (that is the exit-code path being tested). Do
not point it at folders that must stay clean; batch_tally.py is the report-free
alternative.

The three data roots come from arguments or environment variables:
  --baselines-root      LLAMAS_QA_BASELINES          (QA_baselines: Bias/ Darks/ ... copies/)
  --warm-dir            LLAMAS_QA_WARM_DIR           (2026-05-06 warm-incident frames)
  --commissioning-dir   LLAMAS_QA_COMMISSIONING_DIR  (ut20260710_11 commissioning frames)
Cases whose root is unset, or whose file is absent, are reported as MISSING FILE
and skipped.
"""
import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

from llamas_checks.paths import CONFIG_DIR

PY = sys.executable
CAL = str(CONFIG_DIR / "qa_config_cal.yaml")
SCI = str(CONFIG_DIR / "qa_config_science.yaml")

# (label, expectation, root key, path relative to that root, config)
CASES = [
    ("Normal BIAS",         "PASS/WARN", "baselines", "Bias/LLAMAS_2026-04-07_19-23-19.3_CAL22_mef.fits",       CAL),
    ("Normal DARK",         "PASS/WARN", "baselines", "Darks/LLAMAS_2026-04-07_21-13-34.9_CAL22_mef.fits",       CAL),
    ("Normal ARC (ThAr)",   "PASS/WARN", "baselines", "Arcs/LLAMAS_2026-04-07_20-40-49.6_CAL22_mef.fits",        CAL),
    ("Normal LDLS FLAT",    "PASS/WARN", "baselines", "lamp_flats/LLAMAS_2026-04-07_20-37-07.4_CAL22_mef.fits",   CAL),
    ("Normal SKY FLAT",     "PASS/WARN", "baselines", "twilight_flats/LLAMAS_2026-05-02_22-15-33.4_CAL22_mef.fits", CAL),
    ("ODD-STRUCTURE dark",  "FAIL",      "baselines", "copies/LLAMAS_2026-06-04_20-36-33.1_CAL22_mef.fits",       CAL),
    ("ODD-STRUCTURE arc",   "FAIL",      "commissioning", "LLAMAS_2026-07-10_20-22-43.7_CAL22_mef.fits",          CAL),
    ("ODD-STRUCTURE arc #2","FAIL",      "commissioning", "LLAMAS_2026-07-10_20-22-58.8_CAL22_mef.fits",          CAL),
    ("Normal commissioning arc", "PASS/WARN", "commissioning", "LLAMAS_2026-07-10_20-22-28.7_CAL22_mef.fits",    CAL),
    ("WARM sci (shutter fault)", "FAIL", "warm", "LLAMAS_2026-05-06_05-10-56.6_SCI22_mef.fits",                    SCI),
    # 06-08-21.4: camera 3A.green is WARMING -- a dark-current glow that the header CCD
    # temperature does NOT yet show (it reads -92.1, at its -92.9 baseline). The
    # exposure-normalized background-gradient RATE flags it: 3A.green ~2.1 ADU/s vs a
    # healthy <=0.06 -> camera_warming_gradient WARN (severity WARN, so overall WARN =
    # engine exit 0). shutter/temps are otherwise fine. (Was previously mislabelled
    # "normal", and briefly FAILed under the rejected structure approach.)
    ("WARM sci (3A.green glow)", "PASS/WARN", "warm", "LLAMAS_2026-05-06_06-08-21.4_SCI22_mef.fits",         SCI),
    # Commissioning science: files store CCDTEMP1 (no underscore) so temp rules now
    # EVALUATE (were SKIPPED) and temps are healthy. 4A.red has fixed electronic banding
    # (not a smooth dark-current gradient), so the gradient-rate metric does NOT flag it
    # -> PASS. Banding != warming; gross banding on unilluminated cals is still caught by
    # the cal structure FAIL. (Briefly FAILed under the rejected science-structure approach.)
    ("Commissioning sci (4A.red banding)", "PASS/WARN", "commissioning", "LLAMAS_2026-07-11_05-05-22.4_SCI22_mef.fits", SCI),
    ("WARM cal ARC (shutter)",   "FAIL", "warm", "LLAMAS_2026-05-06_05-50-23.7_CAL0_mef.fits",                     CAL),
    ("WARM cal BIAS (shutter ok)", "PASS", "warm", "LLAMAS_2026-05-06_05-56-25.8_CAL0_mef.fits",                   CAL),
    ("WARM cal BIAS (shutter)",  "FAIL", "warm", "LLAMAS_2026-05-06_05-56-30.7_CAL0_mef.fits",                     CAL),
]

EXIT_MEANING = {0: "PASS/WARN", 1: "FAIL", 2: "ERROR"}


def fired_rules(report):
    """Return dict {verdict_effect: [(rule, ext, detail)]} for FAIL/WARN results."""
    out = {"FAIL": [], "WARN": [], "SKIPPED": 0, "PLACEHOLDER": 0, "ERROR": [], "NO_RULES": 0}
    for r in report.get("results", []):
        eff = r.get("verdict_effect")
        status = r.get("status")
        if status == "SKIPPED":
            out["SKIPPED"] += 1
            continue
        if status == "PLACEHOLDER":
            out["PLACEHOLDER"] += 1
            continue
        if status == "NO_RULES_MATCHED":
            out["NO_RULES"] += 1
            continue
        if status == "ERROR":
            out["ERROR"].append((r.get("rule"), r.get("extension"), r.get("message", "")))
            continue
        if r.get("passed") is False and eff in ("FAIL", "WARN"):
            out[eff].append((r.get("rule"), r.get("extension"), r.get("measured_value")))
    return out


def summarize(rules):
    def cnt(lst):
        c = Counter(rule for rule, _, _ in lst)
        return ", ".join(f"{k}×{v}" for k, v in c.most_common())
    parts = []
    if rules["FAIL"]:
        parts.append(f"FAIL[{len(rules['FAIL'])}]: {cnt(rules['FAIL'])}")
    if rules["WARN"]:
        parts.append(f"WARN[{len(rules['WARN'])}]: {cnt(rules['WARN'])}")
    if rules["ERROR"]:
        parts.append(f"ERROR[{len(rules['ERROR'])}]")
    if rules["NO_RULES"]:
        parts.append("NO_RULES_MATCHED")
    if rules["SKIPPED"]:
        parts.append(f"skip={rules['SKIPPED']}")
    if rules["PLACEHOLDER"]:
        parts.append(f"placeholder={rules['PLACEHOLDER']}")
    return "; ".join(parts) if parts else "all rules passed"


def main():
    parser = argparse.ArgumentParser(
        description="Run the 15-case engine test matrix and record exit code + fired rules per case.")
    parser.add_argument("--baselines-root", default=os.environ.get("LLAMAS_QA_BASELINES"),
                        help="QA_baselines root (default: $LLAMAS_QA_BASELINES)")
    parser.add_argument("--warm-dir", default=os.environ.get("LLAMAS_QA_WARM_DIR"),
                        help="2026-05-06 warm-incident frames (default: $LLAMAS_QA_WARM_DIR)")
    parser.add_argument("--commissioning-dir", default=os.environ.get("LLAMAS_QA_COMMISSIONING_DIR"),
                        help="ut20260710_11 commissioning frames (default: $LLAMAS_QA_COMMISSIONING_DIR)")
    parser.add_argument("--out", default="qa_test_results.json",
                        help="output JSON path (default: ./qa_test_results.json)")
    args = parser.parse_args()
    roots = {"baselines": args.baselines_root, "warm": args.warm_dir,
             "commissioning": args.commissioning_dir}

    results = []
    print(f"{'CASE':22s} {'EXIT':4s} {'VERDICT':8s} {'PRODCATG':11s} FIRED RULES")
    print("-" * 110)
    for label, expect, root_key, rel, cfg in CASES:
        root = roots[root_key]
        path = os.path.join(root, rel) if root else None
        if path is None or not os.path.exists(path):
            print(f"{label:22s}  --  MISSING FILE: {path or f'<{root_key} root unset>/{rel}'}")
            continue
        proc = subprocess.run([PY, "-m", "llamas_checks.qa_engine", path, "--config", cfg, "--summary-only"],
                              capture_output=True, text=True)
        code = proc.returncode
        qajson = Path(path).with_suffix(".qa.json")
        verdict, prodcatg, fired, detail = "?", "?", {}, ""
        if qajson.exists():
            rep = json.load(open(qajson))
            verdict = rep.get("overall_verdict", "?")
            prodcatg = (rep.get("metadata", {}) or {}).get("exposure_type") or \
                       (rep.get("selected_rule_set") or rep.get("rule_set") or "?")
            fired = fired_rules(rep)
            detail = summarize(fired)
        row = {"case": label, "expect": expect, "exit_code": code,
               "exit_meaning": EXIT_MEANING.get(code, str(code)),
               "verdict": verdict, "prodcatg": prodcatg, "detail": detail,
               "config": os.path.basename(cfg), "file": os.path.basename(path)}
        results.append(row)
        print(f"{label:22s} {code:<4d} {verdict:8s} {str(prodcatg):11s} {detail}")
        if proc.stderr.strip():
            print(f"    stderr: {proc.stderr.strip()[:200]}")

    out = os.path.abspath(args.out)
    json.dump(results, open(out, "w"), indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
