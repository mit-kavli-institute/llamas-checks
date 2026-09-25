"""Observing-GUI entry point for LLAMAS raw-frame QA checks.

Installed as the ``llamas-checks`` console script (also ``python -m llamas_checks``).
The observing GUI calls this on each newly written frame; the tool is quiet by
default and signals only through its exit code, with an optional JSON report
named after the frame (``--report`` alone writes ``<frame>.qa.json`` in the current
directory, ``--report DIR`` / ``--report-dir DIR`` inside a directory, ``--report
file.json`` to an explicit file) and a ``--verbose`` human-readable summary.
A report is written only for warn/fail verdicts unless ``--report-all`` is given.

Exit codes: 0 = pass, 1 = warn, 2 = fail, 3 = system error
"""

import argparse
import sys

from .llamasQATests import check_image


def main():
    parser = argparse.ArgumentParser(prog="llamas-checks",
                                     description="Run QA checks on a raw LLAMAS frame.")
    parser.add_argument("input_path", help="a single raw FITS/MEF file")
    parser.add_argument("--suite", default="basic_cal",
                        help="config name tried as <calib-root>/<suite>.yaml before the "
                             "PRODCATG auto-selection (default: basic_cal)")
    parser.add_argument("--qa-yaml", default=None,
                        help="explicit QA YAML config; overrides --suite/--calib-root")
    parser.add_argument("--calib-root", default=None,
                        help="directory holding the QA configs (default: the shipped configs/)")
    where = parser.add_mutually_exclusive_group()
    where.add_argument("--report", nargs="?", const=".", default=None, metavar="PATH",
                       help="write the JSON report, named after the frame (<frame>.qa.json): "
                            "with no value into the current directory, with an existing "
                            "directory into that directory, otherwise PATH is the file to write")
    where.add_argument("--report-dir", default=None, metavar="DIR",
                       help="write the JSON report into this directory, named after the "
                            "frame (<frame>.qa.json, the same name the engine gives its "
                            "sidecars); the directory is created when a report is written")
    parser.add_argument("--report-all", action="store_true",
                        help="also write a report when the frame passes; by default a "
                             "report is written only for warn and fail (no file = pass)")
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="print the result summary and failing-rule detail; by default the tool "
             "is quiet and signals only through its exit code "
             "(0=pass, 1=warn, 2=fail, 3=system error).",
    )
    args = parser.parse_args()

    try:
        results = check_image(
            input_path=args.input_path,
            suite=args.suite,
            qa_yaml=args.qa_yaml,
            calib_root=args.calib_root,
            report=args.report,
            verbose=args.verbose,
            report_dir=args.report_dir,
            report_all=args.report_all,
        )
    except Exception as exc:
        # Exactly one stderr line, even for multi-line messages (validator
        # error lists, PyYAML marks), so the GUI log stays one row per frame.
        text = " | ".join(line.strip() for line in str(exc).splitlines() if line.strip())
        print(f"system error: {text}", file=sys.stderr)
        return 3

    status = results.get("status", "pass")  # "pass" | "warn" | "fail"

    # Quiet by default: the exit code IS the interface (0=pass, 1=warn, 2=fail).
    # --verbose opts into the human-readable summary (and the per-rule detail that
    # check_image prints). A system error still reported above (exit 3), regardless.
    if args.verbose:
        message = results.get("message", status)
        print(message, file=sys.stdout if status == "pass" else sys.stderr)
        if results.get("report_path"):
            print(f"report: {results['report_path']}", file=sys.stderr)

    return {"pass": 0, "warn": 1, "fail": 2}.get(status, 2)


if __name__ == "__main__":
    sys.exit(main())
