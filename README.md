# llamas-checks

Automatic quality checks for **raw LLAMAS frames**, run by the LLAMAS observing GUI on each
frame as it is written and usable from the command line at the telescope. The checks work on
multi-extension FITS files straight off the instrument, before any reduction takes place.

They look for four classes of problems: shutter faults, camera warming or loss of cooling, odd detector structure, and
saturation. It runs **per detector** across all 24 extensions, so a fault on one camera is
reported by name rather than hidden in a frame-wide average. 

The package is standalone: it does not import or require the `llamas-pyjamas` reduction pipeline, whose own QA (fibre-trace counts,
wavelength-solution and flat-field validation) runs later on reduced products.

## Install

Python 3.11 or newer, in a conda environment or a venv. Dependencies are `numpy`, `astropy` and
`pyyaml` only.

```bash
# from a checkout (editable, with pytest):
git clone git@github.com:mit-kavli-institute/llamas-checks.git
cd llamas-checks
pip install -e ".[dev]"

# or straight from GitHub:
pip install git+ssh://git@github.com/mit-kavli-institute/llamas-checks.git
```

This installs three commands: `llamas-checks`, `llamas-checks-engine` and
`llamas-checks-validate`. `python -m llamas_checks <file>` is an alias for `llamas-checks`.

## Checking a single frame

This is the normal use at the telescope. Point it at a frame; it reads the frame's `PRODCATG`
header keyword to decide whether to apply the calibration or the science checks, so you do not
tell it which kind of frame it is.

```bash
llamas-checks /path/to/LLAMAS_..._mef.fits -v
```

**Always use `-v` interactively.** Without it the tool is deliberately silent and reports only
through its exit code. With `-v` you get a one-line verdict, a `structure:` line (extensions present, missing
cameras, placeholders), and a line for every check that fired, naming the detector responsible.

- `--report out.json` writes the full per-check detail to a file — useful for the night log or
  for sending to someone else to look at. Nothing is written next to the input frame.
- **Exit codes: `0` pass, `1` warn, `2` fail, `3` system error.**
- It takes one frame at a time. Pointed at a directory it exits `3`.

## Checking a whole night

To sweep a directory of frames — for example, vetting the night's calibrations before reducing —
use the engine directly. This is the only tool that accepts a directory.

```bash
llamas-checks-engine /path/to/night_dir --config qa_config_cal.yaml --summary-only
llamas-checks-engine /path/to/night_dir --config qa_config_science.yaml --summary-only
```

There is no flag to switch between single-frame and directory mode; the tool looks at what you
gave it. Pointed at a directory, it:

- Reads that directory **one level deep only**. Sub-folders are ignored, so a night split into
  `cals/` and `sci/` needs a run per folder.
- Picks up files ending `.fits`, `.fit` or `.fts` and checks them in parallel (`--jobs`,
  default 8).
- Writes a `<frame>.qa.json` report **next to each frame**, in the raw data directory.
- With `--summary-only`, prints one `filename VERDICT` line per frame once the run finishes,
  in filename order.
- Returns a single exit code for the whole sweep: `2` if any frame errored, otherwise `1` if
  any frame failed, otherwise `0`.

### Why calibrations and science are two separate runs

`--config` applies to every file in the directory, and unlike the single-frame tool this mode
does *not* work out each frame's type for itself. A frame that the config you passed does not
cover is neither failed nor errored — it comes back as a **WARN reading `no rules matched`**,
meaning it was not actually checked. On a mixed directory the calibration run checks the biases,
darks, arcs and flats and returns "no rules matched" for every science frame; the science run
does the reverse. Run both, and read each one only for the frames it was meant to cover. A WARN
in a sweep is therefore ambiguous on its face; the per-frame line tells you which kind it is.

### The reports are overwritten by the second run

Each sweep writes a `.qa.json` for **every** frame it looked at, including the ones it had no
rules for, so the science sweep overwrites the calibration sweep's reports for the calibration
frames with a `no rules matched` stub, and vice versa. After running both, only the frames
belonging to the *last* run still have a meaningful report on disk. The console output is not
affected, so keep it:

```bash
llamas-checks-engine /path/to/night_dir --config qa_config_cal.yaml --summary-only | tee cal_qa.log
llamas-checks-engine /path/to/night_dir --config qa_config_science.yaml --summary-only | tee sci_qa.log
```

For the detailed JSON of one frame, re-run `llamas-checks --report` on it, which writes only
where you tell it to.

### Three practical notes

- **Read the summary lines, not the exit code.** A sweep exits `0` when frames merely warned.
- **The exit codes here do not mean the same thing as the single-frame tool's.** There, `1`
  means warn and `2` means fail; in a sweep, `1` means fail and `2` means something errored.
- An empty directory, or one with no FITS files in it, is treated as an error, exit `2`.

## What the checks look for

### Calibration frames

Applies to bias, dark, ThAr arc, LDLS lamp flat and twilight/sky flat. The frame type is taken
from the header product category (`PRODCATG`), not from the `OBJECT` string.

| Check | What it catches | Level |
|---|---|---|
| Shutter vs requested time | shutter stuck open, exposure aborted or truncated | **FAIL** |
| Row / column structure | banding, blooming, or a warming glow on an unilluminated frame | **FAIL** (all cal types) |
| Saturation | light leak or saturated pixels (above 63000 ADU) | **FAIL** on bias, WARN elsewhere |
| CCD temperature — warm / hot / shut-off | cooling degrading or failed | WARN / **FAIL** / **FAIL** |
| Background level and read noise | bias-level drift, excess noise | WARN |

- Whole-frame level and read-noise checks are **not** applied to arcs and flats, because
  brightness scales with exposure time and a fixed level band would be meaningless. The
  unilluminated edge strip is still checked on every frame type.
- The structure limits are set separately for each detector and are deliberately generous:
  naturally structured detectors, and red ThAr arcs that normally saturate, pass comfortably.
  A trip means a **gross** anomaly, not marginal texture.
- Cameras missing from the frame are skipped, never failed. A missing camera appears as a
  placeholder extension (a constant frame of all-ones in real frames, all-zeros from the pipeline
  validator; any other constant frame, e.g. a railed detector, is evaluated normally) or as an
  absent HDU; both are
  listed in the report's `structure` block.

### Science frames

| Check | What it catches | Level |
|---|---|---|
| Shutter vs requested time | you did not get the exposure you asked for | **FAIL** |
| Camera-warming gradient | dark-current glow from a warming detector | WARN |
| Saturation | saturated target or sky (more than 2% of pixels) | WARN |
| CCD temperature — warm / hot / shut-off | cooling degrading or failed | WARN / **FAIL** / **FAIL** |

- There are no background-level or structure checks on science frames. Real sky and
  dispersed-spectrum structure on a bright or long exposure cannot be separated from a detector
  fault by any fixed limit, so attempting it only produces false alarms.
- The warming-gradient check is skipped on exposures shorter than 8 s.

### Warm cameras: the one thing to understand

Each camera is judged against **its own** healthy operating temperature, not against a single
number for the whole array. The detectors' normal temperatures differ by around 26 °C, and the
red cameras run warmest. A camera warns at 8 °C above its own baseline (10 °C for the
reds) and fails at 15 °C (16 °C for the reds), with an absolute failure above −60 °C where the
cooler is at its limit.

**The header temperature lags the actual problem.** A camera can already be growing a
dark-current glow across the detector while its reported temperature still reads normal. This is
why science frames carry the separate warming-gradient check, which measures the glow in the
pixels themselves as a background gradient per second of exposure (warning above 1.3 ADU/s). In
a real case from May 2026 a green camera showed 2.1 ADU/s against a healthy value of 0.06 or
less, while its header temperature sat exactly at its baseline. **A warming-gradient WARN is
worth acting on even when the temperature reads fine.**

### What to do when something fires

- **FAIL — shutter.** The frame is not the exposure you requested. Retake it, and check the
  shutter before continuing the sequence.
- **FAIL — structure.** Inspect the detector named in the output. Do not use the frame as a
  master calibration.
- **FAIL — temperature hot or shut-off.** Cooling needs attention. Data from that camera will
  carry heavy dark current until it is fixed.
- **WARN — camera-warming gradient.** The earliest sign of a warming camera. Watch that detector
  over the next few frames.
- **WARN — level, noise or saturation.** Usually benign drift. Note it and keep observing;
  roughly one in twelve normal calibration frames warns this way.
- **Exit code 3, or an ERROR verdict.** The check did not actually run — typically an unreadable
  or truncated file. This is *not* a pass; re-copy or re-take the frame and run it again.

A check whose header keyword is missing from the frame is reported as *skipped*, never as a
failure.

## CLI reference

### `llamas-checks` — one frame, config auto-selected

```
llamas-checks <file> [-v] [--report out.json] [--qa-yaml cfg.yaml] [--calib-root DIR] [--suite NAME]
```

`--qa-yaml` overrides the auto-selection with an explicit config. `--calib-root` points at a
directory holding your own `qa_config_cal.yaml` / `qa_config_science.yaml` / `qa_config.yaml`
instead of the shipped ones; `--suite NAME` (default `basic_cal`) is tried first as
`<calib-root>/<NAME>.yaml`. If `PRODCATG` is missing or unrecognised the base `qa_config.yaml` is
used, and a frame whose type no rule set matches is reported as a **fail** with a message saying
which identification keyword was missing.

| Exit | Meaning |
|:--:|---|
| 0 | pass |
| 1 | warn |
| 2 | fail (including "could not identify frame type") |
| 3 | system error: missing file, unreadable input, invalid config, directory given |

Quiet by default: `0`/`1`/`2` print nothing; `3` always prints one line to stderr.

### `llamas-checks-engine` — one frame or a directory, explicit config

```
llamas-checks-engine <file-or-dir> --config <yaml> [--jobs N] [--summary-only] [--no-validate]
```

`--config` takes a path, or the bare name of a shipped config (`qa_config_cal.yaml`,
`qa_config_science.yaml`, `qa_config.yaml`). `--no-validate` skips the schema check of the
config before the run. A `<name>.qa.json` is written next to every input frame.

| Exit | Meaning |
|:--:|---|
| 0 | PASS or WARN (every frame, in directory mode) |
| 1 | FAIL (any frame) |
| 2 | ERROR: unreadable file, bad config, off-mode frame, an unevaluable rule, an empty directory, or a path that does not exist |

### `llamas-checks-validate` — check a config file

```
llamas-checks-validate <yaml>
```

Validates a QA config against the engine's schema (extensions, regions, metrics, lookup tables,
`header_check` rules) and prints `VALID: <path>`, or `INVALID: <path>` followed by one `ERROR:`
line per problem. Exit `0` valid, `1` invalid, `2` the file could not be loaded. Run it after
editing or regenerating a YAML.

### `--report` JSON layout

`llamas-checks --report out.json` writes:

```json
{
  "status": "warn",
  "message": "WARN: 2 warn check(s): edge_background_level@2.A.Green, ...",
  "suite": "basic_cal",
  "fits_file": "/path/to/LLAMAS_..._mef.fits",
  "overall_verdict": "WARN",
  "summary": {"total_checks": 190, "evaluated_checks": 154, "skipped_checks": 36,
              "passed_checks": 152, "failed_checks": 2, "fail_effects": 0, "warn_effects": 2,
              "missing_extensions": 0, "placeholder_extensions": 2},
  "structure": {
    "n_extensions": 24,
    "expected_extensions": 24,
    "missing_cameras": [],
    "identity_mismatches": [],
    "placeholder_extensions": ["1.A.Blue", "2.A.Blue"]
  },
  "report": { "...the full engine report: metadata, active_rule_sets, results[]..." }
}
```

`status` (`pass`/`warn`/`fail`) is what the exit code is derived from. `overall_verdict`,
`summary` and `report` are present whenever the engine ran; `report.results` is the per-rule list
(`rule`, `extension`, `hdu_index`, `measured_value`, `limits`, `passed`, `severity`,
`verdict_effect`, `status` = `EVALUATED`/`SKIPPED`/`PLACEHOLDER`/`MISSING`/`ERROR`, `message`).

The `structure` block is present on **every** return path. It comes from a header-only pass
over the file made before the engine runs: `n_extensions` is the number of image extensions
present (`len(hdul) - 1`), `expected_extensions` is 24, `missing_cameras` lists detectors
(`"1.A.Blue"` style, in detector order) whose HDU is absent, `identity_mismatches` lists
extensions whose header `BENCH`/`SIDE`/`COLOR` disagree with the position they occupy
(`{extension, hdu_index, header_identity}`), and `placeholder_extensions` lists extensions
that are present but constant-valued (from the engine results with status `PLACEHOLDER`; empty
when no engine report was produced). It is informational: **the structure block never changes
`status` or the exit code.**

The engine's own `<frame>.qa.json` (written by `llamas-checks-engine`) is the inner `report`
object on its own: `fits_file`, `instrument`, `metadata`, `active_rule_sets`, `overall_verdict`,
`summary`, `results`. An errored frame gets `{"fits_file", "overall_verdict": "ERROR", "error"}`.

## Python API

```python
from llamas_checks.llamasQATests import check_image
from llamas_checks.validate import inspect_structure

result = check_image("/path/to/LLAMAS_..._mef.fits", report="out.json")
print(result["status"], result["message"], result["structure"]["missing_cameras"])
structure = inspect_structure("/path/to/LLAMAS_..._mef.fits")   # header-only, no pixel reads
```

- `check_image(input_path, suite="basic_cal", qa_yaml=None, calib_root=None, report=None,
  verbose=False) -> dict` — the function behind `llamas-checks`. Returns `status`, `message`,
  `suite`, `fits_file`, `structure`, and (when the engine ran) `overall_verdict` and `summary`.
  Raises `QAEngineError` on system-level problems; QA problems are returned, never raised.
- `QAEngine(config).run(Path) -> dict` in `llamas_checks.qa_engine` — the rule engine used by
  both commands; `load_yaml(path)` and `validate_config(config, path)` sit beside it. This is
  the report-free way to run many frames in-process (see `scripts/batch_tally.py`).
- `inspect_structure(fits_path, extensions=None) -> dict` and `detector_label(header)` in
  `llamas_checks.validate` — the structure block on its own; `extensions` is the config's
  extension list, used for the identity comparison.
- `llamas_checks.paths` exposes `CONFIG_DIR`, `BASELINES_DIR` and the shipped config names.

## GUI integration

The observing GUI shells out to `llamas-checks <file>` (or `python -m llamas_checks <file>`
when a console script is not on the PATH) after each frame is written and **branches on the
exit code**: `0` pass, `1` warn, `2` fail, `3` the check itself did not run. Nothing is printed
on `0`–`2`, so there is no output to parse. For detail — which rule fired on which detector, the
missing/placeholder cameras — pass `--report <path>` and read the JSON described above. Do not
give the GUI `llamas-checks-engine`: its exit codes mean different things and it writes reports
into the raw data directory.

## Regenerating thresholds

The per-detector limits in `qa_config_cal.yaml` / `qa_config_science.yaml` are **generated** from
a set of baseline calibration frames (`llamas_checks/baselines/qa_thresholds_derived.json`
holds the derived limits and the observed statistics). To refresh them as more nights arrive,
run the scripts in `scripts/` in order. Each has `--help`; the data roots come from arguments or
from these environment variables, which are only argparse defaults:

| Variable | Argument | Contents |
|---|---|---|
| `LLAMAS_QA_BASELINES` | `--baselines-root` | the `QA_baselines` folder: `copies/` (untouched originals) plus the sorted `Bias/ Darks/ Arcs/ lamp_flats/ twilight_flats/` |
| `LLAMAS_QA_WARM_DIR` | `--warm-dir` | the 2026-05-06 warm-incident frames (validation only) |
| `LLAMAS_QA_COMMISSIONING_DIR` | `--commissioning-dir` | the `ut20260710_11` commissioning frames (validation only) |

For example:

```bash
export LLAMAS_QA_BASELINES=/data/LLAMAS/QA_baselines
export LLAMAS_QA_WARM_DIR=/data/LLAMAS/20260505_06-selected
export LLAMAS_QA_COMMISSIONING_DIR=/data/LLAMAS/ut20260710_11
```

Then:

```bash
python scripts/sort_baselines.py                 # copies/ -> per-type folders + manifest.csv (idempotent)
python scripts/extract_stats.py                  # per-file, per-extension stats -> ./qa_stats_raw.json (--jobs 6)
python scripts/aggregate_thresholds.py           # -> llamas_checks/baselines/qa_thresholds_derived.json + qa_tracking_baselines.csv
python scripts/gen_configs.py                    # -> llamas_checks/configs/qa_config_{cal,science}.yaml
llamas-checks-validate llamas_checks/configs/qa_config_cal.yaml
llamas-checks-validate llamas_checks/configs/qa_config_science.yaml
```

`aggregate_thresholds.py` prints its self-checks (band breaches among normal frames, the held-out
June-2026 odd dark against the dark structure caps, and — when `--warm-dir` is set — the warm
folder against the shutter, edge-background and temperature limits). `extract_stats.py --out`,
`aggregate_thresholds.py --raw/--out-dir` and `gen_configs.py --derived/--out-dir` let you keep
intermediate files elsewhere; the defaults write into the package so a fresh run reproduces the
shipped files exactly (the only text in the YAMLs that is not derived is the one-line header).

Two validation scripts sit beside them:

```bash
python scripts/batch_tally.py        # all baseline files per type, in-process, no .qa.json written -> ./batch_tally.json
python scripts/run_qa_tests.py       # the 15-case matrix through the engine CLI (writes .qa.json next to the inputs) -> ./qa_test_results.json
```

`run_qa_tests.py` skips any case whose root is unset or whose file is absent (`MISSING FILE`).
Thresholds, methodology and the validation results are written up in
[`docs/QA_TESTS_SUMMARY.md`](docs/QA_TESTS_SUMMARY.md).

## Repository layout

```
llamas_checks/                 the package
  QA_assess.py                 `llamas-checks` entry point (exit codes 0/1/2/3)
  llamasQATests.py             check_image(): config auto-selection, structure block, verdict collapse
  qa_engine.py                 QAEngine + `llamas-checks-engine` CLI
  qa_config_validator.py       config schema check + `llamas-checks-validate` CLI
  validate.py                  MEF structure inspection (inspect_structure, detector_label)
  paths.py                     CONFIG_DIR, BASELINES_DIR, shipped config names
  configs/                     qa_config.yaml (base), qa_config_cal.yaml, qa_config_science.yaml (generated)
  baselines/                   qa_thresholds_derived.json, qa_tracking_baselines.csv
scripts/                       threshold pipeline + validation scripts (see above)
tests/                         pytest suite
docs/QA_CHECKS_CATALOGUE.md    every check, its threshold, and the bad-image test cases (start here)
docs/QA_THRESHOLDS_TABLES.md   generated per-detector limits (scripts/gen_thresholds_doc.py)
docs/QA_TESTS_SUMMARY.md       design rationale, provenance and the July-2026 validation run
```

## Tests

```bash
pip install -e ".[dev]"
pytest tests/
```

The tests build small synthetic MEF files, so no instrument data is needed.

What each test asserts, the real bad-frame regression cases and the current thresholds are
catalogued in [`docs/QA_CHECKS_CATALOGUE.md`](docs/QA_CHECKS_CATALOGUE.md).

## Licence

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 MIT Kavli Institute for Astrophysics and
Space Research.
