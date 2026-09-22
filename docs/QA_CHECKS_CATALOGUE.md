# LLAMAS raw-frame QA — check catalogue, thresholds and bad-image test cases

Status as of **2026-09-22** (`llamas-checks` 0.1.0). Every example result below was produced by
re-running the shipped package on the frame named, on this date. The per-detector numbers behind
the checks are in the generated appendix [`QA_THRESHOLDS_TABLES.md`](QA_THRESHOLDS_TABLES.md);
the design rationale and the original July-2026 validation campaign are in
[`QA_TESTS_SUMMARY.md`](QA_TESTS_SUMMARY.md).

Contents

1. [Running the checks](#1-running-the-checks)
2. [Frame types, rule-set selection and geometry](#2-frame-types-rule-set-selection-and-geometry)
3. [Check catalogue, by rule set](#3-check-catalogue-by-rule-set)
4. [Static thresholds](#4-static-thresholds)
5. [How the per-detector thresholds were derived](#5-how-the-per-detector-thresholds-were-derived)
6. [Bad-image test cases (real frames)](#6-bad-image-test-cases-real-frames)
7. [Automated test inventory](#7-automated-test-inventory)
8. [Regenerating](#8-regenerating)

---

## 1. Running the checks

```bash
llamas-checks /path/to/LLAMAS_2026-04-07_19-23-19.3_CAL22_mef.fits -v          # one frame, verdict + fired rules
llamas-checks /path/to/frame.fits --report /path/to/frame_qa.json               # quiet; full detail in the JSON
python -m llamas_checks /path/to/frame.fits -v                                  # same tool, module form
```

The frame type is read from the primary-header `PRODCATG`: `CAL.*` frames get
`qa_config_cal.yaml`, `SCI.*` frames get `qa_config_science.yaml`; an unrecognised or missing
`PRODCATG` falls back to the base `qa_config.yaml` and is reported as unidentified. Nothing is
written next to the input; only `--report` writes a file.

| exit code | meaning | printed without `-v` |
|---|---|---|
| 0 | pass | nothing |
| 1 | warn — flagged, but usable | nothing |
| 2 | fail — at least one FAIL-severity check tripped, or the engine could not evaluate the frame | nothing |
| 3 | system error — unreadable/truncated file, missing or invalid config | one line on stderr |

`llamas-checks-engine <dir> --config qa_config_cal.yaml --summary-only` sweeps a directory with one
config and writes a `<frame>.qa.json` beside every frame; its exit codes are 0 = PASS or WARN,
1 = FAIL, 2 = ERROR. `llamas-checks-validate <yaml>` checks a config file.

## 2. Frame types, rule-set selection and geometry

| rule set | `PRODCATG` | readout modes modelled | typical exposure |
|---|---|---|---|
| BIAS | `CAL.R-BIA` | FAST, SLOW | 0.001 s |
| DARK | `CAL.R-DRK` | SLOW | 600 s |
| LDLS_FLAT | `CAL.R-FLT` | FAST | 0.07–0.5 s |
| SKY_FLAT | `CAL.R-SKY` | FAST, SLOW | 1–60 s |
| ARC_THAR | `CAL.R-ARC` | FAST, SLOW | 0.05–1 s |
| SCIENCE | `SCI.R-*` (e.g. `SCI.R-SL`, `SCI.R-DT`) | any (no per-mode tables) | varies |

- `OBJECT` is never used for selection. The readout mode comes from `READ-MDE`.
- Detectors are 2048 × 2048; the frame has 24 image extensions, HDU 1–24, in
  red → green → blue order for bench-sides 1A, 1B, 2A, 2B, 3A, 3B, 4A, 4B. Detector names in all
  reports are `bench.side.Colour`, e.g. `3.A.Green`.
- Regions: `full_frame`; `bottom_stripe` = rows 2–28 × columns 100–1948 (unilluminated, used as the
  per-detector bias reference on every frame type); `top_stripe` = rows 2020–2046, same columns.
- A missing camera is a **placeholder** extension: every pixel constant at 1 (real frames) or 0
  (pipeline-generated). Placeholders are skipped by every pixel rule and are listed, never failed.
  A frame with a genuinely constant but non-0/1 detector (e.g. railed at 65535) is *not* a
  placeholder and is evaluated normally.
- A rule whose header keyword is absent, or whose lookup table has no entry for this detector and
  readout mode, is **SKIPPED** for that detector only. A rule that cannot be computed (bad region,
  unreadable data) gives an **ERROR** verdict, which `llamas-checks` reports as fail (exit 2), never
  as pass.
- Every result also carries a `structure` block: number of extensions, cameras whose HDU is absent,
  placeholder detectors, and any extension whose `BENCH`/`SIDE`/`COLOR` header disagrees with the
  position the config expects. It is informational and never changes the verdict.

## 3. Check catalogue, by rule set

Severity: **FAIL** blocks the verdict (exit 2); **WARN** flags but passes (exit 1). "per det × mode"
means the limit comes from a lookup table keyed on detector name and readout mode (appendix);
"per det" means keyed on detector only.

### BIAS (`CAL.R-BIA`) — 10 rules

| rule | what is measured | limit | severity |
|---|---|---|---|
| `shutter_exptime_consistency` | \|SEXPTIME − REXPTIME\| | passes within abs 0.214 s **or** rel 10 % | FAIL |
| `edge_background_level` | median of `bottom_stripe` | band [min, max], per det × mode | WARN |
| `frame_level_median` | median of `full_frame` | band [med_min, med_max], per det × mode | WARN |
| `frame_noise_rms` | std of `full_frame` | ≤ rms_max, per det × mode | WARN |
| `row_structure` | row-banding amplitude of `full_frame` (std of the per-row means) | ≤ row_max, per det × mode | FAIL |
| `column_structure` | column-banding amplitude of `full_frame` (std of the per-column means) | ≤ col_max, per det × mode | FAIL |
| `saturation_fraction` | fraction of `full_frame` pixels > 63000 ADU | ≤ frac_max, per det × mode (≈ 5 × 10⁻⁴) | FAIL |
| `ccd_temperature_warm` | `CCDTEMP_1` / `CCDTEMP1` / `CCDTEMP-1` of this extension | within [−140, warn_max], per det | WARN |
| `ccd_temperature_hot` | same keyword | ≤ fail_max, per det | FAIL |
| `ccd_temperature_shutoff` | same keyword | ≤ −60 °C, all cameras | FAIL |

### DARK (`CAL.R-DRK`) — 10 rules

Identical to BIAS with the DARK tables, except: shutter abs tolerance 0.238 s, and
`saturation_fraction` is **WARN** (hot pixels are tolerated on a 600 s dark). Row/column structure
stays FAIL.

### LDLS_FLAT (`CAL.R-FLT`), SKY_FLAT (`CAL.R-SKY`), ARC_THAR (`CAL.R-ARC`) — 9 rules each

| rule | what is measured | limit | severity |
|---|---|---|---|
| `shutter_exptime_consistency` | \|SEXPTIME − REXPTIME\| | abs 0.252 s (LDLS) / 0.348 s (SKY) / 0.328 s (ARC) **or** rel 10 % | FAIL |
| `edge_background_level` | median of `bottom_stripe` | band, per det × mode | WARN |
| `row_structure` / `column_structure` | banding amplitude of `full_frame` | ≤ heavy-tail cap, per det × mode | FAIL |
| `saturation_fraction` | fraction > 63000 ADU | ≤ frac_max, per det × mode | WARN |
| `ccd_temperature_warm` / `_hot` / `_shutoff` | `CCDTEMP*` per extension | as BIAS | WARN / FAIL / FAIL |

No `frame_level_median` or `frame_noise_rms` on illuminated frames: the level scales with exposure
time, so a fixed band is not physical. The structure caps for these types are deliberately loose
(1.5 × the worst normal frame, see §5) so real fibre and line structure passes; only a gross
anomaly trips them. Red ThAr arcs normally saturate, which is why saturation is WARN here and
structure, not saturation, is the defect discriminator.

### SCIENCE (`SCI.R-*`) — 6 rules

| rule | what is measured | limit | severity |
|---|---|---|---|
| `shutter_exptime_consistency` | \|SEXPTIME − REXPTIME\| | abs 1.0 s **or** rel 10 % | FAIL |
| `saturation_fraction` | fraction of `full_frame` > 63000 ADU | ≤ 0.02 | WARN |
| `camera_warming_gradient` | background-gradient **rate**: max(\|median(bottom ¼) − median(top ¼)\|, \|median(left ¼) − median(right ¼)\|) ÷ exposure time (`SEXPTIME` → `REXPTIME` → `EXPTIME`) | ≤ 1.3 ADU/s; skipped when exposure < 8 s | WARN |
| `ccd_temperature_warm` / `_hot` / `_shutoff` | `CCDTEMP*` per extension | as BIAS (same `temp` table) | WARN / FAIL / FAIL |

No level or structure checks on science: sky and dispersed-spectrum structure on a bright or long
exposure cannot be separated from a detector fault by any absolute limit. A warming detector is
caught by the gradient *rate*, because dark current accrues per second while sky gradients do not.

## 4. Static thresholds

| quantity | value | where |
|---|---|---|
| saturation pixel level | > 63000 ADU | all `saturation_fraction` rules |
| science saturation fraction | ≤ 0.02 (WARN) | SCIENCE |
| shutter relative tolerance | 10 % | all types |
| shutter absolute tolerance | BIAS 0.214 s · DARK 0.238 s · LDLS 0.252 s · SKY 0.348 s · ARC 0.328 s · SCIENCE 1.0 s | per type |
| camera-warming gradient rate | ≤ 1.3 ADU/s (WARN), `min_exptime` 8 s | SCIENCE |
| CCD temperature shut-off | > −60 °C → FAIL, every camera | all types |
| CCD temperature cold floor | −140 °C | all types |
| CCD temperature warm / hot margins | baseline + 8 / + 15 °C (green, blue); + 10 / + 16 °C (red) | per camera, all types |
| structure cap floor | 2.0 ADU | all cal types |
| RMS cap floor | 5.0 ADU | BIAS, DARK |
| saturation cap floor | 5 × 10⁻⁴ | all cal types |
| level band floor | ± 15 ADU | BIAS, DARK, edge stripes |

Per-camera temperature limits span a 26 °C range: 1.A.Red has a healthy baseline of −78.3 °C
(WARN above −68.3, FAIL above −62.3) while 1.A.Green sits at −104.7 °C (WARN above −96.7, FAIL
above −89.7). A single global threshold would either flag the reds permanently or miss a warming
blue; the full table is in the appendix.

## 5. How the per-detector thresholds were derived

Inputs: **153 baseline calibration frames** from three epochs, 2026-04-07, 2026-05-02 and
2026-06-30, in Box `LLAMAS_analysis/QA_baselines/` (sorted into `Bias/`, `Darks/`, `Arcs/`,
`lamp_flats/`, `twilight_flats/`, each with a `manifest.csv`). The known-bad 2026-06-04 dark was
held out for validation. The bias subset is 48 frames:

| date | FAST | SLOW |
|---|---|---|
| 2026-04-07 | 14 | 11 |
| 2026-05-02 | 1 | 0 |
| 2026-06-30 | 11 | 11 |

Per (type × readout mode × detector), 3344 normal detector-frame records in total:

- Samples are **MAD sigma-clipped** first, so an anomalous baseline frame cannot inflate its own cap.
- Level and edge-background bands: `median ± max(6 σ_robust, 15 ADU)` — an absolute ADU floor.
- RMS cap: `max(median + 8 σ_robust, 3 × median, 5.0)`.
- Row/column structure and saturation caps are **heavy-tail**: `max(1.5 × unscreened max, median + 8 σ_robust, floor)`
  with floors 2.0 (structure) and 5 × 10⁻⁴ (saturation). Persistently structured detectors keep their
  own loose cap instead of false-failing.
- CCD temperature: each camera's own sigma-clipped median plus the colour-aware margins above.
- Shutter absolute tolerance per type: `2 × max(4σ-clipped |SEXPTIME − REXPTIME|) + 0.2 s`, with the
  relative tolerance fixed at 10 %. The 0.30 s overrun of the 2026-05-02 LDLS flat was clipped as an
  outlier, which is why that frame fails its own type's 0.252 s tolerance (§6.1).

Self-check on the baselines themselves: 2.7 % of normal detector-frames sit outside their
edge-background band (WARN-level drift), none breach a FAIL cap.

Pipeline: `scripts/sort_baselines.py` → `scripts/extract_stats.py` → `scripts/aggregate_thresholds.py`
(writes `llamas_checks/baselines/qa_thresholds_derived.json` and `qa_tracking_baselines.csv`) →
`scripts/gen_configs.py` (writes the two YAMLs) → `llamas-checks-validate`. The appendix tables are
rendered from those files by `scripts/gen_thresholds_doc.py`.

## 6. Bad-image test cases (real frames)

All runs: `llamas-checks <file> -v --report <out>.json`, 2026-09-22. Paths:

- **BASE** = `/Users/slh/Library/CloudStorage/Box-Box/slhughes/LLAMAS_analysis/QA_baselines`
- **COMM** = `/Users/slh/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/ut20260710_11`
- **WARM** = `/Users/slh/Downloads/20260505_06-selected` (the 2026-05-06 warm-camera incident night)

Every frame below except the CAL0 ones has two placeholder cameras, `1.A.Blue` and `4.A.Blue`
(22 live detectors). The CAL0 frames from the incident night have all 24 extensions as placeholders,
so only the header checks run on them; they are still useful shutter cases.

### 6.1 Shutter faults → FAIL (exit 2)

The frame is not the exposure that was requested. `shutter_exptime_consistency` is FAIL severity on
every frame type, so one tripped check fails the frame regardless of the pixels.

| frame | type / mode | REXPTIME → SEXPTIME | measured Δ | tolerance | result |
|---|---|---|---|---|---|
| `WARM/LLAMAS_2026-05-06_05-10-56.6_SCI22_mef.fits` (ZTFJ1312p1031) | SCIENCE, SLOW | 600 → 180.946 s | 419.05 s (70 %) | 1.0 s or 10 % | **FAIL**, exit 2 |
| `COMM/LLAMAS_2026-07-11_10-11-03.4_SCI22_mef.fits` (fiber535) | SCIENCE, SLOW | 900 → 215.807 s | 684.19 s (76 %) | 1.0 s or 10 % | **FAIL**, exit 2 |
| `WARM/LLAMAS_2026-05-06_05-50-23.7_CAL0_mef.fits` (ThAr arc) | ARC_THAR, SLOW | 1 → 122.736 s | 121.74 s | 0.328 s or 10 % | **FAIL**, exit 2 (shutter stuck open) |
| `WARM/LLAMAS_2026-05-06_05-56-30.7_CAL0_mef.fits` (bias) | BIAS, SLOW | 0.001 → 0.352 s | 0.351 s | 0.214 s or 10 % | **FAIL**, exit 2 |
| `BASE/lamp_flats/LLAMAS_2026-05-02_22-58-00.4_CAL22_mef.fits` | LDLS_FLAT, FAST | 0.5 → 0.8 s | 0.30 s (60 %) | 0.252 s or 10 % | **FAIL**, exit 2 (only flat in the baselines that fails) |

Counter-example from the same incident: `WARM/LLAMAS_2026-05-06_05-56-25.8_CAL0_mef.fits`, the
bias taken five seconds earlier, has 0.001 → 0.003 s and **passes** (exit 0). The short-exposure
overhead is what the absolute tolerance is for; the 10 % rule alone would fail every bias.

### 6.2 Odd detector structure → FAIL (exit 2)

Row/column banding above the per-detector heavy-tail cap. On a uniform frame (bias, dark) the caps
are near the 2.0 ADU floor; on illuminated frames they are 1.5 × the worst normal frame.

**Dark with column structure on two green detectors** — `BASE/Darks/LLAMAS_2026-06-04_20-36-33.1_CAL22_mef.fits`
(DARK, SLOW, 600 s; shutter nominal 600 → 600.002 s). Held out of the threshold derivation.

| detector | rule | measured | cap | effect |
|---|---|---|---|---|
| 2.B.Green | `column_structure` | 13.15 | 2.0 | **FAIL** |
| 2.B.Green | `row_structure` | 6.47 | 2.30 | **FAIL** |
| 1.B.Green | `column_structure` | 4.11 | 2.0 | **FAIL** |
| 2.B.Green | `frame_noise_rms` | 52.6 | 33.8 | WARN |
| 1.B.Green | `frame_noise_rms` | 76.6 | 39.8 | WARN |

The other 20 live detectors pass all 133 evaluated checks. Result: `FAIL: 3 fail check(s)`, exit 2.

**Commissioning ThAr arcs with banding on every camera and red blooming** — two consecutive frames:

| frame | column FAIL | row FAIL | other |
|---|---|---|---|
| `COMM/LLAMAS_2026-07-10_20-22-43.7_CAL22_mef.fits` (ARC_THAR, FAST, 0.07 s) | 22 / 22 detectors, 1.8–2.3 × cap (e.g. 1.A.Green 85.6 vs 40.7; 3.B.Blue 173.9 vs 78.6; 1.B.Red 8527 vs 4713) | 21 / 22 (e.g. 4.A.Green 82.8 vs 29.8; 3.B.Red 2626 vs 879) | saturation WARN on all 8 reds (0.015–0.033 vs caps 0.010–0.021) |
| `COMM/LLAMAS_2026-07-10_20-22-58.8_CAL22_mef.fits` (ARC_THAR, FAST) | 22 / 22, 1.0–3.1 × cap (3.A.Blue 176 vs 57.7; 2.A.Blue 182 vs 62.0; 1.A.Red 3293 vs 3263) | 15 / 22 | saturation WARN on all 8 reds (0.029–0.072); edge-background WARN on all 8 reds |

Result for both: **FAIL**, exit 2 (43 and 37 FAIL checks respectively).

Counter-example, same sequence, 15 s earlier: `COMM/LLAMAS_2026-07-10_20-22-28.7_CAL22_mef.fits`
passes 89 / 89 evaluated checks (exit 0); its structure sits at ≤ 0.67 × cap on every detector. The
16 normal commissioning arcs of that night all pass; only these two frames fail.

### 6.3 Camera warming → WARN (exit 1)

`WARM/LLAMAS_2026-05-06_06-08-21.4_SCI22_mef.fits` (LTT4816, SCIENCE, SLOW, 10 s).

| detector | rule | measured | limit | effect |
|---|---|---|---|---|
| 3.A.Green | `camera_warming_gradient` | 2.09 ADU/s | 1.3 ADU/s | WARN |
| 3.A.Green | `ccd_temperature_warm` / `_hot` | −92.1 °C (header) | baseline −91.6 °C; WARN above −83.6, FAIL above −76.6 | pass |

The header temperature of 3.A.Green reads −92.1 °C, below its −91.6 °C healthy median, so the
temperature checks pass. The pixel-level dark-current glow is already there and the gradient rate
flags it. The next-highest rate on the frame is 0.30 ADU/s (2.B.Green); the other 20 live detectors
are ≤ 0.20 ADU/s. Result: `WARN: 1 warn check(s)`, exit 1.
This is the case that motivated the rate metric: two earlier candidates (absolute structure floors,
frame σ) either missed it or false-flagged bright long exposures.

No real frame has yet tripped `ccd_temperature_hot` or `ccd_temperature_shutoff`; those paths are
covered by the synthetic tests in §7 (header injection at baseline + 10 °C and + 20 °C).

### 6.4 Unreadable or truncated frame → exit 3 (`llamas-checks`) / ERROR, exit 2 (engine)

`COMM/LLAMAS_2026-07-11_10-19-11.8_SCI22_mef.fits` (Feige110, 60 s) is 453 568 bytes short of its
declared size. astropy warns `File may have been truncated` and the data of the last extension
cannot be reshaped to 2048 × 2048.

- `llamas-checks`: one stderr line, `system error: cannot reshape array of size 3967936 into shape (2048,2048)`, exit **3**, no report written.
- `llamas-checks-engine`: verdict `ERROR`, exit **2**, a stub `.qa.json` with the error text.

Neither is a pass. Re-copy or re-take the frame.

### 6.5 Frames that must NOT fire (regression guards)

| frame | type | result | why it matters |
|---|---|---|---|
| `BASE/Bias/LLAMAS_2026-04-07_19-23-19.3_CAL22_mef.fits` | BIAS, FAST | WARN, exit 1: `edge_background_level` and `frame_level_median` on 3.A.Green (803 vs ≤ 802), 4.A.Green (889 vs ≤ 881.8), 4.B.Green (822 vs ≤ 819.7) | benign level drift a few ADU over band; exactly the WARN-level monitoring the bands are for, never a FAIL |
| `BASE/Darks/LLAMAS_2026-04-07_21-13-34.9_CAL22_mef.fits` | DARK, SLOW | PASS 133 / 133, exit 0 | normal dark against the structure caps that fail the June dark |
| `BASE/Arcs/LLAMAS_2026-04-07_20-40-49.6_CAL22_mef.fits` | ARC_THAR, FAST | PASS 155 / 155, exit 0 | normal red arc saturation (WARN-tier cap) passes; structure well under cap |
| `BASE/lamp_flats/LLAMAS_2026-04-07_20-37-07.4_CAL22_mef.fits` | LDLS_FLAT, FAST | PASS 155 / 155, exit 0 | real fibre structure passes the heavy-tail caps |
| `BASE/twilight_flats/LLAMAS_2026-05-02_22-15-33.4_CAL22_mef.fits` | SKY_FLAT, FAST | PASS 155 / 155, exit 0 | |
| `COMM/LLAMAS_2026-07-10_20-22-28.7_CAL22_mef.fits` | ARC_THAR, FAST | PASS 89 / 89, exit 0 | normal frame bracketing the two odd arcs |
| `COMM/LLAMAS_2026-07-11_05-05-22.4_SCI22_mef.fits` (LTT7987, 1 s) | SCIENCE, FAST | PASS 89 / 89, exit 0 | 4.A.Red carries fixed horizontal banding; the gradient-rate check ignores it by design (banding is not a time-accruing glow), and `CCDTEMP1` (no underscore) is read correctly: all cameras healthy |
| `WARM/LLAMAS_2026-05-06_05-56-25.8_CAL0_mef.fits` | BIAS, SLOW | PASS, exit 0 | bias from the incident night with a nominal shutter |

Skipped-check counts in the reports come from absent header keywords, not from failures: the two
2026-07-10 arcs above and the 05-10-56.6 science frame carry no `CCDTEMP*` keyword on any extension,
so their 66–72 temperature checks are SKIPPED; the 20-22-58.8 arc and the 2026-07-11 frames do carry
them and are evaluated.

### 6.6 Full-baseline sweep

`scripts/batch_tally.py` runs the cal config over every frame in the five baseline folders
(154 files, in-process, no sidecars).

Re-run 2026-09-22 with the shipped package; the counts are identical to the July-2026 campaign.

| folder | n | PASS | WARN | FAIL |
|---|---|---|---|---|
| Bias | 48 | 46 | 2 | 0 |
| Darks | 7 | 6 | 0 | 1 |
| Arcs | 39 | 39 | 0 | 0 |
| lamp_flats | 36 | 30 | 5 | 1 |
| twilight_flats | 24 | 18 | 6 | 0 |
| **total** | **154** | **139** | **13** | **2** |

The only two FAILs are real: the 2026-05-02 LDLS flat with the shutter overrun (§6.1) and the
held-out 2026-06-04 dark (§6.2). No normal baseline frame fails; the WARNs are per-detector
level/background drift of the kind shown for the 2026-04-07 bias.

## 7. Automated test inventory

68 pytest tests, all on synthetic MEF files built in a temporary directory (no instrument data
needed). Run with `pytest` from the repo root.

### `tests/test_qa_engine_header.py` — engine rules and validator (27)

| test | asserts |
|---|---|
| `test_config_is_valid` | the inline BIAS test config passes `QAConfigValidator` |
| `test_shutter_ok_passes` | 0.001 s requested / 0.003 s actual is within the absolute tolerance |
| `test_shutter_gross_fault_fails` | 600 s requested / 181 s actual → FAIL |
| `test_short_exposure_overhead_passes` | 0.05 → 0.12 s (large relative, tiny absolute) passes via `abs_or_rel_diff` |
| `test_warm_temperature_flags_warn` | CCDTEMP −40 °C outside a static range → WARN |
| `test_missing_temperature_is_skipped_not_failed` | absent CCDTEMP → SKIPPED with `passed` true |
| `test_string_temperature_is_parsed` | a string-valued `"-90.0"` header parses to a number |
| `test_off_mode_lookup_miss_skips_not_crashes` | FAST frame, SLOW-only lookup → only that rule SKIPPED |
| `test_region_out_of_bounds_is_error_verdict` | region beyond the array → overall verdict ERROR, not PASS |
| `test_no_rules_matched_warns` | unknown PRODCATG → WARN with a `NO_RULES_MATCHED` result |
| `test_validator_rejects_per_extension_key_without_per_extension` | config error is caught |
| `test_per_detector_temp_config_is_valid` | per-detector temperature lookup config validates |
| `test_per_detector_temp_at_baseline_passes` | −90 °C at a −90 baseline passes |
| `test_per_detector_temp_warm_flags_warn` | −80 °C, above warn_max −82 but below fail_max −75 → WARN |
| `test_per_detector_temp_hot_flags_fail` | −70 °C, above fail_max → FAIL |
| `test_temp_keyword_variant_no_separator_is_read` | `CCDTEMP1` spelling is read (commissioning files) |
| `test_temp_absent_all_variants_is_skipped` | none of the three spellings present → SKIPPED |
| `test_validator_rejects_lookup_with_nonrange_op` | `expected_from_lookup` only allowed with op `range` |
| `test_shipped_config_reds_get_extra_headroom` | shipped cal YAML: red warn→fail band is 6 °C, green 7 °C |
| `test_science_rate_config_is_valid` | gradient-rate config validates |
| `test_flat_frame_passes_rate` | no gradient → rate ≈ 0 → pass |
| `test_short_exposure_gradient_warns` | 45 ADU gradient in 10 s ≈ 4.5 ADU/s → WARN |
| `test_same_gradient_long_exposure_passes` | same gradient over 100 s ≈ 0.45 ADU/s → pass |
| `test_bright_fibers_flat_background_passes` | sparse bright fibres are ignored by the quarter-block medians |
| `test_missing_exposure_time_skips` | no exposure keyword → SKIPPED, never a fail |
| `test_below_min_exposure_skips` | exposure < `min_exptime` → SKIPPED |
| `test_shipped_science_config_has_warming_rate` | shipped science YAML has the WARN rate rule and no structure rules |

### `tests/test_validate.py` — FITS structure validation and placeholders (19)

| test | asserts |
|---|---|
| `test_complete_file_has_no_missing_cameras` | 24 extensions → `missing_cameras` empty |
| `test_skipped_hdus_are_reported_in_idx_order` | HDUs 3, 15, 24 absent → `1.A.Blue, 3.A.Blue, 4.B.Blue`, in that order |
| `test_swapped_bench_is_an_identity_mismatch` | swapped BENCH headers on HDUs 1 and 7 → two `identity_mismatches` |
| `test_colour_only_extensions_skip_identity_check` | config extensions without bench/side are not checked |
| `test_inspect_structure_out_of_range_hdu_index_is_skipped` | a 1-extension file does not raise |
| `test_placeholder_all_ones_int16` | all-ones frame is a placeholder (real-frame convention) |
| `test_placeholder_all_zeros_uint16` | all-zeros frame is a placeholder (pipeline convention) |
| `test_placeholder_comment_marker_on_noisy_data` | the COMMENT marker alone marks a placeholder |
| `test_noisy_data_is_not_placeholder` | normal noise is not a placeholder |
| `test_railed_constant_frame_is_not_placeholder` | constant 65535 or pedestal frames are real faults, evaluated normally |
| `test_cam_name_only_extensions_are_counted_present` | `CAM_NAME`-only headers still identify the camera |
| `test_all_nan_is_not_placeholder` | all-NaN is not a placeholder |
| `test_detector_label_from_bench_side_color` | `BENCH/SIDE/COLOR` → `1.A.Red` |
| `test_detector_label_from_cam_name` | `CAM_NAME 2B_green` → `2.B.Green` |
| `test_detector_label_none_without_identity` | no identity keywords → `None` |
| `test_fix_extensions_round_trip` | `validate_and_fix_extensions` restores 24 extensions with placeholders |
| `test_check_image_reports_structure_on_single_extension` | `result["structure"]` lists 23 missing cameras; status unchanged |
| `test_check_image_reports_placeholder_extensions` | placeholder detector appears in `structure.placeholder_extensions` |
| `test_check_image_structure_present_with_report_and_fail` | structure block present on the early "unidentified type" fail path |

### `tests/test_cli.py` — command-line contract (22)

| test | asserts |
|---|---|
| `test_module_entry_quiet_and_report` | `python -m llamas_checks` is silent on 0/1/2 and the report JSON has `status`, `structure`, `report` |
| `test_no_sidecar_written_next_to_input` | no `.qa.json` appears beside the input |
| `test_shutter_fault_exits_fail_and_verbose_reports` | shutter fault → exit 2; `-v` prints the rule to stderr |
| `test_missing_input_is_system_error` | missing file → exit 3, one stderr line starting `system error` |
| `test_shipped_configs_validate` ×3 | `llamas-checks-validate` exits 0 on cal, science and base YAML |
| `test_import_hygiene_no_pipeline_dependencies` | importing `llamas_checks` loads none of ray, pypeit, matplotlib, scipy or the pipeline |
| `test_engine_pass_exits_0_and_writes_sidecar` | engine PASS → exit 0 and `<frame>.qa.json` |
| `test_engine_fail_exits_1` | engine FAIL → exit 1 |
| `test_engine_error_verdict_exits_2` | engine ERROR → exit 2 |
| `test_engine_batch_with_jobs_is_clean` | directory mode with `--jobs 2`: two sidecars, empty stderr |
| `test_engine_bare_config_name_resolves_to_shipped_copy` | `--config qa_config_cal.yaml` finds the packaged file |
| `test_engine_wrong_path_to_shipped_name_errors` | a wrong directory path is not silently replaced → exit 2 |
| `test_error_verdict_is_fail_with_message_and_report` | `check_image` maps ERROR to status fail and still writes the report |
| `test_error_verdict_through_cli_exits_2_not_3` | same through the CLI → exit 2 |
| `test_calib_root_default_suite_selects_prodcatg_config` | `--calib-root` with the default suite picks the config by PRODCATG |
| `test_suite_name_selects_root_suite_yaml` | `--suite name` selects `<root>/name.yaml` |
| `test_verbose_structure_line_lists_placeholder` | `-v` prints `structure: … placeholder: 1.A.Green` |
| `test_multiline_system_error_prints_one_line` ×2 | invalid config / YAML syntax error → exactly one stderr line, exit 3 |
| `test_help_shows_console_script_name` | `--help` shows `usage: llamas-checks` |

### Regression drivers on real data (not part of `pytest`)

- `scripts/run_qa_tests.py` — the 15-case matrix of §6 through the engine CLI (writes sidecars
  beside the inputs on purpose; run it on copies). Needs `LLAMAS_QA_BASELINES`,
  `LLAMAS_QA_WARM_DIR`, `LLAMAS_QA_COMMISSIONING_DIR`.
- `scripts/batch_tally.py` — the 154-file baseline sweep of §6.6, in-process, no sidecars.

## 8. Regenerating

```bash
pytest                                                    # 68 synthetic tests
python scripts/gen_thresholds_doc.py                      # refresh QA_THRESHOLDS_TABLES.md from the YAMLs
python scripts/batch_tally.py --baselines-root "$LLAMAS_QA_BASELINES" --out batch_tally.json
python scripts/run_qa_tests.py                            # 15-case matrix (copies of the frames recommended)
```

After changing thresholds (`aggregate_thresholds.py` → `gen_configs.py` → `llamas-checks-validate`),
re-run the three commands above and update §6 with any changed measured-vs-cap numbers.
