# LLAMAS raw-frame QA — check catalogue, thresholds and bad-image test cases

Status as of **2026-09-22** (`llamas-checks` 0.1.0, branch `fail-issues`: trimmed-mean
structure metrics, MAD rms and WARN/FAIL severity tiers). Every example result below was produced by re-running the shipped
package on the frame named, on this date. The per-detector numbers behind the checks are in the
generated appendix [`QA_THRESHOLDS_TABLES.md`](QA_THRESHOLDS_TABLES.md); pictures of frames that
pass, warn and fail are in [`QA_EXAMPLES.md`](QA_EXAMPLES.md); the design rationale and the
original July-2026 validation campaign are in [`QA_TESTS_SUMMARY.md`](QA_TESTS_SUMMARY.md).

**Severity policy.** FAIL is reserved for defects that make the frame unusable for its purpose
(shutter fault, a saturated unilluminated edge stripe, gross banding on a bias/dark, a camera past
its temperature limit). Anything that is merely high relative to the baselines is WARN. This
replaces the pre-2026-09 behaviour in which one detector 1.5× above its structure cap FAILed
22 of 24 FAST biases on a normal night.

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
llamas-checks /path/to/frame.fits --report                                      # quiet; ./frame.qa.json written only if warn/fail
llamas-checks /path/to/frame.fits --report /path/to/reports                     # same, into an existing directory
python -m llamas_checks /path/to/frame.fits -v                                  # same tool, module form
```

The frame type is read from the primary-header `PRODCATG`: `CAL.*` frames get
`qa_config_cal.yaml`, `SCI.*` frames get `qa_config_science.yaml`; an unrecognised or missing
`PRODCATG` falls back to the base `qa_config.yaml` and is reported as unidentified. Nothing is
written next to the input; only `--report` / `--report-dir` write a file, and only when the frame
**warns or fails** (`--report-all` also writes passing frames). The file is named after the frame,
`<frame>.qa.json`, the same name the engine gives its sidecars: bare `--report` puts it in the
current directory, `--report DIR` / `--report-dir DIR` in a directory, and `--report out.json`
names an explicit file.

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

Metric definitions used below (all robust to isolated hot pixels, hot-column fragments and
cosmic rays since 2026-09):

- **row / column structure** — standard deviation of the per-row (per-column) **2 %-trimmed
  means** of the region (the lowest and highest 2 % of each line are dropped before averaging).
  Whole-row/column offsets (banding, bars) and smooth glow gradients move it; a cluster of
  saturated pixels or a cosmic-ray trail (< 2 % of a line) does not. A plain median profile was
  tried and rejected: it missed the June-4 bars, which occupy a minority of each column.
- **robust rms** — 1.4826 × median absolute deviation of the pixels (read noise).
- **median**, **fraction > 63000 ADU** — as named.

### BIAS (`CAL.R-BIA`) — 13 rules

| rule | what is measured | limit | severity |
|---|---|---|---|
| `shutter_exptime_consistency` | \|SEXPTIME − REXPTIME\| | passes within abs 0.214 s **or** rel 10 % | FAIL |
| `edge_background_level` | median of `bottom_stripe` | band [min, max], per det × mode | WARN |
| `edge_saturated` | median of `bottom_stripe` | ≤ 63000 ADU (static) | FAIL |
| `frame_level_median` | median of `full_frame` | band [med_min, med_max], per det × mode | WARN |
| `frame_noise_rms` | robust rms of `full_frame` | ≤ rms_max, per det × mode | WARN |
| `row_structure` / `column_structure` | banding amplitude of `full_frame` | ≤ row_max / col_max, per det × mode | WARN |
| `row_structure_gross` / `column_structure_gross` | same metric | ≤ 4 × row_max / col_max (`row_fail_max`, `col_fail_max`) | FAIL |
| `saturation_fraction` | fraction of `full_frame` pixels > 63000 ADU | ≤ frac_max, per det × mode (≈ 5 × 10⁻⁴) | WARN |
| `saturation_gross` | same metric | ≤ 0.01 (static) | FAIL |
| `ccd_temperature_warm` | `CCDTEMP_1` / `CCDTEMP1` / `CCDTEMP-1` of this extension | within [−140, warn_max], per det | WARN |
| `ccd_temperature_hot` | same keyword | ≤ fail_max, per det | FAIL |
| `ccd_temperature_shutoff` | same keyword | ≤ −60 °C, all cameras | FAIL |

### DARK (`CAL.R-DRK`) — 13 rules

Identical to BIAS with the DARK tables; shutter abs tolerance 0.238 s.

### LDLS_FLAT (`CAL.R-FLT`), ARC_THAR (`CAL.R-ARC`) — 10 rules each (fixed lamp)

| rule | what is measured | limit | severity |
|---|---|---|---|
| `shutter_exptime_consistency` | \|SEXPTIME − REXPTIME\| | abs 0.252 s (LDLS) / 0.328 s (ARC) **or** rel 10 % | FAIL |
| `edge_background_level` | median of `bottom_stripe` | band, per det × mode | WARN |
| `edge_saturated` | median of `bottom_stripe` | ≤ 63000 ADU | FAIL |
| `row_structure` / `column_structure` | banding amplitude of `full_frame` | ≤ heavy-tail cap, per det × mode | FAIL |
| `saturation_fraction` | fraction > 63000 ADU | ≤ frac_max, per det × mode | WARN |
| `ccd_temperature_warm` / `_hot` / `_shutoff` | `CCDTEMP*` per extension | as BIAS | WARN / FAIL / FAIL |

### SKY_FLAT (`CAL.R-SKY`) — 10 rules

As LDLS/ARC with the SKY tables and shutter abs tolerance 0.348 s, except that
`row_structure` / `column_structure` are **WARN**: the structure metric scales with the twilight
brightness, which varies by design (four "good blue" twilights on 2026-09-06 sat 1.0–1.2× above
the caps).

No `frame_level_median` or `frame_noise_rms` on illuminated frames: the level scales with exposure
time, so a fixed band is not physical. The structure caps for these types are deliberately loose
(1.5 × the worst normal frame, see §5) so real fibre and line structure passes; only a gross
anomaly trips them. LDLS exposure times are the observer's choice (the calibration script
suggests 0.07 / 0.15 / 0.3 / 0.5 s and most nights use it, but nothing enforces it); at the
longer exposures taken for the blue channel the red detectors rail over most of the fibre area,
so red saturation is not a defect on an LDLS flat and the per-detector red saturation caps,
derived from baseline nights that include such frames, are effectively open. What marks an
overexposed or light-flooded flat is `edge_saturated`: the unilluminated stripe never exceeds
~5000 ADU on a normal frame of any type or exposure time in the baselines.

### SCIENCE (`SCI.R-*`) — 7 rules

| rule | what is measured | limit | severity |
|---|---|---|---|
| `shutter_exptime_consistency` | \|SEXPTIME − REXPTIME\| | abs 1.0 s **or** rel 10 % | FAIL |
| `saturation_fraction` | fraction of `full_frame` > 63000 ADU | ≤ 0.02 | WARN |
| `edge_saturated` | median of `bottom_stripe` | ≤ 63000 ADU | FAIL |
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
| gross saturation fraction | > 0.01 → FAIL (`saturation_gross`) | BIAS, DARK |
| edge stripe saturated | `bottom_stripe` median > 63000 ADU → FAIL (`edge_saturated`) | every rule set |
| gross structure factor | FAIL above 4 × the per-detector WARN cap (`*_gross`) | BIAS, DARK |
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

Per (type × readout mode × detector), 3344 normal detector-frame records in total (re-derived
2026-09-22 with the trimmed-mean structure profiles and the MAD rms; the per-extension statistics
are committed as `llamas_checks/baselines/qa_stats_raw.json`, so everything below is reproducible
without the frames):

- Samples are **MAD sigma-clipped** first, so an anomalous baseline frame cannot inflate its own cap.
- Level and edge-background bands: `median ± max(6 σ_robust, 15 ADU)` — an absolute ADU floor.
- RMS cap: `max(median + 8 σ_robust, 3 × median, 5.0)` on the robust rms; caps are now read-noise
  level (5 ADU on most SLOW detectors, 11–13 ADU FAST, 74 ADU on 4.A.Red FAST) instead of the
  25–3800 ADU plain-std caps that hot pixels used to inflate.
- Row/column structure and saturation caps are **heavy-tail**: `max(1.5 × unscreened max, median + 8 σ_robust, floor)`
  with floors 2.0 (structure) and 5 × 10⁻⁴ (saturation). Persistently structured detectors keep their
  own loose cap instead of false-failing. On BIAS/DARK the cap is the WARN tier; the FAIL tier is 4 × the cap.
- CCD temperature: each camera's own sigma-clipped median plus the colour-aware margins above.
- Shutter absolute tolerance per type: `2 × max(4σ-clipped |SEXPTIME − REXPTIME|) + 0.2 s`, with the
  relative tolerance fixed at 10 %. The 0.30 s overrun of the 2026-05-02 LDLS flat was clipped as an
  outlier, which is why that frame fails its own type's 0.252 s tolerance (§6.1).

Self-check on the baselines themselves: 0.75 % of normal detector-frames (25 of 3344) sit outside
their edge-background band (WARN-level drift); the held-out June-4 dark trips the DARK/SLOW
structure caps on 2.B.Green (row 6.7 vs 2.19, column 13.3 vs 2.0, i.e. 6.6 × the cap).

Pipeline: `scripts/sort_baselines.py` → `scripts/extract_stats.py` → `scripts/aggregate_thresholds.py`
(writes `llamas_checks/baselines/qa_thresholds_derived.json` and `qa_tracking_baselines.csv`) →
`scripts/gen_configs.py` (writes the two YAMLs) → `llamas-checks-validate`. The appendix tables are
rendered from those files by `scripts/gen_thresholds_doc.py`.

## 6. Bad-image test cases (real frames)

All runs: `llamas-checks <file> -v --report <out>.json`, 2026-09-22 (before the report gate;
today the passing cases would write no file without `--report-all`). Paths:

- **BASE** = `/Users/slh/Library/CloudStorage/Box-Box/slhughes/LLAMAS_analysis/QA_baselines`
- **COMM** = `/Users/slh/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/ut20260710_11`
- **WARM** = `/Users/slh/Downloads/20260505_06-selected` (the 2026-05-06 warm-camera incident night)
- **S06** = `/Users/slh/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/20260906_07_cals`
- **S07** = `/Users/slh/Library/CloudStorage/Box-Box/slhughes/Llamas_Commissioning_Data/20260907_08`

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

Row/column banding measured with the 2 %-trimmed-mean profile. On a uniform frame (bias, dark) the
WARN caps are near the 2.0 ADU floor and the FAIL tier is 4 × the cap; on fixed-lamp illuminated
frames (LDLS, ThAr) the heavy-tail cap (1.5 × the worst normal frame) is itself the FAIL.

**Dark with periodic column bars on 2.B.Green** — `BASE/Darks/LLAMAS_2026-06-04_20-36-33.1_CAL22_mef.fits`
(DARK, SLOW, 600 s; shutter nominal 600 → 600.002 s). Held out of the threshold derivation.

| detector | rule | measured | limit | effect |
|---|---|---|---|---|
| 2.B.Green | `column_structure_gross` | 13.28 | 8.0 (4 × 2.0) | **FAIL** |
| 2.B.Green | `column_structure` | 13.28 | 2.0 | WARN |
| 2.B.Green | `row_structure` | 6.68 | 2.19 | WARN |

Result: `FAIL: 1 fail check(s): column_structure_gross@2.B.Green`, exit 2. The 1.B.Green "column
structure" that the plain-mean metric also flagged on this frame (4.11 vs 2.0) was hot pixels: with
the trimmed profile it measures 0.14 and passes, which is the intended behaviour.

**Commissioning ThAr arcs with banding on every camera and red blooming** — two consecutive frames:

| frame | column FAIL | row FAIL | excess over cap | other |
|---|---|---|---|---|
| `COMM/LLAMAS_2026-07-10_20-22-43.7_CAL22_mef.fits` (ARC_THAR, FAST, 0.07 s) | 22 / 22 detectors (e.g. 1.A.Green 79.0 vs 34.3; 3.B.Blue 165.7 vs 69.4; 1.B.Red 8514 vs 3569) | 21 / 22 | 2.2–12.6 × | edge-background WARN on 4.A.Red, saturation WARN on the reds |
| `COMM/LLAMAS_2026-07-10_20-22-58.8_CAL22_mef.fits` (ARC_THAR, FAST) | 22 / 22 (1.A.Green 99.8; 3.B.Blue 204.2; 1.B.Red 7382) | 21 / 22 | 1.3–5.4 × | edge-background WARN on all 8 reds, saturation WARN on the reds |

Result for both: **FAIL**, exit 2 (43 FAIL checks each).

Counter-example, same sequence, 15 s earlier: `COMM/LLAMAS_2026-07-10_20-22-28.7_CAL22_mef.fits`
passes 111 / 111 evaluated checks (exit 0). The 26 ThAr arcs of 2026-09-06/07 all pass as well.

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

### 6.5 Saturated / light-flooded flats → FAIL (exit 2) — `edge_saturated`

The unilluminated bottom stripe (y 2–28, x 100–1948) of a detector never exceeds ~5000 ADU on a
normal frame of any type, including the longer LDLS exposures that rail the red *fibres*. A
stripe median above 63000 ADU means light is flooding pixels no fibre illuminates.

| frame | type / mode / REXPTIME | `edge_saturated` FAIL on | other | result |
|---|---|---|---|---|
| `S07/LLAMAS_2026-09-07_18-17-12.8_CAL22_mef.fits` | LDLS_FLAT, FAST, 0.07 s | 18 detectors (every live one except 1.A.Red 59028, 1.B.Blue 8680, 2.B.Green 62398, 4.A.Green 62678) | edge-background WARN on all 22, blue saturation 0.68–0.76 | **FAIL**, exit 2 (was WARN) |
| `S07/LLAMAS_2026-09-07_18-18-55.0_CAL22_mef.fits` | LDLS_FLAT, FAST, 0.3 s | 1.B.Red, 2.A.Red, 2.B.Red, 3.B.Red, 4.A.Red, 4.B.Red (64053–65535) | 13 edge-background WARNs; greens railed, blues 2–4 × a normal 0.3 s step | **FAIL**, exit 2 (was WARN) |
| `BASE/lamp_flats/LLAMAS_2026-06-30_19-33-58.0_CAL22_mef.fits` | LDLS_FLAT, FAST, 0.15 s (SEXPTIME 0.246 s, inside the shutter tolerance) | 1.B.Red, 2.A.Red, 2.B.Red, 3.B.Red | shutter overrun of 64 % | **FAIL**, exit 2 (was PASS; newly detected baseline anomaly, left in the set because the level bands are MAD-robust) |

`S07/LLAMAS_2026-09-07_18-16-59.5_CAL22_mef.fits` (the frame before 18-17-12.8) is railed the same
way and also fails; `18-17-27.1` (0.07 s, 15 s later) is a normal flat and passes, so the flooding
was intermittent within the sequence.

Counter-examples: the normal 0.07 s flat `S06/LLAMAS_2026-09-06_18-52-19.2` (red medians 15–26k,
stripes at 1000–2300 ADU) and the normal 0.3 s flat `S06/…18-53-43.1` (reds railed, stripes ≤ 4900)
both pass 111 / 111.

### 6.6 Benign frames that used to FAIL (regression guards for the 2026-09 revision)

Every one of these was a FAIL under the plain-mean structure / plain-std rms rules; none is a
defect. Values are the new metric against the new cap.

| frame | type / mode | old FAIL reason | now | why |
|---|---|---|---|---|
| `S06/LLAMAS_2026-09-06_19-14-52.1_CAL22_mef.fits` (and 33 of the other 34 FAST biases of 09-06/09-07) | BIAS, FAST | `row_structure` 4.A.Red 382.5 vs 259.8 | **WARN**, exit 1: `row_structure` 387.8 vs 202.4, `column_structure` 739.0 vs 706.5 | 4.A.Red's variable left-edge glow ran 1.4–2.1 × the April–June cap on those nights; the 4 × FAIL tier (809.7 / 2825.8) is far away. Adding a September epoch to the baselines would widen this cap (§9) |
| `S06/LLAMAS_2026-09-06_20-33-17.7_CAL22_mef.fits` | BIAS, FAST | 2.A.Red row 6.1 / col 11.2 vs 2.0, rms 159 vs 18 | **WARN**, exit 1: only the 4.A.Red row structure (317.4 vs 202.4) | ~20 saturated pixels in one column of 2.A.Red; trimmed profile 0.36, robust rms unaffected |
| `S06/LLAMAS_2026-09-06_19-08-53.8_CAL22_mef.fits` | BIAS, SLOW | 3.A.Red row 3.4 / col 5.6 vs 2.0, rms 106 vs 25 | **PASS** 221 / 221 | hot pixels (trimmed column profile 0.06) |
| `S07/LLAMAS_2026-09-07_18-28-08.0_CAL22_mef.fits` | BIAS, SLOW | 3.B.Red row 3.0 / col 3.3 vs 2.0, rms 79 vs 22 | **PASS** 221 / 221 | hot pixels |
| `S06/LLAMAS_2026-09-06_19-28-14.2_CAL22_mef.fits` | DARK, SLOW | 2.B.Green row 2.37 / col 4.57, rms 86 vs 34 | **PASS** 221 / 221 | hot pixels on top of a ~5 ADU glow gradient (trimmed column profile 0.11) |
| `S07/LLAMAS_2026-09-07_19-15-34.6_CAL22_mef.fits` | DARK, SLOW | 5 structure FAILs on 1.A.Red / 2.A.Green / 4.A.Red, rms 85–112 | **PASS** 221 / 221 | hot pixels and cosmic rays after 600 s |
| `S06/LLAMAS_2026-09-06_22-22-03.4_CAL22_mef.fits` ("good blue" in the observer log) | SKY_FLAT, SLOW | 4 structure FAILs at 1.00–1.04 × cap | **WARN**, exit 1: 14 edge-background, 9 saturation, 5 structure WARNs | twilight brighter than the baseline set; structure on sky flats is WARN |
| `S06/LLAMAS_2026-09-06_22-19-43.5_CAL22_mef.fits` ("saturated" in the observer log) | SKY_FLAT, FAST | 13 structure FAILs | **WARN**, exit 1: 42 WARNs (edge background, saturation 0.04–0.27, structure) | a too-bright twilight is a WARN for the observer, not a rejected frame |

### 6.7 Frames that must NOT fire (regression guards)

| frame | type | result | why it matters |
|---|---|---|---|
| `BASE/Bias/LLAMAS_2026-04-07_19-23-19.3_CAL22_mef.fits` | BIAS, FAST | WARN, exit 1: `edge_background_level` and `frame_level_median` on 3.A.Green (803 vs ≤ 802), 4.A.Green (889 vs ≤ 881.8), 4.B.Green (822 vs ≤ 819.7) | benign level drift a few ADU over band; exactly the WARN-level monitoring the bands are for, never a FAIL |
| `BASE/Darks/LLAMAS_2026-04-07_21-13-34.9_CAL22_mef.fits` | DARK, SLOW | PASS 221 / 221, exit 0 | normal dark against the structure caps that fail the June dark |
| `BASE/Arcs/LLAMAS_2026-04-07_20-40-49.6_CAL22_mef.fits` | ARC_THAR, FAST | PASS 177 / 177, exit 0 | normal red arc saturation (WARN-tier cap) passes; structure well under cap |
| `BASE/lamp_flats/LLAMAS_2026-04-07_20-37-07.4_CAL22_mef.fits` | LDLS_FLAT, FAST | PASS 177 / 177, exit 0 | real fibre structure passes the heavy-tail caps |
| `BASE/twilight_flats/LLAMAS_2026-05-02_22-15-33.4_CAL22_mef.fits` | SKY_FLAT, FAST | PASS 177 / 177, exit 0 | |
| `S06/LLAMAS_2026-09-06_18-21-00.7_CAL22_mef.fits` | BIAS, FAST | PASS 221 / 221, exit 0 | FAST bias with the usual 4.A.Red pattern at a low amplitude |
| `S06/LLAMAS_2026-09-06_18-53-43.1_CAL22_mef.fits` | LDLS_FLAT, FAST, 0.3 s | PASS 111 / 111, exit 0 | reds railed by a normal blue-channel exposure, stripes unsaturated: must not trip `edge_saturated` |
| `COMM/LLAMAS_2026-07-10_20-22-28.7_CAL22_mef.fits` | ARC_THAR, FAST | PASS 111 / 111, exit 0 | normal frame bracketing the two odd arcs |
| `COMM/LLAMAS_2026-07-11_05-05-22.4_SCI22_mef.fits` (LTT7987, 1 s) | SCIENCE, FAST | PASS 111 / 111, exit 0 | 4.A.Red carries fixed horizontal banding; the gradient-rate check ignores it by design (banding is not a time-accruing glow), and `CCDTEMP1` (no underscore) is read correctly: all cameras healthy |
| `WARM/LLAMAS_2026-05-06_05-56-25.8_CAL0_mef.fits` | BIAS, SLOW | PASS, exit 0 | bias from the incident night with a nominal shutter |

Skipped-check counts in the reports come from absent header keywords, not from failures: the two
2026-07-10 arcs above and the 05-10-56.6 science frame carry no `CCDTEMP*` keyword on any extension,
so their 66–72 temperature checks are SKIPPED; the 20-22-58.8 arc and the 2026-07-11 frames do carry
them and are evaluated.

### 6.8 Full-baseline sweep

`scripts/batch_tally.py` runs the cal config over every frame in the five baseline folders
(154 files, in-process, no sidecars).

Re-run 2026-09-22 with the revised rules (trimmed-mean structure, MAD rms, WARN/FAIL tiers,
`edge_saturated`). The only change against the July-2026 campaign is the 2026-06-30 LDLS flat
with railed red edge stripes moving from WARN to FAIL (§6.5).

| folder | n | PASS | WARN | FAIL |
|---|---|---|---|---|
| Bias | 48 | 46 | 2 | 0 |
| Darks | 7 | 6 | 0 | 1 |
| Arcs | 39 | 39 | 0 | 0 |
| lamp_flats | 36 | 30 | 4 | 2 |
| twilight_flats | 24 | 18 | 6 | 0 |
| **total** | **154** | **139** | **12** | **3** |

The three FAILs are real: the 2026-05-02 LDLS flat with the shutter overrun (§6.1), the held-out
2026-06-04 dark (§6.2, via `column_structure_gross`) and the 2026-06-30 LDLS flat with saturated
edge stripes (§6.5). No normal baseline frame fails; the WARNs are per-detector level/background
drift of the kind shown for the 2026-04-07 bias.

### 6.9 September 2026 commissioning nights (the nights that motivated the revision)

In-process sweep (no sidecars) of every raw frame in `S06` (79) and `S07` (121; the 14
`*_mef_white.fits` white-light products in that folder are not raw frames and are skipped as
unreadable). Before the revision the observer's log recorded 22 of 24 FAST biases, 1 SLOW bias,
1 dark and 4 twilight flats as FAIL on 09-06 alone, and the two railed LDLS flats as WARN.

| type / mode | n | PASS | WARN | FAIL | notes |
|---|---|---|---|---|---|
| BIAS FAST | 35 | 1 | 34 | 0 | every WARN is 4.A.Red `row_structure` (1.4–2.1 × cap); 11 also `column_structure` (≤ 1.12 ×) |
| BIAS SLOW | 22 | 22 | 0 | 0 | the two hot-pixel frames (§6.6) now pass |
| DARK SLOW | 7 | 7 | 0 | 0 | the two glow darks (§6.6) now pass |
| ARC FAST / SLOW | 18 / 6 | 24 | 0 | 0 | |
| LDLS FAST | 28 | 21 | 4 | 3 | FAILs: `18-16-59.5`, `18-17-12.8`, `18-18-55.0`, all `edge_saturated` (§6.5); WARNs: 1.B.Red edge band on the 0.3 s step |
| SKY FAST / SLOW | 11 / 19 | 16 | 14 | 0 | bright twilights: edge band, saturation and structure WARNs |
| SCIENCE FAST / SLOW | 40 | 40 | 0 | 0 | |

No frame on either night FAILs for a reason other than the light-flooded flats.

## 7. Automated test inventory

99 pytest tests, all on synthetic MEF files built in a temporary directory (no instrument data
needed). Run with `pytest` from the repo root.

### `tests/test_severity_tiers.py` — robust metrics and WARN/FAIL tiers (20)

| test | asserts |
|---|---|
| `test_config_is_valid`, `test_validator_accepts_robust_std` | the tier test config validates; `robust_std` is a known metric type |
| `test_trimmed_structure_ignores_hot_column_fragment` | 4 railed pixels in one 200-row column leave row/column structure < 0.1 |
| `test_trimmed_structure_measures_full_column_bar` | a 200-column full-height +10 ADU bar gives column structure > 2 |
| `test_trimmed_structure_measures_row_banding` | alternate rows +3 ADU → row structure 1.5 |
| `test_robust_std_is_read_noise_and_blind_to_hot_pixels` | robust rms = 1.4826 × MAD, unchanged by hot pixels; the plain std is not |
| `test_structure_above_cap_is_warn_not_fail` | 1.5 × cap → `row_structure` WARN, `row_structure_gross` passes, overall WARN |
| `test_gross_structure_fails` | 5 × cap → `row_structure_gross` FAIL, overall FAIL |
| `test_hot_pixel_cluster_bias_passes` | a bias with a saturated hot-pixel cluster is PASS |
| `test_modest_saturation_is_warn_only` | 1e-3 saturated → `saturation_fraction` WARN, `saturation_gross` passes |
| `test_gross_saturation_fails` | 2 % saturated → `saturation_gross` FAIL |
| `test_railed_edge_stripe_fails` | bottom stripe at 65535 → `edge_saturated` FAIL |
| `test_bright_but_unsaturated_edge_stripe_passes_edge_rule` | stripe +4000 ADU passes `edge_saturated` |
| `test_shipped_uniform_frames_have_two_tier_structure` (×2) | shipped BIAS/DARK: structure WARN + `*_gross` FAIL, saturation WARN + `saturation_gross` FAIL at 0.01 |
| `test_shipped_gross_caps_are_four_times_warn_caps` (×2) | every `row_fail_max`/`col_fail_max` in the shipped tables is 4 × the WARN cap |
| `test_shipped_illuminated_structure_severities` | shipped SKY structure WARN, LDLS/ARC FAIL, no gross tier on illuminated sets |
| `test_every_shipped_rule_set_has_edge_saturated_fail` | all six rule sets carry `edge_saturated` FAIL at 63000 ADU on `bottom_stripe` |
| `test_shipped_rms_metric_is_robust` | the shipped `rms` metric is `robust_std` |

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

### `tests/test_cli.py` — command-line contract (33)

| test | asserts |
|---|---|
| `test_module_entry_quiet_and_report` | `python -m llamas_checks` is silent on 0/1/2 and the report JSON has `status`, `structure`, `report`, `report_path` |
| `test_no_sidecar_written_next_to_input` | no `.qa.json` appears beside the input, with `--report` or `--report-dir` elsewhere |
| `test_pass_writes_no_report_by_default` | passing frame + `--report PATH` → exit 0, no file |
| `test_pass_writes_report_with_report_all` | `--report-all` writes the passing report; `-v` prints `report: <path>` |
| `test_warn_report_dir_names_report_after_frame` | warn + `--report-dir` → `DIR/LLAMAS_…_mef.qa.json`, same name as the engine sidecar |
| `test_bare_report_names_after_frame_in_cwd` | bare `--report` → `./LLAMAS_…_mef.qa.json`, nothing beside the frame |
| `test_report_pointed_at_directory_names_after_frame` | `--report <existing dir>` → `<dir>/<frame>.qa.json` |
| `test_report_refuses_fits_path` | `--report something.fits` → exit 3, the FITS file is untouched |
| `test_error_verdict_written_with_report_dir` | ERROR verdict (status fail) is written under `--report-dir` |
| `test_pass_with_report_dir_creates_nothing` | a pass does not even create the report directory |
| `test_report_and_report_dir_are_exclusive` | both options → argparse error; both arguments → `QAEngineError` |
| `test_check_image_report_path_key` | `result["report_path"]` is `None` on pass and the derived path on warn |
| `test_report_path_for_matches_engine_sidecar_name` | `report_path_for` with and without a directory |
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
| `test_calib_root_default_suite_selects_prodcatg_config` | `--calib-root` with the default suite picks the config by PRODCATG (`--report-all` to capture the passing report) |
| `test_suite_name_selects_root_suite_yaml` | `--suite name` selects `<root>/name.yaml` |
| `test_verbose_structure_line_lists_placeholder` | `-v` prints `structure: … placeholder: 1.A.Green` |
| `test_multiline_system_error_prints_one_line` ×2 | invalid config / YAML syntax error → exactly one stderr line, exit 3 |
| `test_help_shows_console_script_name` | `--help` shows `usage: llamas-checks` |

### Regression drivers on real data (not part of `pytest`)

- `scripts/run_qa_tests.py` — the 23-case matrix of §6 through the engine CLI (writes sidecars
  beside the inputs on purpose; run it on copies). Needs `LLAMAS_QA_BASELINES`,
  `LLAMAS_QA_WARM_DIR`, `LLAMAS_QA_COMMISSIONING_DIR`.
- `scripts/batch_tally.py` — the 154-file baseline sweep of §6.6, in-process, no sidecars.

## 8. Regenerating

```bash
pytest                                                    # 99 synthetic tests
python scripts/gen_thresholds_doc.py                      # refresh QA_THRESHOLDS_TABLES.md from the YAMLs
python scripts/gen_examples_doc.py                        # refresh QA_EXAMPLES.md + docs/images/ (all five data roots)
python scripts/batch_tally.py --baselines-root "$LLAMAS_QA_BASELINES" --out batch_tally.json
python scripts/run_qa_tests.py                            # 23-case matrix (copies of the frames recommended)
```

After changing metrics or thresholds (`extract_stats.py` → `aggregate_thresholds.py` → `gen_configs.py`
→ `llamas-checks-validate`), re-run the commands above and update §6 with any changed
measured-vs-cap numbers.
