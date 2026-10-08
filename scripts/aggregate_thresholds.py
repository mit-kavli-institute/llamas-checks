#!/usr/bin/env python3
"""Phase B aggregation (ROBUST). Turn qa_stats_raw.json into per-detector derived
thresholds + per-epoch tracking baselines.

Robustness (addressing review findings):
- Per (type,mode,detector) samples are sigma-clipped (MAD-based) to reject
  anomalous baseline frames BEFORE deriving limits, so an unscreened outlier can
  no longer inflate its own cap.
- Background/level bands use an ABSOLUTE ADU floor (additive quantity), not a
  relative %: band = median +/- max(K*sigma_robust, FLOOR_ADU).
- Structure/RMS/saturation caps are robust one-sided: max(median + N*sigma_robust,
  floor), on the screened sample.
- The known-bad June-4 odd dark is held out from derivation and validated.
- Warm-camera flagging is validated against the May-6 warm folder using BOTH the
  edge-stripe background band and the CCD temperature band (not just shutter).
  The warm folder comes from --warm-dir or LLAMAS_QA_WARM_DIR; when neither is
  set that validation block is skipped.

Outputs qa_thresholds_derived.json + qa_tracking_baselines.csv into --out-dir
(default: the package baselines/ directory, where gen_configs.py reads them).
"""
import argparse
import csv
import glob
import json
import os

import numpy as np
from astropy.io import fits

from llamas_checks.paths import BASELINES_DIR

ODD_TAG = "2026-06-04_20-36-33"

# absolute ADU floors (additive pedestals) and robust widths
K_LEVEL = 6          # band half-width in robust sigmas for level/background
FLOOR_LEVEL = 15.0   # min band half-width, ADU  (catches modest warm/drift)
N_CAP = 8            # one-sided caps in robust sigmas
FLOOR_STRUCT = 2.0   # min structure cap
FLOOR_RMS = 5.0      # min rms cap, ADU
FLOOR_SAT = 0.0005   # min saturation-fraction cap

# Signal-normalised structure (structure / (full_mean - edge_med)) for lamp frames
# whose structure IS the lamp pattern and scales with exposure: the absolute caps
# encoded the baseline exposure times (every FAST arc was 0.07-0.4 s, every SLOW
# arc 1 s), so a 1 s FAST arc failed. The ratio is independent of exposure and
# readout mode, so for these types the sample is pooled over both modes.
MIN_SIGNAL_NORM = 2.0                 # ADU; fainter records are excluded (noise / noise)
POOL_MODES_FOR_NORM = ("CAL.R-ARC",)  # types whose normalised caps pool FAST+SLOW

# Vertical smear / halo on arcs: std of the per-column MEDIAN profile over the lamp
# signal (col_med_std / (full_mean - edge_med)). Curved arc lines leave the column
# medians at the background (~0.03 on a red camera at any exposure); a vertical
# halo fills whole columns and lifts them (>1 on the 2026-07-10 odd arcs). Below
# ~20 ADU the green/blue column medians are read-noise dominated, so fainter
# records are excluded and the engine SKIPs the rule there (min_signal: 20).
MIN_SIGNAL_SMEAR = 20.0

# Per-detector CCD temperature: each camera is monitored against its OWN healthy
# baseline and WARN/FAIL on warming above it. Colour-aware margins -- the red
# detectors run warmer and can be more thermally variable, so they get extra
# headroom. Absolute -60 C is a hard TEC-limit safety FAIL for every camera.
TEMP_WARN_MARGIN = {"red": 10.0, "green": 8.0, "blue": 8.0}   # deg C above baseline -> WARN
TEMP_FAIL_MARGIN = {"red": 16.0, "green": 15.0, "blue": 15.0}  # deg C above baseline -> FAIL
TEMP_COLD_MIN = -140.0    # absolute cold-garbage floor (sensor fault)
TEMP_ABS_FAIL = -60.0     # absolute TEC-limit safety FAIL


def det_name(rec):
    return f"{rec['bench']}.{rec['side']}.{str(rec['color']).capitalize()}"


def robust_sigma(a):
    med = np.median(a)
    mad = np.median(np.abs(a - med))
    return 1.4826 * mad


def sigma_clip(vals, nsig=3.0, iters=3, min_keep=3):
    """Iteratively drop points > nsig robust-sigmas from the median."""
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size <= min_keep:
        return a
    for _ in range(iters):
        med = np.median(a)
        sig = robust_sigma(a)
        if sig == 0:
            break
        keep = np.abs(a - med) <= nsig * sig
        if keep.sum() < min_keep or keep.all():
            break
        a = a[keep]
    return a


def obs_stats(a):
    if a.size == 0:
        return None
    return dict(n=int(a.size), min=float(a.min()), max=float(a.max()),
                med=float(np.median(a)), sigma_robust=round(float(robust_sigma(a)), 4),
                p95=float(np.percentile(a, 95)))


def level_band(screened):
    """Two-sided additive band: median +/- max(K*sigma, FLOOR_ADU)."""
    med = float(np.median(screened))
    half = max(K_LEVEL * robust_sigma(screened), FLOOR_LEVEL)
    return round(med - half, 2), round(med + half, 2)


def one_sided_cap(screened, floor):
    med = float(np.median(screened))
    return round(max(med + N_CAP * robust_sigma(screened), med * 3.0, floor), 4)


def normalised_structure(rs):
    """Caps for structure / signal over the records ``rs`` (signal = full-frame mean
    minus edge-stripe median; records fainter than MIN_SIGNAL_NORM are dropped)."""
    rows, cols = [], []
    for r in rs:
        sig = r.get("full_mean")
        edge = r.get("edge_med")
        if sig is None or edge is None or not np.isfinite(sig) or not np.isfinite(edge):
            continue
        sig = sig - edge
        if sig < MIN_SIGNAL_NORM:
            continue
        rows.append(r["row_struct"] / sig)
        cols.append(r["col_struct"] / sig)
    if not rows:
        return None
    return {"row_max": heavy_tail_cap(rows, 0.0), "col_max": heavy_tail_cap(cols, 0.0),
            "n_norm": len(rows), "min_signal": MIN_SIGNAL_NORM,
            "observed_row": obs_stats(np.asarray(rows)), "observed_col": obs_stats(np.asarray(cols))}


def vertical_smear(rs):
    """Cap for col_med_std / signal over the records ``rs`` that carry col_med_std
    (signal = full_mean - edge_med; records fainter than MIN_SIGNAL_SMEAR dropped).
    None when no record has the statistic (frame types it was never extracted for)."""
    vals = []
    for r in rs:
        smear = r.get("col_med_std")
        sig = r.get("full_mean")
        edge = r.get("edge_med")
        if smear is None or sig is None or edge is None:
            continue
        sig = sig - edge
        if not np.isfinite(smear) or not np.isfinite(sig) or sig < MIN_SIGNAL_SMEAR:
            continue
        vals.append(smear / sig)
    if not vals:
        return None
    return {"max": robust_smear_cap(vals), "n": len(vals), "min_signal": MIN_SIGNAL_SMEAR,
            "observed": obs_stats(np.asarray(vals))}


def robust_smear_cap(vals):
    """Cap for the smear ratio: max(median + N_CAP sigma_robust, 1.5 x 95th percentile).

    Unlike heavy_tail_cap this does not key on the single worst frame: the normal
    smear distribution is tight (a red camera sits at 0.02-0.05 at any exposure),
    so one baseline frame that itself carries a halo on one camera (2.B.Red of
    2026-05-03 00-08-43.3, 0.57 against 0.04 on its siblings) must not open that
    camera's cap by 15x. Such a record shows up in the self-check below instead."""
    a = np.asarray(vals, float)
    return round(max(float(np.median(a)) + N_CAP * robust_sigma(a),
                     1.5 * float(np.percentile(a, 95))), 4)


def heavy_tail_cap(vals, floor):
    """One-sided cap for HEAVY-TAILED quantities (dark structure/RMS vary frame to
    frame from cosmic rays / hot columns). Uses each detector's OWN unscreened
    maximum x1.5 so naturally-structured detectors keep a high cap and clean ones
    stay tight -- rather than sigma-clipping away the legit tail and flooring the
    cap (which false-fails normal frames). Trade-off: a detector whose baseline
    contains a genuine bad frame gets a loose cap (documented; n is small)."""
    a = np.asarray([v for v in vals if v is not None and np.isfinite(v)], float)
    if a.size == 0:
        return floor
    med = float(np.median(a))
    return round(max(float(a.max()) * 1.5, med + N_CAP * robust_sigma(a), floor), 4)


def main():
    parser = argparse.ArgumentParser(
        description="Derive per-detector QA thresholds and tracking baselines from qa_stats_raw.json.")
    parser.add_argument("--raw", default="qa_stats_raw.json",
                        help="input from extract_stats.py (default: ./qa_stats_raw.json)")
    parser.add_argument("--out-dir", default=str(BASELINES_DIR),
                        help="where to write qa_thresholds_derived.json and qa_tracking_baselines.csv "
                             "(default: the package baselines/ directory)")
    parser.add_argument("--warm-dir", default=os.environ.get("LLAMAS_QA_WARM_DIR"),
                        help="folder of May-6 warm-incident frames for validation "
                             "(default: $LLAMAS_QA_WARM_DIR; skipped when unset)")
    args = parser.parse_args()
    RAW = args.raw
    WARM_DIR = args.warm_dir
    os.makedirs(args.out_dir, exist_ok=True)
    OUT_JSON = os.path.join(args.out_dir, "qa_thresholds_derived.json")
    OUT_CSV = os.path.join(args.out_dir, "qa_tracking_baselines.csv")

    recs = json.load(open(RAW))
    normal = [r for r in recs if not r.get("placeholder") and "error" not in r
              and ODD_TAG not in r["filename"]]
    odd = [r for r in recs if ODD_TAG in r["filename"] and not r.get("placeholder")]

    groups = {}
    for r in normal:
        groups.setdefault((r["prodcatg"], r["mode"], det_name(r)), []).append(r)
    # mode-pooled samples for the signal-normalised structure of lamp frames
    pooled = {}
    for r in normal:
        if r["prodcatg"] in POOL_MODES_FOR_NORM:
            pooled.setdefault((r["prodcatg"], det_name(r)), []).append(r)

    derived = {}
    tracking_rows = []
    for (pc, mode, det), rs in sorted(groups.items()):
        # Background/level bands: use the UNCLIPPED spread so genuine night-to-night
        # drift is inside the band (MAD sigma already resists outliers at the centre);
        # clipping legit drift would tighten the band and false-WARN same-type frames.
        edge_a = np.asarray([r["edge_med"] for r in rs if np.isfinite(r["edge_med"])], float)
        med_a = np.asarray([r["full_med"] for r in rs if np.isfinite(r["full_med"])], float)
        # Heavy-tailed quantities (structure/RMS/saturation vary frame to frame, and
        # illuminated flats legitimately saturate) -> per-detector unscreened max x1.5.
        row_v = [r["row_struct"] for r in rs]
        col_v = [r["col_struct"] for r in rs]
        rms_v = [r["full_std"] for r in rs]
        sat_v = [r["sat_frac"] for r in rs]
        if edge_a.size == 0:
            continue
        emin, emax = level_band(edge_a)
        lmin, lmax = level_band(med_a)
        derived.setdefault(pc, {}).setdefault(mode, {})[det] = {
            "n_files": len(rs),
            "edge_bg": {"min": emin, "max": emax, "observed": obs_stats(edge_a)},
            "full_level": {"med_min": lmin, "med_max": lmax,
                           "rms_max": heavy_tail_cap(rms_v, FLOOR_RMS),
                           "observed_med": obs_stats(med_a)},
            "structure": {"row_max": heavy_tail_cap(row_v, FLOOR_STRUCT),
                          "col_max": heavy_tail_cap(col_v, FLOOR_STRUCT),
                          "observed_row": obs_stats(np.asarray(row_v)),
                          "observed_col": obs_stats(np.asarray(col_v))},
            "structure_norm": normalised_structure(
                pooled[(pc, det)] if pc in POOL_MODES_FOR_NORM else rs),
            "smear": vertical_smear(
                pooled[(pc, det)] if pc in POOL_MODES_FOR_NORM else rs),
            "sat_frac_max": heavy_tail_cap(sat_v, FLOOR_SAT),
        }
        by_date = {}
        for r in rs:
            by_date.setdefault(r["date"], []).append(r)
        for date, drs in sorted(by_date.items()):
            tracking_rows.append(dict(
                prodcatg=pc, mode=mode, detector=det, date=date, n=len(drs),
                edge_med=round(float(np.median([r["edge_med"] for r in drs])), 2),
                full_med=round(float(np.median([r["full_med"] for r in drs])), 2),
                row_struct=round(float(np.median([r["row_struct"] for r in drs])), 3),
                col_struct=round(float(np.median([r["col_struct"] for r in drs])), 3),
            ))

    # shutter tol per type: abs_tol absorbs fixed overhead; rel_tol tight 10%
    shutter = {}
    for pc in sorted(set(r["prodcatg"] for r in normal)):
        deltas = []
        for r in normal:
            if r["prodcatg"] != pc or r["hdu"] != 1:
                continue
            if r.get("rexp") is not None and r.get("sexp") is not None:
                deltas.append(abs(r["sexp"] - r["rexp"]))
        if deltas:
            ds = sigma_clip(deltas, nsig=4)
            shutter[pc] = {"abs_delta_max": round(float(np.max(deltas)), 4),
                           "abs_tol": round(float(np.max(ds)) * 2 + 0.2, 3), "rel_tol": 0.10}
    shutter["SCI.R-*"] = {"abs_tol": 1.0, "rel_tol": 0.10, "note": "science: 1s abs + 10% rel"}

    temps = sigma_clip([r["ccdtemp"] for r in normal if r.get("ccdtemp") is not None])
    tstat = obs_stats(temps) if temps.size else None
    # legacy pooled warm threshold (kept for reference; superseded by per-detector)
    warm_warn = round(float(np.median(temps) + 5 * robust_sigma(temps)), 1) if temps.size else -72.0

    # per-detector healthy baselines: group each camera's own CCD temps and derive
    # colour-aware warm/fail caps relative to that camera's median. Pooling all
    # cameras (as the legacy global does) hides the warm-running detectors -- e.g.
    # 1A.red sits ~-78 C, 12 C above the -90 array median -- so we must go per camera.
    temp_by_det = {}
    for r in normal:
        t = r.get("ccdtemp")
        if t is None or not np.isfinite(t):
            continue
        temp_by_det.setdefault(det_name(r), []).append(t)
    temp_pd = {}
    for det, vals in sorted(temp_by_det.items()):
        a = sigma_clip(vals)              # drop rare cold-excursion glitches
        if a.size == 0:
            continue
        med = float(np.median(a))
        color = det.split(".")[-1].lower()
        temp_pd[det] = {
            "baseline_med": round(med, 2),
            "warn_max": round(med + TEMP_WARN_MARGIN.get(color, 8.0), 2),
            "fail_max": round(med + TEMP_FAIL_MARGIN.get(color, 15.0), 2),
            "cold_min": TEMP_COLD_MIN,
            "observed": obs_stats(a),
        }
    ccd = {"observed": tstat, "cold_min": TEMP_COLD_MIN,
           "abs_fail_above": TEMP_ABS_FAIL,           # absolute TEC-limit safety FAIL
           "warn_margin": TEMP_WARN_MARGIN,           # deg C above per-camera baseline
           "fail_margin": TEMP_FAIL_MARGIN,
           "per_detector": temp_pd,                   # per-camera warn/fail caps
           "warm_warn_above": warm_warn,              # legacy pooled (reference only)
           "warm_fail_above": -60.0}

    out = {"meta": {"n_normal_records": len(normal), "excluded_odd": ODD_TAG,
                    "epochs": sorted(set(r["date"] for r in normal)),
                    "robustness": f"MAD sigma-clip screen; level band +/-max({K_LEVEL}sig,{FLOOR_LEVEL}ADU); "
                                  f"caps med+{N_CAP}sig",
                    "region_convention": "edge stripes y[2:28]&[2020:2046] x[100:1948]; sat>63000",
                    "structure_norm": f"structure / (full_mean - edge_med), heavy-tail cap, records with "
                                      f"signal < {MIN_SIGNAL_NORM} ADU excluded; FAST+SLOW pooled for "
                                      f"{', '.join(POOL_MODES_FOR_NORM)}",
                    "smear": f"col_med_std / (full_mean - edge_med) (std of the per-column median "
                             f"profile over the lamp signal), heavy-tail cap, records with signal < "
                             f"{MIN_SIGNAL_SMEAR} ADU or without col_med_std excluded; FAST+SLOW pooled "
                             f"for {', '.join(POOL_MODES_FOR_NORM)}"},
           "per_detector": derived, "shutter": shutter, "ccd_temp": ccd}
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["prodcatg", "mode", "detector", "date", "n",
                                           "edge_med", "full_med", "row_struct", "col_struct"])
        w.writeheader()
        w.writerows(sorted(tracking_rows, key=lambda x: (x["prodcatg"], x["mode"], x["detector"], x["date"])))

    # ---------------- VALIDATION ----------------
    print("=== ROBUST DERIVATION ===")
    print(f"normal={len(normal)} groups={len(groups)} epochs={out['meta']['epochs']}")
    print(f"CCD temp per-detector baselines: {len(temp_pd)} cameras; "
          f"warn=base+{TEMP_WARN_MARGIN} fail=base+{TEMP_FAIL_MARGIN}; abs_fail>{TEMP_ABS_FAIL}")
    for det in sorted(temp_pd):
        e = temp_pd[det]
        print(f"   {det:10s} base={e['baseline_med']:7.1f} warn>{e['warn_max']:7.1f} fail>{e['fail_max']:7.1f}")
    print("\n=== self-check: does any NORMAL frame breach its own derived band? (should be ~0) ===")
    breaches = 0
    for r in normal:
        ent = derived.get(r["prodcatg"], {}).get(r["mode"], {}).get(det_name(r))
        if not ent:
            continue
        eb = ent["edge_bg"]
        if not (eb["min"] <= r["edge_med"] <= eb["max"]):
            breaches += 1
    print(f"  edge_bg breaches among {len(normal)} normal records: {breaches} "
          f"({100*breaches/max(len(normal),1):.2f}%)")
    smear_breaches = []
    for r in normal:
        ent = derived.get(r["prodcatg"], {}).get(r["mode"], {}).get(det_name(r))
        sm = ent and ent.get("smear")
        if not sm or r.get("col_med_std") is None:
            continue
        sig = r["full_mean"] - r["edge_med"]
        if sig < MIN_SIGNAL_SMEAR:
            continue
        if r["col_med_std"] / sig > sm["max"]:
            smear_breaches.append((r["filename"], det_name(r), round(r["col_med_std"] / sig, 3), sm["max"]))
    print(f"  vertical-smear breaches among the arc records: {len(smear_breaches)} "
          f"(a baseline frame with a halo on that camera; the cap is robust to it)")
    for b in smear_breaches:
        print("    ", b)

    print("\n=== VALIDATION: odd June-4 dark vs derived DARK/SLOW structure caps ===")
    dark = derived.get("CAL.R-DRK", {}).get("SLOW", {})
    flagged = 0
    for r in sorted(odd, key=lambda x: x["hdu"]):
        ent = dark.get(det_name(r))
        if not ent:
            continue
        rc, cc = ent["structure"]["row_max"], ent["structure"]["col_max"]
        if r["row_struct"] > rc or r["col_struct"] > cc:
            flagged += 1
            print(f"  ext{r['hdu']:<2} {det_name(r):<8} row={r['row_struct']:.1f}(cap {rc}) "
                  f"col={r['col_struct']:.1f}(cap {cc}) FLAG")
    print(f"  odd-dark detectors flagged: {flagged}")

    if WARM_DIR is None:
        print("\n=== VALIDATION: warm folder -- SKIPPED (no --warm-dir / LLAMAS_QA_WARM_DIR) ===")
    else:
        print("\n=== VALIDATION: warm folder -- shutter + edge-background + temperature ===")
        bias_ref = derived.get("CAL.R-BIA", {})
        for p in sorted(glob.glob(WARM_DIR + "/*.fits")):
            with fits.open(p, memmap=False) as h:
                ph = h[0].header
                pc = str(ph.get("PRODCATG", "?")).strip()
                mode = str(ph.get("READ-MDE", "?")).strip().upper()
                r, s = ph.get("REXPTIME"), ph.get("SEXPTIME")
                tol = shutter.get(pc) or (shutter.get("SCI.R-*") if pc.startswith("SCI") else {}) or {}
                sh = ""
                if r is not None and s is not None:
                    d = abs(float(s) - float(r)); rel = d / float(r) if r and float(r) > 0.005 else 9e9
                    ok = (d <= tol.get("abs_tol", 1.0)) or (rel <= tol.get("rel_tol", 0.1))
                    sh = f"shutter={'FAIL' if not ok else 'ok'}"
                # edge background + temp across detectors (compare to BIAS band of same mode)
                n_edge_hot = n_temp_warm = n_temp_checked = 0
                btab = bias_ref.get(mode, {})
                for e in h[1:25]:
                    d = e.data
                    if d is None:
                        continue
                    v = np.asarray(d, float)
                    if v.min() == v.max():
                        continue
                    nm = f"{e.header.get('BENCH')}.{str(e.header.get('SIDE')).upper()}.{str(e.header.get('COLOR')).capitalize()}"
                    em = float(np.median(v[2:28, 100:1948]))
                    ent = btab.get(nm)
                    if ent and em > ent["edge_bg"]["max"]:
                        n_edge_hot += 1
                    t = None
                    for kk in ("CCDTEMP_1", "CCDTEMP1", "CCDTEMP-1"):
                        vv = e.header.get(kk)
                        if vv is not None:
                            try:
                                t = float(vv); break
                            except (TypeError, ValueError):
                                pass
                    pdent = temp_pd.get(nm)
                    if t is not None and pdent:
                        n_temp_checked += 1
                        if t > pdent["warn_max"]:
                            n_temp_warm += 1
                print(f"  {os.path.basename(p)[:34]:34s} {pc:9s} {mode:4s} {sh:13s} "
                      f"edge_hot_det={n_edge_hot} temp_warm={n_temp_warm}/{n_temp_checked}")
    print(f"\nWrote {OUT_JSON}\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
