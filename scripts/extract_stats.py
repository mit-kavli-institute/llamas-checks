#!/usr/bin/env python3
"""Phase B extraction: per-file, per-extension QA statistics for all baseline cal
MEF files in copies/. Parallel over files. Writes a flat JSON list of records to
qa_stats_raw.json for downstream aggregation/threshold derivation.

Region convention (2048x2048; illuminated fibre stack ~y[32:2004]):
  bottom_stripe = y[2:28],   x[100:1948]   (unilluminated)
  top_stripe    = y[2020:2046], x[100:1948] (unilluminated)
These stripes are the per-detector bias/background reference on ANY frame type.
Structure metrics follow the engine definition (std of the per-row/column
2 %-trimmed-mean profile, qa_engine.line_profiles); row_med_std / col_med_std are
the std of the per-row / per-column MEDIAN profiles from the same sort, and
col_med_smooth_std the same after the engine's running median over
DEFAULT_SMEAR_SMOOTH columns (the vertical-smear metric on arcs is
col_med_smooth_std / (full_mean - edge_med)); full_std is the robust 1.4826*MAD
sigma (full_std_plain = np.std).
Placeholder/missing extensions (constant-valued frames) are recorded with
placeholder=True and no statistics.

The copies/ folder comes from --copies-dir, or <baselines-root>/copies where the
root is --baselines-root or the LLAMAS_QA_BASELINES environment variable.
``--update EXISTING.json --files GLOB`` recomputes only the matching frames and
merges their records into an existing qa_stats_raw.json (records with the same
filename are replaced, everything else is kept), so a new statistic can be added
for one frame type without re-reading every baseline.
"""
import argparse
import concurrent.futures
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

from llamas_checks.qa_engine import DEFAULT_SMEAR_SMOOTH, line_profiles, running_median

SAT = 63000.0
BOT = (slice(2, 28), slice(100, 1948))       # (y, x)
TOP = (slice(2020, 2046), slice(100, 1948))

def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def extract_file(path):
    recs = []
    try:
        with fits.open(path, memmap=False) as h:
            ph = h[0].header
            base = dict(
                filename=os.path.basename(path),
                prodcatg=str(ph.get("PRODCATG", "?")).strip(),
                mode=str(ph.get("READ-MDE", "?")).strip().upper(),
                rexp=fnum(ph.get("REXPTIME")),
                sexp=fnum(ph.get("SEXPTIME")),
                dexp=fnum(ph.get("DEXPTIME")),
                date=str(ph.get("DATE-OBS", "?")).strip(),
            )
            for idx, e in enumerate(h[1:25], start=1):
                d = e.data
                col = str(e.header.get("COLOR", "?")).strip().lower()
                bench = e.header.get("BENCH")
                side = str(e.header.get("SIDE", "?")).strip().upper()
                ccdt = fnum(e.header.get("CCDTEMP_1"))
                rec = dict(base, hdu=idx, color=col, bench=bench, side=side,
                           ccdtemp=ccdt, placeholder=False)
                if d is None:
                    rec.update(placeholder=True, reason="nodata")
                    recs.append(rec); continue
                v = np.asarray(d, dtype=np.float64)
                vmin, vmax = float(np.nanmin(v)), float(np.nanmax(v))
                if vmin == vmax:  # constant-valued placeholder / missing camera
                    rec.update(placeholder=True, reason="constant", const=vmin)
                    recs.append(rec); continue
                finite = v[np.isfinite(v)]
                bot = v[BOT]; top = v[TOP]
                # Engine definitions: structure = std of the per-row/column 2 %-trimmed
                # means, rms = 1.4826*MAD (both immune to hot pixels / cosmic rays);
                # *_med_std = std of the per-line medians from the same sort (smear).
                rowprof, rowmed = line_profiles(v, axis=1)   # collapse X -> profiles over rows
                colprof, colmed = line_profiles(v, axis=0)   # collapse Y -> profiles over cols
                full_med = float(np.nanmedian(v))
                rec.update(
                    full_med=full_med,
                    full_std=float(1.4826 * np.nanmedian(np.abs(finite - full_med))),
                    full_std_plain=float(np.nanstd(v)),
                    full_mean=float(np.nanmean(v)),
                    full_p99=float(np.nanpercentile(finite, 99)),
                    full_max=vmax,
                    sat_frac=float(np.count_nonzero(finite > SAT) / finite.size),
                    bot_med=float(np.nanmedian(bot)),
                    bot_std=float(np.nanstd(bot)),
                    top_med=float(np.nanmedian(top)),
                    top_std=float(np.nanstd(top)),
                    edge_med=float(np.nanmedian(np.concatenate([bot.ravel(), top.ravel()]))),
                    row_struct=float(np.nanstd(rowprof[np.isfinite(rowprof)])),
                    col_struct=float(np.nanstd(colprof[np.isfinite(colprof)])),
                    row_med_std=float(np.nanstd(rowmed[np.isfinite(rowmed)])),
                    col_med_std=float(np.nanstd(colmed[np.isfinite(colmed)])),
                    # smear statistic as the engine computes it: narrow vertical lines
                    # removed by a running median before the std
                    col_med_smooth_std=float(np.nanstd(
                        running_median(np.nan_to_num(colmed, nan=np.nanmedian(colmed)),
                                       DEFAULT_SMEAR_SMOOTH))),
                )
                recs.append(rec)
    except Exception as exc:
        recs.append(dict(filename=os.path.basename(path), error=str(exc)))
    return recs

def main():
    parser = argparse.ArgumentParser(
        description="Extract per-file, per-extension QA statistics from the baseline copies/ folder.")
    parser.add_argument("--baselines-root", default=os.environ.get("LLAMAS_QA_BASELINES"),
                        help="QA_baselines root containing copies/ (default: $LLAMAS_QA_BASELINES)")
    parser.add_argument("--copies-dir", default=None,
                        help="folder of *_mef.fits baselines (default: <baselines-root>/copies)")
    parser.add_argument("--out", default="qa_stats_raw.json",
                        help="output JSON path (default: ./qa_stats_raw.json)")
    parser.add_argument("--jobs", type=int, default=6,
                        help="parallel worker processes (default: 6)")
    parser.add_argument("--update", default=None, metavar="EXISTING",
                        help="merge the new records into this existing qa_stats_raw.json "
                             "(records of the same filename are replaced); written to --out, "
                             "or in place when --out is left at its default")
    parser.add_argument("--files", default=None, metavar="GLOB",
                        help="frames to process instead of <copies-dir>/*_mef.fits "
                             "(e.g. '<root>/Arcs/*_mef.fits'); implies nothing about --update")
    args = parser.parse_args()
    if args.files is not None:
        files = sorted(glob.glob(args.files))
    else:
        if args.copies_dir is None:
            if not args.baselines_root:
                parser.error("--copies-dir, --baselines-root (or LLAMAS_QA_BASELINES) or --files is required")
            args.copies_dir = os.path.join(args.baselines_root, "copies")
        files = sorted(glob.glob(os.path.join(args.copies_dir, "*_mef.fits")))
    if args.update is not None and args.out == parser.get_default("out"):
        args.out = args.update
    out = os.path.abspath(args.out)
    print(f"[extract] {len(files)} files", flush=True)
    all_recs = []
    done = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(extract_file, p): p for p in files}
        for fut in concurrent.futures.as_completed(futs):
            all_recs.extend(fut.result())
            done += 1
            if done % 10 == 0:
                print(f"[extract] {done}/{len(files)} files", flush=True)
    if args.update is not None:
        with open(args.update) as fh:
            existing = json.load(fh)
        replaced = {r["filename"] for r in all_recs}
        kept = [r for r in existing if r.get("filename") not in replaced]
        print(f"[extract] merging into {args.update}: {len(existing)} records, "
              f"{len(existing) - len(kept)} replaced, {len(all_recs)} new", flush=True)
        all_recs = kept + all_recs
    with open(out, "w") as fh:
        json.dump(all_recs, fh)
    errs = [r for r in all_recs if "error" in r]
    print(f"[extract] DONE {len(all_recs)} records -> {out} | file-errors={len(errs)}", flush=True)
    if errs:
        print("  ", errs[:5], flush=True)

if __name__ == "__main__":
    sys.exit(main())
