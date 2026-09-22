#!/usr/bin/env python3
"""Phase B extraction: per-file, per-extension QA statistics for all baseline cal
MEF files in copies/. Parallel over files. Writes a flat JSON list of records to
qa_stats_raw.json for downstream aggregation/threshold derivation.

Region convention (2048x2048; illuminated fibre stack ~y[32:2004]):
  bottom_stripe = y[2:28],   x[100:1948]   (unilluminated)
  top_stripe    = y[2020:2046], x[100:1948] (unilluminated)
These stripes are the per-detector bias/background reference on ANY frame type.
Structure metrics follow the engine definition (std of the collapsed profile).
Placeholder/missing extensions (constant-valued frames) are skipped.

The copies/ folder comes from --copies-dir, or <baselines-root>/copies where the
root is --baselines-root or the LLAMAS_QA_BASELINES environment variable.
"""
import argparse
import concurrent.futures
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits

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
                rowprof = np.nanmean(v, axis=1)   # collapse X -> profile over rows
                colprof = np.nanmean(v, axis=0)   # collapse Y -> profile over cols
                rec.update(
                    full_med=float(np.nanmedian(v)),
                    full_std=float(np.nanstd(v)),
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
    args = parser.parse_args()
    if args.copies_dir is None:
        if not args.baselines_root:
            parser.error("--copies-dir or --baselines-root is required (or set LLAMAS_QA_BASELINES)")
        args.copies_dir = os.path.join(args.baselines_root, "copies")
    out = os.path.abspath(args.out)

    files = sorted(glob.glob(os.path.join(args.copies_dir, "*_mef.fits")))
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
    with open(out, "w") as fh:
        json.dump(all_recs, fh)
    errs = [r for r in all_recs if "error" in r]
    print(f"[extract] DONE {len(all_recs)} records -> {out} | file-errors={len(errs)}", flush=True)
    if errs:
        print("  ", errs[:5], flush=True)

if __name__ == "__main__":
    sys.exit(main())
