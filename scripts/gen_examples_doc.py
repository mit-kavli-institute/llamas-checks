#!/usr/bin/env python3
"""Generate docs/QA_EXAMPLES.md: real LLAMAS frames that PASS, WARN and FAIL, with
images, the rules that fired and the previous (pre-2026-09) verdict.

For every case the shipped config is selected from the frame's PRODCATG exactly as
``llamas-checks`` does, ``QAEngine.run()`` is executed in-process (nothing is ever
written next to the input frames), a 6x4 mosaic of the 24 detectors and a zoom of
the worst detector are rendered with matplotlib, and a markdown section is emitted.

Data roots come from arguments or environment variables (a case whose root is unset
or whose file is missing is reported as MISSING and skipped):
  --baselines-root      LLAMAS_QA_BASELINES          Box LLAMAS_analysis/QA_baselines
  --warm-dir            LLAMAS_QA_WARM_DIR           2026-05-06 warm-incident frames
  --commissioning-dir   LLAMAS_QA_COMMISSIONING_DIR  ut20260710_11 commissioning frames
  --sept06-dir          LLAMAS_QA_SEPT06_DIR         Llamas_Commissioning_Data/20260906_07_cals
  --sept07-dir          LLAMAS_QA_SEPT07_DIR         Llamas_Commissioning_Data/20260907_08

Images are written as JPEG (noise-like frames do not compress as PNG) into
<out-dir>/images/, the document to <out-dir>/QA_EXAMPLES.md.
"""
import argparse
import os
from pathlib import Path

import matplotlib
import numpy as np
from astropy.io import fits

from llamas_checks.paths import CAL_CONFIG_NAME, CONFIG_DIR, SCIENCE_CONFIG_NAME
from llamas_checks.qa_engine import QAEngine, QAEngineError, load_yaml, trimmed_mean_profile

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot import)

REPO_ROOT = Path(__file__).resolve().parents[1]
EXIT_FOR_VERDICT = {"PASS": 0, "WARN": 1, "FAIL": 2, "ERROR": 2}
SEVERITY_COLOR = {"FAIL": "#d62728", "WARN": "#ff9f1c"}
BIN = 4                      # 2048 -> 512 px per detector before plotting
PLACEHOLDER_VALUES = (0.0, 1.0)

# Detector to zoom on when the text discusses a specific camera; otherwise the zoom is the
# detector with the largest excess over its limit (FAIL rules first), or the first live one.
ZOOM_OVERRIDE = {
    "pass_bias_fast": "4.A.Red",
    "pass_ldls_long": "1.B.Red",
    "pass_bias_slow_hotpixels": "3.A.Red",
    "pass_dark_glow_2bgreen": "2.B.Green",
    "pass_dark_glow_multi": "1.A.Red",
    "warn_bias_fast_hotpixels": "2.A.Red",
}

# (slug, title, root key, relative path, previous verdict, explanation)
CASES = {
    "PASS": [
        ("pass_bias_fast", "FAST bias", "sept06",
         "LLAMAS_2026-09-06_18-21-00.7_CAL22_mef.fits", "PASS",
         "Normal FAST bias. 4.A.Red shows its usual left-edge glow and horizontal streaks; "
         "with the trimmed-mean structure metric this fixed pattern stays inside its cap."),
        ("pass_bias_slow", "SLOW bias", "sept06",
         "LLAMAS_2026-09-06_19-04-16.9_CAL22_mef.fits", "PASS",
         "Normal SLOW bias: flat pedestal, read noise only."),
        ("pass_dark", "600 s dark", "sept06",
         "LLAMAS_2026-09-06_19-39-23.0_CAL22_mef.fits", "PASS",
         "Normal 600 s dark: level at the bias pedestal, no banding, a few hot pixels."),
        ("pass_ldls_short", "LDLS flat, 0.07 s", "sept06",
         "LLAMAS_2026-09-06_18-52-19.2_CAL22_mef.fits", "PASS",
         "A short LDLS exposure (the calibration script suggests 0.07 s): fibres well exposed in the reds "
         "(peak ~65k in the brightest fibres), unilluminated edge stripes at the bias level."),
        ("pass_ldls_long", "LDLS flat, 0.3 s (reds railed by a normal blue-channel exposure)", "sept06",
         "LLAMAS_2026-09-06_18-53-43.1_CAL22_mef.fits", "PASS",
         "A longer LDLS exposure of the kind taken for the blue channel. The red detectors are "
         "saturated over most of the illuminated area -- this is normal and expected, so the "
         "per-detector saturation caps do not fire. The edge stripes stay unsaturated."),
        ("pass_arc", "ThAr arc", "sept06",
         "LLAMAS_2026-09-06_18-55-49.0_CAL22_mef.fits", "PASS",
         "Normal ThAr arc: emission-line spectra on every live detector."),
        ("pass_sky", "Twilight sky flat", "sept06",
         "LLAMAS_2026-09-06_22-28-03.6_CAL22_mef.fits", "PASS",
         "Normal twilight flat at a brightness comparable to the baseline set."),
        ("pass_bias_slow_hotpixels", "SLOW bias with hot pixels on 3.A.Red (was FAIL)", "sept06",
         "LLAMAS_2026-09-06_19-08-53.8_CAL22_mef.fits",
         "FAIL (3.A.Red row 3.4 vs 2.0, column 5.6 vs 2.0; rms 106 vs 25)",
         "A few saturated pixels on 3.A.Red drove the old per-column MEAN profile and the plain "
         "std, tripping three rules on a frame that is pure read noise. The trimmed-mean profile "
         "and the MAD-based rms ignore them: nothing fires."),
        ("pass_dark_glow_2bgreen", "600 s dark with a mild 2.B.Green glow (was FAIL)", "sept06",
         "LLAMAS_2026-09-06_19-28-14.2_CAL22_mef.fits",
         "FAIL (2.B.Green row 2.37 vs 2.30, column 4.57 vs 2.0; rms 86 vs 34)",
         "A smooth ~5 ADU large-scale gradient (amplifier glow) plus the usual hot pixels of a "
         "600 s integration. The hot pixels, not the glow, produced the old column excess; with "
         "the trimmed-mean profile the frame sits inside every cap."),
        ("pass_dark_glow_multi", "600 s dark with glow on three detectors (was FAIL)", "sept07",
         "LLAMAS_2026-09-07_19-15-34.6_CAL22_mef.fits",
         "FAIL (structure on 1.A.Red, 2.A.Green, 4.A.Red at 1.4-2.9x cap; rms 85-112)",
         "A ~16 ADU vignetting-like glow on 1.A.Red and milder gradients on two other detectors "
         "after 600 s. All of the old excess came from hot pixels and cosmic rays; the frame passes."),
        ("pass_science_banding", "Science frame with 4.A.Red fixed banding", "commissioning",
         "LLAMAS_2026-07-11_05-05-22.4_SCI22_mef.fits", "PASS",
         "Science frames run shutter, gross saturation, edge saturation, camera-warming rate and "
         "CCD temperature only. The 4.A.Red electronic banding is a fixed pattern, not a "
         "dark-current gradient, so the warming-rate check correctly ignores it."),
    ],
    "WARN": [
        ("warn_bias_fast_4ared", "FAST bias with strong 4.A.Red banding (was FAIL)", "sept06",
         "LLAMAS_2026-09-06_19-14-52.1_CAL22_mef.fits", "FAIL (row_structure @ 4.A.Red 382.5 vs 259.8)",
         "The amplitude of the 4.A.Red left-edge glow and streaks varies from frame to frame and "
         "on 2026-09-06/07 ran 1.4-2x above the April-June baseline cap on most FAST biases. That "
         "is now a WARN; a FAIL needs 4x the cap (the *_gross rules), which only a real defect "
         "such as the June-4 bars reaches. Adding a September epoch to the baselines would widen "
         "this cap (see the catalogue)."),
        ("warn_bias_fast_hotpixels", "FAST bias with a saturated hot-pixel cluster on 2.A.Red (was FAIL)", "sept06",
         "LLAMAS_2026-09-06_20-33-17.7_CAL22_mef.fits",
         "FAIL (2.A.Red row 6.1 vs 2.0, column 11.2 vs 2.0; rms 159 vs 18)",
         "About twenty railed pixels in one column of 2.A.Red drove the old per-column MEAN "
         "profile and the plain std, tripping three rules on a detector that is pure read noise. "
         "The trimmed-mean profile and the MAD-based rms ignore them; the only thing left is the "
         "4.A.Red banding WARN shared by most FAST biases of that night."),
        ("warn_sky_bright", "Bright twilight flat (was FAIL)", "sept06",
         "LLAMAS_2026-09-06_22-22-03.4_CAL22_mef.fits",
         "FAIL (row/column structure 1.00-1.04x cap on four blue/green detectors)",
         "Twilight brightness varies by design and the structure metric scales with it, so "
         "structure on sky flats is WARN only. The observer logged this frame as 'good blue'. "
         "The many edge-background and saturation WARNs are the brightness itself."),
        ("warn_ldls_edge", "LDLS flat, 0.3 s, 1.B.Red edge stripe high", "sept06",
         "LLAMAS_2026-09-06_18-53-57.5_CAL22_mef.fits", "WARN (edge_background_level @ 1.B.Red)",
         "The unilluminated stripe on 1.B.Red is brighter than the baseline band (scattered "
         "light at a longer exposure) but far from saturated."),
        ("warn_science_warming", "Science frame with a warming camera (3.A.Green)", "warm",
         "LLAMAS_2026-05-06_06-08-21.4_SCI22_mef.fits", "WARN (camera_warming_gradient @ 3.A.Green)",
         "Dark-current glow growing at ~2.1 ADU/s on 3.A.Green while the header CCD temperature "
         "still reads its baseline value. Advisory: check the camera."),
    ],
    "FAIL": [
        ("fail_ldls_railed_all", "LDLS flat, 0.07 s, railed on every detector", "sept07",
         "LLAMAS_2026-09-07_18-17-12.8_CAL22_mef.fits", "WARN (28 warn checks, exit 1)",
         "Light flooded the whole detector: medians of 64-65k on all 22 live detectors, and the "
         "unilluminated bottom stripe itself is at the ADC ceiling. The new edge_saturated rule "
         "makes this a FAIL. A normal 0.07 s flat (see PASS) has red medians of 15-26k."),
        ("fail_ldls_railed_reds", "LDLS flat, 0.3 s, edge stripes railed in the reds", "sept07",
         "LLAMAS_2026-09-07_18-18-55.0_CAL22_mef.fits", "WARN (13 warn checks, exit 1)",
         "Reds and greens railed and the red edge stripes at 65535, with the blues 2-4x brighter "
         "than a normal 0.3 s step. Railed reds alone are normal at this exposure (see PASS); a saturated "
         "unilluminated stripe is not."),
        ("fail_dark_odd_bars", "600 s dark with periodic column bars (June 4)", "baselines",
         "copies/LLAMAS_2026-06-04_20-36-33.1_CAL22_mef.fits", "FAIL (column_structure @ 2.B.Green)",
         "Full-height vertical bars across 2.B.Green, ~13 ADU peak to peak: an unambiguous "
         "electronic defect, several times the gross (4x) cap."),
        ("fail_arc_odd", "ThAr arc with odd structure on every detector (July 10)", "commissioning",
         "LLAMAS_2026-07-10_20-22-43.7_CAL22_mef.fits", "FAIL (column_structure on 22/22 detectors)",
         "Blooming/excess structure on every live detector relative to the fixed-lamp ThAr "
         "baseline. Structure on fixed-lamp frames (ThAr, LDLS) remains a FAIL at the cap."),
        ("fail_shutter_science", "Science frame with a shutter fault (600 s requested)", "warm",
         "LLAMAS_2026-05-06_05-10-56.6_SCI22_mef.fits", "FAIL (shutter_exptime_consistency)",
         "SEXPTIME 181 s against REXPTIME 600 s: the shutter closed early."),
        ("fail_shutter_bias", "Bias with the shutter open", "warm",
         "LLAMAS_2026-05-06_05-56-30.7_CAL0_mef.fits", "FAIL (shutter_exptime_consistency)",
         "SEXPTIME 0.352 s for a 0.001 s bias: not a bias."),
        ("fail_ldls_baseline_overrun", "Baseline LDLS flat with a shutter overrun (June 30)", "baselines",
         "lamp_flats/LLAMAS_2026-06-30_19-33-58.0_CAL22_mef.fits", "PASS (undetected)",
         "REXPTIME 0.15 s but SEXPTIME 0.246 s (inside the shutter tolerance), leaving the red "
         "edge stripes at 65535. Newly caught by edge_saturated; it stays in the baseline set "
         "because the level bands are MAD-robust and it is documented here."),
    ],
    "ERROR": [
        ("error_truncated", "Truncated file", "commissioning",
         "LLAMAS_2026-07-11_10-19-11.8_SCI22_mef.fits", "exit 3",
         "The file cannot be read to the end; `llamas-checks` exits 3 (system error), the batch "
         "engine reports ERROR."),
    ],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate docs/QA_EXAMPLES.md with images.")
    parser.add_argument("--baselines-root", default=os.environ.get("LLAMAS_QA_BASELINES"))
    parser.add_argument("--warm-dir", default=os.environ.get("LLAMAS_QA_WARM_DIR"))
    parser.add_argument("--commissioning-dir", default=os.environ.get("LLAMAS_QA_COMMISSIONING_DIR"))
    parser.add_argument("--sept06-dir", default=os.environ.get("LLAMAS_QA_SEPT06_DIR"))
    parser.add_argument("--sept07-dir", default=os.environ.get("LLAMAS_QA_SEPT07_DIR"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "docs"),
                        help="where QA_EXAMPLES.md and images/ go (default: the repo docs/)")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    return parser.parse_args()


def config_for(path):
    prodcatg = str(fits.getheader(path, 0).get("PRODCATG", "")).strip().upper()
    name = SCIENCE_CONFIG_NAME if prodcatg.startswith("SCI") else CAL_CONFIG_NAME
    return load_yaml(CONFIG_DIR / name)


def is_placeholder(data):
    finite = data[np.isfinite(data)]
    return finite.size == 0 or (finite.min() == finite.max() and finite.min() in PLACEHOLDER_VALUES)


def binned(data, factor=BIN):
    ny, nx = data.shape[-2] // factor * factor, data.shape[-1] // factor * factor
    view = data[:ny, :nx].reshape(ny // factor, factor, nx // factor, factor)
    return view.mean(axis=(1, 3))


def fired_by_extension(report):
    """{extension: worst verdict_effect} over EVALUATED, failed results."""
    worst = {}
    rank = {"WARN": 1, "FAIL": 2}
    for r in report["results"]:
        if r["status"] != "EVALUATED" or r["passed"] or r["extension"] is None:
            continue
        if rank.get(r["verdict_effect"], 0) > rank.get(worst.get(r["extension"]), 0):
            worst[r["extension"]] = r["verdict_effect"]
    return worst


def worst_extension(report, extensions):
    """Extension with the largest value/limit excess among fired rules (FAIL rules outrank
    WARN rules), else the first configured one."""
    best, best_key = None, (-1, 0.0)
    for r in report["results"]:
        if r["status"] != "EVALUATED" or r["passed"] or r["extension"] is None:
            continue
        lim, value = r["limits"] or {}, r["measured_value"]
        ratio = value / lim["max"] if "max" in lim and lim["max"] and value > lim["max"] else 1.0
        key = (1 if r["verdict_effect"] == "FAIL" else 0, ratio)
        if key >= best_key:
            best, best_key = r["extension"], key
    if best is None:
        best = extensions[0]["name"]
    return best


def render(path, report, config, images_dir, slug, quality):
    extensions = config["extensions"]
    fired = fired_by_extension(report)
    zoom_name = ZOOM_OVERRIDE.get(slug) or worst_extension(report, extensions)
    stats_line = ""
    rendered = {}            # extension name -> binned image (placeholders are not rendered)
    fig, axes = plt.subplots(4, 6, figsize=(9.6, 6.6))
    with fits.open(path, memmap=False) as hdul:
        for ax, ext in zip(axes.ravel(), extensions):
            ax.set_xticks([]); ax.set_yticks([])
            idx = ext["hdu_index"]
            data = None if idx >= len(hdul) else hdul[idx].data
            if data is None or is_placeholder(np.asarray(data, dtype=float)):
                ax.set_facecolor("#bbbbbb"); ax.set_title(f"{ext['name']} (missing)", fontsize=7)
                continue
            data = np.asarray(data, dtype=float)
            small = binned(data)
            lo, hi = np.percentile(small, [1, 99])
            ax.imshow(small, origin="lower", cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
            ax.set_title(ext["name"], fontsize=7)
            color = SEVERITY_COLOR.get(fired.get(ext["name"]))
            if color:
                for spine in ax.spines.values():
                    spine.set_edgecolor(color); spine.set_linewidth(3)
            rendered[ext["name"]] = (small, data)
    fig.suptitle(f"{Path(path).name}  ->  {report['overall_verdict']}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    mosaic = images_dir / f"{slug}_mosaic.jpg"
    fig.savefig(mosaic, format="jpg", dpi=72, pil_kwargs={"quality": quality})
    plt.close(fig)

    if not rendered:                      # every camera is a placeholder (e.g. the CAL0 frames)
        return mosaic.name, None, None, "no live detector in this frame"
    if zoom_name not in rendered:
        zoom_name = next(iter(rendered))
    zoom, data = rendered[zoom_name]
    rows = trimmed_mean_profile(data, 1); cols = trimmed_mean_profile(data, 0)
    med = np.median(data); mad = 1.4826 * np.median(np.abs(data - med))
    stats_line = (f"median {med:.0f} ADU, robust rms {mad:.1f}, row structure {np.std(rows):.2f}, "
                  f"column structure {np.std(cols):.2f}, saturated fraction {np.mean(data > 63000):.3f}")
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    lo, hi = np.percentile(zoom, [1, 99])
    ax.imshow(zoom, origin="lower", cmap="gray", vmin=lo, vmax=max(hi, lo + 1))
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{zoom_name}  (stretch {lo:.0f}-{hi:.0f} ADU)", fontsize=9)
    fig.tight_layout()
    zoom_path = images_dir / f"{slug}_zoom.jpg"
    fig.savefig(zoom_path, format="jpg", dpi=72, pil_kwargs={"quality": quality})
    plt.close(fig)
    return mosaic.name, zoom_path.name, zoom_name, stats_line


def rules_table(report, limit=12):
    rows = [r for r in report["results"]
            if r["status"] == "EVALUATED" and not r["passed"]]
    rows.sort(key=lambda r: (r["verdict_effect"] != "FAIL", r["rule"], r["extension"] or ""))
    if not rows:
        return "All evaluated checks passed.\n"
    out = ["| severity | rule | detector | value | limit |", "|---|---|---|---|---|"]
    for r in rows[:limit]:
        lim = r["limits"] or {}
        lim_s = ", ".join(f"{k} {v:.4g}" for k, v in lim.items())
        value = r["measured_value"]
        value_s = f"{value:.4g}" if isinstance(value, (int, float)) else str(value)
        out.append(f"| {r['verdict_effect']} | `{r['rule']}` | {r['extension'] or '-'} | {value_s} | {lim_s} |")
    if len(rows) > limit:
        out.append(f"| ... | +{len(rows) - limit} more | | | |")
    return "\n".join(out) + "\n"


def header_info(path):
    h = fits.getheader(path, 0)
    return (f"`{h.get('OBJECT', '?')}` / PRODCATG `{h.get('PRODCATG', '?')}` / "
            f"READ-MDE `{h.get('READ-MDE', '?')}` / REXPTIME {h.get('REXPTIME', '?')} s / "
            f"SEXPTIME {h.get('SEXPTIME', '?')} s")


def main():
    args = parse_args()
    roots = {"baselines": args.baselines_root, "warm": args.warm_dir,
             "commissioning": args.commissioning_dir, "sept06": args.sept06_dir, "sept07": args.sept07_dir}
    out_dir = Path(args.out_dir); images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    doc = ["# LLAMAS raw-frame QA — worked examples: what passes, warns and fails", "",
           "*Generated by `scripts/gen_examples_doc.py` from real frames; regenerate rather than edit.*", "",
           "Every frame below was run through the shipped rules exactly as `llamas-checks <file>` does.",
           "Exit codes: **0 PASS**, **1 WARN**, **2 FAIL**, **3 unreadable file**. Each mosaic shows the 24",
           "detectors in HDU order (one bench per row: A-side red/green/blue, then B-side), each panel stretched",
           "to its own 1–99 % range; missing cameras are grey; a panel outlined **red** fired a FAIL rule,",
           "**orange** a WARN rule. The zoom is the detector with the largest excess over its limit (or the",
           "first live detector when nothing fired). \"Previously\" is the verdict before the 2026-09 revision",
           "(2 %-trimmed-mean structure profiles, MAD rms, WARN/FAIL tiers, `edge_saturated`).", "",
           "See `QA_CHECKS_CATALOGUE.md` for every rule and threshold.", ""]
    summary = []
    for group, cases in CASES.items():
        doc += [f"## {group}", ""]
        for slug, title, root_key, rel, previous, explanation in cases:
            root = roots[root_key]
            path = Path(root) / rel if root else None
            if path is None or not path.is_file():
                doc += [f"### {title}", "", f"**MISSING FILE** — `{rel}` under `{root_key}` root "
                        f"({'unset' if root is None else root}).", ""]
                summary.append((title, previous, "MISSING", "-"))
                continue
            try:
                config = config_for(path)
                report = QAEngine(config).run(path)
                verdict = report["overall_verdict"]
                exit_code = EXIT_FOR_VERDICT[verdict]
            except (QAEngineError, OSError, ValueError) as exc:
                verdict, exit_code, report, config = "unreadable", 3, None, None
                error = f"{type(exc).__name__}: {exc}"
            doc += [f"### {title}", "", f"- File: `{path}`", f"- Header: {header_info(path)}",
                    f"- **Verdict: {verdict}** (`llamas-checks` exit {exit_code}); previously: {previous}", "",
                    explanation, ""]
            if report is None:
                doc += [f"```", error, "```", ""]
                summary.append((title, previous, verdict, str(exit_code)))
                continue
            mosaic, zoom, zoom_name, stats = render(path, report, config, images_dir, slug, args.jpeg_quality)
            doc += [f"![{slug} mosaic](images/{mosaic})", ""]
            if zoom is not None:
                doc += [f"![{slug} zoom](images/{zoom})", "", f"Zoomed detector {zoom_name}: {stats}.", ""]
            else:
                doc += [f"({stats}; only the header checks run.)", ""]
            doc += [rules_table(report), ""]
            summary.append((title, previous, verdict, str(exit_code)))
            print(f"[examples] {group:5s} {verdict:5s} exit {exit_code}  {path.name}", flush=True)

    doc += ["## Summary of verdict changes", "", "| case | previously | now | exit |", "|---|---|---|---|"]
    doc += [f"| {t} | {p} | **{v}** | {e} |" for t, p, v, e in summary]
    doc.append("")
    (out_dir / "QA_EXAMPLES.md").write_text("\n".join(doc), encoding="utf-8")
    total = sum(p.stat().st_size for p in images_dir.glob("*.jpg"))
    print(f"[examples] wrote {out_dir / 'QA_EXAMPLES.md'}; images {total / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
