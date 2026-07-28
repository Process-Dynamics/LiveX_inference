#!/usr/bin/env python3
"""Step-by-step visualization of the flat-field pipeline on ONE chosen frame.

Runs the full pipeline for a single frame and writes one image per stage:

    step1_raw.png                  the raw input frame
    step2_flat.png                 the flat (mean of the reference images)
    step3_division_no_reg.png      frame / flat WITHOUT registration
    step4_registered_division.png  frame / locally-registered flat (the pipeline)
    step5_temporal_mean.png        trailing mean over the last --mean-win frames
    step6_final.png                contrast stretch + Gaussian = pipeline output
    overview.png                   all six panels laid out in a 2x3 grid

Comparing step 3 with step 4 shows what the tiled non-rigid registration
removes; step 4 -> 5 shows the noise suppression of the temporal average.
Steps 1-5 are displayed with their own 1%/99% stretch so they are comparable.

The few frames this script registers are always cached (corr_<stem>.npy in
<output>/_corr_cache/, same format flatfield_correct.py --cache writes), so
re-running it on the same frame finishes in seconds.

Usage:
  python visualize_steps.py --ref flats/ --input frames/ --frame 1352 --output viz/
  (--frame matches the trailing number or the full stem of a file in --input)
"""

import argparse
import os
import time

import cv2
import numpy as np

import flatfield_correct as fc

# ---- default paths (EDIT these for your data; CLI arguments override them) ----
REF_PATH = r"path/to/reference_images"
INPUT_PATH = r"path/to/frames"
OUTPUT_DIR = r"path/to/viz_output"
FRAME = ""                       # trailing number (e.g. "1352") or full stem

LABEL_H = 90                     # white header strip on each overview panel, px


def disp(img):
    """Display version of a float image: its own 1%/99% stretch -> 8-bit."""
    p1, p99 = np.percentile(img, [1, 99])
    return (np.clip((img - p1) / (p99 - p1 + 1e-9), 0, 1) * 255).astype(np.uint8)


def save(path, img8):
    fc.save_atomic(path, lambda p: fc._write_png(p, img8))
    print(f"  [out] {os.path.basename(path)}")


def labeled(img8, text, scale):
    """Panel with a white header strip carrying the step label."""
    h, w = img8.shape
    small = cv2.resize(img8, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
    strip = np.full((LABEL_H // scale + 24, small.shape[1]), 255, np.uint8)
    cv2.putText(strip, text, (12, strip.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, 0, 2, cv2.LINE_AA)
    return np.vstack([strip, small])


def pick_frame(items, key):
    """Match --frame against the full filename stem, or against the number at the
    END of the stem ('mg41078_0056_frame_001352' is frame 1352, not 410780056001352)."""
    if not key:
        raise SystemExit("Pick a frame: --frame <trailing number or filename stem> "
                         f"(available: {items[0][0]} .. {items[-1][0]})")
    want = str(key).strip()
    for j, (stem, path) in enumerate(items):
        num = fc.frame_number(path)
        if stem == want or (num is not None and want.lstrip("0") == str(num)):
            return j
    raise SystemExit(f"--frame {key!r} not found in the input folder "
                     f"(available: {items[0][0]} .. {items[-1][0]})")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=REF_PATH,
                    help="reference (flat) image OR folder; mean = flat.")
    ap.add_argument("--input", default=INPUT_PATH,
                    help="image OR folder the chosen frame comes from.")
    ap.add_argument("--output", default=OUTPUT_DIR,
                    help="output folder (its _corr_cache is shared with "
                         "flatfield_correct.py --cache).")
    ap.add_argument("--frame", default=FRAME,
                    help="which frame to visualize: trailing number or full stem.")
    ap.add_argument("--mean-win", type=int, default=3,
                    help="trailing average window for step 5 (default 3).")
    ap.add_argument("--tile", type=int, default=fc.TILE,
                    help="registration tile size (must match the run being "
                         "visualized if reusing its cache).")
    ap.add_argument("--tile-step", type=int, default=fc.TSTEP,
                    help="tile stride; overlap = tile - step.")
    ap.add_argument("--sigma", type=float, default=fc.SIGMA,
                    help="final Gaussian sigma for step 6.")
    args = ap.parse_args()

    t0 = time.perf_counter()
    for label, p in (("--ref", args.ref), ("--input", args.input)):
        if not os.path.exists(p):
            raise SystemExit(
                f"{label} not found: {p}\n"
                "Pass --ref/--input/--output/--frame on the command line, or edit "
                "the constants at the top of this script (they ship as "
                "'path/to/...' placeholders).")
    files = fc.list_images(args.input)
    items = [(os.path.splitext(os.path.basename(f))[0], f) for f in files]
    stems = [s for s, _ in items]
    if len(set(stems)) != len(stems):
        raise SystemExit("Duplicate basenames in --input; cache entries would collide.")
    j = pick_frame(items, args.frame)      # fail before creating anything

    os.makedirs(args.output, exist_ok=True)
    cache = os.path.join(args.output, "_corr_cache")
    os.makedirs(cache, exist_ok=True)
    fc.check_recipe(cache,
                    {"ref": fc.ref_fingerprint(args.ref), "tile": args.tile,
                     "tile_step": args.tile_step},
                    guard=any(f.startswith("corr_") for f in os.listdir(cache)),
                    holds="cached registrations")
    flat = fc.build_flat(args.ref, cache)
    fc.check_tiling(args.tile, args.tile_step, flat.shape)
    stem, path = items[j]
    N = max(1, args.mean_win)
    window = items[max(0, j - (N - 1)):j + 1]
    print(f"[frame]  {stem}  (window for step 5: {[s for s, _ in window]})")

    # register the window (cache-aware; the chosen frame is window[-1])
    corrs = []
    for s, p in window:
        cp = os.path.join(cache, f"corr_{s}.npy")
        if os.path.exists(cp):
            print(f"[reg]    {s}: cached")
            corrs.append(np.load(cp))
        else:
            t1 = time.perf_counter()
            c = fc.register_frame(flat, fc.read_norm(p), args.tile, args.tile_step)
            fc.save_atomic(cp, lambda q: fc._write_npy(q, c))
            print(f"[reg]    {s}: {time.perf_counter() - t1:.1f}s")
            corrs.append(c)

    raw = fc.read_norm(path)
    corr = corrs[-1]
    plain = raw / (flat + fc.EPS)                       # step 3: no registration
    mean = np.mean(corrs, axis=0).astype(np.float32)    # step 5
    p1, p99 = np.percentile(mean, [1, 99])              # step 6 = pipeline finish
    final = cv2.GaussianBlur(
        np.clip((mean - p1) / (p99 - p1 + 1e-9), 0, 1).astype(np.float32),
        (0, 0), sigmaX=args.sigma)
    final = (final * 255).astype(np.uint8)

    steps = [
        ("step1_raw", f"1. raw frame ({stem})", disp(raw)),
        ("step2_flat", "2. flat = mean of reference images", disp(flat)),
        ("step3_division_no_reg", "3. frame / flat (NO registration)", disp(plain)),
        ("step4_registered_division", "4. frame / registered flat", disp(corr)),
        ("step5_temporal_mean", f"5. trailing mean{len(corrs)}", disp(mean)),
        ("step6_final", f"6. final: 1/99 stretch + Gaussian s{args.sigma:g}", final),
    ]
    for name, _, img8 in steps:
        save(os.path.join(args.output, f"{name}.png"), img8)

    panels = [labeled(img8, text, scale=4) for _, text, img8 in steps]
    rows = [np.hstack(panels[:3]), np.hstack(panels[3:])]
    save(os.path.join(args.output, "overview.png"), np.vstack(rows))
    print(f"[total]  {time.perf_counter() - t0:.1f}s -> {args.output}")


if __name__ == "__main__":
    main()
