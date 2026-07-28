#!/usr/bin/env python3
"""Flat-field correction with tiled non-rigid registration for time-resolved X-ray imaging.

Pipeline (per frame):
  1. flat      = mean of the reference image(s)  (--ref: one image or a folder)
  2. corrected = frame / locally-registered flat
       The flat is warped to the frame independently in overlapping tiles
       (template-matching landmarks + thin-plate-spline, Hann-blended), which
       removes both the static background structure and its slow drift.
  3. finish    = 1%/99% contrast stretch -> Gaussian blur -> 8-bit PNG.

Frames are registered in parallel across worker processes, and PNGs stream out
one by one during the run rather than all at the end. Nothing except the output
PNGs is written to disk; --cache additionally keeps every frame's registration
result (~38 MB per frame) so an interrupted run can resume and --sigma /
--mean-win can be changed without registering again.

By default (--mean-win 3) each output frame is the average of the last 3
registered frames - a trailing/causal window pushed through a ring buffer, so
results are consumed in frame order (the first mean-win - 1 outputs are
averages of fewer frames, since the window has not filled yet). --mean-win 1
disables the averaging; that path writes each frame as soon as ITS OWN
registration completes, in whatever order workers finish (not window order,
since there is no window). The 1%/99% stretch endpoints are computed per output
frame (adapts to slow background drift such as liquid-composition change). A
single-image input is processed alone.

Usage:
  python flatfield_correct.py --ref ref_dir_or_image --input dir_or_image --output out_dir
  python flatfield_correct.py --ref flats/ --input frames/ --output out/ --mean-win 1
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import json
import multiprocessing as mp
import re
import sys
import time

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator

cv2.setNumThreads(1)
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

VALID_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")
EPS = 1e-6

# ---- default paths (EDIT these for your data; CLI arguments override them) ----
REF_PATH = r"path/to/reference_images"    # one image or a folder -> averaged into the flat
INPUT_PATH = r"path/to/frames"            # one image or a folder of frames to correct
OUTPUT_DIR = r"path/to/output"            # corrected PNGs are written here

# ---- tiling / finish defaults (CLI-overridable) ----
TILE = 512                # tile size for local registration
TSTEP = 384               # tile stride (overlap = TILE - TSTEP, Hann-blended)
SIGMA = 3.0               # final Gaussian sigma

# ---- registration knobs (validated defaults; rarely need editing) ----
PATCH, SEARCH, NCC = 41, 26, 0.4     # template size / search radius / min match score
GRID = 45                            # landmark grid spacing inside a tile
NCORNER, QUAL, MINDIST, BLK = 120, 0.005, 22, 11   # extra corner landmarks
TPS_GS, TPS_SM, ITERS = 12, 0.3, 12  # TPS grid step / smoothing / refine iterations
RMARGIN = 160                        # context margin around each tile
TARGET_GRID_PTS = 700                # cap on TPS control points (solve is O(N^3))
DEFAULT_WORKERS = max(1, min(14, (os.cpu_count() or 4) - 2))

# A worker killed abruptly by the OS (most commonly the OOM killer) is silently
# respawned by multiprocessing.Pool, but the task it was running is lost, so the
# result iterator then waits forever for a result that can never arrive. Bound the
# wait so that failure is loud and immediate instead of an indefinite, silent hang.
# Registration is ~60-100s/frame even at 4096x2304 with the default tiling, so this
# is generous headroom, not a real per-frame budget.
FRAME_TIMEOUT_S = 1800


# ----------------------------------------------------------------- I/O helpers

def read_norm(path):
    """Read an image as float32 [0, 1] grayscale, normalizing by its dtype range."""
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise RuntimeError(f"Failed to read {path}")
    if im.ndim == 3:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if im.dtype == np.uint8:
        return im.astype(np.float32) / 255.0
    if im.dtype == np.uint16:
        return im.astype(np.float32) / 65535.0
    return im.astype(np.float32)


def list_images(path):
    """One image file -> [that file]; a folder -> all images in it, sorted by the
    trailing number in the filename (falls back to name order)."""
    if os.path.isfile(path):
        if not path.lower().endswith(VALID_EXTS):
            raise SystemExit(f"Not a supported image file: {path}")
        return [path]
    if not os.path.isdir(path):
        raise SystemExit(f"Path not found: {path}")
    files = [os.path.join(path, f) for f in os.listdir(path)
             if f.lower().endswith(VALID_EXTS)]
    if not files:
        raise SystemExit(f"No images ({'/'.join(VALID_EXTS)}) in {path}")

    return sorted(files, key=lambda f: ((0, frame_number(f)) if frame_number(f) is not None
                                        else (1, 0), os.path.basename(f)))


def frame_number(path):
    """The number at the end of the filename (before the extension), or None.
    Only the TRAILING number counts: 'mg41078_0056_frame_001352.tiff' is frame 1352,
    not 410780056001352."""
    m = re.search(r"(\d+)(?=\.[^.]+$)", os.path.basename(path))
    return int(m.group(1)) if m else None


def save_atomic(path, writer):
    """Write via a temp file + os.replace, so an interrupted run can never leave a
    truncated file behind that a later run would silently accept as complete
    (cache entries and output PNGs are both validated by existence alone, and a
    live consumer may be reading the output folder while the run proceeds).

    The temp name keeps the real name and appends '.<pid>.part', so it does NOT
    match *.png / corr_*.npy: a consumer globbing the output folder can never pick
    up a half-written file. Writers therefore get a path with no usable extension
    and must not rely on one (see _write_png / _write_npy)."""
    tmp = f"{path}.{os.getpid()}.part"
    try:
        writer(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass          # never mask the original error with a cleanup failure
        raise


def _write_npy(path, arr):
    """np.save to an explicit file object: it must not append '.npy' to the name."""
    with open(path, "wb") as fh:
        np.save(fh, arr)


def _write_png(path, img8):
    """Encode first, then write raw bytes: the file name carries no extension."""
    ok, buf = cv2.imencode(".png", img8)
    if not ok:
        raise IOError("PNG encoding failed")
    with open(path, "wb") as fh:
        fh.write(buf.tobytes())


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)


def ref_fingerprint(ref_path):
    """Identify the reference set by name+size+mtime of each file: enough to catch a
    different folder, a different frame range, or edited files, without reading them."""
    return [(os.path.basename(f), os.path.getsize(f), int(os.path.getmtime(f)))
            for f in list_images(ref_path)]


def _describe_recipe(r):
    names = [f[0] for f in r.get("ref", [])]
    span = (f"{names[0]} .. {names[-1]}" if len(names) > 1
            else (names[0] if names else "-"))
    rest = ", ".join(f"{k} {v}" for k, v in sorted(r.items()) if k != "ref")
    return f"ref {len(names)} file(s) [{span}], {rest}"


def check_recipe(directory, recipe, guard, holds):
    """Existing files in `directory` were made with the settings in its recipe.json.
    When `guard` says such files are present, refuse to add files made with different
    settings. Used for the `--cache` directory only, so nothing is ever written next
    to the output PNGs."""
    path = os.path.join(directory, "recipe.json")
    if guard and os.path.exists(path):
        try:
            old = json.load(open(path, encoding="utf-8"))
        except (ValueError, OSError) as e:
            raise SystemExit(f"{path} is unreadable ({e}). Delete it (and the files it "
                             f"describes) or use a different --output.")
        if old != json.loads(json.dumps(recipe)):   # normalize tuples -> lists
            raise SystemExit(
                f"{directory} already holds {holds} made with different settings.\n"
                f"  existing : {_describe_recipe(old)}\n"
                f"  now      : {_describe_recipe(recipe)}\n"
                "Mixing them would silently produce an inconsistent result. Use a "
                "different --output, or remove the existing files first.")
    save_atomic(path, lambda p: _write_json(p, recipe))


def build_flat(ref_path, cache_dir=None):
    """Mean of all reference images (cached as flat.npy when a cache dir is given)."""
    flat_cache = os.path.join(cache_dir, "flat.npy") if cache_dir else None
    if flat_cache and os.path.exists(flat_cache):
        flat = np.load(flat_cache)
        print(f"[flat]   cached  shape {flat.shape}")
        return flat
    files = list_images(ref_path)
    print(f"[flat]   building from {len(files)} reference image(s)...")
    t0 = time.perf_counter()
    acc = None
    for f in files:
        im = read_norm(f)
        acc = im if acc is None else acc + im
    flat = acc / len(files)
    if flat_cache:
        save_atomic(flat_cache, lambda p: _write_npy(p, flat))
    print(f"[flat]   mean of {len(files)} reference image(s) "
          f"({time.perf_counter() - t0:.1f}s)  shape {flat.shape}")
    return flat


# ------------------------------------------------- registration (per tile, TPS)

def _u8(z):
    lo, hi = np.percentile(z, [1, 99])
    return (np.clip((z - lo) / (hi - lo + 1e-9), 0, 1) * 255).astype(np.uint8)


def _subpix(res):
    """Sub-pixel peak of a template-matching response map."""
    _, mx, _, ml = cv2.minMaxLoc(res)
    px, py = ml
    h, w = res.shape
    dx = dy = 0.0
    if 0 < px < w - 1:
        d = res[py, px - 1] - 2 * res[py, px] + res[py, px + 1]
        if abs(d) > 1e-9:
            dx = 0.5 * (res[py, px - 1] - res[py, px + 1]) / d
    if 0 < py < h - 1:
        d = res[py - 1, px] - 2 * res[py, px] + res[py + 1, px]
        if abs(d) > 1e-9:
            dy = 0.5 * (res[py - 1, px] - res[py + 1, px]) / d
    return px + dx, py + dy, mx


def _ransac_inliers(P, Q, thr=2.5):
    """Iteratively drop landmark matches inconsistent with a global affine fit."""
    inl = np.ones(len(P), bool)
    for _ in range(4):
        A, *_ = np.linalg.lstsq(np.c_[P[inl], np.ones(inl.sum())], Q[inl], rcond=None)
        r = np.linalg.norm(Q - np.c_[P, np.ones(len(P))] @ A, axis=1)
        mad = np.median(r[inl]) + 1e-6
        nw = r < max(thr * mad, 1.0)
        if nw.sum() < 6 or np.array_equal(nw, inl):
            return nw
        inl = nw
    return inl


def correct_box(flat, frame, box):
    """Register one box (iterative dense-landmark TPS): warp the FLAT to match the
    FRAME and return (xm, ym, xM, yM, corrected_patch = frame / warped_flat).

    On a featureless tile (too few landmarks match) this falls back to a single
    ECC-estimated translation instead - in the OPPOSITE direction (the FRAME is
    warped to match the FLAT, then divided by the untouched flat), because ECC
    estimates a frame->flat translation directly and re-inverting it is unnecessary
    for a fallback whose only job is a coarse, low-confidence nudge."""
    FH, FW = flat.shape
    x, y, w, h = box
    xm, ym = max(0, x - RMARGIN), max(0, y - RMARGIN)
    xM, yM = min(FW, x + w + RMARGIN), min(FH, y + h + RMARGIN)
    fM, iM = flat[ym:yM, xm:xM], frame[ym:yM, xm:xM]
    fU, iU = _u8(fM), _u8(iM)
    hh, ww = fM.shape
    b = PATCH // 2 + SEARCH + 2
    # coarsen the landmark grid on large boxes so the O(N^3) TPS solve stays bounded
    gstep = max(GRID, int(round((max(1.0, (ww * hh) / TARGET_GRID_PTS)) ** 0.5)))
    grid = [(int(px), int(py))
            for py in np.arange(b, hh - b, gstep) for px in np.arange(b, ww - b, gstep)]
    co = cv2.goodFeaturesToTrack(fU, NCORNER, QUAL, MINDIST, blockSize=BLK)
    co = [(int(px), int(py)) for (px, py) in (co.reshape(-1, 2) if co is not None else [])
          if b < px < ww - b and b < py < hh - b]
    LM = sorted(set(grid + co))
    mx, my = np.meshgrid(np.arange(ww, dtype=np.float32), np.arange(hh, dtype=np.float32))

    def warp(Dx, Dy):
        return cv2.remap(fM, (mx - Dx), (my - Dy), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    def track(wU):
        P, Q = [], []
        for (px, py) in LM:
            tm = wU[py - PATCH // 2:py + PATCH // 2 + 1,
                    px - PATCH // 2:px + PATCH // 2 + 1]
            if tm.std() < 4:
                continue
            sx0, sy0 = px - PATCH // 2 - SEARCH, py - PATCH // 2 - SEARCH
            sr = iU[sy0:py + PATCH // 2 + SEARCH + 1, sx0:px + PATCH // 2 + SEARCH + 1]
            if sr.shape[0] <= tm.shape[0] or sr.shape[1] <= tm.shape[1]:
                continue
            fx, fy, nc = _subpix(cv2.matchTemplate(sr, tm, cv2.TM_CCOEFF_NORMED))
            if nc < NCC:
                continue
            P.append([px, py])
            Q.append([sx0 + fx + PATCH // 2, sy0 + fy + PATCH // 2])
        P, Q = np.array(P, float), np.array(Q, float)
        if len(P) < 8:
            return None, None
        inl = _ransac_inliers(P, Q)
        return P[inl], (Q - P)[inl]

    Dx = np.zeros((hh, ww), np.float32)
    Dy = np.zeros((hh, ww), np.float32)
    nin = -1
    for _ in range(ITERS):
        P, V = track(_u8(warp(Dx, Dy)))
        if P is None:
            break
        nin = len(P)
        gx, gy = np.meshgrid(np.arange(0, ww, TPS_GS), np.arange(0, hh, TPS_GS))
        g = np.c_[gx.ravel(), gy.ravel()]
        rbf = RBFInterpolator(P, V, kernel="thin_plate_spline", smoothing=TPS_SM)
        Dg = rbf(g).reshape(gx.shape[0], gx.shape[1], 2)
        Dx += cv2.resize(Dg[..., 0].astype(np.float32), (ww, hh),
                         interpolation=cv2.INTER_CUBIC)
        Dy += cv2.resize(Dg[..., 1].astype(np.float32), (ww, hh),
                         interpolation=cv2.INTER_CUBIC)
    if nin < 0:  # featureless tile -> uniform ECC translation fallback
        wp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 300, 1e-7)
        try:
            _, wp = cv2.findTransformECC(fU.astype(np.float32) / 255,
                                         iU.astype(np.float32) / 255,
                                         wp, cv2.MOTION_TRANSLATION, crit, None, 5)
            al = cv2.warpAffine(iM, wp, (ww, hh),
                                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                                borderMode=cv2.BORDER_REFLECT)
        except cv2.error:
            al = iM
        return xm, ym, xM, yM, al / (fM + EPS)
    return xm, ym, xM, yM, iM / (warp(Dx, Dy) + EPS)


# --------------------------------------------------- whole-frame tiled register

def tile_starts(total, tile, step):
    ss = list(range(0, max(total - tile, 0) + 1, step))
    if ss[-1] != total - tile:
        ss.append(total - tile)
    return ss


def check_tiling(tile, tstep, shape):
    """Tiles must cover the whole frame: a stride larger than the tile leaves gaps,
    and a tile larger than the image makes tile_starts emit a negative offset. Either
    one silently yields uncovered pixels (0/0 = NaN, which a percentile stretch then
    spreads over the entire frame as black), so reject both up front."""
    H, W = shape
    if tstep < 1 or tstep > tile:
        raise SystemExit(f"tile step {tstep} must be between 1 and the tile size "
                         f"{tile}; a larger step would leave gaps between tiles.")
    if tile > min(H, W):
        raise SystemExit(f"tile {tile} is larger than the image ({W}x{H}); "
                         f"use a tile of at most {min(H, W)}.")


def register_frame(flat, frame, tile=TILE, tstep=TSTEP):
    """Full-frame tiled registration: warp the flat onto the frame tile by tile
    (correct_box) and Hann-blend the per-tile corrections `frame / warped_flat`
    into one float image. This is THE core operation of the pipeline."""
    H, W = flat.shape
    if frame.shape != flat.shape:
        raise RuntimeError(f"frame shape {frame.shape} != flat {flat.shape}")
    check_tiling(tile, tstep, flat.shape)
    tbs = [(tx, ty, tile, tile)
           for ty in tile_starts(H, tile, tstep) for tx in tile_starts(W, tile, tstep)]
    wh = np.hanning(tile).astype(np.float64) + 1e-3
    w2 = np.outer(wh, wh)
    num = np.zeros((H, W))
    den = np.zeros((H, W))
    for tb in tbs:
        xm, ym, xM, yM, loc = correct_box(flat, frame, tb)
        tx, ty, tw, th = tb
        patch = loc[(ty - ym):(ty - ym + th), (tx - xm):(tx - xm + tw)]
        num[ty:ty + th, tx:tx + tw] += w2 * patch
        den[ty:ty + th, tx:tx + tw] += w2
    if not np.all(den > 0):                 # belt and braces: never emit NaN
        raise RuntimeError(f"tiling left {int((den <= 0).sum())} pixel(s) uncovered "
                           f"(tile {tile}, step {tstep}, image {W}x{H})")
    return (num / den).astype(np.float32)


def _finish_write(img, sigma, outp):
    """Per-frame 1%/99% stretch -> Gaussian(sigma) -> 8-bit PNG."""
    p1, p99 = np.percentile(img, [1, 99])
    s = np.clip((img - p1) / (p99 - p1 + 1e-9), 0, 1)
    s = cv2.GaussianBlur(s.astype(np.float32), (0, 0), sigmaX=sigma)
    u8 = (s * 255).astype(np.uint8)
    save_atomic(outp, lambda p: _write_png(p, u8))


_W = {}


def _init(flat, cache, emit_png, outdir, tile, tstep, sigma, send_corr):
    """cache: cache dir, or None to keep nothing on disk (default).
    emit_png: the worker writes the PNG itself (no temporal averaging).
    send_corr: return the registered array to the parent (needed for the trailing
    average when there is no cache to read it back from)."""
    cv2.setNumThreads(1)
    _W.update(flat=flat, cache=cache, emit=emit_png, out=outdir, sigma=sigma,
              tile=tile, tstep=tstep, send=send_corr)


def _proc(item):
    """Register one frame against the flat. Returns (stem, seconds, array): seconds
    is None if the frame came from an existing cache entry, and array is None unless
    the parent needs it (see _init)."""
    stem, path = item
    t0 = time.perf_counter()
    cp = os.path.join(_W["cache"], f"corr_{stem}.npy") if _W["cache"] else None
    if cp and os.path.exists(cp):
        corr, dt = None, None
    else:
        frame = read_norm(path)
        if frame.shape != _W["flat"].shape:
            raise RuntimeError(f"{os.path.basename(path)}: image is "
                               f"{frame.shape[1]}x{frame.shape[0]}, but the flat is "
                               f"{_W['flat'].shape[1]}x{_W['flat'].shape[0]} - every "
                               "input and reference image must have one resolution")
        corr = register_frame(_W["flat"], frame, _W["tile"], _W["tstep"])
        if cp:
            save_atomic(cp, lambda p: _write_npy(p, corr))
        dt = time.perf_counter() - t0
    if _W["emit"]:
        _finish_write(corr if corr is not None else np.load(cp), _W["sigma"],
                      os.path.join(_W["out"], f"{stem}.png"))
    return stem, dt, (corr if _W["send"] else None)


def _drain(result_iter, n, what):
    """Consume a pool.imap()/imap_unordered() iterator with a per-item timeout, so a
    worker killed by the OS fails the run loudly instead of hanging it forever."""
    for i in range(n):
        try:
            yield result_iter.next(timeout=FRAME_TIMEOUT_S)
        except mp.TimeoutError:
            raise SystemExit(
                f"No result after {FRAME_TIMEOUT_S}s waiting for {what} {i + 1}/{n}. "
                "A worker process was very likely killed by the OS (commonly the "
                "OOM killer) - multiprocessing does not report this on its own, so "
                "the run would otherwise hang forever. Lower --workers to reduce "
                "peak memory, then retry.")


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=REF_PATH,
                    help="reference (flat) image OR folder of reference images; their "
                         "mean is the flat. Use structure-free (e.g. pre-event) frames.")
    ap.add_argument("--input", default=INPUT_PATH,
                    help="image OR folder of images to correct.")
    ap.add_argument("--output", default=OUTPUT_DIR,
                    help="output folder for corrected PNGs.")
    ap.add_argument("--mean-win", type=int, default=3,
                    help="average each output over the last N registered frames "
                         "(trailing/causal window). Default 3; set 1 to disable. "
                         "A single-image input effectively uses 1.")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"parallel worker processes (default {DEFAULT_WORKERS}).")
    ap.add_argument("--tile", type=int, default=TILE, help="registration tile size.")
    ap.add_argument("--tile-step", type=int, default=TSTEP,
                    help="tile stride; overlap = tile - step, Hann-blended.")
    ap.add_argument("--sigma", type=float, default=SIGMA, help="final Gaussian sigma.")
    ap.add_argument("--cache", action="store_true",
                    help="keep every frame's registration result in "
                         "<output>/_corr_cache/ (~38 MB per frame at 4096x2304). "
                         "Lets an interrupted run resume, and lets --sigma / "
                         "--mean-win be changed without registering again. Off by "
                         "default: nothing but the output PNGs is written.")
    args = ap.parse_args()

    t_run0 = time.perf_counter()
    # validate the inputs BEFORE creating anything, so a mistyped path (or the
    # untouched placeholder constants) cannot leave an empty output tree behind
    for label, p in (("--ref", args.ref), ("--input", args.input)):
        if not os.path.exists(p):
            raise SystemExit(
                f"{label} not found: {p}\n"
                "Pass --ref/--input/--output on the command line, or edit the "
                "REF_PATH / INPUT_PATH / OUTPUT_DIR constants at the top of this "
                "script (they ship as 'path/to/...' placeholders).")
    files = list_images(args.input)
    items = [(os.path.splitext(os.path.basename(f))[0], f) for f in files]
    stems = [s for s, _ in items]
    if len(set(stems)) != len(stems):
        raise SystemExit("Duplicate basenames in --input; outputs would collide.")
    N = max(1, args.mean_win)

    os.makedirs(args.output, exist_ok=True)
    # Nothing but PNGs is written to the output folder. Every run rewrites every
    # frame, so the folder is always internally consistent and needs no guard.
    cache = os.path.join(args.output, "_corr_cache") if args.cache else None
    if cache:
        os.makedirs(cache, exist_ok=True)
        # Cache entries ARE reused, so they do need one: a flat.npy or a
        # registration built from a different --ref/tiling would silently produce
        # wrong images. recipe.json lives inside _corr_cache/, never beside the PNGs.
        check_recipe(cache,
                     {"ref": ref_fingerprint(args.ref), "tile": args.tile,
                      "tile_step": args.tile_step},
                     guard=any(f.startswith("corr_") for f in os.listdir(cache)),
                     holds="cached registrations")

    flat = build_flat(args.ref, cache)
    check_tiling(args.tile, args.tile_step, flat.shape)
    print(f"[input]  {len(items)} frame(s): {stems[0]} .. {stems[-1]}")
    print(f"[recipe] tile{args.tile} step{args.tile_step} grid{GRID}, "
          f"per-frame 1/99, sigma{args.sigma:g}, "
          f"{'trailing mean' + str(N) if N > 1 else 'no temporal mean'} | "
          f"workers {args.workers} | cache {'on' if cache else 'off'}")

    def cpath(stem):
        return os.path.join(cache, f"corr_{stem}.npy")

    if N == 1:
        # ---- streaming: register + write PNG per frame the moment it completes
        #      (each frame is self-contained). With --cache, frames already
        #      registered but not yet written are finished from the cache first.
        if cache:
            wrote = 0
            for stem, _ in items:
                outp = os.path.join(args.output, f"{stem}.png")
                if not os.path.exists(outp) and os.path.exists(cpath(stem)):
                    _finish_write(np.load(cpath(stem)), args.sigma, outp)
                    wrote += 1
            if wrote:
                print(f"[catchup] wrote {wrote} PNG(s) from existing cache")
        todo = [(s, p) for s, p in items
                if not os.path.exists(os.path.join(args.output, f"{s}.png"))]
        print(f"[stream] frames to process: {len(todo)}")
        times = []
        if todo:
            with mp.Pool(max(1, min(args.workers, len(todo))), initializer=_init,
                         initargs=(flat, cache, True, args.output, args.tile,
                                   args.tile_step, args.sigma, False)) as pool:
                it = pool.imap_unordered(_proc, todo)
                for k, (stem, dt, _) in enumerate(_drain(it, len(todo), "frame")):
                    if dt is not None:
                        times.append(dt)
                    print(f"  [stream {k + 1}/{len(todo)}] {stem} -> PNG"
                          + (f"  {dt:.1f}s" if dt is not None else "  (cached)"))
        if times:
            print(f"[timing] per-frame register->PNG single-core "
                  f"mean {np.mean(times):.1f}s median {np.median(times):.1f}s")
    else:
        # ---- mean-N: frames are registered by all workers in parallel, but their
        #      results are consumed IN ORDER (pool.imap) and pushed through a ring
        #      buffer, so each PNG is written as soon as its trailing window is
        #      complete - PNGs stream out during the run at full parallelism,
        #      exactly the ring-buffer behaviour a live deployment uses
        missing = [s for s in stems if not os.path.exists(
            os.path.join(args.output, f"{s}.png"))]
        if not missing:
            # everything is already there - do not even build a pool
            print(f"[finish] mean{N} wrote 0 PNG(s) (all {len(items)} already exist)")
            times = []
        else:
            if len(missing) < len(items) and not cache:
                # partial resume without --cache: frames whose PNG already exists
                # still have to be RE-registered, because their registration feeds
                # the trailing window of later, still-missing frames. This is not
                # wasted by mistake - it is the cost of resuming without --cache.
                print(f"[resume] {len(items) - len(missing)}/{len(items)} output(s) "
                      f"already exist, but --cache is off, so the trailing window "
                      f"still requires re-registering every frame from the start "
                      f"to rebuild it. Use --cache on the original run to make a "
                      f"resume only redo the {len(missing)} missing frame(s).")
            print(f"[reg]    registering {len(items)} frame(s) with "
                  f"{max(1, min(args.workers, len(items)))} worker(s)...")
            ring = collections.deque(maxlen=N)
            wrote, times = 0, []
            with mp.Pool(max(1, min(args.workers, len(items))), initializer=_init,
                         initargs=(flat, cache, False, None, args.tile, args.tile_step,
                                   args.sigma, not cache)) as pool:
                it = pool.imap(_proc, items)
                for j, (stem, dt, corr) in enumerate(_drain(it, len(items), "frame")):
                    ring.append(corr if corr is not None else np.load(cpath(stem)))
                    if dt is not None:
                        times.append(dt)
                    reg = f"reg {dt:.1f}s" if dt is not None else "cached"
                    outp = os.path.join(args.output, f"{stem}.png")
                    if os.path.exists(outp):
                        # PNG already there: the frame still has to pass through the
                        # ring buffer so the following windows are correct
                        print(f"  [{j + 1}/{len(items)}] {stem}  ({reg}, PNG exists)")
                        continue
                    _finish_write(np.mean(list(ring), axis=0).astype(np.float32),
                                  args.sigma, outp)
                    wrote += 1
                    print(f"  [{j + 1}/{len(items)}] {stem} -> PNG  ({reg})")
            print(f"[finish] mean{N} wrote {wrote} PNG(s)")
        if times:
            print(f"[timing] per-frame registration single-core "
                  f"mean {np.mean(times):.1f}s median {np.median(times):.1f}s")

    print(f"[total]  wall {time.perf_counter() - t_run0:.1f}s")
    print("[done]")


if __name__ == "__main__":
    main()
