"""live_inference_3head_2.5D.py

LIVE 3-head inference: watch a folder while preprocessed frames arrive, run the
single-backbone 3-head V5aModel on each frame as soon as it can be processed, and keep
three SMALL output files up to date after every frame:

    <out>/live_stats.csv      per-frame AREA RATIOS (total/alpha/beta/overlap, px + %)
                              + IMAGE-LEVEL CLASS PROBABILITIES for Liquid / Alpha /
                              Beta (clsmax_* = max over sliding-window tiles = presence,
                              clsmean_* = mean = extent). Append-only, one row per frame.
    <out>/alpha_counts.csv    TRACKED crystal count per frame (cumulative + active),
                              cumulative booked at the BIRTH frame. REWRITTEN after
                              every frame, because confirmation and the beta veto are
                              retroactive: a birth 4 frames ago is only counted once the
                              track confirms, and a track can be vetoed later.
    <out>/per_instance.csv    one row per (frame, track): frame_name, frame_idx,
                              track_id, x, y, area, detected, merged, state,
                              birth_frame, confirm_frame - the same schema
                              export_3head_masks_2.5D.py writes, so
                              annotate_alpha_count.py --suspect can read it. REWRITTEN
                              after every frame (states are retroactive too).

This is the LIVE counterpart of export_3head_masks_2.5D.py with the outputs cut down to
numbers only: no mask PNGs, no composite, no video (--save_masks re-enables binary mask
dumps for debugging / verification). The inference core - offline DINOv3 build, 2.5D
input, sliding-window at original resolution with Hann-blended overlapping tiles,
softmax[:,0] readout, alpha/beta intersected with total - is copied from that script
unchanged, and the tracker replays the exact same frozen rule set
(CONFIRM_HITS=4, grace 0, relink off, beta veto 0.05, birth-frame booking), so a
finished live run produces the SAME alpha_counts.csv / per_instance.csv the offline
export would.

2.5D NEIGHBORS AND THE ONE-FRAME LAG. Training input is (t-1, t, t+1), so frame t is
only processed once its SUCCESSOR exists on disk: the live output always lags the
newest frame by one. When the run ENDS (idle timeout / --once / Ctrl+C) the trailing
frame is processed with the center-frame fallback for t+1 - exactly what the offline
script does at a sequence's last frame, so the lag never loses a frame. A
single-frame (non-2.5D) checkpoint needs no successor and has no lag.

LIVE TRACKING = the offline two-pass algorithm, made incremental. Pass 1 (per-frame
matching, merge/split handling, confirmation state machine, lifetime beta
accumulation) runs frame by frame as frames arrive - the code is the same, only fed
incrementally. Pass 2 (beta veto + curve rebuild) is CHEAP, so it simply reruns after
every frame over the tracks seen so far; the CSVs are rewritten with the result. Two
consequences worth knowing:
  * the tail of the live cumulative curve is PROVISIONAL: crystals born within the
    last CONFIRM_HITS-1 frames have not had time to confirm yet, and a track's
    lifetime beta fraction can still cross the veto threshold later;
  * the offline rule "a crystal born too close to the end confirms with however many
    frames remain" needs the sequence length, which live does not know - it is applied
    once at FINALIZE. Under the frozen rules (grace 0) this relaxation can only ever
    fire on the very last frame, so the finalized numbers match the offline algorithm
    exactly; the equivalence is covered by a regression test against
    export_3head_masks_2.5D.py --counts_only on the real 0056 masks.

ORDERING IS ENFORCED, NOT ASSUMED. Frames are ordered by the trailing number in the
filename (frame_001150.png -> 1150). Every image file must carry one, duplicates are
an error, and a frame that appears on disk BEHIND the already-processed front (the
preprocessing wrote frames out of order) raises immediately - tracking against wrong
neighbors would silently corrupt every number downstream. Rerun with --once after the
acquisition instead if your producer cannot guarantee order. A file is only picked up
once its mtime is at least --settle_sec old (0 = off; raise it if the producer
writes frames straight to their final name rather than renaming them into place).

RESTART = REPROCESS. All tracker state lives in memory; on start the script processes
every frame already in the folder (output files are rewritten from scratch), then
keeps polling. To resume after a crash mid-acquisition just start it again.

Usage:
    # live: poll until no new frame for 10 min, then finalize
    python live_inference_3head_2.5D.py --input D:/live/0056_aligned \\
        --output D:/live/0056_live --idle_timeout 600
    # replay a finished folder (also the equivalence-test mode)
    python live_inference_3head_2.5D.py --input ... --output ... --once
Ctrl+C finalizes with the frames processed so far and exits cleanly.
"""

import argparse
import csv
import os
import re
import signal
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Locate the 3-HEAD model code (alpha_beta_diff_script/model.py). EDIT MODEL_CODE_DIR
#     if that code is NOT a subfolder next to this script. "" = auto. There is also an OLD
#     2-head v5a/model.py — if it gets imported instead you get "Unexpected key(s) ...
#     decoder.head_alpha/head_beta" at load; this block prints which model.py it imported. ---
MODEL_CODE_DIR = ""
_cands = ([MODEL_CODE_DIR] if MODEL_CODE_DIR else []) + \
         [os.path.join(_SCRIPT_DIR, "alpha_beta_diff_script"), _SCRIPT_DIR]
_model_dir = next((c for c in _cands if os.path.isfile(os.path.join(c, "model.py"))), None)
if _model_dir is None:
    raise RuntimeError("Cannot find model.py. Looked in: " + " | ".join(_cands) +
                       ". Put alpha_beta_diff_script/ next to this script or set MODEL_CODE_DIR.")
sys.path.insert(0, _model_dir)
import model as _model_mod            # noqa: E402
from model import V5aModel            # noqa: E402
print(f"[import] V5aModel <- {_model_mod.__file__}")

# ============================================================================
# Build the DINOv3 backbone WITHOUT contacting HuggingFace
# ============================================================================
# Same offline rebind as export_3head_masks_2.5D.py: the gated HF repo is never
# needed because the fine-tuned checkpoint carries all 211 backbone tensors and
# load_state_dict(strict=True) overwrites every one of them. Verified bit-identical
# to the HF path there (295/295 tensors, all logits max|diff| = 0).
DINOV3_VITB16_CONFIG = {
    "architectures": ["DINOv3ViTModel"], "attention_dropout": 0.0,
    "drop_path_rate": 0.0, "hidden_act": "gelu", "hidden_size": 768,
    "image_size": 224, "initializer_range": 0.02, "intermediate_size": 3072,
    "key_bias": False, "layer_norm_eps": 1e-05, "layerscale_value": 1.0,
    "mlp_bias": True, "model_type": "dinov3_vit", "num_attention_heads": 12,
    "num_channels": 3, "num_hidden_layers": 12, "num_register_tokens": 4,
    "patch_size": 16, "pos_embed_jitter": None, "pos_embed_rescale": 2.0,
    "pos_embed_shift": None, "proj_bias": True, "query_bias": True,
    "rope_theta": 100.0, "torch_dtype": "float32", "use_gated_mlp": False,
    "value_bias": True,
}
SUPPORTED_BACKBONE = "facebook/dinov3-vitb16-pretrain-lvd1689m"


def _local_backbone_config(backbone_name, **overrides):
    """DINOv3 config from the dict above instead of the HuggingFace Hub."""
    if backbone_name != SUPPORTED_BACKBONE:
        raise RuntimeError(
            f"Only the config for {SUPPORTED_BACKBONE} is bundled in this script, but "
            f"the checkpoint asks for {backbone_name!r}.\nFix: copy that model's "
            "config.json from its HuggingFace page into DINOV3_VITB16_CONFIG above and "
            "update SUPPORTED_BACKBONE (architecture only - no weights involved).")
    from transformers import AutoConfig
    cfg = dict(DINOV3_VITB16_CONFIG)
    cfg.update(overrides)
    return AutoConfig.for_model(cfg.pop("model_type"), **cfg)


class _OfflineAutoConfig:
    @staticmethod
    def from_pretrained(name, **kw):
        return _local_backbone_config(name, **kw)


class _OfflineAutoModel:
    @staticmethod
    def from_pretrained(name, **kw):
        from transformers import AutoModel
        # from_config = skeleton with random init, never touches the Hub
        return AutoModel.from_config(_local_backbone_config(name), **kw)


_model_mod.AutoConfig = _OfflineAutoConfig
_model_mod.AutoModel = _OfflineAutoModel
print("[import] DINOv3 built from the inlined config (no HuggingFace account needed)")


# ============================================================================
# EDIT THESE (hardcoded defaults — CLI args override if provided)
# ============================================================================
# Folder the preprocessing (live_tiled_align.py / flatfield pipeline) writes PNGs into.
INPUT_PATH = ""

# ONE output folder for the three CSVs (and mask dumps if --save_masks).
OUTPUT_DIR = ""

# Step-3 model, inference-only slim (slim_ckpt_inference.py, ep6 best_metric 0.6854).
# Local copy for the live machine (loads in ~1 s; the NAS original at
# \\psg-ds2422plus\home\Attentionmap_Temporal_Classification\Checkpoints\3head\
# is bit-identical but takes minutes over SMB).
CKPT = r"D:\v5a_ckpts\2.5D_3head_best_v4(new_annotation_step3)_no_optimizer.pt"

DEVICE = "cuda"

# image-level cls token order (v4_cct CLASS_NAMES, token indices 0..5) and the subset
# reported in live_stats.csv (the user-facing three).
CLS_NAMES = ["Col", "Eq", "Alpha", "Liquid", "Beta", "HotTear"]
LIVE_CLS = ["Liquid", "Alpha", "Beta"]

# Per-head crystal/foreground thresholds (softmax class-0 prob >= thr).
DPT_THRESHOLD_TOTAL = 0.5
DPT_THRESHOLD_ALPHA = 0.5
DPT_THRESHOLD_BETA = 0.5
INTERSECT_WITH_TOTAL = True       # alpha/beta gated by the total head (train.py readout)

# Sliding-window inference (identical to export_3head_masks_2.5D.py) - EXCEPT the
# default batch size. On the 12 GB 5070 Ti laptop this script targets, batch 4 does
# not fit: the Windows driver silently spills to shared system memory instead of
# raising OOM, and a frame takes ~44 s at ~54 W. Measured 2026-08-04 on real frames:
#     gpu_batch_size 1 -> 4.6 s/frame (peak alloc  5.0 GB, no spill)
#     gpu_batch_size 2 -> 24.8 s/frame (peak alloc  9.4 GB, spilled)
#     gpu_batch_size 4 -> ~44  s/frame (pegged at 12 GB, deep spill)
# On a >=24 GB card pass --gpu_batch_size 4. Results are numerically batch-size
# sensitive only at the last bit (kernel reduction order), never materially.
GPU_BATCH_SIZE = 1
TILE_OVERLAP = 512
TILE_BLEND_HANN = True

# ---- live loop timing ----
POLL_SEC = 2.0        # folder re-scan interval while waiting for frames
SETTLE_SEC = 0.0      # a file is only picked up once its mtime is this old. OFF by
                      # default: a producer that writes atomically (temp name, then
                      # rename - what the flat-field pipeline does) never exposes a
                      # partial file, so the wait would be pure added latency. Set
                      # it to ~2 if frames are written straight to their final name,
                      # where a truncated read would abort the run.
IDLE_TIMEOUT = 0.0    # finalize after this many seconds without a NEW frame; 0 = wait
                      # forever (Ctrl+C to finalize)

SAVE_MASKS = False    # dump alpha_only/beta_only/total binary PNGs (debug/verification)

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ----------------- unicode-safe PNG writer -----------------

def imwrite_unicode(path: str, img: np.ndarray):
    """cv2.imwrite that tolerates non-ASCII paths (Windows). Raises on failure (no silent
    skip)."""
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise IOError(f"cv2.imencode failed for {path}")
    buf.tofile(path)


# ----------------- input building -----------------
# Same construction as export_3head_masks_2.5D.py's build_25d_pil/build_full_input,
# but with the neighbors passed in EXPLICITLY: the live loop owns the (growing) frame
# list, so a folder-listing cache keyed at first access - fine offline where the
# folder is complete - would go stale here as new frames arrive.

def _load_gray_np(path) -> np.ndarray:
    return np.array(Image.open(str(path)).convert("L"), dtype=np.uint8)


def build_input_tensor(cur_path, prev_path, next_path, use_25d, norm_mean, norm_std):
    """(3, H, W) normalized tensor at ORIGINAL resolution. 2.5D channels are
    (t-1, t, t+1) grayscale; a missing neighbor (sequence boundary) falls back to the
    center frame, exactly like the offline exporter. Returns (tensor, n_neighbors)."""
    g_t = _load_gray_np(cur_path)
    if use_25d:
        H, W = g_t.shape

        def load_or_center(npath):
            if npath is not None and Path(npath).is_file():
                g = _load_gray_np(npath)
                if g.shape != (H, W):
                    g = np.array(Image.fromarray(g).resize((W, H), Image.BILINEAR))
                return g, True
            return g_t, False

        g_tm1, fm = load_or_center(prev_path)
        g_tp1, fp = load_or_center(next_path)
        stack = np.stack([g_tm1, g_t, g_tp1], axis=-1)
        n_neigh = int(fm) + int(fp)
    else:
        stack = np.stack([g_t, g_t, g_t], axis=-1)
        n_neigh = -1
    arr = stack.astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    mean = torch.tensor(norm_mean).view(3, 1, 1)
    std = torch.tensor(norm_std).view(3, 1, 1)
    return (tensor - mean) / std, n_neigh


# ----------------- Model loading (same as export_3head_masks_2.5D.py) -----------------

def load_model(ckpt_path: str, device: torch.device):
    """Load a 3-head V5aModel from a checkpoint (best.pt / last.pt / slim)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "config" not in ckpt or "model" not in ckpt:
        raise RuntimeError(f"Checkpoint {ckpt_path} missing 'config'/'model' key.")
    cfg = ckpt["config"]
    model = V5aModel(cfg).to(device)
    if not getattr(model, "three_head", False):
        raise RuntimeError(
            f"Constructed model is NOT 3-head (model.three_head=False). V5aModel was "
            f"imported from {_model_mod.__file__} — almost certainly the OLD 2-head "
            f"model.py, not alpha_beta_diff_script/model.py. Put alpha_beta_diff_script/ "
            f"next to the script or set MODEL_CODE_DIR at the top.")
    sd = {k.replace("module.", "", 1): v for k, v in ckpt["model"].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    ep = ckpt.get("epoch", "?")
    best = ckpt.get("best_metric", "?")
    print(f"[load] {os.path.basename(ckpt_path)}  epoch={ep} best_metric={best} (3-head)")
    return model, cfg


def _resolve_cfg(cfg, input_mode):
    d = cfg["data"]
    target_size = int(d["target_size"])
    norm_mean = list(d.get("normalize_mean", [0.5, 0.5, 0.5]))
    norm_std = list(d.get("normalize_std", [0.5, 0.5, 0.5]))
    cfg_use_25d = bool(d.get("use_25d", False))
    use_25d = cfg_use_25d if input_mode == "auto" else (input_mode == "25d")
    return target_size, norm_mean, norm_std, use_25d


def _tile_starts(total, tile, stride):
    """Start offsets covering [0, total) with `tile`-sized windows at `stride`, always
    including the flush-to-edge start so the last strip is never dropped."""
    if total <= tile:
        return [0]
    starts = list(range(0, total - tile + 1, stride))
    if starts[-1] != total - tile:
        starts.append(total - tile)
    return starts


def _hann2d(tile):
    """2-D Hann window (outer product of 1-D Hann), floored so a tile never contributes
    exactly zero weight anywhere (guards against a pixel covered only by tile edges)."""
    w1 = np.hanning(tile + 2)[1:-1].astype(np.float32)   # drop the two zero endpoints
    w2 = np.outer(w1, w1).astype(np.float32)
    return np.maximum(w2, 1e-3)


@torch.no_grad()
def sliding_window_three_probs(model, tensor, device, tile_size, gpu_batch_size,
                               tile_overlap=TILE_OVERLAP, blend_hann=TILE_BLEND_HANN):
    """Sliding-window inference at ORIGINAL resolution (no downscaling) returning THREE
    foreground-probability maps from ONE forward per tile: (total, alpha, beta), each
    (H, W) float, plus the per-tile image-level class probabilities (n_tiles, C).
    Raises if the checkpoint is not 3-head (alpha/beta logits None — NO silent
    fallback). Copied verbatim from export_3head_masks_2.5D.py."""
    gpu_batch_size = max(1, int(gpu_batch_size))
    _, h, w = tensor.shape
    if h < tile_size or w < tile_size:
        pad_h = max(0, tile_size - h)
        pad_w = max(0, tile_size - w)
        pad_top = pad_h // 2
        pad_left = pad_w // 2
        padded = F.pad(tensor.unsqueeze(0),
                       (pad_left, pad_w - pad_left, pad_top, pad_h - pad_top),
                       mode="replicate").squeeze(0)
    else:
        pad_top = pad_left = 0
        padded = tensor
    _, ph, pw = padded.shape

    stride = max(1, tile_size - max(0, int(tile_overlap)))
    positions = [(t, l)
                 for t in _tile_starts(ph, tile_size, stride)
                 for l in _tile_starts(pw, tile_size, stride)]
    wnd = _hann2d(tile_size) if (blend_hann and len(positions) > 1) else np.ones(
        (tile_size, tile_size), dtype=np.float32)

    psum_t = np.zeros((ph, pw), dtype=np.float32)
    psum_a = np.zeros((ph, pw), dtype=np.float32)
    psum_b = np.zeros((ph, pw), dtype=np.float32)
    cnt = np.zeros((ph, pw), dtype=np.float32)
    cls_tiles = []   # per-tile image-level class probabilities, (n_tiles, C) after concat
    for cs in range(0, len(positions), gpu_batch_size):
        batch_pos = positions[cs:cs + gpu_batch_size]
        chunk = torch.stack([padded[:, at:at + tile_size, al:al + tile_size]
                             for (at, al) in batch_pos], dim=0).to(device)
        with torch.amp.autocast(device.type, dtype=torch.bfloat16,
                                enabled=(device.type == "cuda")):
            out = model(chunk, None)
        sa = out.get("seg_logits_alpha", None)
        sb = out.get("seg_logits_beta", None)
        if sa is None or sb is None:
            raise RuntimeError(
                "Checkpoint is NOT a 3-head model (seg_logits_alpha/beta is None). "
                "live_inference_3head_2.5D.py requires a three_head=True checkpoint.")
        pt = F.softmax(out["seg_logits"].float(), dim=1)[:, 0].cpu().numpy()
        pa = F.softmax(sa.float(), dim=1)[:, 0].cpu().numpy()
        pb = F.softmax(sb.float(), dim=1)[:, 0].cpu().numpy()
        cls_tiles.append(torch.sigmoid(out["cls_logits"].float()).cpu().numpy())
        for k, (at, al) in enumerate(batch_pos):
            psum_t[at:at + tile_size, al:al + tile_size] += pt[k] * wnd
            psum_a[at:at + tile_size, al:al + tile_size] += pa[k] * wnd
            psum_b[at:at + tile_size, al:al + tile_size] += pb[k] * wnd
            cnt[at:at + tile_size, al:al + tile_size] += wnd
    if not np.all(cnt > 0):
        raise RuntimeError("Sliding-window blend left uncovered pixels (cnt==0); "
                           "check tile_size / tile_overlap.")
    denom = cnt
    sl = (slice(pad_top, pad_top + h), slice(pad_left, pad_left + w))
    return ((psum_t / denom)[sl], (psum_a / denom)[sl], (psum_b / denom)[sl],
            np.concatenate(cls_tiles, axis=0))


def derive_masks(total_prob, alpha_prob, beta_prob, thr_total, thr_alpha, thr_beta,
                 intersect_total):
    """Return a dict of boolean masks. The heads are independent, so alpha and beta MAY
    overlap; `overlap` is that shared region (reported in the stats)."""
    total = total_prob >= thr_total
    alpha_raw = alpha_prob >= thr_alpha
    beta_raw = beta_prob >= thr_beta
    if intersect_total:
        alpha = alpha_raw & total
        beta = beta_raw & total
    else:
        alpha = alpha_raw
        beta = beta_raw
    return {"total": total, "alpha": alpha, "beta": beta, "overlap": alpha & beta}


# ============================================================================
# Tracked Alpha counting — frozen rule set, incremental
# ============================================================================
# Constants and the per-frame matching code are copied from
# export_3head_masks_2.5D.py (Rule v3, frozen 2026-08-03: confirm 4, grace 0, relink
# off, beta veto 0.05, cumulative booked at the BIRTH frame). Keep them in sync.
CNT_MASK_THR = 200      # mask pixel > this = Alpha (kept for --save_masks readback parity)
CNT_CLOSE_KERNEL = 15   # morphological closing: merge fragmented dendrite arms (0 = off)
CNT_MIN_AREA_PX = 200   # ignore components smaller than this (ORIGINAL resolution)
CNT_IOU_THRESHOLD = 0.10        # 1:1 match accepted above this IoU
CNT_MERGE_IOU_THRESHOLD = 0.05  # overlap counted for merge/split detection above this
CNT_CONFIRM_HITS = 4    # a birth is only COUNTED after this many consecutive detections
CNT_TENTATIVE_GRACE = 0
CNT_RELINK_TENTATIVE = False
CNT_RELINK_WINDOW = 5
CNT_BETA_VETO_FRAC = 0.05   # drop confirmed tracks whose LIFETIME beta overlap exceeds
                            # this fraction; NEGATIVE = veto off


def _cnt_components(alpha_mask):
    """Binary mask -> closing -> connected components -> min-area filter. Uses
    connectedComponentsWithStats so per-component work is O(bbox), not O(image)."""
    mask = (alpha_mask != 0).astype(np.uint8)
    if CNT_CLOSE_KERNEL > 0:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (CNT_CLOSE_KERNEL, CNT_CLOSE_KERNEL))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker)
    n_cc, cc_labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    comps = []
    for cid in range(1, n_cc):
        area = int(stats[cid, cv2.CC_STAT_AREA])
        if area < CNT_MIN_AREA_PX:
            continue
        x, y = int(stats[cid, cv2.CC_STAT_LEFT]), int(stats[cid, cv2.CC_STAT_TOP])
        w, h = int(stats[cid, cv2.CC_STAT_WIDTH]), int(stats[cid, cv2.CC_STAT_HEIGHT])
        comps.append({"local_mask": (cc_labels[y:y + h, x:x + w] == cid).astype(np.uint8),
                      "bbox": (x, y, w, h), "area": area})
    return comps


def _cnt_iou(ca, cb):
    ba, bb = ca["bbox"], cb["bbox"]
    if (ba[0] + ba[2] <= bb[0] or bb[0] + bb[2] <= ba[0]
            or ba[1] + ba[3] <= bb[1] or bb[1] + bb[3] <= ba[1]):
        return 0.0
    ix1, iy1 = max(ba[0], bb[0]), max(ba[1], bb[1])
    ix2 = min(ba[0] + ba[2], bb[0] + bb[2])
    iy2 = min(ba[1] + ba[3], bb[1] + bb[3])
    ra = ca["local_mask"][iy1 - ba[1]:iy2 - ba[1], ix1 - ba[0]:ix2 - ba[0]]
    rb = cb["local_mask"][iy1 - bb[1]:iy2 - bb[1], ix1 - bb[0]:ix2 - bb[0]]
    inter = int(np.logical_and(ra, rb).sum())
    if inter == 0:
        return 0.0
    union = ca["area"] + cb["area"] - inter
    return inter / union if union > 0 else 0.0


def _cnt_match(prev, curr):
    """-> matches[(pi,cj)], merges[([pi..],cj)], splits[(pi,[cj..])], births[cj]."""
    from scipy.optimize import linear_sum_assignment

    np_, nc = len(prev), len(curr)
    if np_ == 0:
        return [], [], [], list(range(nc))
    if nc == 0:
        return [], [], [], []
    iou = np.zeros((np_, nc))
    for i in range(np_):
        for j in range(nc):
            iou[i, j] = _cnt_iou(prev[i], curr[j])

    curr_to_prev = defaultdict(list)
    for i in range(np_):
        for j in range(nc):
            if iou[i, j] > CNT_MERGE_IOU_THRESHOLD:
                curr_to_prev[j].append(i)
    matches, merges, splits = [], [], []
    matched_prev, matched_curr = set(), set()
    for j, plist in curr_to_prev.items():
        if len(plist) > 1:                      # several previous -> one now = MERGE
            merges.append((plist, j))
            matched_curr.add(j)
            matched_prev.update(plist)

    prev_to_curr = defaultdict(list)
    for j in range(nc):
        if j in matched_curr:
            continue
        for i in range(np_):
            if i in matched_prev:
                continue
            if iou[i, j] > CNT_MERGE_IOU_THRESHOLD:
                prev_to_curr[i].append(j)
    for i, clist in prev_to_curr.items():
        if len(clist) > 1:                      # one previous -> several now = SPLIT
            splits.append((i, clist))
            matched_prev.add(i)
            matched_curr.update(clist)

    rem_p = [i for i in range(np_) if i not in matched_prev]
    rem_c = [j for j in range(nc) if j not in matched_curr]
    if rem_p and rem_c:                         # the rest: optimal 1:1 assignment
        sub = np.zeros((len(rem_p), len(rem_c)))
        for ii, i in enumerate(rem_p):
            for jj, j in enumerate(rem_c):
                sub[ii, jj] = iou[i, j]
        ri, ci = linear_sum_assignment(-sub)
        for r, c in zip(ri, ci):
            if sub[r, c] > CNT_IOU_THRESHOLD:
                matches.append((rem_p[r], rem_c[c]))
                matched_prev.add(rem_p[r])
                matched_curr.add(rem_c[c])
    births = [j for j in range(nc) if j not in matched_curr]
    return matches, merges, splits, births


def _cnt_prune_discards(recent, fidx, window):
    recent[:] = [d for d in recent if fidx - d["frame"] <= window]


def _cnt_birth_relink(recent, new_comp, fidx, window):
    """The matched discard record (or None) if a just-born CC spatially matches a
    tentative discarded within `window` frames. Removes the match (counted once)."""
    for k, d in enumerate(recent):
        if fidx - d["frame"] <= window and _cnt_iou(new_comp, d["comp"]) > CNT_MERGE_IOU_THRESHOLD:
            recent.pop(k)
            return d
    return None


def _cnt_centroid(comp):
    """Centre of MASS of a component (memoised - a confirmed track that stops being
    detected carries the same component forward for many frames)."""
    c = comp.get("_centroid")
    if c is None:
        x, y, _, _ = comp["bbox"]
        ys, xs = np.nonzero(comp["local_mask"])
        c = (float(x + xs.mean()), float(y + ys.mean()))
        comp["_centroid"] = c
    return c


class LiveTracker:
    """export_3head_masks_2.5D.py's track_alpha_counts(), split into an incremental
    update() (its per-frame loop body, verbatim) and cheap recomputable pass-2 pieces
    (veto_set / curves / finalize). Equivalence with the offline two-pass version:

      * update() is the SAME code fed one frame at a time - matching, merge/split id
        bookkeeping, grace/relink, lifetime beta accumulation, per-instance rows;
      * the ONLY thing the offline pass 1 knows that live cannot is the sequence
        length n, used in `required = min(CONFIRM_HITS, n - birth)` to let a crystal
        born near the end confirm with the frames it has. Live uses the strict
        CONFIRM_HITS during the run and applies the relaxation in finalize() by
        replaying each unconfirmed track's recorded (detected, streak) history against
        the now-known n. The streak values were produced by the real state machine, so
        the replay reproduces the offline confirmation frame exactly. Under grace=0
        the relaxation can only ever fire for a track detected on every frame from its
        birth through the LAST frame (any miss discards it in both versions before the
        relaxed threshold is reachable), so live and offline never diverge in matching
        behavior either - the flip happens after the final frame's matching.
      * veto + both curves are rebuilt from scratch after every frame (they are cheap),
        which is exactly the offline pass 2 run early - and run one last time in
        finalize(), where it gives the offline result.
    """

    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        self.frames = []                    # processed frame names, in order
        self.confirm_frame = {}             # tid -> frame index where it confirmed
        self.birth_of = {}                  # tid -> birth frame index
        self.appeared_per_frame = []        # per frame: set of appeared tids
        self.beta_px = defaultdict(int)     # lifetime beta-overlap accumulators
        self.area_px = defaultdict(int)
        self.recent_discards = []
        self.pi_rows = []                   # (fidx, tid, cx, cy, area, detected, merged)
        self.hist = defaultdict(list)       # tid -> [(fidx, detected, streak)], all frames
                                            # the track was alive; feeds finalize()
        self.finalized = False

    @property
    def n(self):
        return len(self.frames)

    def update(self, frame_name, alpha_mask, beta_mask):
        """Advance the tracker by one frame. alpha_mask/beta_mask are BOOLEAN arrays -
        the same pixels the offline path would read back from the 0/255 PNGs."""
        if self.finalized:
            raise RuntimeError("LiveTracker.update() after finalize()")
        fidx = self.n
        self.frames.append(frame_name)
        tracks = self.tracks

        comps = _cnt_components(alpha_mask)
        if beta_mask is not None:
            for c in comps:
                bx, by, bw, bh = c["bbox"]
                c["beta_px"] = int(
                    beta_mask[by:by + bh, bx:bx + bw][c["local_mask"] > 0].sum())
        else:
            for c in comps:
                c["beta_px"] = 0
        _cnt_prune_discards(self.recent_discards, fidx, CNT_RELINK_WINDOW)

        # A CONFIRMED track stays matchable forever - a solidified crystal does not
        # vanish, so if it drops out for a frame it reconnects to its own id. A
        # TENTATIVE one may miss up to CNT_TENTATIVE_GRACE consecutive frames.
        active_ids = [tid for tid, t in tracks.items()
                      if t["state"] == "confirmed"
                      or fidx - t["last_frame"] <= 1 + CNT_TENTATIVE_GRACE]
        prev_comps = [tracks[tid]["comp"] for tid in active_ids]
        matches, merges, splits, births = _cnt_match(prev_comps, comps)

        cc_ids = defaultdict(list)
        for pi, cj in matches:
            cc_ids[cj].append(active_ids[pi])
        for plist, cj in merges:                # merge: keep every id, count unchanged
            for pi in plist:
                cc_ids[cj].append(active_ids[pi])
        for pi, clist in splits:                # split: the fragments are one crystal
            for cj in clist:
                cc_ids[cj].append(active_ids[pi])
        relink_of = {}                          # new tid -> inherited discard record
        for cj in births:                       # birth: tentative, NOT counted yet
            cc_ids[cj].append(self.next_id)
            if CNT_RELINK_TENTATIVE:
                rl = _cnt_birth_relink(self.recent_discards, comps[cj], fidx,
                                       CNT_RELINK_WINDOW)
                if rl is not None:
                    relink_of[self.next_id] = rl
            self.next_id += 1
        appeared = {tid for ids in cc_ids.values() for tid in ids}
        self.appeared_per_frame.append(appeared)

        for cj, comp in enumerate(comps):       # lifetime beta accumulation
            for tid in cc_ids.get(cj, []):
                self.beta_px[tid] += comp["beta_px"]
                self.area_px[tid] += comp["area"]

        new_tracks = {}
        for cj, comp in enumerate(comps):
            ids = cc_ids.get(cj, [])
            if len(ids) <= 1:
                for tid in ids:
                    new_tracks[tid] = {"comp": comp, "last_frame": fidx}
            else:
                # merged blob: each id keeps its OWN previous component, so when the
                # blob splits again each piece re-attaches to its own region
                for tid in ids:
                    new_tracks[tid] = {"comp": tracks.get(tid, {}).get("comp", comp),
                                       "last_frame": fidx}
        for tid in active_ids:
            if tid not in new_tracks:
                if tracks[tid]["state"] == "confirmed":
                    new_tracks[tid] = tracks[tid]
                elif fidx - tracks[tid]["last_frame"] <= CNT_TENTATIVE_GRACE:
                    new_tracks[tid] = tracks[tid]   # within grace: keep waiting
                else:                               # discard, but remember for relink
                    self.recent_discards.append(
                        {"comp": tracks[tid]["comp"], "frame": fidx,
                         "hits": tracks[tid]["streak"],
                         "birth": tracks[tid]["birth_frame"]})

        for tid, t in new_tracks.items():
            old = tracks.get(tid)
            detected = tid in appeared
            if old is None:
                rl = relink_of.get(tid)
                if rl is not None:                  # flicker: resume the old account
                    t.update(state="tentative", birth_frame=rl["birth"],
                             streak=rl["hits"] + 1)
                else:
                    t.update(state="tentative", birth_frame=fidx, streak=1)
            else:
                t["birth_frame"] = old["birth_frame"]
                if old["state"] == "confirmed":
                    t.update(state="confirmed", streak=old["streak"])
                else:
                    t.update(state="tentative",
                             streak=old["streak"] + 1 if detected else old["streak"])
            if t["state"] == "tentative":
                # STRICT threshold while live: the sequence length is unknown, so the
                # offline "born near the end" relaxation is applied in finalize().
                if detected and t["streak"] >= CNT_CONFIRM_HITS:
                    t["state"] = "confirmed"
                    self.confirm_frame[tid] = fidx
            self.birth_of[tid] = t["birth_frame"]
            self.hist[tid].append((fidx, detected, t["streak"]))

        merged_of = {tid: (1 if len(ids) > 1 else 0)
                     for ids in cc_ids.values() for tid in ids}
        for tid, t in new_tracks.items():
            cx, cy = _cnt_centroid(t["comp"])
            self.pi_rows.append((fidx, tid, cx, cy, t["comp"]["area"],
                                 int(tid in appeared), merged_of.get(tid, 0)))
        self.tracks = new_tracks

    def veto_set(self):
        """Confirmed tracks whose lifetime beta overlap exceeds the veto fraction.
        Recomputed from the running accumulators - a track can enter this set frames
        after it confirmed, which is why the counts CSV is rewritten every frame."""
        if CNT_BETA_VETO_FRAC < 0:
            return set()
        return {tid for tid in self.confirm_frame
                if self.area_px[tid] > 0
                and self.beta_px[tid] / self.area_px[tid] > CNT_BETA_VETO_FRAC}

    def curves(self, vetoed):
        """(cumulative, active) lists over all processed frames, beta-vetoed tracks
        removed everywhere, cumulative booked at the BIRTH frame."""
        n = self.n
        kept = [self.birth_of[tid] for tid in self.confirm_frame if tid not in vetoed]
        births = np.zeros(n, dtype=np.int64)
        for bf in kept:
            births[bf] += 1
        cum = np.cumsum(births).tolist()
        act = [len(self.appeared_per_frame[f] - vetoed) for f in range(n)]
        return cum, act

    def finalize(self):
        """End of sequence: apply the offline 'born near the end' confirmation rule,
        now that n is known. Replays each unconfirmed track's recorded (detected,
        streak) history against required = min(CONFIRM_HITS, n - birth); the first
        frame that satisfies it is the confirmation frame the offline pass would have
        produced. Idempotent; call once when the run ends."""
        if self.finalized:
            return
        n = self.n
        for tid, h in self.hist.items():
            if tid in self.confirm_frame:
                continue
            required = min(CNT_CONFIRM_HITS, n - self.birth_of[tid])
            for fidx, detected, streak in h:
                if detected and streak >= required:
                    self.confirm_frame[tid] = fidx
                    break
        self.finalized = True


# ----------------- CSV writers -----------------

STATS_FIELDS = (["frame_idx", "frame", "n_neighbors_25d",
                 "area_px", "total_px", "alpha_px", "beta_px", "overlap_px",
                 "total_pct", "alpha_pct", "beta_pct", "overlap_pct"]
                + [f"{agg}_{name}" for name in LIVE_CLS for agg in ("clsmax", "clsmean")])


def append_stats_row(csv_path, row, first):
    """live_stats.csv is append-only: one immutable row per frame."""
    with open(csv_path, "w" if first else "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=STATS_FIELDS)
        if first:
            w.writeheader()
        w.writerow(row)


def _atomic_rewrite(path, write_fn):
    """Write via a temp file + os.replace, so a concurrent reader (a plotting script
    polling the CSVs) never sees a half-written file. On Windows os.replace itself
    fails with PermissionError during the few ms a polling reader holds the target
    open, so retry briefly. A reader that KEEPS the file open (a CSV opened in Excel
    locks it for good) still fails loudly after the retries - skipping the write
    silently would leave a stale curve on disk."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        write_fn(csv.writer(f))
    for _ in range(20):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.1)
    os.replace(tmp, path)   # last attempt: let the real error propagate


def rewrite_counts_csv(path, tracker, vetoed):
    cum, act = tracker.curves(vetoed)

    def _w(w):
        w.writerow(["frame_idx", "frame", "cumulative_count", "active_count"])
        for i, (s, c, a) in enumerate(zip(tracker.frames, cum, act)):
            w.writerow([i, s, c, a])
    _atomic_rewrite(path, _w)
    return cum, act


def rewrite_per_instance_csv(path, tracker, vetoed):
    """Same schema and state semantics as the offline per_instance.csv: 'confirmed'
    from the BIRTH frame of a track that reached confirmation (so the file agrees with
    the birth-booked cumulative curve), 'vetoed' overrides everything."""
    def _w(w):
        w.writerow(["frame_name", "frame_idx", "track_id", "x", "y", "area",
                    "detected", "merged", "state", "birth_frame", "confirm_frame"])
        for fidx, tid, cx, cy, ar, det, mrg in tracker.pi_rows:
            cf = tracker.confirm_frame.get(tid)
            bf = tracker.birth_of.get(tid)
            state = ("vetoed" if tid in vetoed else
                     "confirmed" if (cf is not None and bf is not None and fidx >= bf)
                     else "tentative")
            w.writerow([tracker.frames[fidx], fidx, tid, round(cx, 2), round(cy, 2),
                        ar, det, mrg, state, "" if bf is None else bf,
                        "" if cf is None else cf])
    _atomic_rewrite(path, _w)


# ----------------- frame discovery -----------------

def _frame_number(name: str):
    m = re.search(r"(\d+)(?=\.[^.]+$)", name)
    return int(m.group(1)) if m else None


def scan_ready_frames(input_dir: str, settle_sec: float):
    """Sorted [(number, path)] of image files whose mtime is at least settle_sec old
    (settle_sec = 0 accepts them as soon as they appear, which is right for a
    producer that renames finished files into place). Zero-byte files are skipped
    either way. Every image file MUST carry a trailing frame number and numbers must
    be unique - live ordering depends on them."""
    now = time.time()
    out = {}
    for p in Path(input_dir).iterdir():
        if not (p.is_file() and p.suffix.lower() in VALID_EXTS):
            continue
        num = _frame_number(p.name)
        if num is None:
            raise RuntimeError(
                f"[scan] image file without a trailing frame number: {p.name} - live "
                "mode orders frames by that number. Remove the file or rename it.")
        try:
            st = p.stat()
        except FileNotFoundError:
            continue                       # deleted between listing and stat
        if now - st.st_mtime < settle_sec or st.st_size == 0:
            continue                       # still being written
        if num in out:
            raise RuntimeError(f"[scan] duplicate frame number {num}: "
                               f"{Path(out[num]).name} vs {p.name}")
        out[num] = str(p)
    return sorted(out.items())


# ----------------- per-frame work -----------------

class _DeferSigint:
    """Delay Ctrl+C across the tracker-update + CSV-rewrite critical section, so an
    interrupt can never leave the tracker half-advanced (frames appended but curves
    not, etc). Inference - the long part - stays interruptible: it mutates nothing."""

    def __enter__(self):
        self._hit = False
        self._old = signal.signal(signal.SIGINT, self._handler)
        return self

    def _handler(self, signum, frame):
        self._hit = True
        print("\n[interrupt] finishing the current frame's bookkeeping ...", flush=True)

    def __exit__(self, exc_type, exc, tb):
        signal.signal(signal.SIGINT, self._old)
        if self._hit and exc_type is None:
            raise KeyboardInterrupt
        return False


def process_frame(idx, cur, prev_path, next_path, *, ctx):
    """Inference + stats row + tracker update + output rewrite for ONE frame.
    `cur` = (number, path). Returns the wall seconds the frame took."""
    t0 = time.time()
    num, path = cur
    stem = Path(path).stem
    tensor, n_neigh = build_input_tensor(path, prev_path, next_path, ctx["use_25d"],
                                         ctx["norm_mean"], ctx["norm_std"])
    total_prob, alpha_prob, beta_prob, cls_tiles = sliding_window_three_probs(
        ctx["model"], tensor, ctx["device"], ctx["target_size"], ctx["gpu_batch_size"],
        tile_overlap=ctx["tile_overlap"], blend_hann=ctx["tile_blend_hann"])
    m = derive_masks(total_prob, alpha_prob, beta_prob, ctx["thr_total"],
                     ctx["thr_alpha"], ctx["thr_beta"], ctx["intersect_total"])
    t_infer = time.time() - t0     # GPU forward incl. input build; masks AND the
                                   # image-level cls probs come out of this one pass

    if ctx["save_masks"]:
        for key in ("alpha", "beta", "total"):
            sub = {"alpha": "alpha_only", "beta": "beta_only", "total": "total"}[key]
            imwrite_unicode(os.path.join(ctx["out_dir"], sub, f"{stem}.png"),
                            m[key].astype(np.uint8) * 255)

    # ---- stats row (area ratios + the three requested class probabilities) ----
    t0 = time.time()
    area = float(m["total"].size)
    tp, ap, bp = int(m["total"].sum()), int(m["alpha"].sum()), int(m["beta"].sum())
    ov = int(m["overlap"].sum())
    row = {
        "frame_idx": idx, "frame": os.path.basename(path), "n_neighbors_25d": n_neigh,
        "area_px": int(area), "total_px": tp, "alpha_px": ap, "beta_px": bp,
        "overlap_px": ov,
        "total_pct": round(tp / area * 100.0, 4), "alpha_pct": round(ap / area * 100.0, 4),
        "beta_pct": round(bp / area * 100.0, 4), "overlap_pct": round(ov / area * 100.0, 4),
    }
    for name in LIVE_CLS:
        ci = CLS_NAMES.index(name)
        row[f"clsmax_{name}"] = round(float(cls_tiles[:, ci].max()), 4)
        row[f"clsmean_{name}"] = round(float(cls_tiles[:, ci].mean()), 4)
    append_stats_row(ctx["stats_csv"], row, first=(idx == 0))
    t_stats = time.time() - t0     # area fractions + cls columns + CSV append

    # ---- tracking (in-memory masks; identical pixels to the offline PNG readback) ----
    tracker = ctx["tracker"]
    with _DeferSigint():
        t0 = time.time()
        tracker.update(f"{stem}.png",
                       m["alpha"], m["beta"] if CNT_BETA_VETO_FRAC >= 0 else None)
        t_track = time.time() - t0     # component extraction + matching + state machine
        t0 = time.time()
        vetoed = tracker.veto_set()
        cum, act = rewrite_counts_csv(ctx["counts_csv"], tracker, vetoed)
        rewrite_per_instance_csv(ctx["pi_csv"], tracker, vetoed)
        t_settle = time.time() - t0    # veto + curve rebuild + both CSV rewrites

    dt = t_infer + t_stats + t_track + t_settle
    print(f"[{idx:4d}] {stem}  total {row['total_pct']:6.2f}%  alpha {row['alpha_pct']:6.2f}%  "
          f"beta {row['beta_pct']:6.2f}%  |  active {act[-1]:4d}  cum {cum[-1]:4d}"
          f"{' (' + str(len(vetoed)) + ' vetoed)' if vetoed else ''}  |  "
          + "  ".join(f"{n[0]} {row['clsmax_' + n]:.2f}" for n in LIVE_CLS)
          + f"  |  {dt:5.1f}s (infer {t_infer:.1f} stats {t_stats:.2f} "
          + f"track {t_track:.2f} settle {t_settle:.2f})", flush=True)
    return dt


def finalize_run(ctx, reason):
    """Apply the end-of-sequence confirmation relaxation, rerun the veto, rewrite both
    tracker CSVs one last time, and print the summary. After this the outputs are the
    ones the offline exporter would have produced for the same frame set."""
    tracker = ctx["tracker"]
    if tracker.n == 0:
        print(f"[finalize] {reason}; no frames were processed - nothing to write.")
        return
    tracker.finalize()
    vetoed = tracker.veto_set()
    cum, act = rewrite_counts_csv(ctx["counts_csv"], tracker, vetoed)
    rewrite_per_instance_csv(ctx["pi_csv"], tracker, vetoed)
    print(f"[finalize] {reason}; {tracker.n} frame(s) processed.")
    print(f"[finalize] final cumulative {cum[-1]}, active on last frame {act[-1]}, "
          f"{len(vetoed)} track(s) beta-vetoed "
          f"(rules: confirm {CNT_CONFIRM_HITS}, grace {CNT_TENTATIVE_GRACE}, "
          f"relink {'on' if CNT_RELINK_TENTATIVE else 'off'}, "
          f"veto {'off' if CNT_BETA_VETO_FRAC < 0 else CNT_BETA_VETO_FRAC})")
    print(f"[finalize] outputs: {ctx['stats_csv']}  |  {ctx['counts_csv']}  |  "
          f"{ctx['pi_csv']}")


# ----------------- main -----------------

RULE_ARGS = {"confirm_hits": "CNT_CONFIRM_HITS", "grace": "CNT_TENTATIVE_GRACE",
             "relink": "CNT_RELINK_TENTATIVE", "beta_veto": "CNT_BETA_VETO_FRAC",
             "min_area": "CNT_MIN_AREA_PX", "close_kernel": "CNT_CLOSE_KERNEL",
             "iou": "CNT_IOU_THRESHOLD", "relink_window": "CNT_RELINK_WINDOW"}


def main():
    ap = argparse.ArgumentParser(
        description="LIVE 3-head inference: area ratios + tracked crystal counts + "
                    "Liquid/Alpha/Beta image-level probabilities, updated as frames "
                    "arrive.")
    ap.add_argument("--input", default=INPUT_PATH,
                    help="folder the preprocessed frames arrive in")
    ap.add_argument("--output", default=OUTPUT_DIR, help="output folder for the CSVs")
    ap.add_argument("--ckpt", default=CKPT, help="3-head V5aModel ckpt (slim or full)")
    ap.add_argument("--device", default=DEVICE)
    ap.add_argument("--gpu_batch_size", type=int, default=GPU_BATCH_SIZE,
                    help="max sliding-window crops per forward (reduce if OOM)")
    ap.add_argument("--tile_overlap", type=int, default=TILE_OVERLAP,
                    help="sliding-window tile overlap in px (stride=tile-overlap)")
    ap.add_argument("--no_tile_blend_hann", dest="tile_blend_hann",
                    action="store_false", default=TILE_BLEND_HANN,
                    help="merge overlapping tiles by flat average instead of Hann weight")
    ap.add_argument("--dpt_threshold_total", type=float, default=DPT_THRESHOLD_TOTAL)
    ap.add_argument("--dpt_threshold_alpha", type=float, default=DPT_THRESHOLD_ALPHA)
    ap.add_argument("--dpt_threshold_beta", type=float, default=DPT_THRESHOLD_BETA)
    ap.add_argument("--no_intersect_total", dest="intersect_total", action="store_false",
                    default=INTERSECT_WITH_TOTAL,
                    help="do NOT gate alpha/beta by the total head")
    ap.add_argument("--input_mode", default="auto", choices=["auto", "25d", "single"],
                    help="override the checkpoint's 2.5D flag (default: follow ckpt)")

    ap.add_argument("--poll_sec", type=float, default=POLL_SEC,
                    help="folder re-scan interval, seconds")
    ap.add_argument("--settle_sec", type=float, default=SETTLE_SEC,
                    help="wait this long after a file's last write before reading "
                         "it (0 = off; raise it only if frames are written straight "
                         "to their final name instead of being renamed into place)")
    ap.add_argument("--idle_timeout", type=float, default=IDLE_TIMEOUT,
                    help="finalize after this many seconds without a new frame; "
                         "0 = wait forever (Ctrl+C to finalize)")
    ap.add_argument("--once", action="store_true",
                    help="process the frames present NOW, finalize, exit (replay mode; "
                         "gives the same numbers as the offline exporter)")
    ap.add_argument("--save_masks", action="store_true", default=SAVE_MASKS,
                    help="also dump alpha_only/beta_only/total binary mask PNGs")

    # ---- counting rules. Defaults ARE the frozen set (Rule v3); these exist for
    #      ablations, same flags as export_3head_masks_2.5D.py. ----
    ap.add_argument("--confirm_hits", type=int, default=CNT_CONFIRM_HITS,
                    help="consecutive detections before a birth is counted")
    ap.add_argument("--grace", type=int, default=CNT_TENTATIVE_GRACE,
                    help="frames a tentative may miss and survive")
    ap.add_argument("--relink", dest="relink", action="store_true",
                    default=CNT_RELINK_TENTATIVE,
                    help="a birth inherits a just-discarded tentative's account")
    ap.add_argument("--no_relink", dest="relink", action="store_false")
    ap.add_argument("--beta_veto", type=float, default=CNT_BETA_VETO_FRAC,
                    help="drop confirmed tracks whose LIFETIME beta overlap exceeds "
                         "this fraction; NEGATIVE switches the veto off")
    ap.add_argument("--min_area", type=int, default=CNT_MIN_AREA_PX,
                    help="ignore components smaller than this, px")
    ap.add_argument("--close_kernel", type=int, default=CNT_CLOSE_KERNEL,
                    help="morphological closing kernel; 0 = off")
    ap.add_argument("--iou", type=float, default=CNT_IOU_THRESHOLD,
                    help="1:1 track match accepted above this IoU")
    ap.add_argument("--relink_window", type=int, default=CNT_RELINK_WINDOW,
                    help="frames a discarded tentative stays relinkable")
    args = ap.parse_args()

    rules = {g: getattr(args, a) for a, g in RULE_ARGS.items()}
    changed = {k: v for k, v in rules.items() if v != globals()[k]}
    if changed:
        print("[rules] overridden: "
              + "  ".join(f"{k[4:].lower()}={v}" for k, v in sorted(changed.items())))
    globals().update(rules)

    if not args.input:
        raise ValueError("--input is required (folder the preprocessed frames arrive in)")
    if not Path(args.input).is_dir():
        raise NotADirectoryError(f"--input is not a directory: {args.input}")
    if not args.output:
        raise ValueError("--output is required")
    if not args.ckpt or not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"checkpoint not found: {args.ckpt}")

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available())
                          else "cpu")
    model, cfg = load_model(args.ckpt, device)
    target_size, norm_mean, norm_std, use_25d = _resolve_cfg(cfg, args.input_mode)
    print(f"Device: {device}")
    print(f"Input mode: {'2.5D (t-1,t,t+1) - one-frame lag while live' if use_25d else 'single-frame'}; "
          f"target_size={target_size}; gpu_batch_size={args.gpu_batch_size}")
    print(f"DPT thr: total={args.dpt_threshold_total} alpha={args.dpt_threshold_alpha} "
          f"beta={args.dpt_threshold_beta}  intersect_total={args.intersect_total}")
    print(f"Input:  {args.input}\nOutput: {args.output}")

    out_dir = args.output
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if args.save_masks:
        for sub in ("alpha_only", "beta_only", "total"):
            Path(out_dir, sub).mkdir(exist_ok=True)
    for name in ("live_stats.csv", "alpha_counts.csv", "per_instance.csv"):
        if Path(out_dir, name).exists():
            print(f"[warn] {name} already exists in the output folder and will be "
                  "rewritten from scratch (restart = reprocess).")

    ctx = dict(model=model, device=device, target_size=target_size,
               norm_mean=norm_mean, norm_std=norm_std, use_25d=use_25d,
               gpu_batch_size=args.gpu_batch_size, tile_overlap=args.tile_overlap,
               tile_blend_hann=args.tile_blend_hann,
               thr_total=args.dpt_threshold_total, thr_alpha=args.dpt_threshold_alpha,
               thr_beta=args.dpt_threshold_beta, intersect_total=args.intersect_total,
               save_masks=args.save_masks, out_dir=out_dir,
               stats_csv=os.path.join(out_dir, "live_stats.csv"),
               counts_csv=os.path.join(out_dir, "alpha_counts.csv"),
               pi_csv=os.path.join(out_dir, "per_instance.csv"),
               tracker=LiveTracker())

    ready = scan_ready_frames(args.input, args.settle_sec)
    k = 0                                   # frames processed so far
    print(f"[start] {len(ready)} frame(s) already in the folder"
          + (" - catching up first." if ready else "."))

    if args.once:
        if not ready:
            raise RuntimeError(f"--once but no ready frames in {args.input}")
        for i, cur in enumerate(ready):
            process_frame(i, cur,
                          prev_path=ready[i - 1][1] if i > 0 else None,
                          next_path=ready[i + 1][1] if i + 1 < len(ready) else None,
                          ctx=ctx)
        finalize_run(ctx, "--once replay complete")
        return

    last_new = time.time()
    last_wait_note = 0.0
    try:
        while True:
            fresh = scan_ready_frames(args.input, args.settle_sec)
            if {n for n, _ in fresh} - {n for n, _ in ready}:
                last_new = time.time()
            if k > 0:
                # ORDER GUARD. The first k frames in sorted order must still be exactly
                # the k already processed, and (2.5D) the pending head - which frame
                # k-1 consumed as its (t+1) neighbor - must still sort right behind
                # them. A file inserting anywhere in that prefix means the producer
                # wrote frames out of order (or deleted/renamed one): the neighbors
                # already used are wrong and every downstream number would be silently
                # off. Stop loudly; rerun with --once after the acquisition instead.
                expect = [n for n, _ in ready[:k]]
                if use_25d and k < len(ready):
                    expect.append(ready[k][0])
                fresh_nums = [n for n, _ in fresh]
                if fresh_nums[:len(expect)] != expect:
                    raise RuntimeError(
                        f"[order] the frame order changed behind the processed front "
                        f"(expected leading frame numbers {expect[-4:]}..., found "
                        f"{fresh_nums[:len(expect)][-4:]}...): the preprocessing wrote "
                        f"out of order, or a consumed frame was deleted/renamed. Live "
                        f"tracking against wrong neighbors would corrupt every output "
                        f"- rerun in --once mode after the acquisition instead.")
            ready = fresh

            # a frame is processed once its SUCCESSOR is ready (2.5D neighbor), so
            # while live we always keep exactly one trailing frame pending
            limit = len(ready) - 1 if use_25d else len(ready)
            while k < limit:
                process_frame(k, ready[k],
                              prev_path=ready[k - 1][1] if k > 0 else None,
                              next_path=ready[k + 1][1] if k + 1 < len(ready) else None,
                              ctx=ctx)
                k += 1
                last_new = time.time()

            idle = time.time() - last_new
            if args.idle_timeout > 0 and idle >= args.idle_timeout:
                # sequence over: the trailing frame gets the center-frame fallback,
                # exactly like the offline exporter's last frame
                while k < len(ready):
                    process_frame(k, ready[k],
                                  prev_path=ready[k - 1][1] if k > 0 else None,
                                  next_path=None, ctx=ctx)
                    k += 1
                finalize_run(ctx, f"no new frame for {idle:.0f}s (--idle_timeout "
                                  f"{args.idle_timeout:.0f}s)")
                return
            if idle > 60 and time.time() - last_wait_note > 60:
                pend = len(ready) - k
                print(f"[wait] {k} processed, {pend} pending, idle {idle:.0f}s"
                      + ("" if args.idle_timeout > 0 else " (no --idle_timeout; Ctrl+C "
                         "to finalize)"), flush=True)
                last_wait_note = time.time()
            time.sleep(max(0.1, args.poll_sec))
    except KeyboardInterrupt:
        # End of run. In the steady 2.5D state exactly ONE frame is pending - the
        # trailing frame held back for its (t+1) neighbor - and dropping it would
        # make a Ctrl+C run one frame shorter than the same folder replayed with
        # --once, which also shifts finalize()'s `required = min(CONFIRM_HITS,
        # n - birth)` and can change the final count. So process that one frame with
        # the center-frame fallback, exactly like the idle-timeout path. A LONGER
        # backlog means Ctrl+C landed during catch-up, where the user wants out now
        # and the remainder could be arbitrarily large: skip it and say so.
        print()
        pending = max(0, len(ready) - k)
        if pending == 1:
            try:
                process_frame(k, ready[k], prev_path=ready[k - 1][1] if k > 0 else None,
                              next_path=None, ctx=ctx)
                k += 1
                pending = 0
            except KeyboardInterrupt:      # second Ctrl+C: really stop now
                print("[interrupt] trailing frame abandoned.")
                pending = 1
        finalize_run(ctx, f"Ctrl+C ({k} frame(s) processed"
                          + (f", {pending} pending skipped)" if pending else ")"))


if __name__ == "__main__":
    main()
