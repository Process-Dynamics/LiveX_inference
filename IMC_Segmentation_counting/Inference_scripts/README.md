# Live crystal segmentation, tracking and counting for in-situ X-ray radiography

Streaming inference for time-resolved solidification experiments. Point the script
at the folder your preprocessing writes frames into; as each frame lands it is
segmented into α / β / total-crystal, the crystals are tracked and counted across
frames, and three small CSV files are kept up to date — so the numbers are
available *while the experiment runs*, not only afterwards.

A single DINOv3 + DPT backbone produces all of it in **one forward pass per
frame**: three segmentation heads (total crystal, α, β) plus an image-level class
probability for each phase.

```
live_inference_3head_2.5D.py   the pipeline (run this)
model.py                       the network definition (imported, do not run)
requirements.txt               dependencies
```

**Contents**
[Part 1 — What it does](#part-1--what-it-does) ·
[Part 2 — Installation](#part-2--installation) ·
[Part 3 — The model checkpoint](#part-3--the-model-checkpoint) ·
[Part 4 — Usage](#part-4--usage) ·
[Part 5 — Outputs](#part-5--outputs) ·
[Part 6 — Options](#part-6--options) ·
[Part 7 — Performance](#part-7--performance) ·
[Part 8 — Troubleshooting](#part-8--troubleshooting)

---

## Part 1 — What it does

For every frame that appears in the input folder:

1. **Build the network input.** The model is *2.5D*: it sees three consecutive
   frames stacked as the three channels of one RGB image — `(t-1, t, t+1)`. This
   gives it the local time context that separates a real crystal from a
   fluctuation. Frames are ordered by the number at the end of the filename
   (`frame_001150.png` → 1150), and the neighbours are the adjacent files in that
   order, not strictly ±1 in frame number.

2. **Sliding-window forward pass at full resolution.** The frame is *not*
   downscaled. It is covered with overlapping tiles of the checkpoint's training
   size (2048 px, 512 px overlap), each tile is run through the network, and the
   probability maps are stitched back with a Hann window so a pixel near a tile
   edge — which the network saw with almost no context — barely contributes,
   while the tile that has it near its centre dominates. Without the overlap a
   visible seam appears wherever the image width is an exact multiple of the tile
   size.

3. **Read out three masks.** Each head outputs a foreground probability; the
   thresholded masks are

   ```
   total = P_total ≥ 0.5
   α     = (P_α ≥ 0.5) ∩ total
   β     = (P_β ≥ 0.5) ∩ total
   ```

   The α and β heads are independent, so a pixel may be **both** — the shared
   region is reported as `overlap_px` rather than being forced into one class.

4. **Track and count the α crystals.** Crystals are matched between consecutive
   frames by IoU with Hungarian assignment, so each one keeps its identity over
   time and `cumulative_count` reflects distinct crystals rather than the number
   of connected components in a frame. A new track has to be seen for
   `--confirm_hits` frames before it counts, and is then credited to the frame it
   first appeared in; tracks that turn out to overlap the β mask are dropped
   (`--beta_veto`).

5. **Write the three CSVs** (below), then wait for the next frame.

### Three input frames per output frame

Every frame is processed from three images: the frame itself plus one before and
one after. These do **not** have to be consecutive at the camera's frame rate —
they are simply the nearest available frames on either side, i.e. the neighbouring
files in the input folder. You can therefore feed the model a subsampled sequence
(every 2nd, every 5th frame, …) without changing anything; only the ordering
matters, not the spacing.

Because frame *t* needs its successor, it can only be processed once that
successor has arrived, so the live output trails the newest file by one frame.
When the run ends — idle timeout, `--once`, or Ctrl+C — that trailing frame is
processed with the centre frame substituted for *t+1*, so no frame is lost.

---

## Part 2 — Installation

Python 3.10+ with a CUDA-capable GPU (CPU works but is far slower).

```bash
pip install -r requirements.txt
```

Tested with Python 3.11, `torch 2.11.0+cu128`, `transformers 5.10.2`,
`opencv-python 4.13`, `numpy 2.4`, `scipy 1.17`, `pillow 12.2`.
Install the torch build that matches your CUDA version — see
<https://pytorch.org/get-started/locally/>.

`model.py` must sit next to the script (it does in this folder).

---

## Part 3 — The model checkpoint

Download the model:

### [⬇️ Model.pt](https://drive.google.com/file/d/1Es1jUSh8w0TKpEAqqtPos2CAQ7nox_ph/view?usp=sharing)

409 MB. Pass its path with `--ckpt`, or edit the `CKPT` constant near the top of
the script to avoid typing it every time.

The checkpoint carries the weights and the training config (input size,
normalisation, 2.5D flag — all read automatically).

---

## Part 4 — Usage

### Folder layout

```
live_inference_3head_2.5D.py     the script
model.py                         must sit beside it
Model.pt                         the downloaded checkpoint (any path, --ckpt)

preprocessed_frames/             --input : where your preprocessing writes frames
├── frame_001150.png
├── frame_001152.png
├── frame_001154.png
└── ...

results/                         --output : created if it does not exist
├── live_stats.csv
├── alpha_counts.csv
└── per_instance.csv
```

The input folder holds nothing but the frames — one flat folder, no sub-folders,
no other images. The output folder is written by the script alone; point a fresh
one at every run, or expect its three CSVs to be overwritten.

### Live, while the experiment runs

Start it before or during acquisition — a missing or still-empty input folder is
fine, it waits.

```bash
python live_inference_3head_2.5D.py \
    --input  /path/to/preprocessed_frames \
    --output /path/to/results \
    --ckpt   /path/to/Model.pt
```

### Input requirements

* One folder of image files (`.png`, `.jpg`, `.tif`, …), all the same size.
* **Every file must end in a frame number** — `frame_001150.png`, `img_42.tif`.
  Ordering depends on it, so a file without one is an error rather than a guess,
  and duplicate numbers are rejected.
* Frames must appear in increasing order. If a file shows up *behind* the frame
  the pipeline has already consumed as a neighbour, the run stops with an error
  instead of silently producing numbers computed against the wrong context — rerun
  with `--once` afterwards if your producer cannot guarantee ordering.

Restarting reprocesses the whole folder from scratch; all tracker state is in
memory and the output files are rewritten.

---

## Part 5 — Outputs

Three CSVs in `--output`:

### `live_stats.csv` — per-frame areas and class probabilities

One row per frame, appended and never modified.

| column | meaning |
|---|---|
| `frame_idx`, `frame` | index in the processed sequence, original filename |
| `n_neighbors_25d` | how many real 2.5D neighbours were available (2 = both) |
| `area_px` | total pixels in the frame |
| `total_px`, `alpha_px`, `beta_px` | crystal pixels per head |
| `overlap_px` | pixels claimed by both the α and β heads |
| `total_pct`, `alpha_pct`, `beta_pct`, `overlap_pct` | the same as % of the frame — **the area fractions** |
| `clsmax_Liquid`, `clsmax_Alpha`, `clsmax_Beta` | image-level class probability, max over sliding-window tiles → *is this phase present anywhere in the frame* |
| `clsmean_…` | the same averaged over tiles → *how much of the frame it covers* |

### `alpha_counts.csv` — the crystal count

| column | meaning |
|---|---|
| `frame_idx`, `frame` | as above |
| `cumulative_count` | distinct α crystals that have appeared up to this frame (monotonic) |
| `active_count` | how many are visible in this frame (can fall — crystals merge) |

**This file is rewritten in full after every frame, not appended to — the whole
curve keeps changing as the run proceeds, not just its end.** The reason is the
β-veto: α tracks are filtered against the β mask, and a track's β overlap is only
known over its whole lifetime, so a crystal counted at frame *t* can be identified
as β structure hundreds of frames later and must then be removed from *every*
frame back to *t*. Confirmation works the same way, crediting a crystal back to
the frame it first appeared in a few frames after the fact. Past rows are
therefore corrected continuously, and the file always holds the best estimate
given the frames seen so far. The rewrite is atomic (temp file + rename), so a
plotting script polling the file never sees it half-written.

### `per_instance.csv` — one row per (frame, crystal)

`frame_name, frame_idx, track_id, x, y, area, detected, merged, state,
birth_frame, confirm_frame` — the position and size of every tracked crystal in
every frame it existed, with its state (`tentative` / `confirmed` / `vetoed`).
Rows are written per **track**, so a crystal keeps its own centroid and area
through a merge instead of jumping to the merged blob's centre, and a track that
is momentarily undetected still has a row. Rewritten every frame for the same
reason as above.

`--save_masks` additionally dumps the binary masks to `alpha_only/`, `beta_only/`
and `total/` (one PNG per frame, 0/255) — useful for verification, off by default
since it is by far the largest output.

---

## Part 6 — Options

### Run control

| flag | default | meaning |
|---|---|---|
| `--input` | — | folder the frames arrive in (required) |
| `--output` | — | folder for the CSVs (required) |
| `--ckpt` | see script | checkpoint path |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--once` | off | process what is there now, finalise, exit |
| `--idle_timeout` | `0` | finalise after N seconds with no new frame; 0 = wait forever |
| `--poll_sec` | `2` | how often the folder is re-scanned |
| `--settle_sec` | `0` | wait this long after a file's last write before reading it — only needed if frames are written straight to their final name instead of being renamed into place |
| `--save_masks` | off | also write the binary mask PNGs |

### Inference

| flag | default | meaning |
|---|---|---|
| `--gpu_batch_size` | `1` | tiles per forward pass |
| `--tile_overlap` | `512` | sliding-window overlap in px (0 reproduces the old seam behaviour) |
| `--no_tile_blend_hann` | off | merge overlapping tiles by flat average instead of a Hann weight |
| `--dpt_threshold_total/_alpha/_beta` | `0.5` | per-head foreground thresholds |
| `--no_intersect_total` | off | do not gate the α/β heads with the total head |
| `--input_mode` | `auto` | override the checkpoint's 2.5D flag (`25d` / `single`) |

### Counting rules

The defaults are a validated rule set — tuned against hand-annotated ground truth
on two sequences and frozen. Change them only for ablation studies.

| flag | default | meaning |
|---|---|---|
| `--confirm_hits` | `4` | consecutive detections before a birth is counted |
| `--beta_veto` | `0.05` | drop tracks whose lifetime β overlap exceeds this fraction; a **negative** value disables the veto |
| `--min_area` | `200` | ignore components smaller than this (px, full resolution) |
| `--close_kernel` | `15` | morphological closing kernel that reconnects fragmented arms; 0 = off |
| `--iou` | `0.10` | IoU above which a 1:1 frame-to-frame match is accepted |

The β-veto is the only filter applied to the tracks. `--beta_veto 0.05` is a
genuine optimum rather than "tighter is better" — 0.1 over-counts, 0.03 already
undershoots, 0.02 badly so.

---

## Part 7 — Performance

Measured on a 136-frame sequence of 4096×2304 frames, RTX 5070 Ti Laptop (12 GB):

| stage | mean per frame |
|---|---|
| inference (masks **and** class probabilities, one pass) | 4.89 s |
| area fractions + CSV append | 0.017 s |
| tracking update | 0.091 s |
| veto, curve rebuild, rewrite of both CSVs | 0.022 s |
| **total** | **5.02 s** |

Inference is 97% of the cost; the per-frame full rewrite of the counting files is
negligible against it.

### `--gpu_batch_size`

| batch | seconds/frame | peak allocation |
|---|---|---|
| 1 | **4.6** | 5.0 GB |
| 2 | 24.8 | 9.4 GB |
| 4 | ~44 | pinned at 12 GB |

On a 24 GB or larger card use `--gpu_batch_size 4`.

---

## Part 8 — Troubleshooting

**`image file without a trailing frame number`** — a file in the input folder does
not end in digits before its extension. Remove or rename it.

**`the frame order changed behind the processed front`** — a frame appeared out of
order, or one that had already been used as a neighbour was deleted or renamed.
The run stops rather than continue with a corrupted context; rerun with `--once`
once acquisition is complete.

**CUDA out of memory** — lower `--gpu_batch_size` to 1, then `--tile_overlap` (to
256 or 0). Note that image viewers holding a 4096×2304 PNG open are surprisingly
expensive on GPU memory.

**A CSV stops updating / a permission error on Windows** — a reader is holding the
file open exclusively. Excel does this; pandas and plotting scripts do not.

**`Sliding-window blend left uncovered pixels`** — `--tile_overlap` is at or above
the tile size, so the stride became non-positive. Lower it.
