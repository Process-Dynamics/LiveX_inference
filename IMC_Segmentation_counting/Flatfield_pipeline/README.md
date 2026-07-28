# Flat-field correction with tiled non-rigid registration

Flat-field (background) removal for time-resolved X-ray image sequences where the
background structure **drifts and deforms** over time, so a plain `frame / flat`
division leaves residual structure. The flat is locally re-registered to every
frame before the division.

```
flatfield_correct.py   the pipeline: correct one image or a whole folder
visualize_steps.py     per-stage visualization of one chosen frame
requirements.txt       dependencies
.gitignore             keeps pipeline output (PNGs, npy cache) out of git
```

## Part 1 — Pipeline

For each frame:

1. **Flat construction** — the reference images (`--ref`, one image or a folder)
   are averaged into a single flat.

2. **Tiled non-rigid registration** — the background drift is spatially varying,
   so no single global warp can fit it. Instead the frame is covered with
   **overlapping tiles** (default 512 px, stride 384 px → 128 px overlap) and the
   flat is registered to the frame **independently in every tile**:

   1. lay a uniform landmark grid over the tile and add detected corner points;
   2. find each landmark's displacement by normalized template matching
      (sub-pixel refined);
   3. reject inconsistent matches with an affine-RANSAC test;
   4. interpolate the surviving displacements into a dense field with a
      thin-plate spline, warp the flat, and repeat from step 2 for up to 12
      refinement iterations;
   5. if a tile has too few matchable landmarks (featureless, e.g. uniform
      liquid), fall back to a single uniform translation estimated by ECC —
      in the opposite direction (the frame is nudged to match the flat, then
      divided by the untouched flat), since ECC estimates that direction
      directly and this fallback is only a coarse, low-confidence patch;
   6. the tile's correction is `frame / warped_flat` (or, on the ECC fallback,
      `warped_frame / flat`).

   Because the tiles overlap, every pixel receives values from up to four
   tiles; they are merged with a **Hann-window weight** (tile centre trusted
   most, tile edge least), which removes the tile seams from the stitched
   full-frame result.

3. **Image processing** — the registered result is turned into the output image
   in this order:

   1. **temporal averaging** (`--mean-win`, default 3): average the current
      frame's registered result with the previous two — a **trailing** window,
      so it is causal and streamable with a ring buffer (the first
      `--mean-win - 1` outputs are averages of fewer frames, since the window
      has not filled yet — frame 1 is unaveraged, frame 2 is a mean of 2, and
      so on). Suppresses photon noise and thereby increases the contrast of
      slow-moving structures (e.g. crystals) against the fluctuating
      background, without mixing in future frames. `--mean-win 1` disables it;
      a single-image input is processed alone either way;
   2. **contrast stretch**: a 1%/99% percentile stretch whose endpoints are
      computed per output frame — this adapts automatically to slow
      background drift (beam decay, liquid-composition change during
      solidification);
   3. **Gaussian blur** (`--sigma`, default 3);
   4. write as 8-bit PNG, named `<input stem>.png`.

Frames are registered in parallel across worker processes, and output PNGs
appear one by one while the run proceeds rather than all at the end — with
`--mean-win 1` each frame is written the moment its own registration finishes
(in whatever order workers finish); with the default trailing average, results
are consumed in frame order through the ring buffer instead, so a window is
only written once every frame it needs is ready.

Nothing but the output PNGs is written to disk by default (peak memory is
roughly 150 MB per worker for the registration accumulators, plus another
~38 MB per in-flight frame when `--mean-win` > 1, since each worker has to
pickle its result back to the main process to feed the ring buffer). `--cache`
additionally keeps every frame's registration result in `<output>/_corr_cache/`
(~38 MB per frame at 4096×2304, so ~5 GB for a 136-frame sequence).

The output folder contains nothing but the PNGs. A frame whose PNG already
exists is skipped, which is what makes an interrupted run resumable — so if you
change `--sigma`, `--mean-win` or `--ref` and want the images regenerated,
delete them first (an existing PNG is never rewritten, whatever the settings).
With `--cache`, a `recipe.json` inside `_corr_cache/` records the reference
files and tiling the cached registrations came from, and reusing that cache
with a different `--ref`, `--tile` or `--tile-step` is refused with an error
naming what changed — a stale flat would otherwise silently produce wrong
images.

## Part 2 — Requirement installation

Python 3.9 or newer:

```bash
pip install -r requirements.txt
```

## Part 3 — Usage

```bash
python flatfield_correct.py --ref path/to/reference_images \
    --input path/to/frames --output path/to/output
```

`--ref` and `--input` each take either a single image or a folder of images.
(On Windows PowerShell the line-continuation character is `` ` ``, not `\`.)

Inputs may be 8-bit or 16-bit TIFF/PNG/JPG/BMP (any mix; images are normalized
by their dtype range). All images and the reference must share one resolution.
Outputs are named `<input stem>.png`.

## Part 4 — Options

| Option | Default | Meaning |
|---|---|---|
| `--ref` | `REF_PATH` constant | reference image or folder; mean = flat |
| `--input` | `INPUT_PATH` constant | image or folder to correct |
| `--output` | `OUTPUT_DIR` constant | output folder |
| `--mean-win` | 3 | trailing average over the last N frames (1 = off) |
| `--workers` | 14 | parallel CPU worker processes |
| `--tile` | 512 | registration tile size (px) |
| `--tile-step` | 384 | tile stride; overlap = tile − step |
| `--sigma` | 3.0 | final Gaussian sigma |
| `--cache` | off | keep per-frame registration results on disk (resume + cheap re-finish, ~38 MB per frame) |

The three path constants sit at the top of the script — edit them once for your
data and run without arguments, or pass the paths on the command line.

Registration internals (landmark spacing, template size, TPS smoothing, …) are
constants at the top of the script with comments.

## Part 5 — Step-by-step visualization

`visualize_steps.py` runs the full pipeline on ONE chosen frame and writes an
image per stage, plus a combined `overview.png`:

```bash
python visualize_steps.py --ref path/to/reference_images \
    --input path/to/frames --frame 1352 --output path/to/viz_output
```

Stages: raw frame → flat → unregistered division (for comparison) → registered
division → trailing temporal mean → final output. Comparing stages 3 and 4
shows what the tiled registration removes; 4 vs 5 shows the noise suppression
of the temporal average. `--frame` matches the trailing number or the full
filename stem. This script always caches the few frames it registers (and reads
a cache left by a `--cache` run), so re-running it on the same frame — for
example to try another `--sigma` — finishes in seconds.

For what the output should look like, see this
[example step visualization](https://drive.google.com/file/d/1cvfTFywiSHQJAZNRRIi7SDGZOswF8v1-/view?usp=sharing)
(produced from the example data below).

## Part 6 — Example data

A complete example sequence (synchrotron X-ray radiography of an alloy
solidifying, 4096×2304, 16-bit TIFF) is available for reproducing the results
and for checking that an online/live deployment produces the same output as
this offline pipeline:

| Download | Content |
|---|---|
| ⬇️ [raw frames](https://drive.google.com/file/d/1BCOltXXZoOp27af25FTfnHwi9R1bf2yo/view?usp=sharing) | 136 raw frames of the sequence to correct (`--input`) |
| ⬇️ [reference frames](https://drive.google.com/file/d/1P5OU06zykZvwLtrJxTPSdKOunnilt6tU/view?usp=sharing) | 21 pre-nucleation frames used as the flat (`--ref`) |
| ⬇️ [corrected output](https://drive.google.com/file/d/1mhhfK7Az4h9EDcxv_HL67LMnleEIrnA8/view?usp=sharing) | the 136 corrected PNGs this pipeline produces from the two above, at default settings |

To reproduce that output, unzip the first two archives and run:

```bash
python flatfield_correct.py --ref raw_images_0056_frame0~20/ \
    --input raw_images_0056/ --output corrected/
```

The PNGs in `corrected/` should be byte-identical to the corrected-output
archive. That makes the archive a regression fixture: any other implementation
of this pipeline — an online version running during acquisition, a port to
another language — can be validated by processing the same frames and comparing
against it. To inspect a single frame stage by stage instead, run
`visualize_steps.py` on the same data (see above).
