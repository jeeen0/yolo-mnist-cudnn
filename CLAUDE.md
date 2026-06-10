# yolo-mnist-cudnn — project brain

This file is read by Claude Code at the start of every session on the Orin.
It defines the goal, the hard constraints, where the score comes from, how to
measure, and what "done" means. When asked to "read CLAUDE.md and reach the
done criteria," self-correct against the loop at the bottom.

## Goal
Detect handwritten digits in a video with YOLO, save each as a 28x28 PGM, and
classify them with the mnistCUDNN example — maximizing **recognition accuracy**
and **minimizing latency**, which together are 80% of the (relative) grade.
Specifically: keep the example's 1/3/5 samples correct while making newly
collected '6' and '8' samples recognized.

## Grading spec (official notice — authoritative)
- The clip shows **10~15 digits one after another** (e.g. 6 for 2s, 8 for 3s,
  8 for 1.5s, 6 for 2s). YOLO must save **exactly ONE PGM per digit APPEARANCE**,
  not one per frame — N digits shown -> N PGMs, in order. The example clip
  `clips/number.mp4` pans across a whiteboard with NO blank gaps (the next digit
  slides in as the previous slides out), so appearances are split on a detection
  gap **OR a box-center horizontal jump** (see `scripts/03_pipeline.py`).
- PGM filename convention: `frame_<frameidx:06d>_digit_<lab>_conf_<c.cc>_<seq:04d>.pgm`
  (single-class YOLO doesn't know the digit, so `<lab>` is a placeholder `x`; the
  graded value is mnistCUDNN's classification, not the name).
- Evaluation = run mnistCUDNN over the PGMs (the first 10~15, i.e. the number of
  digits shown) and print this EXACT format:
  ```
  INPUT: <pgm filename>
  Result of classification: <digit>
  ... (one block per image)
  Total Images : <N>
  Total Time   : <sec>     # MNIST-only in-program time, summed
  Digit 0 : <count> ... Digit 9 : <count>
  ```
  Implemented as the `--dir=<folder> [--limit=N]` harness in `mnistCUDNN.cpp`.

## Hard constraints (do NOT violate)
1. **Do not change the 9-stage forward structure or its order** in mnistCUDNN.
   Optimize *around* it (algorithm choice, precision, context reuse, timers),
   never the network shape or stage sequence.
2. **1/3/5 must stay correct.** Every change is gated by the 1/3/5 regression
   check in `scripts/04_eval.py`. If it fails, the change is rejected.
3. If fine-tuning weights (reserve path), the LeNet shape and the 8 `.bin`
   layout must match the example exactly (see `finetune/finetune_lenet.py`).
   Back up the original `.bin` before overwriting.

## Where the score comes from (effort priority)
- **Preprocessing / MNIST normalization (`preprocess.py`)** — biggest accuracy
  lever. White digit on black, stroke thickened, scaled to 20x20, centered by
  center-of-mass into 28x28. Tune `THICKEN_FRAC` and `CLOSE_K` for faint scans.
- **Latency patches in mnistCUDNN** — remove per-run algorithm search (hardcode
  the fastest algo for the fixed 28x28 input), run a single precision (try FP16
  on Orin tensor cores), reuse handles/descriptors/workspace/device buffers
  across images, use pinned memory for H2D. Then print `LATENCY_MS=<ms>`.
- **YOLO** — single class "digit"; detect + crop reliably AND emit one PGM per
  appearance (the dedup logic in `03_pipeline.py` is part of the score).
- **Fine-tuning** — done (`finetune/finetune_lenet.py`): the main lever that lifted
  6/8. Always honor the 1/3/5 gate; ship only after the Orin 04 check.

## Files
- `preprocess.py` — MNIST normalization; shared by labeling + pipeline.
- `scripts/01_make_dataset.py` — auto-label photos -> YOLO dataset.
- `scripts/02_train_yolo.py` — train + export (DESKTOP GPU).
- `scripts/03_pipeline.py` — video -> YOLO -> crop -> 28x28 PGM (Orin); ONE PGM
  per digit appearance (gap- OR box-center-jump segmentation; most-centered frame).
- `scripts/04_eval.py` — classify PGMs, per-digit accuracy + latency, 1/3/5 gate.
- `finetune/finetune_lenet.py` — fine-tune LeNet -> 8 `.bin` (init from original,
  freeze conv1/conv2, augment our 6/8, ship gate). `finetune/eval_extra.py` —
  OOD check on USPS/EMNIST/ARDIS.
- `mnistCUDNN/mnistCUDNN.cpp` — patched: `LATENCY_MS=`, hardcoded conv algo, and
  the `--dir` spec-format harness. `mnistCUDNN/` also has Makefile + `.bin` + PGMs.

## Build / run
- Desktop: `pip install -r requirements.txt`; `python scripts/01_make_dataset.py`;
  `python scripts/02_train_yolo.py`. Copy `best.pt` to the Orin.
- Orin: build the engine `yolo export model=best.pt format=engine half=True
  device=0`; build mnistCUDNN with `make` in `mnistCUDNN/`; then `./run_all.sh`.

## Latency measurement (source of truth)
Latency must be the **program's own reported time including H2D/D2H and I/O**.
The patched mnistCUDNN must print `LATENCY_MS=<float>` covering load->classify
(wrap `classify_example` start..end, `cudaDeviceSynchronize()` before stopping).
`04_eval.py` parses that token; until it exists it falls back to wall-clock and
prints a WARNING — that fallback is a placeholder, not a score. Measure with
several repeats and report the median (warmup-robust).

## Confirmed facts from the lab slides (lab10) — drive the mnistCUDNN patches
- Default run does BOTH `Testing single precision` and `Testing half precision`,
  classifies one/three/five (hardcoded), prints `Result of classification: 1 3 5`
  then `Test passed!`. Single-image path: `--image=<pgm>` (checkCmdLineFlag
  "image", slide 866) prints `Result of classification: <one digit>`.
- `convoluteForward` runs `cudnnFindConvolutionForwardAlgorithm` every call when
  `convAlgorithm < 0` (prints `Testing cudnnFindConvolutionForwardAlgorithm`).
  Input is fixed 28x28 -> hardcode the fastest algo to kill the search.
- There is currently NO latency in the output. Must add `LATENCY_MS=`.
- The 9 stages: convolute, pool, convolute, pool, fullyConnected, activation
  (ReLU), lrn, fullyConnected, softmax. DO NOT reorder.
- Full patch instructions with exact spots: `docs/mnistCUDNN_patch_guide.md`.

## Latency levers (in priority order, all leave the 9 stages intact)
1. Add `LATENCY_MS=` timer (patch 1) — required to even have a score.
2. Hardcode conv algorithm, skip the per-run search (patch 2) — biggest win.
3. Run ONE precision, not both FP32+FP16 (patch 3) — measure which is faster.
4. (reserve) reuse handles/buffers, pinned memory (patch 4) — for multi-image.

## Current status (desktop work done; Orin validation pending)
- `02` -> `best.pt` (single-class digit detector, mAP50 ~0.995). Engine builds on Orin.
- Fine-tune done: `finetune/weights/*.bin` (v3 = init-from-original + conv freeze +
  augmentation). Proxy + OOD (USPS/EMNIST/ARDIS): 6 improves everywhere, 8 mostly,
  1/3/5 preserved. NOT yet applied to `mnistCUDNN/data` — Orin backs up + swaps + 04 gate.
- `03` appearance-dedup validated on `clips/number.mp4`: 4 appearances -> 4 PGMs
  reading 5,6,5,6. Thresholds (`--gap/--jump/--conf`) tuned to that clip.
- `mnistCUDNN.cpp` patched (LATENCY_MS, hardcoded algo, `--dir` harness). NOT yet
  compiled here (no cuDNN on desktop) — Orin `make` is the real check.

## Done criteria
- `01` builds the dataset; `02` produces `best.pt`; Orin builds `best.engine`.
- `03` turns a multi-digit clip into ONE correct-looking 28x28 PGM per appearance
  (count == number of digits shown), spec filename.
- mnistCUDNN `--dir` prints the spec format (INPUT/Result per image, Total Images,
  Total Time, per-digit counts) with a real `LATENCY_MS`-based time (no wall-clock).
- `04` gate: 1/3/5 still all correct AND 6/8 recognized (per-digit table in
  `runtime/results/results.csv`).

## Work loop (self-correct)
1. `python scripts/03_pipeline.py --video clips/<clip>.mp4 --weights best.engine`
   -> one PGM per appearance in `runtime/pgm`. Sanity-check the count == digits shown.
2. `./mnistCUDNN/mnistCUDNN --dir=runtime/pgm --limit=<#digits>` for the spec output;
   `scripts/04_eval.py` for the 1/3/5 gate + per-digit/latency table.
3. If a digit misclassifies -> tune `preprocess.py` knobs or re-run the fine-tune.
   If PGM count is wrong -> tune `03`'s `--gap/--jump/--conf`. If latency is high ->
   next mnistCUDNN latency patch.
4. Re-run. Never ship a change that fails the 1/3/5 gate.
5. Stop when the done criteria hold.
