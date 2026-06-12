#!/usr/bin/env python3
"""Synthesize a multi-digit handwriting test clip from the real 6/8 photos.

There is only ONE real handwriting video on the box (clips/number.mp4, which
reads 5,6,5,6 and never exercises our fine-tuned 6/8). The grading clip shows
10~15 digits one after another, panning across a board with NO blank gaps. This
builds exactly that kind of clip from our own 6.zip/8.zip photos so we can test
the full 03 -> mnistCUDNN path end-to-end on the digits we actually tuned for.

How it mimics the grading clip (and number.mp4: 640x480, 30 fps):
  - each chosen photo is cropped to its digit (preprocess.digit_mask/bbox) and
    pasted, centered, onto a 640x480 white tile;
  - tiles are concatenated left-to-right into one wide canvas;
  - a 640-wide window pans across at constant speed, so one digit is centered at
    a time and the next slides in as the previous slides out (no blank gap).
    Adjacent digits therefore split on the box-CENTER horizontal jump, exactly
    the dedup branch 03_pipeline.py is built around.

The ground-truth digit order is printed and written to <out>.labels.txt so the
classification result can be checked position-by-position.
"""
import os
import sys
import glob
import argparse
import subprocess

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import digit_mask, digit_bbox  # noqa: E402

W, H = 640, 480           # match number.mp4
FPS = 30


def crop_digit(path, pad_frac=0.35):
    """Return a color crop tightly around the digit, or a center crop on failure."""
    img = cv2.imread(path)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    box = digit_bbox(digit_mask(gray))
    h, w = gray.shape
    if box is None:
        s = min(h, w)
        y0, x0 = (h - s) // 2, (w - s) // 2
        return img[y0:y0 + s, x0:x0 + s]
    x, y, bw, bh = box
    px, py = int(bw * pad_frac), int(bh * pad_frac)
    x0, y0 = max(0, x - px), max(0, y - py)
    x1, y1 = min(w, x + bw + px), min(h, y + bh + py)
    return img[y0:y1, x0:x1]


PAPER = 205   # uniform paper-gray background (NOT white) so no transition frame

def fit_digit(crop, digit_px=360):
    """Resize `crop` so its longest side is ~digit_px (keeps aspect)."""
    if crop is None or crop.size == 0:
        return np.full((digit_px, digit_px // 2, 3), PAPER, np.uint8)
    ch, cw = crop.shape[:2]
    scale = digit_px / max(ch, cw)
    nw, nh = max(1, int(cw * scale)), max(1, int(ch * scale))
    return cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)


def build_canvas(seq, pitch):
    """Lay digits on ONE continuous paper-gray strip, centers `pitch` px apart.

    Two requirements, in tension, that fix the artifacts found earlier:
      - pitch ~= W (640): exactly ONE digit is centered per window at a time.
        pitch << W puts two digits in frame so YOLO's pick flips frame-to-frame,
        the box center oscillates, and the appearance-splitter shatters one digit
        into dozens (observed: pitch 400 -> 104 PGMs for 14 digits).
      - a uniform PAPER-gray background, never white: a tile-with-white-margins
        layout produced all-white transition frames that YOLO mis-detected as
        partial/white-box spurious appearances. A continuous gray strip with the
        digit's own paper blended in has no such frame.
    This matches the grading clip: pan across a board, next digit slides in as the
    previous slides out, no blank gaps.
    """
    crops = [fit_digit(crop_digit(p)) for _, p in seq]
    lead = W // 2                          # so digit 0 starts centered at offset 0
    width = lead * 2 + pitch * (len(seq) - 1)
    canvas = np.full((H, width, 3), PAPER, np.uint8)
    for i, cr in enumerate(crops):
        nh, nw = cr.shape[:2]
        cx = lead + i * pitch
        x0, y0 = cx - nw // 2, (H - nh) // 2
        canvas[y0:y0 + nh, x0:x0 + nw] = cr
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/hw_src",
                    help="dir with 6/ and 8/ subfolders of jpgs")
    ap.add_argument("--out", default="clips/handwriting_test.mp4")
    ap.add_argument("--order", default="6,8,6,8,8,6,8,6,6,8,6,8",
                    help="comma digit order; each consumes the next photo of that class")
    ap.add_argument("--secs-per-digit", type=float, default=2.5)
    ap.add_argument("--pitch", type=int, default=620,
                    help="px between digit centers (~W => one centered digit at a time)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    pools = {}
    for d in ("6", "8"):
        files = sorted(glob.glob(os.path.join(args.src, d, "*.jpg")))
        rng = np.random.RandomState(args.seed)
        rng.shuffle(files)
        pools[d] = files
    idx = {"6": 0, "8": 0}

    order = [t.strip() for t in args.order.split(",") if t.strip()]
    seq = []
    for d in order:
        if idx[d] >= len(pools[d]):
            print(f"ran out of '{d}' photos", file=sys.stderr); sys.exit(1)
        seq.append((d, pools[d][idx[d]])); idx[d] += 1

    print(f"building {len(seq)}-digit clip, order = {' '.join(order)}")
    canvas = build_canvas(seq, args.pitch)

    frames_per_digit = int(args.secs_per_digit * FPS)
    n_steps = (len(seq) - 1) * frames_per_digit + 1   # land exactly on last digit
    max_off = args.pitch * (len(seq) - 1)             # offset where last digit is centered
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(args.out, fourcc, FPS, (W, H))
    for k in range(n_steps):
        off = int(round(max_off * k / max(1, n_steps - 1)))
        vw.write(canvas[:, off:off + W])
    vw.release()

    # re-encode to H.264 yuv420p so every player/decoder (incl. OpenCV) is happy
    tmp = args.out + ".tmp.mp4"
    os.replace(args.out, tmp)
    r = subprocess.run(["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", args.out],
                       capture_output=True, text=True)
    os.remove(tmp) if r.returncode == 0 else os.replace(tmp, args.out)

    labels_path = args.out + ".labels.txt"
    with open(labels_path, "w") as f:
        f.write(" ".join(order) + "\n")
    print(f"wrote {args.out} ({n_steps} frames, {n_steps/FPS:.1f}s) and {labels_path}")
    print(f"ground truth: {' '.join(order)}")


if __name__ == "__main__":
    main()
