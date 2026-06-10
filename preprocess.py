"""
MNIST-style 28x28 normalization for handwritten digits.

This is the single most important accuracy lever in the project. LeNet (the
Caffe weights mnistCUDNN ships with) already knows 0-9; the reason raw photos
of '6'/'8' get misclassified is that the saved 28x28 does not look like an
MNIST sample. This module turns a photo (or a YOLO-cropped frame) into the
exact distribution LeNet expects:

  white digit on black background  (matches THRESH_BINARY_INV direction)
  stroke thickened so thin pen/pencil lines look like MNIST ink
  digit scaled to fit a 20x20 box, preserving aspect ratio
  placed by center-of-mass into the middle of a 28x28 canvas (LeCun's recipe)

Both 01_make_dataset.py (auto-labeling) and 03_pipeline.py (runtime) import
digit_mask / digit_bbox / normalize_to_mnist so the labeling logic and the
inference logic share exactly one definition of "where is the digit".

Two knobs decide most of the quality on faint scans:
  THICKEN_FRAC  - how much to dilate the stroke (relative to digit size)
  CLOSE_K       - morphological close kernel, reconnects broken strokes
"""

import cv2
import numpy as np

# ---- tuning knobs ---------------------------------------------------------
THICKEN_FRAC = 0.025   # stroke dilation as a fraction of the digit bbox size
CLOSE_K      = 3       # close kernel (px) to reconnect broken thin strokes
TARGET_BOX   = 20      # digit is scaled to fit this box inside the 28 canvas
CANVAS       = 28      # final MNIST size
# ---------------------------------------------------------------------------


def digit_mask(gray):
    """Binarize a grayscale crop into a white-digit-on-black mask.

    Uses Otsu + INV so dark ink on light paper becomes white foreground.
    A morphological close reconnects thin/broken pencil strokes.
    """
    # Light blur removes paper texture before Otsu picks a threshold.
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255,
                            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if CLOSE_K > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CLOSE_K, CLOSE_K))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def digit_bbox(mask):
    """Return (x, y, w, h) of the digit's foreground component, or None.

    The digit is the largest component that does NOT touch the crop border.
    On clean paper photos the digit is already the largest blob, but on video
    frames a phone bezel / glare strip along the edge becomes a big bright
    region after INV-threshold and out-sizes the thin digit stroke; those
    strips hug the border, so we drop border-touching components first and
    fall back to the global largest only if nothing sits fully inside.
    This same bbox is reused to write YOLO labels.
    """
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    H, W = mask.shape

    def touches_border(b):
        x, y, w, h = b
        return x <= 1 or y <= 1 or (x + w) >= (W - 1) or (y + h) >= (H - 1)

    scored = [(cv2.contourArea(c), cv2.boundingRect(c)) for c in cnts]
    inside = [t for t in scored if not touches_border(t[1])]
    pool = inside if inside else scored
    area, box = max(pool, key=lambda t: t[0])
    if area < 20:                        # reject noise-only frames
        return None
    return box


def _center_of_mass_shift(img28):
    """Shift the 28x28 so the ink center-of-mass lands on the canvas center."""
    m = cv2.moments(img28, binaryImage=False)
    if m["m00"] == 0:
        return img28
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    sx = int(round(CANVAS / 2.0 - cx))
    sy = int(round(CANVAS / 2.0 - cy))
    M = np.float32([[1, 0, sx], [0, 1, sy]])
    return cv2.warpAffine(img28, M, (CANVAS, CANVAS),
                          flags=cv2.INTER_LINEAR, borderValue=0)


def normalize_to_mnist(crop_bgr):
    """Full pipeline: BGR crop -> 28x28 uint8 MNIST-style image.

    crop_bgr is either a whole photo (one digit on paper) or a YOLO box crop.
    Returns a (28,28) uint8 array (white digit, black bg) or None if no digit.
    """
    if crop_bgr.ndim == 3:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop_bgr

    mask = digit_mask(gray)
    box = digit_bbox(mask)
    if box is None:
        return None
    x, y, w, h = box

    # Tighten to the digit, with a small margin so dilation has room.
    pad = int(0.08 * max(w, h)) + 2
    x0 = max(0, x - pad); y0 = max(0, y - pad)
    x1 = min(mask.shape[1], x + w + pad); y1 = min(mask.shape[0], y + h + pad)
    roi = mask[y0:y1, x0:x1]

    # Thicken the stroke proportionally to digit size -> MNIST ink weight.
    if THICKEN_FRAC > 0:
        ksz = max(1, int(THICKEN_FRAC * max(w, h)))
        if ksz % 2 == 0:
            ksz += 1
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
        roi = cv2.dilate(roi, k)

    # Scale longest side to TARGET_BOX, preserve aspect ratio.
    rh, rw = roi.shape
    scale = TARGET_BOX / float(max(rh, rw))
    nw = max(1, int(round(rw * scale)))
    nh = max(1, int(round(rh * scale)))
    resized = cv2.resize(roi, (nw, nh), interpolation=cv2.INTER_AREA)

    # Paste into the center of a 28x28 canvas, then refine by center-of-mass.
    canvas = np.zeros((CANVAS, CANVAS), dtype=np.uint8)
    ox = (CANVAS - nw) // 2
    oy = (CANVAS - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    canvas = _center_of_mass_shift(canvas)
    return canvas


def save_pgm(img28, path):
    """Write a binary (P5) 8-bit PGM, the format mnistCUDNN loads."""
    h, w = img28.shape
    with open(path, "wb") as f:
        f.write(b"P5\n%d %d\n255\n" % (w, h))
        f.write(img28.astype(np.uint8).tobytes())


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("usage: python preprocess.py <input_image> <output.pgm>")
        sys.exit(1)
    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    if img is None:
        print("could not read", sys.argv[1]); sys.exit(1)
    out = normalize_to_mnist(img)
    if out is None:
        print("no digit found in", sys.argv[1]); sys.exit(1)
    save_pgm(out, sys.argv[2])
    print("wrote", sys.argv[2])
