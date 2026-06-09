"""
Video -> YOLO detect -> crop -> MNIST normalize -> 28x28 PGM.

Runs on the Orin. For each input video it scans frames, lets the YOLO engine
find the digit box, crops it, normalizes with preprocess.normalize_to_mnist,
and writes a 28x28 PGM that mnistCUDNN can classify.

To avoid dumping one PGM per frame, we keep only the most confident detection
per video by default (the grading clips show one digit). Use --all-frames to
emit every accepted frame instead.

  python scripts/03_pipeline.py --video clip.mp4 --weights best.engine \
      --out runtime/pgm

Falls back to best.pt if no .engine is given (slower, but works anywhere).
"""

import os
import sys
import glob
import argparse

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import normalize_to_mnist, save_pgm  # noqa: E402


def load_model(weights):
    from ultralytics import YOLO
    return YOLO(weights)


def best_detection(model, frame, conf):
    """Return (box_xyxy, confidence) of the top detection, or (None, 0)."""
    res = model.predict(frame, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None, 0.0
    boxes = res.boxes
    i = int(boxes.conf.argmax())
    xyxy = boxes.xyxy[i].tolist()
    return xyxy, float(boxes.conf[i])


def crop(frame, xyxy, pad_frac=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    x1 = max(0, int(x1 - px)); y1 = max(0, int(y1 - py))
    x2 = min(w, int(x2 + px)); y2 = min(h, int(y2 + py))
    return frame[y1:y2, x1:x2]


def process_video(model, path, out_dir, conf, all_frames, stride):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("could not open", path)
        return 0

    stem = os.path.splitext(os.path.basename(path))[0]
    written = 0
    best_conf, best_img = 0.0, None
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % stride == 0:
            xyxy, c = best_detection(model, frame, conf)
            if xyxy is not None:
                norm = normalize_to_mnist(crop(frame, xyxy))
                if norm is not None:
                    if all_frames:
                        p = os.path.join(out_dir, f"{stem}_f{fidx:05d}.pgm")
                        save_pgm(norm, p)
                        written += 1
                    elif c > best_conf:
                        best_conf, best_img = c, norm
        fidx += 1
    cap.release()

    if not all_frames and best_img is not None:
        p = os.path.join(out_dir, f"{stem}.pgm")
        save_pgm(best_img, p)
        written = 1
        print(f"  {stem}: best conf {best_conf:.3f} -> {p}")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="single video file")
    ap.add_argument("--videos", help="glob, e.g. 'clips/*.mp4'")
    ap.add_argument("--weights", default="best.engine")
    ap.add_argument("--out", default="runtime/pgm")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--stride", type=int, default=2,
                    help="process every Nth frame")
    ap.add_argument("--all-frames", action="store_true")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    os.makedirs(args.out, exist_ok=True)

    vids = []
    if args.video:
        vids.append(args.video)
    if args.videos:
        vids += sorted(glob.glob(args.videos))
    if not vids:
        print("give --video or --videos")
        sys.exit(1)

    if not os.path.exists(args.weights):
        print("weights not found:", args.weights)
        sys.exit(1)

    model = load_model(args.weights)
    total = 0
    for v in vids:
        total += process_video(model, v, args.out, args.conf,
                               args.all_frames, args.stride)
    print(f"\nwrote {total} pgm(s) to {args.out}")


if __name__ == "__main__":
    main()
