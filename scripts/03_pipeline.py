"""
Video -> YOLO detect -> crop -> MNIST normalize -> 28x28 PGM, ONE PGM PER
DIGIT APPEARANCE (the grading spec).

A single clip shows 10~15 digits one after another (e.g. 6 for 2s, then 8 for
3s, ...). A naive detector would dump one PGM per *frame* (dozens per digit).
The spec wants exactly ONE PGM per *appearance*: as many PGMs as digits shown.

How we segment appearances (single-class YOLO -- "is there a digit", not which):
  - a digit is "present" on frames where YOLO detects a box (conf >= --conf)
  - an appearance ENDS on EITHER boundary signal:
      (1) the digit is gone for --gap consecutive processed frames (a clean
          swap with a blank moment), OR
      (2) the box CENTER jumps horizontally by > --jump (the old digit pans out
          one side while the next enters the other side -- detection never
          drops, but the center teleports). This is what real pan clips do.
    Either boundary also splits two of the same digit shown back-to-back.
  - within an appearance we keep the single highest-confidence frame and write
    one PGM for it when the appearance closes.

Output filename matches the spec example:
  frame_<frameidx:06d>_digit_<lab>_conf_<conf:.2f>_<seq:04d>.pgm
Single-class YOLO doesn't know which digit it is, so <lab> is a placeholder
('x') -- the graded value is mnistCUDNN's "Result of classification", not the
name. (Pass --label to stamp a known digit, e.g. for building a labeled set.)

  python scripts/03_pipeline.py --video clips/number.mp4 --weights best.engine \
      --out runtime/pgm

Falls back to best.pt if no .engine is given (slower, works anywhere).
"""

import os
import sys
import glob
import argparse

import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess import normalize_to_mnist, save_pgm  # noqa: E402


class AppearanceSegmenter:
    """Turn a per-frame detection stream into one pick per appearance.

    Feed update(fidx, conf, norm) for every processed frame (conf=0/norm=None
    when nothing was detected). It returns a pick dict when an appearance just
    closed, else None. Call finalize() once at the end for a trailing pick.
    Pure (no I/O) so the segmentation can be unit-tested without a video.
    """

    def __init__(self, gap=8, min_present=2, min_conf=0.0, jump=0.35):
        self.gap = gap                  # absent frames that end an appearance
        self.min_present = min_present  # reject blips shorter than this
        self.min_conf = min_conf
        self.jump = jump                # box-center jump that ends an appearance
        self.seq = 0
        self._reset()

    def _reset(self):
        self.active = False
        self.best_conf = -1.0
        self.best_norm = None
        self.best_fidx = -1
        self.best_score = -1e9     # higher = better (most-centered frame)
        self.present = 0
        self.miss = 0
        self.last_cx = None

    def _close(self):
        pick = None
        if (self.best_norm is not None and self.present >= self.min_present
                and self.best_conf >= self.min_conf):
            self.seq += 1
            pick = {"seq": self.seq, "fidx": self.best_fidx,
                    "conf": self.best_conf, "norm": self.best_norm}
        self._reset()
        return pick

    def update(self, fidx, conf, cx, norm):
        if norm is not None:                       # digit present this frame
            boundary = None
            if (self.active and cx is not None and self.last_cx is not None
                    and abs(cx - self.last_cx) > self.jump):
                boundary = self._close()           # center teleported -> new digit
            self.active = True
            self.miss = 0
            self.present += 1
            # pick the MOST-CENTERED frame (digit fully in view), not max-conf:
            # conf stays high even while a digit slides off-frame, so centeredness
            # is the better "clean view" signal. cx unknown -> fall back to conf.
            score = -abs(cx - 0.5) if cx is not None else (conf - 1.0)
            if score > self.best_score:
                self.best_score = score
                self.best_conf, self.best_norm, self.best_fidx = conf, norm, fidx
            self.last_cx = cx
            return boundary
        if self.active:                            # nothing detected this frame
            self.miss += 1
            if self.miss >= self.gap:
                return self._close()
        return None

    def finalize(self):
        return self._close() if self.active else None


def load_model(weights):
    from ultralytics import YOLO
    return YOLO(weights)


def best_detection(model, frame, conf):
    res = model.predict(frame, conf=conf, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return None, 0.0, None
    boxes = res.boxes
    i = int(boxes.conf.argmax())
    xyxy = boxes.xyxy[i].tolist()
    cx = (xyxy[0] + xyxy[2]) / 2.0 / frame.shape[1]   # normalized center x
    return xyxy, float(boxes.conf[i]), cx


def crop(frame, xyxy, pad_frac=0.15):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = xyxy
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    x1 = max(0, int(x1 - px)); y1 = max(0, int(y1 - py))
    x2 = min(w, int(x2 + px)); y2 = min(h, int(y2 + py))
    return frame[y1:y2, x1:x2]


def write_pick(pick, out_dir, label):
    fn = (f"frame_{pick['fidx']:06d}_digit_{label}"
          f"_conf_{pick['conf']:.2f}_{pick['seq']:04d}.pgm")
    path = os.path.join(out_dir, fn)
    save_pgm(pick["norm"], path)
    return path


def process_video(model, path, out_dir, conf, stride, gap, min_present, jump, label):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print("could not open", path)
        return 0
    seg = AppearanceSegmenter(gap=gap, min_present=min_present, jump=jump)
    written, fidx = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx % stride == 0:
            xyxy, c, cx = best_detection(model, frame, conf)
            norm = normalize_to_mnist(crop(frame, xyxy)) if xyxy is not None else None
            pick = seg.update(fidx, c, cx, norm)
            if pick:
                p = write_pick(pick, out_dir, label)
                written += 1
                print(f"  appearance {pick['seq']:02d}: frame {pick['fidx']} "
                      f"conf {pick['conf']:.2f} -> {os.path.basename(p)}")
        fidx += 1
    cap.release()
    last = seg.finalize()
    if last:
        p = write_pick(last, out_dir, label)
        written += 1
        print(f"  appearance {last['seq']:02d}: frame {last['fidx']} "
              f"conf {last['conf']:.2f} -> {os.path.basename(p)}")
    print(f"  {os.path.basename(path)}: {written} appearance PGM(s)")
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", help="single video file")
    ap.add_argument("--videos", help="glob, e.g. 'clips/*.mp4'")
    ap.add_argument("--weights", default="best.engine")
    ap.add_argument("--out", default="runtime/pgm")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--gap", type=int, default=8,
                    help="absent processed-frames that end an appearance")
    ap.add_argument("--min-present", type=int, default=2,
                    help="reject appearances shorter than this many frames")
    ap.add_argument("--jump", type=float, default=0.35,
                    help="normalized box-center jump that ends an appearance")
    ap.add_argument("--label", default="x",
                    help="digit stamped in the filename (single-class YOLO "
                         "doesn't know it; default placeholder 'x')")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    os.makedirs(args.out, exist_ok=True)

    vids = ([args.video] if args.video else []) + \
           (sorted(glob.glob(args.videos)) if args.videos else [])
    if not vids:
        print("give --video or --videos"); sys.exit(1)
    if not os.path.exists(args.weights):
        print("weights not found:", args.weights); sys.exit(1)

    model = load_model(args.weights)
    total = 0
    for v in vids:
        total += process_video(model, v, args.out, args.conf, args.stride,
                               args.gap, args.min_present, args.jump, args.label)
    print(f"\nwrote {total} appearance PGM(s) to {args.out}")


if __name__ == "__main__":
    main()
