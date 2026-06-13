"""
LIVE CAMERA demo — the SAME assignment-notice pipeline as the video path, but
fed by a webcam instead of a clip file (for an in-class live demonstration).

Flow (identical to the grading spec, see ../CLAUDE.md):
  camera frame -> YOLO detect "digit" -> crop -> MNIST normalize -> 28x28 PGM,
  ONE PGM PER DIGIT APPEARANCE, spec filename
  frame_<frameidx:06d>_digit_x_conf_<c.cc>_<seq:04d>.pgm
then mnistCUDNN classifies the saved PGMs and prints the official spec format
(INPUT / Result of classification / Total Images / Total Time / per-digit).

The appearance segmentation, frame-pick and preprocessing are REUSED verbatim
from scripts/03_pipeline.py + preprocess.py, so the live path produces byte-for-
byte the same PGMs the graded video path would for the same digit — nothing about
the scored pipeline changes; only the frame source does.

Two demo modes:
  (default, auto)   show a digit to the camera; when it is held steadily then
                    removed (or you slide in the next), one PGM is saved and the
                    digit is classified on-screen instantly. Mirrors the
                    "one PGM per appearance" rule the clip path uses.
  --manual          YOLO box is drawn live; press SPACE to capture+classify the
                    current frame on demand (you control the timing).

On-screen feedback is instant: a persistent `mnistCUDNN --daemon` process loads the
weights+CUDA context once and classifies each frame in ~3ms (vs ~325ms cold per
--image), so the shown digit tracks what is held in real time. When you quit
(press q / ESC), it ALSO runs mnistCUDNN --dir over the whole saved folder and
prints the official assignment-notice format — that --dir output is the graded one.

  python live/live_demo.py                      # auto mode, default camera 0
  python live/live_demo.py --manual --cam 1     # spacebar capture, camera 1
  python live/live_demo.py --no-display         # headless (console only)

YOLO runs on CPU here (torch has no CUDA on this Orin), so high-res webcams can
lag; --cam-w/--cam-h and --stride keep it responsive.
"""

import os
import re
import sys
import glob
import time
import tempfile
import argparse
import subprocess

import cv2

# Reuse the EXACT scored pipeline pieces — no reimplementation, no drift.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from preprocess import normalize_to_mnist, save_pgm  # noqa: E402
import importlib                            # noqa: E402
_pipe = importlib.import_module("03_pipeline")  # module name starts with a digit
AppearanceSegmenter = _pipe.AppearanceSegmenter
best_detection = _pipe.best_detection
crop = _pipe.crop
write_pick = _pipe.write_pick
load_model = _pipe.load_model
_pick_score = _pipe._pick_score

PRED_RE = re.compile(r"classification:\s*([0-9 ]+)")
LAT_RE = re.compile(r"LATENCY_MS\s*=\s*([0-9]*\.?[0-9]+)")


class LiveSegmenter:
    """Live-camera appearance splitter that ALSO splits on a digit-content change.

    The clip AppearanceSegmenter (scripts/03_pipeline.py) splits an appearance on
    a detection GAP or a box-center JUMP. On a handheld whiteboard demo you often
    erase a digit and write the next one in the SAME spot: no clean gap (YOLO
    keeps firing on the half-erased strokes) and no center jump (same place). The
    clip logic would then merge both into ONE appearance -> the new digit gets no
    PGM and the screen stays frozen on the old digit.

    So here we additionally classify the held digit on a debounce: when the
    recognised digit CHANGES (confirmed over `confirm` reads) we close the
    previous run (saving its best frame) and start a fresh one. Classification is
    sparse -- every `live_every` processed frames -- because each --image call
    reloads the mnistCUDNN process; that is fast enough to follow a hand writing,
    not a per-frame cost. `classify_fn(norm)->str|None` is injected so this stays
    unit-testable without the binary. update() returns (display_digit, pick|None);
    `pick` is a save dict {seq,fidx,conf,norm} exactly like the clip segmenter.
    """

    def __init__(self, classify_fn, gap=8, jump=0.35, live_every=5,
                 confirm=2, min_present=2):
        self.classify_fn = classify_fn
        self.gap = gap
        self.jump = jump
        self.live_every = live_every
        self.confirm = confirm
        self.min_present = min_present
        self.seq = 0
        self._reset()

    def _reset(self):
        self.active = False
        self.best_norm = None
        self.best_conf = -1.0
        self.best_fidx = -1
        self.best_score = -1e9
        self.present = 0
        self.miss = 0
        self.last_cx = None
        self.run_digit = None        # confirmed digit of the open run
        self.since_cls = 0
        self.pending_d = None
        self.pending_n = 0

    def _close(self):
        pick = None
        if self.best_norm is not None and self.present >= self.min_present:
            self.seq += 1
            pick = {"seq": self.seq, "fidx": self.best_fidx,
                    "conf": self.best_conf, "norm": self.best_norm}
        self._reset()
        return pick

    def _start_run_from(self, fidx, conf, cx, norm, digit):
        """Begin a new run seeded with the CURRENT frame (used after a split)."""
        self.active = True
        self.present = 1
        self.miss = 0
        self.last_cx = cx
        self.run_digit = digit
        self.since_cls = 0
        self.pending_d = None
        self.pending_n = 0
        fg = float((norm > 40).mean())
        self.best_norm = norm
        self.best_conf = conf
        self.best_fidx = fidx
        self.best_score = _pick_score(cx, conf, fg, 0.0)

    def update(self, fidx, conf, cx, norm, bh=0.0):
        # ---- nothing detected this frame ----
        if norm is None:
            if self.active:
                self.miss += 1
                if self.miss >= self.gap:
                    return self.run_digit, self._close()
            return self.run_digit, None

        # ---- digit present ----
        # (1) hard split: box center teleported (slide-in of a new digit)
        boundary = None
        if (self.active and cx is not None and self.last_cx is not None
                and abs(cx - self.last_cx) > self.jump):
            boundary = self._close()
            self._start_run_from(fidx, conf, cx, norm, None)
            return self.run_digit, boundary

        self.active = True
        self.miss = 0
        self.present += 1
        self.last_cx = cx
        # track the best-looking frame of the run (same composite as the clip path)
        fg = float((norm > 40).mean())
        score = _pick_score(cx, conf, fg, bh)
        if score > self.best_score:
            self.best_score = score
            self.best_conf, self.best_norm, self.best_fidx = conf, norm, fidx

        # (2) sparse classification -> drives the on-screen digit AND content-split
        self.since_cls += 1
        if self.run_digit is None or self.since_cls >= self.live_every:
            self.since_cls = 0
            d = self.classify_fn(norm)
            if d is not None:
                if self.run_digit is None:
                    self.run_digit = d                 # first attribution
                elif d != self.run_digit:
                    # confirm the change over `confirm` reads to ignore a single
                    # misread, then split: save the old run, open a new one here.
                    if self.pending_d == d:
                        self.pending_n += 1
                    else:
                        self.pending_d, self.pending_n = d, 1
                    if self.pending_n >= self.confirm:
                        pick = self._close()
                        self._start_run_from(fidx, conf, cx, norm, d)
                        return d, pick
                else:
                    self.pending_d, self.pending_n = None, 0
        return self.run_digit, None

    def finalize(self):
        return self._close() if self.active else None


def classify_one(binary, pgm):
    """Classify a single PGM via the binary's --image path. Returns (digit|None)."""
    if not os.path.exists(binary):
        return None
    try:
        proc = subprocess.run([binary, f"--image={pgm}"],
                              capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    out = proc.stdout + "\n" + proc.stderr
    hits = PRED_RE.findall(out)
    if hits and hits[-1].split():
        return hits[-1].split()[0]
    return None


class PersistentClassifier:
    """Long-lived mnistCUDNN --daemon process: pay the ~325ms CUDA/cuDNN cold-start
    ONCE, then classify each PGM in ~3ms (measured 100x faster than per-call
    --image). This is what lets the live loop classify EVERY frame in real time
    instead of throttling to one slow call every few frames. Falls back to the
    cold per-call path if the daemon can't start, so the demo still runs anywhere.
    """

    def __init__(self, binary):
        self.binary = binary
        self.proc = None
        self.tmp = os.path.join(tempfile.gettempdir(), "live_daemon_tmp.pgm")
        if not os.path.exists(binary):
            return
        try:
            self.proc = subprocess.Popen(
                [binary, "--daemon"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, bufsize=1)
            for line in self.proc.stdout:           # skip init chatter until READY
                if line.strip() == "READY":
                    break
            else:
                self.proc = None
        except Exception:
            self.proc = None

    @property
    def ok(self):
        return self.proc is not None and self.proc.poll() is None

    def classify_path(self, path):
        if not self.ok:
            return classify_one(self.binary, path)   # fallback: cold subprocess
        try:
            self.proc.stdin.write(path + "\n"); self.proc.stdin.flush()
            for line in self.proc.stdout:            # read until the result line
                if line.startswith("DIGIT="):
                    d = line.strip().split("=", 1)[1]
                    return d if d not in ("-1", "") else None
            self.proc = None
            return None
        except Exception:
            self.proc = None
            return classify_one(self.binary, path)

    def classify(self, norm):
        save_pgm(norm, self.tmp)
        return self.classify_path(self.tmp)

    def close(self):
        if self.ok:
            try:
                self.proc.stdin.write("quit\n"); self.proc.stdin.flush()
                self.proc.wait(timeout=3)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass


def run_dir_spec(binary, pgm_dir, limit):
    """Run the official --dir spec harness over the saved folder and echo it."""
    if not os.path.exists(binary):
        print("\n[skip] binary not found, cannot run the --dir spec output:", binary)
        return
    cmd = [binary, f"--dir={pgm_dir}"]
    if limit > 0:
        cmd.append(f"--limit={limit}")
    print("\n" + "=" * 60)
    print("OFFICIAL spec output (mnistCUDNN --dir):  " + " ".join(cmd))
    print("=" * 60)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout
    # Echo every spec line; hide the noisy device/setup chatter.
    for line in out.splitlines():
        if (line.startswith("INPUT:") or line.startswith("Result of classification:")
                or line.startswith("Total ") or line.startswith("Digit ")
                or line.strip() == ""):
            print(line)
    m = LAT_RE.search(out)
    if m:
        print(f"LATENCY_MS={m.group(1)}")


# ---- on-screen drawing helpers -------------------------------------------
FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_hud(frame, xyxy, conf, last_digit, count, mode, banner=None):
    h, w = frame.shape[:2]
    if xyxy is not None:
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"digit {conf:.2f}", (x1, max(20, y1 - 8)),
                    FONT, 0.6, (0, 255, 0), 2)
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 0), -1)
    cv2.putText(frame, f"mode:{mode}  saved:{count}", (10, 27),
                FONT, 0.7, (255, 255, 255), 2)
    hint = "SPACE=capture  q/ESC=quit" if mode == "manual" else "q/ESC=quit"
    cv2.putText(frame, hint, (w - 360, 27), FONT, 0.55, (180, 180, 180), 1)
    if last_digit is not None:
        cv2.putText(frame, str(last_digit), (w - 110, h - 20),
                    FONT, 3.0, (0, 255, 255), 6)
        cv2.putText(frame, "last", (w - 120, h - 110),
                    FONT, 0.6, (0, 255, 255), 2)
    if banner:
        cv2.rectangle(frame, (0, h - 40), (w, h), (40, 40, 40), -1)
        cv2.putText(frame, banner, (10, h - 13), FONT, 0.6, (0, 255, 0), 2)
    return frame


def open_cam(idx, cw, ch):
    cap = cv2.VideoCapture(idx)
    if cw:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cw)
    if ch:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, ch)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.path.join(ROOT, "best.pt"),
                    help="YOLO weights (.engine if built on the Orin, else best.pt)")
    ap.add_argument("--bin", default=os.path.join(ROOT, "mnistCUDNN", "mnistCUDNN"),
                    help="patched mnistCUDNN classifier binary")
    ap.add_argument("--out", default=os.path.join(ROOT, "runtime", "live_pgm"),
                    help="folder for the saved appearance PGMs")
    ap.add_argument("--cam", type=int, default=0, help="camera index (/dev/videoN)")
    ap.add_argument("--cam-w", type=int, default=1280)
    ap.add_argument("--cam-h", type=int, default=720)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--stride", type=int, default=1,
                    help="run YOLO every Nth frame (>=2 if the CPU detector lags)")
    ap.add_argument("--gap", type=int, default=8,
                    help="absent processed-frames that end an appearance")
    ap.add_argument("--min-present", type=int, default=2)
    ap.add_argument("--jump", type=float, default=0.35,
                    help="normalized box-center jump that ends an appearance")
    ap.add_argument("--live-every", type=int, default=1,
                    help="re-classify the held digit every Nth processed frame "
                         "(1 = every frame; cheap via the daemon)")
    ap.add_argument("--confirm", type=int, default=2,
                    help="consecutive reads of a new digit needed to switch/split "
                         "(higher = steadier, lower = snappier)")
    ap.add_argument("--label", default="x",
                    help="digit stamped in the PGM filename (placeholder)")
    ap.add_argument("--manual", action="store_true",
                    help="SPACE captures the current frame instead of auto-appearance")
    ap.add_argument("--no-display", action="store_true",
                    help="headless: no window, log to console (auto mode only)")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe the out folder before starting")
    ap.add_argument("--limit", type=int, default=0,
                    help="--dir limit at the end (0 = all saved)")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        print("weights not found:", args.weights); sys.exit(1)
    os.makedirs(args.out, exist_ok=True)
    if args.fresh:
        for p in glob.glob(os.path.join(args.out, "*.pgm")):
            os.remove(p)

    print("loading YOLO:", args.weights)
    model = load_model(args.weights)
    cap = open_cam(args.cam, args.cam_w, args.cam_h)
    if not cap.isOpened():
        print("could not open camera index", args.cam); sys.exit(1)

    headless = args.no_display
    if not headless:
        try:
            cv2.namedWindow("live", cv2.WINDOW_NORMAL)
        except Exception:
            print("[warn] no GUI available, falling back to --no-display")
            headless = True

    mode = "manual" if args.manual else "auto"
    # A digit held in front of the camera is classified on a debounce. That live
    # digit (a) drives the on-screen number so it always reflects what is CURRENTLY
    # shown, and (b) lets the segmenter split a same-spot rewrite (erase 3, write 2
    # in place) that has neither a detection gap nor a center jump — see
    # LiveSegmenter. Classification goes through a PERSISTENT daemon (~3ms/call)
    # so we can classify every frame in real time; cold per-call would be ~325ms.
    clf = PersistentClassifier(args.bin)
    print(f"   classifier: {'daemon (~3ms/img)' if clf.ok else 'cold --image fallback (~325ms/img)'}")

    seg = LiveSegmenter(clf.classify, gap=args.gap, jump=args.jump,
                        live_every=args.live_every, confirm=args.confirm,
                        min_present=args.min_present)
    fidx = 0
    saved = 0
    last_digit = None
    banner = None
    banner_until = 0.0

    def handle_pick(pick, screen_digit=None):
        # Save the appearance PGM (spec) + classify the SAVED file for the
        # console/banner. The on-screen number is owned by the live classification
        # (screen_digit), so we don't overwrite it here unless none was given
        # (manual mode, which has no live path).
        nonlocal saved, last_digit, banner, banner_until
        p = write_pick(pick, args.out, args.label)
        saved += 1
        digit = clf.classify_path(p)
        if screen_digit is None:
            last_digit = digit if digit is not None else last_digit
        banner = f"saved #{pick['seq']:02d} {os.path.basename(p)} -> {digit}"
        banner_until = time.time() + 2.5
        print(f"  appearance {pick['seq']:02d}: frame {pick['fidx']} "
              f"conf {pick['conf']:.2f} -> {os.path.basename(p)}  digit={digit}")

    print(f"== LIVE {mode} mode ==  out={args.out}")
    print("   show a digit to the camera; q/ESC to finish and print the spec output")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera read failed"); break

            xyxy, c, cx = (None, 0.0, None)
            if fidx % args.stride == 0:
                xyxy, c, cx = best_detection(model, frame, args.conf)

            if not args.manual:
                # AUTO: live segmentation — splits on gap, center-jump, OR a
                # change in the recognised digit (same-spot rewrite). The live
                # digit owns the screen; pick (when a run closes) is saved.
                if fidx % args.stride == 0:
                    norm = normalize_to_mnist(crop(frame, xyxy)) if xyxy is not None else None
                    bh = (xyxy[3] - xyxy[1]) / frame.shape[0] if xyxy is not None else 0.0
                    disp, pick = seg.update(fidx, c, cx, norm, bh)
                    if disp is not None:
                        last_digit = disp
                    if pick:                       # a run closed -> SAVE its best
                        handle_pick(pick, screen_digit=last_digit)

            if headless:
                fidx += 1
                # headless auto mode runs until Ctrl-C
                continue

            if banner and time.time() > banner_until:
                banner = None
            draw_hud(frame, xyxy, c, last_digit, saved, mode, banner)
            cv2.imshow("live", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):           # q or ESC
                break
            if args.manual and key == ord(" "):
                if xyxy is not None:
                    norm = normalize_to_mnist(crop(frame, xyxy))
                    if norm is not None:
                        seg.seq += 1
                        handle_pick({"seq": seg.seq, "fidx": fidx,
                                     "conf": c, "norm": norm})
                    else:
                        banner = "no clean digit in crop — try again"
                        banner_until = time.time() + 1.5
                else:
                    banner = "no digit detected — hold it steady"
                    banner_until = time.time() + 1.5
            fidx += 1
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if not args.manual:
            last = seg.finalize()
            if last:
                handle_pick(last)
        cap.release()
        clf.close()
        if not headless:
            cv2.destroyAllWindows()

    print(f"\nsaved {saved} appearance PGM(s) to {args.out}")
    if saved:
        run_dir_spec(args.bin, args.out, args.limit)


if __name__ == "__main__":
    main()
