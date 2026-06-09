"""
Classify PGMs with mnistCUDNN, measure per-digit accuracy and latency.

Runs on the Orin. For every PGM whose filename starts with its true label
(e.g. 6_0007.pgm, 8_0031.pgm, or the YOLO-pipeline outputs you rename), this:

  1. invokes the mnistCUDNN binary on the PGM
  2. parses the predicted class and the reported latency
  3. records correctness + latency
  4. enforces the 1/3/5 regression gate (those samples must still pass)
  5. writes runtime/results/results.csv and prints a per-digit summary

Latency source of truth: the project rule is "total time measured by the
program's own output, including H2D/D2H and I/O". So we FIRST look for a
latency token printed by the (patched) binary -- LATENCY_MS=<float>. If the
binary doesn't print one yet, we fall back to wall-clock around the process
and print a clear warning that this includes process startup and is only a
placeholder until the LATENCY_MS timer is added (see CLAUDE.md).

  python scripts/04_eval.py --bin ./mnistCUDNN/mnistCUDNN \
      --pgm-dir runtime/pgm --samples-dir mnistCUDNN/data

The 1/3/5 regression samples are the example PGMs shipped with mnistCUDNN
(one_28x28.pgm / three_28x28.pgm / five_28x28.pgm). Point --samples-dir at them.

How the binary is invoked (confirmed from the lab slides):
  - Default (no flag): the example hardcodes one/three/five and prints
    "Result of classification: 1 3 5". We do NOT use that here.
  - With --image=<path> (checkCmdLineFlag(argc, argv, "image"), slide line 866):
    it classifies ONE pgm and prints "Result of classification: <digit>".
    That single-image path is what we drive, one pgm at a time.
"""

import os
import re
import csv
import sys
import time
import glob
import argparse
import subprocess
from collections import defaultdict

LATENCY_RE = re.compile(r"LATENCY_MS\s*=\s*([0-9]*\.?[0-9]+)")
# Single-image path prints e.g. "Result of classification: 6".
# Grab the LAST run of digits after "classification:" and take its first digit.
PRED_RE = re.compile(r"classification:\s*([0-9 ]+)")


def true_label_from_name(path):
    """Infer the ground-truth digit from a filename, or None.

    one_28x28 -> 1, three_28x28 -> 3, five_28x28 -> 5, 6_0007 -> 6, 8_x -> 8.
    """
    name = os.path.splitext(os.path.basename(path))[0].lower()
    words = {"one": "1", "three": "3", "five": "5"}
    for w, d in words.items():
        if name.startswith(w):
            return d
    m = re.match(r"([0-9])", name)         # 6_0007 -> '6'
    return m.group(1) if m else None


def run_one(binary, pgm, image_flag):
    """Run the binary on one PGM. Returns (pred, latency_ms, used_wallclock).

    image_flag is the CLI form to pass a single image, e.g. "--image" so we
    call `binary --image=<pgm>` (matches checkCmdLineFlag/getCmdLineArgumentString).
    """
    cmd = [binary, f"{image_flag}={pgm}"] if image_flag else [binary, pgm]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = (time.perf_counter() - t0) * 1000.0
    out = proc.stdout + "\n" + proc.stderr

    pred = None
    hits = PRED_RE.findall(out)
    if hits:
        digits = hits[-1].split()        # last "classification:" line
        if digits:
            pred = digits[0]             # single-image path -> one digit

    m = LATENCY_RE.search(out)
    if m:
        return pred, float(m.group(1)), False
    return pred, wall, True


def collect(binary, files, runs, image_flag):
    """Run each file `runs` times; keep median latency, last prediction."""
    rows = []
    used_wall = False
    for f in files:
        truth = true_label_from_name(f)
        lats, pred = [], None
        for _ in range(runs):
            p, lat, wall = run_one(binary, f, image_flag)
            used_wall = used_wall or wall
            lats.append(lat)
            pred = p
        lats.sort()
        med = lats[len(lats) // 2]
        rows.append({
            "file": os.path.basename(f),
            "truth": truth,
            "pred": pred,
            "correct": int(pred == truth) if truth else "",
            "latency_ms": round(med, 4),
        })
    return rows, used_wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="path to mnistCUDNN binary")
    ap.add_argument("--pgm-dir", default="runtime/pgm")
    ap.add_argument("--samples-dir", default="mnistCUDNN/data",
                    help="dir with 1/3/5 regression PGMs")
    ap.add_argument("--runs", type=int, default=5,
                    help="repeats per image (median, warmup-robust)")
    ap.add_argument("--image-flag", default="--image",
                    help="CLI flag to pass one image; set '' if the patched "
                         "binary takes the path as a bare argument")
    ap.add_argument("--out", default="runtime/results/results.csv")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    if not os.path.exists(args.bin):
        print("binary not found:", args.bin); sys.exit(1)

    pgms = sorted(glob.glob(os.path.join(args.pgm_dir, "*.pgm")))
    samples = []
    for pat in ("one*", "three*", "five*"):   # one_28x28.pgm etc.
        samples += glob.glob(os.path.join(args.samples_dir, pat + ".pgm"))
    samples = sorted(set(samples))

    if not pgms and not samples:
        print("no PGMs found in", args.pgm_dir, "or", args.samples_dir)
        sys.exit(1)

    rows, used_wall = collect(args.bin, samples + pgms, args.runs,
                              args.image_flag)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "file", "truth", "pred", "correct", "latency_ms"])
        w.writeheader()
        w.writerows(rows)

    # ---- per-digit summary -----------------------------------------------
    by = defaultdict(lambda: [0, 0, 0.0])   # truth -> [correct, total, lat_sum]
    for r in rows:
        if not r["truth"]:
            continue
        c, t, s = by[r["truth"]]
        by[r["truth"]] = [c + (r["correct"] == 1), t + 1, s + r["latency_ms"]]

    print("\n digit |  acc   | n  | mean latency (ms)")
    print("-------+--------+----+-------------------")
    all_lat = []
    for d in sorted(by):
        c, t, s = by[d]
        all_lat.append(s / t)
        print(f"   {d}   | {c}/{t:<3} | {t:<2} | {s / t:8.3f}")
    if all_lat:
        print(f"\n overall mean latency: {sum(all_lat) / len(all_lat):.3f} ms")

    # ---- 1/3/5 regression gate -------------------------------------------
    reg = {d: by.get(d) for d in ("1", "3", "5") if d in by}
    failed = [d for d, v in reg.items() if v and v[0] < v[1]]
    if reg:
        if failed:
            print("\n[REGRESSION FAIL] 1/3/5 no longer all correct:", failed)
            print("  -> do NOT ship this change. See CLAUDE.md hard constraint.")
        else:
            print("\n[regression ok] 1/3/5 all still correct.")

    if used_wall:
        print("\n[WARN] latency = wall-clock around the process (includes "
              "startup).\n  Add a LATENCY_MS=<ms> print to the patched "
              "mnistCUDNN to report the real in-program time. See CLAUDE.md.")

    print("\nwrote", args.out)
    if failed:
        sys.exit(2)        # non-zero so run_all.sh / Claude Code sees the fail


if __name__ == "__main__":
    main()
