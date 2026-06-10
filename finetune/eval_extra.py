"""Evaluate the LeNet (original vs our fine-tuned) on INDEPENDENT handwritten
digit datasets (not MNIST) -> honest out-of-distribution generalization check,
especially for 6/8. All inputs are coerced to the LeNet domain: white digit on
black, 28x28, [0,1].

  python finetune/eval_extra.py --orig mnistCUDNN/data --ft finetune/weights
"""
import argparse
import numpy as np
import cv2
import torch
import torchvision.datasets as D

import importlib.util, os
_spec = importlib.util.spec_from_file_location(
    "ft", os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_lenet.py"))
ftmod = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(ftmod)
LeNet, load_original = ftmod.LeNet, ftmod.load_original


def to_lenet(arr):
    """uint8 HxW -> 28x28 float [0,1], white-on-black (auto-fix polarity)."""
    a = arr.astype(np.float32)
    if a.shape != (28, 28):
        a = cv2.resize(a, (28, 28), interpolation=cv2.INTER_AREA)
    a /= 255.0
    # white-on-black expected (background mostly 0). If the border is bright,
    # the image is black-on-white -> invert.
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if border.mean() > 0.5:
        a = 1.0 - a
    return a


def load_set(name):
    """Return (X uint8 Nx28x28-ish list, Y) for a torchvision handwritten set."""
    root = f"finetune/extra/{name.lower()}"
    if name == "USPS":
        ds = D.USPS(root, train=False, download=True)
        return [(np.array(im), lab) for im, lab in ds]
    if name == "EMNIST":
        # 'digits' split = NIST handwritten digits, different writers than MNIST.
        ds = D.EMNIST(root, split="digits", train=False, download=True)
        # EMNIST images are stored transposed vs MNIST -> transpose to upright.
        return [(np.array(im).T, lab) for im, lab in ds]
    raise ValueError(name)


def evaluate(model, data, dev):
    X = torch.tensor(np.stack([to_lenet(a) for a, _ in data]))[:, None].to(dev)
    Y = torch.tensor([l for _, l in data])
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(X[i:i+4000]).argmax(1).cpu() for i in range(0, len(X), 4000)])
    per = {}
    for d in range(10):
        m = Y == d
        per[d] = (int((pred[m] == d).sum()), int(m.sum()))
    return (pred == Y).float().mean().item(), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig", default="mnistCUDNN/data")
    ap.add_argument("--ft", default="finetune/weights")
    ap.add_argument("--sets", nargs="+", default=["USPS", "EMNIST"])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    orig = LeNet().to(dev); load_original(orig, args.orig)
    ft = LeNet().to(dev); load_original(ft, args.ft)

    for name in args.sets:
        data = load_set(name)
        ao, po = evaluate(orig, data, dev)
        af, pf = evaluate(ft, data, dev)
        print(f"\n=== {name}  ({len(data)} imgs) ===")
        print(f"  overall: ORIG {ao*100:.2f}%  ->  FT {af*100:.2f}%")
        for d in range(10):
            co, no = po[d]; cf, nf = pf[d]
            tag = "  <-- 6/8" if d in (6, 8) else ("  <-- 1/3/5" if d in (1, 3, 5) else "")
            print(f"   {d}: {co/no*100:5.1f}% -> {cf/nf*100:5.1f}%   (n={no}){tag}")


if __name__ == "__main__":
    main()
