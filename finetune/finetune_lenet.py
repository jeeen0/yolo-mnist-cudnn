"""
Fine-tune a LeNet (Caffe-style) on MNIST + our 6/8 samples, then export the
weights as the exact binary blobs mnistCUDNN loads. KEPT IN RESERVE: only run
this if preprocessing alone doesn't get 6/8 high enough, because matching the
binary layout is fiddly and the structure/order must NOT change.

WHY the structure must match the example:
  conv1 : 20 x 1  x 5 x 5   (+ bias 20)
  conv2 : 50 x 20 x 5 x 5   (+ bias 50)
  ip1   : 500 x 800         (+ bias 500)   # 800 = 50 * 4 * 4
  ip2   : 10  x 500         (+ bias 10)
mnistCUDNN reads conv1.bin, conv1.bias.bin, conv2.bin, conv2.bias.bin,
ip1.bin, ip1.bias.bin, ip2.bin, ip2.bias.bin -- each a raw float32 blob in
the order above. We replicate that layout exactly so the 9-stage forward in
mnistCUDNN is byte-for-byte compatible (the grading constraint).

Pipeline:
  - load MNIST (torchvision)
  - load our 6/8 PGMs (already MNIST-normalized by preprocess.py) and OVERSAMPLE
    them so the net adapts to our handwriting without forgetting 1/3/5
  - train, then dump the 8 .bin files into --out

  python finetune/finetune_lenet.py --our-pgm-dir runtime/pgm --out finetune/weights

Then copy the 8 .bin files over mnistCUDNN's weight files and rebuild nothing
(weights are loaded at runtime). KEEP A BACKUP of the original .bin first.
"""

import os
import csv
import glob
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import datasets, transforms


class LeNet(nn.Module):
    """Same shape as the Caffe LeNet mnistCUDNN ships with."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)      # 28 -> 24
        self.conv2 = nn.Conv2d(20, 50, 5)     # 12 -> 8
        self.ip1 = nn.Linear(50 * 4 * 4, 500)
        self.ip2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)    # 24 -> 12
        x = F.max_pool2d(self.conv2(x), 2)    # 8  -> 4
        x = x.view(x.size(0), -1)
        x = F.relu(self.ip1(x))
        return self.ip2(x)


def read_pgm(path):
    """Read a binary P5 PGM into a float32 [0,1] (28,28) array."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5"
        dims = f.readline().split()
        w, h = int(dims[0]), int(dims[1])
        f.readline()  # maxval
        data = np.frombuffer(f.read(w * h), dtype=np.uint8)
    return data.reshape(h, w).astype(np.float32) / 255.0


def load_our_digits(pgm_dir):
    """Load our PGMs whose filename starts with the true label digit."""
    xs, ys = [], []
    for p in sorted(glob.glob(os.path.join(pgm_dir, "*.pgm"))):
        name = os.path.basename(p)
        if not name[0].isdigit():
            continue
        label = int(name[0])
        xs.append(read_pgm(p))
        ys.append(label)
    if not xs:
        return None
    x = torch.tensor(np.stack(xs)).unsqueeze(1)         # (N,1,28,28)
    y = torch.tensor(ys, dtype=torch.long)
    return TensorDataset(x, y)


def dump_bin(tensor, path):
    """Write a tensor as a raw little-endian float32 blob."""
    tensor.detach().cpu().numpy().astype("<f4").tofile(path)


def export_weights(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pairs = [
        ("conv1.bin", model.conv1.weight), ("conv1.bias.bin", model.conv1.bias),
        ("conv2.bin", model.conv2.weight), ("conv2.bias.bin", model.conv2.bias),
        ("ip1.bin", model.ip1.weight),     ("ip1.bias.bin", model.ip1.bias),
        ("ip2.bin", model.ip2.weight),     ("ip2.bias.bin", model.ip2.bias),
    ]
    for fname, t in pairs:
        dump_bin(t, os.path.join(out_dir, fname))
    print("wrote 8 .bin files to", out_dir)
    print("  back up the originals before overwriting mnistCUDNN's weights!")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--our-pgm-dir", default="runtime/pgm")
    ap.add_argument("--out", default="finetune/weights")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--oversample", type=int, default=30,
                    help="repeat our 6/8 set N times so the net adapts")
    ap.add_argument("--data", default="finetune/mnist_data")
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tf = transforms.Compose([transforms.ToTensor()])
    mnist = datasets.MNIST(args.data, train=True, download=True, transform=tf)

    sets = [mnist]
    ours = load_our_digits(args.our_pgm_dir)
    if ours is not None:
        sets += [ours] * args.oversample
        print(f"mixing in {len(ours)} of our digits, oversampled x{args.oversample}")
    else:
        print("no our-digit PGMs found; training on plain MNIST only")

    loader = DataLoader(ConcatDataset(sets), batch_size=128, shuffle=True)

    model = LeNet().to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        model.train()
        tot, corr, loss_sum = 0, 0, 0.0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * y.size(0)
            corr += (out.argmax(1) == y).sum().item()
            tot += y.size(0)
        print(f"epoch {ep + 1}/{args.epochs}  loss {loss_sum / tot:.4f}  "
              f"acc {corr / tot:.4f}")

    export_weights(model, args.out)


if __name__ == "__main__":
    main()
