"""
Fine-tune the mnistCUDNN LeNet on our 6/8 handwriting and export the 8 .bin
blobs the example loads -- WITHOUT breaking 1/3/5 and WITHOUT overfitting.

The structure/order and the .bin layout match the example exactly (grading
constraint): conv1(20x1x5x5) pool conv2(50x20x5x5) pool ip1(500x800) relu
LRN ip2(10x500) softmax. We export conv1.bin, conv1.bias.bin, conv2.bin,
conv2.bias.bin, ip1.bin, ip1.bias.bin, ip2.bin, ip2.bias.bin -- raw float32,
that order. (The LRN has no weights, so including it changes nothing in the
exported layout; it is kept so the training forward matches mnistCUDNN.)

WHY this recipe (the 1st attempt regressed 8 by training from scratch):
  - INIT from the original .bin (true fine-tune, not random init)
  - FREEZE conv1/conv2; train only ip1/ip2 -> preserves the features that
    already nail 8, so no catastrophic forgetting
  - mix in ALL of MNIST -> keeps 1/3/5 (and every digit) correct
  - hold out a stratified VAL split of our 6/8 for HONEST measurement
  - AUGMENT the train split on the fly (rotation/scale/shift/thickness/blur/
    exposure/noise) so it generalizes to unseen video instead of memorizing
    the ~139 photos. Robustness is measured on a PERTURBED val ("new-video
    proxy") so memorization can't hide.
  - low lr (1e-4) + early stop on robustness, keeping 8 and the 1/3/5 gate

Ship gate (adopt only if): 1/3/5 sample gate passes AND our 8 >= 67/69 AND
total > 133/139 AND robustness >= original. Final decision is still made on the
Orin via scripts/04_eval.py (real binary + video pipeline) -- this is a proxy.

  python finetune/finetune_lenet.py --our-pgm-dir finetune/our_pgm --out finetune/weights
"""
import os
import glob
import argparse

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, Dataset
from torchvision import datasets, transforms

# (name, shape) in the exact order/layout the example stores them.
WEIGHTS = [
    ("conv1.bin", (20, 1, 5, 5)), ("conv1.bias.bin", (20,)),
    ("conv2.bin", (50, 20, 5, 5)), ("conv2.bias.bin", (50,)),
    ("ip1.bin", (500, 800)), ("ip1.bias.bin", (500,)),
    ("ip2.bin", (10, 500)), ("ip2.bias.bin", (10,)),
]


class LeNet(nn.Module):
    """Faithful to mnistCUDNN's 9-stage forward (incl. the LRN after ip1+relu)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 50, 5)
        self.ip1 = nn.Linear(800, 500)
        self.ip2 = nn.Linear(500, 10)
        self.lrn = nn.LocalResponseNorm(5, alpha=1e-4, beta=0.75, k=1.0)

    def forward(self, x):
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = x.flatten(1)
        x = F.relu(self.ip1(x))
        x = self.lrn(x.unsqueeze(-1).unsqueeze(-1)).flatten(1)
        return self.ip2(x)


def _lb(path, shape):
    a = np.fromfile(path, dtype="<f4")
    n = int(np.prod(shape))
    if a.size != n:
        raise SystemExit(f"[ft] {os.path.basename(path)}: expected {n} floats, "
                         f"got {a.size} -> layout mismatch, STOP.")
    return torch.from_numpy(a.reshape(shape).copy())


def load_original(model, bin_dir):
    g = {n: _lb(os.path.join(bin_dir, n), s) for n, s in WEIGHTS}
    model.load_state_dict({
        "conv1.weight": g["conv1.bin"], "conv1.bias": g["conv1.bias.bin"],
        "conv2.weight": g["conv2.bin"], "conv2.bias": g["conv2.bias.bin"],
        "ip1.weight": g["ip1.bin"], "ip1.bias": g["ip1.bias.bin"],
        "ip2.weight": g["ip2.bin"], "ip2.bias": g["ip2.bias.bin"]})


def export_weights(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = {"conv1.bin": model.conv1.weight, "conv1.bias.bin": model.conv1.bias,
         "conv2.bin": model.conv2.weight, "conv2.bias.bin": model.conv2.bias,
         "ip1.bin": model.ip1.weight, "ip1.bias.bin": model.ip1.bias,
         "ip2.bin": model.ip2.weight, "ip2.bias.bin": model.ip2.bias}
    for name, _ in WEIGHTS:
        m[name].detach().cpu().numpy().astype("<f4").tofile(os.path.join(out_dir, name))
    print(f"wrote 8 .bin files to {out_dir}  (back up the originals first!)")


def read_pgm(path):
    """Binary P5 reader tolerant of comment (#) lines -> float32 [0,1] (28,28)."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"P5"
        vals = []
        while len(vals) < 3:                       # width, height, maxval
            line = f.readline()
            if line.startswith(b"#"):
                continue
            vals += line.split()
        w, h = int(vals[0]), int(vals[1])
        return np.frombuffer(f.read(w*h), np.uint8).reshape(h, w).astype(np.float32)/255.0


def load_our(pgm_dir):
    """Our PGMs whose filename starts with the true label digit -> {label:[arr]}."""
    by = {}
    for p in sorted(glob.glob(os.path.join(pgm_dir, "*.pgm"))):
        c = os.path.basename(p)[0]
        if c.isdigit():
            by.setdefault(int(c), []).append(read_pgm(p))
    return by


def _aniso_affine(ang, sx, sy, cx=14.0, cy=14.0):
    """2x3 affine: anisotropic scale (sx,sy) about center, then rotate by ang."""
    R = cv2.getRotationMatrix2D((cx, cy), ang, 1.0)
    S = np.array([[sx, 0.0, cx - sx*cx],
                  [0.0, sy, cy - sy*cy],
                  [0.0, 0.0, 1.0]], np.float32)
    R3 = np.vstack([R, [0.0, 0.0, 1.0]]).astype(np.float32)
    return (R3 @ S)[:2]


def _perspective(out, rng, jit):
    """Warp the 4 image corners by +/-jit px to mimic a tilted card/hand."""
    src = np.float32([[0, 0], [27, 0], [27, 27], [0, 27]])
    dst = src + rng.uniform(-jit, jit, src.shape).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(out, H, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0.0)


def _motion_blur(out, rng):
    """Line kernel at a random angle 0-180deg, length 3-9px (camera/hand motion)."""
    L = int(rng.choice([3, 5, 7, 9]))
    k = np.zeros((L, L), np.float32)
    k[L // 2, :] = 1.0                                  # horizontal line...
    M = cv2.getRotationMatrix2D((L/2 - 0.5, L/2 - 0.5), rng.uniform(0, 180), 1.0)
    k = cv2.warpAffine(k, M, (L, L))                    # ...rotated to a random angle
    s = k.sum()
    return cv2.filter2D(out, -1, k/s) if s > 0 else out


def _lighting(out, rng):
    """Multiply by a smooth gradient (linear ramp or off-center vignette), ~0.5-1.3."""
    h, w = out.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if rng.random() < 0.5:                              # directional linear ramp
        a = rng.uniform(0, 2*np.pi)
        d = np.cos(a)*xx + np.sin(a)*yy
        d = (d - d.min()) / (np.ptp(d) + 1e-6)
        lo, hi = rng.uniform(0.5, 0.9), rng.uniform(1.0, 1.3)
        g = lo + (hi - lo) * d
    else:                                               # off-center radial vignette
        cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        r = np.sqrt((xx - cx)**2 + (yy - cy)**2); r /= (r.max() + 1e-6)
        g = 1.3 - 0.8 * r
    return out * g.astype(np.float32)


def _jpeg(out, rng):
    """Round-trip through JPEG at q in [30,70] to add webcam/codec blocking."""
    q = int(rng.integers(30, 71))
    u8 = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", u8, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    if not ok:
        return out
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0


def augment(img01, rng, strong=False):
    """Live-camera-style perturbation of a 28x28 white-on-black [0,1] image.

    Applied in capture-pipeline order: geometric -> blur -> photometric ->
    compression/noise. preprocess.normalize_to_mnist re-centers/rescales, so the
    augments that survive (perspective/shear, blur, lighting, stroke) are leaned
    on hardest, plus modest position/scale jitter for imperfect centering.
    """
    s = 1.4 if strong else 1.0
    # 1. geometric: rotation + anisotropic (foreshortening) scale + wider shift
    ang = rng.uniform(-18*s, 18*s)
    sx = rng.uniform(1 - 0.3*s, 1 + 0.3*s); sy = rng.uniform(1 - 0.3*s, 1 + 0.3*s)
    sc = rng.uniform(0.7, 1.35)                          # overall scale spread
    M = _aniso_affine(ang, sx*sc, sy*sc)
    M[0, 2] += rng.uniform(-5*s, 5*s); M[1, 2] += rng.uniform(-5*s, 5*s)
    out = cv2.warpAffine(img01, M, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0.0)
    if rng.random() < 0.35:
        out = _perspective(out, rng, jit=rng.uniform(3, 5)*s)
    # stroke thickness (dilate/erode) is geometric too
    r = rng.random()
    if r < 0.30:
        out = cv2.dilate(out, np.ones((2, 2), np.float32))
    elif r < 0.50:
        out = cv2.erode(out, np.ones((2, 2), np.float32))
    # 2. blur: motion (hand/camera) then defocus (Gaussian)
    if rng.random() < 0.30:
        out = _motion_blur(out, rng)
    if rng.random() < 0.40:
        k = int(rng.choice([3, 5, 7])); out = cv2.GaussianBlur(out, (k, k), 0)
    # 3. photometric: global exposure then uneven lighting/glare
    out = out * rng.uniform(0.75, 1.25)
    if rng.random() < 0.30:
        out = _lighting(out, rng)
    out = np.clip(out, 0, 1)
    # 4. compression then sensor noise (operate on the displayed uint8 image)
    if rng.random() < 0.30:
        out = _jpeg(out, rng)
    if rng.random() < 0.40:
        out = out + rng.normal(0, rng.uniform(0.06, 0.09), out.shape).astype(np.float32)
    if rng.random() < 0.20:                              # salt-and-pepper sprinkle
        m = rng.random(out.shape) < 0.005
        out[m] = (rng.random(int(m.sum())) < 0.5).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


class AugOurs(Dataset):
    """Our 6/8 train images, returned freshly augmented every access."""
    def __init__(self, xs, ys, mult, augment_on=True):
        self.xs, self.ys, self.mult, self.on = xs, ys, mult, augment_on
        self.rng = np.random.default_rng(0)

    def __len__(self):
        return len(self.xs) * self.mult

    def __getitem__(self, i):
        j = i % len(self.xs)
        a = augment(self.xs[j], self.rng) if self.on else self.xs[j]
        return torch.tensor(a)[None], torch.tensor(self.ys[j], dtype=torch.long)


def make_perturbed(xs, ys, k, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for img, lab in zip(xs, ys):
        for _ in range(k):
            X.append(augment(img, rng, strong=True)); Y.append(lab)
    return torch.tensor(np.stack(X))[:, None], torch.tensor(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="mnistCUDNN/data", help="original .bin to init from")
    ap.add_argument("--our-pgm-dir", default="finetune/our_pgm")
    ap.add_argument("--out", default="finetune/weights")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--mult", type=int, default=40, help="augmented copies of our train set")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--data", default="finetune/mnist_data")
    ap.add_argument("--no-aug", action="store_true", help="disable augmentation")
    ap.add_argument("--no-freeze", action="store_true", help="train conv layers too")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(root)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    if not all(os.path.exists(os.path.join(args.bin_dir, n)) for n, _ in WEIGHTS):
        raise SystemExit(f"[ft] original .bin not found in {args.bin_dir}. "
                         f"Point --bin-dir at mnistCUDNN/data.")

    # stratified train/val split of our 6/8
    by = load_our(args.our_pgm_dir)
    if not by:
        raise SystemExit(f"[ft] no labeled PGMs in {args.our_pgm_dir} "
                         f"(expected 6_*.pgm / 8_*.pgm). Run B-1 first.")
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for lab, arrs in by.items():
        idx = np.random.permutation(len(arrs)); nval = int(len(arrs)*args.val_frac)
        for j, k in enumerate(idx):
            (va_x if j < nval else tr_x).append(arrs[k])
            (va_y if j < nval else tr_y).append(lab)
    full = {lab: torch.tensor(np.stack(a))[:, None].to(dev) for lab, a in by.items()}
    vaX = torch.tensor(np.stack(va_x))[:, None].to(dev); vaY = torch.tensor(va_y)
    pX, pY = make_perturbed(va_x, va_y, k=12, seed=123); pX = pX.to(dev)
    print(f"[ft] our split: train {len(tr_y)} / val {len(va_y)} ; perturbed-val {len(pY)} "
          f"| aug={'off' if args.no_aug else 'on'} freeze_conv={not args.no_freeze}")

    mnist = datasets.MNIST(args.data, train=True, download=True,
                           transform=transforms.ToTensor(),
                           target_transform=lambda y: torch.tensor(y, dtype=torch.long))
    our_tr = AugOurs(tr_x, tr_y, args.mult, augment_on=not args.no_aug)
    loader = DataLoader(ConcatDataset([mnist, our_tr]), batch_size=128, shuffle=True)

    sample = [(read_pgm(os.path.join(args.bin_dir, f"{n}_28x28.pgm")), l)
              for n, l in [("one", 1), ("three", 3), ("five", 5)]]
    mtest = datasets.MNIST(args.data, train=False, download=True, transform=transforms.ToTensor())
    MX = torch.stack([mtest[i][0] for i in range(len(mtest))]).to(dev)
    MY = torch.tensor([mtest[i][1] for i in range(len(mtest))])

    def perlab(m, X, Y, lab):
        m.eval()
        with torch.no_grad():
            p = m(X).argmax(1).cpu()
        msk = Y == lab; return int((p[msk] == lab).sum()), int(msk.sum())
    def macc(m, d):
        m.eval()
        with torch.no_grad():
            return (m(MX[MY == d]).argmax(1).cpu() == d).float().mean().item()
    def gate(m):
        m.eval()
        with torch.no_grad():
            return all(int(m(torch.tensor(im)[None, None].to(dev)).argmax(1)) == l
                       for im, l in sample)
    def report(tag, m):
        cv6, _ = perlab(m, vaX, vaY, 6); cv8, _ = perlab(m, vaX, vaY, 8)
        pv6, pn6 = perlab(m, pX, pY, 6); pv8, pn8 = perlab(m, pX, pY, 8)
        f6c, f6n = perlab(m, full[6], torch.full((full[6].shape[0],), 6), 6)
        f8c, f8n = perlab(m, full[8], torch.full((full[8].shape[0],), 8), 8)
        rob = (pv6 + pv8) / (pn6 + pn8)
        print(f"[{tag}] cleanVal 6={cv6} 8={cv8} | robustVal {rob*100:.1f}% | "
              f"full 6={f6c}/{f6n} 8={f8c}/{f8n} | "
              f"MNIST135={macc(m,1):.3f}/{macc(m,3):.3f}/{macc(m,5):.3f} gate={gate(m)}")
        return rob, f6c + f8c, f8c, gate(m)

    print("\n=== baseline (ORIGINAL weights) ===")
    orig = LeNet().to(dev); load_original(orig, args.bin_dir)
    rob0, _, _, _ = report("ORIG", orig)

    print("\n=== fine-tune ===")
    model = LeNet().to(dev); load_original(model, args.bin_dir)
    if not args.no_freeze:
        for p in list(model.conv1.parameters()) + list(model.conv2.parameters()):
            p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    best = None
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        rob, tot, f8c, g = report(f"ep{ep+1:2d}", model)
        score = (g and f8c >= 67, rob)          # keep 8 + gate, then maximize robustness
        if best is None or score > best[0]:
            best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()},
                    (rob, tot, f8c, g))
    model.load_state_dict(best[1])
    rob, tot, f8c, g = best[2]

    print("\n=== verdict ===")
    print(f"robustness(perturbed-val): ORIG {rob0*100:.1f}%  ->  fine-tuned {rob*100:.1f}%")
    ship = g and f8c >= 67 and tot > 133 and rob >= rob0
    if ship:
        export_weights(model, args.out)
        print(f"[ft] ADOPT (proxy) ✓  total={tot}/139  8={f8c}/69  gate={g}  rob>=orig")
        print( "     -> scp the 8 .bin to the Orin and confirm with scripts/04_eval.py (B-4).")
    else:
        print(f"[ft] REJECT ✗ keep ORIGINAL (need gate & 8>=67 & total>133 & rob>=orig; "
              f"got gate={g} 8={f8c} total={tot} rob={rob:.3f} vs {rob0:.3f}). No .bin written.")


if __name__ == "__main__":
    main()
