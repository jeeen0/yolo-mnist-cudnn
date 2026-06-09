"""Fine-tune LeNet v3 — augmentation for GENERALIZATION to new video.

v2 hit 139/139 on our photos but that is partly MEMORIZATION; the real test is
unseen video, where YOLO crops, framing, stroke thickness, blur and lighting all
vary. v3 trains ip1/ip2 (conv stays frozen) on our 6/8 with ON-THE-FLY realistic
augmentation that mimics those video variations, so the adaptation generalizes
instead of memorizing 139 specific images.

Honesty: we can't run real video here, so we estimate generalization with a
held-out val split that is then PERTURBED (the "new-video proxy"): each val image
is augmented K times and accuracy is measured on those. A model that overfits the
clean photos will drop on perturbed-val; a model that generalizes will hold up.

Reports an A/B of ORIGINAL vs v2 vs v3 on: clean-val, perturbed-val (robustness),
full set, MNIST 1/3/5, and the 1/3/5 sample gate. Ships to --out only if it keeps
8 and the 1/3/5 gate AND is at least as robust as v2 on perturbed-val.

  python finetune/finetune_v3.py --bin-dir mnistCUDNN/data --our-pgm-dir finetune/our_pgm
"""
import os, argparse, importlib.util
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, Dataset
from torchvision import datasets, transforms

# reuse the faithful model + loaders from v2
_spec = importlib.util.spec_from_file_location(
    "v2", os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_v2.py"))
v2 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(v2)
LeNet, load_original, read_pgm, load_our, export_bins, WEIGHTS = (
    v2.LeNet, v2.load_original, v2.read_pgm, v2.load_our, v2.export_bins, v2.WEIGHTS)


def augment(img01, rng, strong=False):
    """Realistic video-style perturbation of a 28x28 white-on-black [0,1] image."""
    s = 1.4 if strong else 1.0
    ang = rng.uniform(-14*s, 14*s)
    sc = rng.uniform(1 - 0.18*s, 1 + 0.18*s)
    M = cv2.getRotationMatrix2D((14, 14), ang, sc)
    M[0, 2] += rng.uniform(-2.5*s, 2.5*s); M[1, 2] += rng.uniform(-2.5*s, 2.5*s)
    out = cv2.warpAffine(img01, M, (28, 28), flags=cv2.INTER_LINEAR, borderValue=0.0)
    r = rng.random()                                    # stroke thickness
    if r < 0.30:
        out = cv2.dilate(out, np.ones((2, 2), np.float32))
    elif r < 0.50:
        out = cv2.erode(out, np.ones((2, 2), np.float32))
    if rng.random() < 0.4:                              # blur (video softness)
        k = int(rng.choice([3, 5])); out = cv2.GaussianBlur(out, (k, k), 0)
    out = out * rng.uniform(0.75, 1.25)                 # exposure
    if rng.random() < 0.4:                              # sensor noise
        out = out + rng.normal(0, 0.06, out.shape).astype(np.float32)
    return np.clip(out, 0, 1).astype(np.float32)


class AugOurs(Dataset):
    """Our 6/8 train images, each returned freshly augmented every access."""
    def __init__(self, xs, ys, mult):
        self.xs = xs; self.ys = ys; self.mult = mult
        self.rng = np.random.default_rng(0)
    def __len__(self): return len(self.xs) * self.mult
    def __getitem__(self, i):
        j = i % len(self.xs)
        a = augment(self.xs[j], self.rng)
        return torch.tensor(a)[None], torch.tensor(self.ys[j], dtype=torch.long)


def make_perturbed(xs, ys, k, seed):
    """Fixed perturbed test set: each image augmented k times (deterministic)."""
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for img, lab in zip(xs, ys):
        for _ in range(k):
            X.append(augment(img, rng, strong=True)); Y.append(lab)
    return torch.tensor(np.stack(X))[:, None], torch.tensor(Y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="mnistCUDNN/data")
    ap.add_argument("--our-pgm-dir", default="finetune/our_pgm")
    ap.add_argument("--out", default="finetune/weights_v3")
    ap.add_argument("--v2-dir", default="finetune/weights", help="v2 weights for A/B")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--mult", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--data", default="finetune/mnist_data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # stratified split of our 6/8
    by = load_our(args.our_pgm_dir)
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for lab, arrs in by.items():
        idx = np.random.permutation(len(arrs)); nval = int(len(arrs)*args.val_frac)
        for j, k in enumerate(idx):
            (va_x if j < nval else tr_x).append(arrs[k]); (va_y if j < nval else tr_y).append(lab)
    full = {lab: torch.tensor(np.stack(a))[:, None].to(dev) for lab, a in by.items()}
    vaX = torch.tensor(np.stack(va_x))[:, None].to(dev); vaY = torch.tensor(va_y)
    pX, pY = make_perturbed(va_x, va_y, k=12, seed=123); pX = pX.to(dev)
    print(f"[v3] split train {len(tr_y)} / val {len(va_y)} ; perturbed-val {len(pY)}")

    mnist = datasets.MNIST(args.data, train=True, download=True,
                           transform=transforms.ToTensor(),
                           target_transform=lambda y: torch.tensor(y, dtype=torch.long))
    aug = AugOurs(tr_x, tr_y, args.mult)
    loader = DataLoader(ConcatDataset([mnist, aug]), batch_size=128, shuffle=True, num_workers=0)

    # sample 1/3/5 gate + MNIST test for broad check
    sample = [(read_pgm(os.path.join(args.bin_dir, f"{n}_28x28.pgm")), l)
              for n, l in [("one", 1), ("three", 3), ("five", 5)]]
    mtest = datasets.MNIST(args.data, train=False, download=True, transform=transforms.ToTensor())
    MX = torch.stack([mtest[i][0] for i in range(len(mtest))]).to(dev)
    MY = torch.tensor([mtest[i][1] for i in range(len(mtest))])

    def acc(m, X, Y):
        m.eval()
        with torch.no_grad():
            p = torch.cat([m(X[i:i+4000]).argmax(1).cpu() for i in range(0, len(X), 4000)])
        return (p == Y).float().mean().item()
    def perlab(m, X, Y, lab):
        m.eval()
        with torch.no_grad(): p = m(X).argmax(1).cpu()
        msk = Y == lab; return int((p[msk] == lab).sum()), int(msk.sum())
    def gate(m):
        m.eval()
        with torch.no_grad():
            return all(int(m(torch.tensor(im)[None, None].to(dev)).argmax(1)) == l for im, l in sample)
    def report(tag, m):
        cv6, cn6 = perlab(m, vaX, vaY, 6); cv8, cn8 = perlab(m, vaX, vaY, 8)
        pv6, pn6 = perlab(m, pX, pY, 6); pv8, pn8 = perlab(m, pX, pY, 8)
        f6 = perlab(m, full[6], torch.tensor([6]*full[6].shape[0]), 6)
        f8 = perlab(m, full[8], torch.tensor([8]*full[8].shape[0]), 8)
        m135 = tuple(round(acc(m, MX[MY==d], MY[MY==d]), 4) for d in (1, 3, 5))
        print(f"[{tag}] cleanVal 6={cv6}/{cn6} 8={cv8}/{cn8} | "
              f"PERTURBval 6={pv6}/{pn6} 8={pv8}/{pn8} ({(pv6+pv8)/(pn6+pn8)*100:.1f}%) | "
              f"full 6={f6[0]}/{f6[1]} 8={f8[0]}/{f8[1]} | MNIST135={m135} gate={gate(m)}")
        return (pv6+pv8)/(pn6+pn8), f8[0], gate(m)

    # ---- A/B baselines -------------------------------------------------------
    orig = LeNet().to(dev); load_original(orig, args.bin_dir)
    print("\n=== baselines ===")
    rob_orig, _, _ = report("ORIG", orig)
    if all(os.path.exists(os.path.join(args.v2_dir, n)) for n, _ in WEIGHTS):
        m2 = LeNet().to(dev); load_original(m2, args.v2_dir)
        rob_v2, _, _ = report("v2  ", m2)
    else:
        rob_v2 = rob_orig
        print("[v2  ] (weights not found, skipping)")

    # ---- train v3 ------------------------------------------------------------
    model = LeNet().to(dev); load_original(model, args.bin_dir)
    for p in list(model.conv1.parameters()) + list(model.conv2.parameters()):
        p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    print("\n=== train v3 (conv frozen, augmented) ===")
    best = None
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        rob, f8c, g = report(f"ep{ep+1:2d}", model)
        score = (g and f8c >= 67, rob)     # keep 8 + gate, then maximize robustness
        if best is None or score > best[0]:
            best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()}, rob)
    model.load_state_dict(best[1])

    print("\n=== verdict ===")
    rob_v3, f8c, g = report("v3* ", model)
    ship = g and f8c >= 67 and rob_v3 >= rob_v2 - 1e-9
    print(f"robustness(perturbed-val acc): ORIG={rob_orig*100:.1f}%  "
          f"v2={rob_v2*100:.1f}%  v3={rob_v3*100:.1f}%")
    if ship:
        export_bins(model, args.out)
        print(f"[v3] ADOPT ✓ -> {args.out}  (gate ok, 8 kept, robustness >= v2)")
    else:
        print(f"[v3] REJECT ✗ (gate={g} 8={f8c} rob_v3={rob_v3:.3f} vs v2 {rob_v2:.3f})")


if __name__ == "__main__":
    main()
