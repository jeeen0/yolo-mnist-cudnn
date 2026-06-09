"""Fine-tune LeNet v2 — conservative, anti-forgetting (implements docs B-1~B-4).

The 1st attempt trained from scratch (random init, lr 1e-3, oversample 30, no
val) and REGRESSED 8 (catastrophic forgetting): original 133/139 (6:66/70,
8:67/69) -> 1st FT 129/139 (8:65/69). This v2 fixes that:

  - INIT from the original .bin (true fine-tune, not scratch)
  - FREEZE conv1/conv2 (keep the feature extractor that already nails 8)
  - hold out a stratified VAL split of our 6/8 -> honest measurement
  - oversample LOW (default 12), lr LOW (1e-4), EARLY STOP on val-6 without
    dropping val-8
  - model is faithful to mnistCUDNN (incl. LRN N=5 a=1e-4 b=0.75 k=1) so the
    proxy A/B tracks the real binary (verified: original proxy == 133/139)

Ship gate (docs): adopt ONLY if proxy A/B beats original AND keeps 8, i.e.
8 >= 67/69 AND sample 1/3/5 all correct AND total > 133/139. Else keep original.

  python finetune/finetune_v2.py --bin-dir mnistCUDNN/data --our-pgm-dir finetune/our_pgm
"""
import os, glob, argparse, shutil
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import datasets, transforms

WEIGHTS = [("conv1.bin",(20,1,5,5)),("conv1.bias.bin",(20,)),
           ("conv2.bin",(50,20,5,5)),("conv2.bias.bin",(50,)),
           ("ip1.bin",(500,800)),("ip1.bias.bin",(500,)),
           ("ip2.bin",(10,500)),("ip2.bias.bin",(10,))]


class LeNet(nn.Module):
    """Faithful to mnistCUDNN's 9-stage forward (incl. LRN after ip1+relu)."""
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


def lb(p, shape):
    a = np.fromfile(p, dtype="<f4"); return torch.from_numpy(a.reshape(shape).copy())


def load_original(model, d):
    g = {n: lb(os.path.join(d, n), s) for n, s in WEIGHTS}
    model.load_state_dict({
        "conv1.weight": g["conv1.bin"], "conv1.bias": g["conv1.bias.bin"],
        "conv2.weight": g["conv2.bin"], "conv2.bias": g["conv2.bias.bin"],
        "ip1.weight": g["ip1.bin"], "ip1.bias": g["ip1.bias.bin"],
        "ip2.weight": g["ip2.bin"], "ip2.bias": g["ip2.bias.bin"]})


def read_pgm(p):
    """Binary P5 reader tolerant of comment (#) lines in the header."""
    with open(p, "rb") as f:
        assert f.readline().strip() == b"P5"
        vals = []
        while len(vals) < 3:                      # width, height, maxval
            line = f.readline()
            if line.startswith(b"#"):
                continue
            vals += line.split()
        w, h = int(vals[0]), int(vals[1])
        return np.frombuffer(f.read(w*h), np.uint8).reshape(h, w).astype(np.float32)/255.0


def load_our(pgm_dir):
    by = {}
    for p in sorted(glob.glob(os.path.join(pgm_dir, "*.pgm"))):
        c = os.path.basename(p)[0]
        if c.isdigit():
            by.setdefault(int(c), []).append(read_pgm(p))
    return by  # {label: [arr,...]}


def export_bins(model, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = {"conv1.bin": model.conv1.weight, "conv1.bias.bin": model.conv1.bias,
         "conv2.bin": model.conv2.weight, "conv2.bias.bin": model.conv2.bias,
         "ip1.bin": model.ip1.weight, "ip1.bias.bin": model.ip1.bias,
         "ip2.bin": model.ip2.weight, "ip2.bias.bin": model.ip2.bias}
    for name, _ in WEIGHTS:
        m[name].detach().cpu().numpy().astype("<f4").tofile(os.path.join(out_dir, name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin-dir", default="mnistCUDNN/data")
    ap.add_argument("--our-pgm-dir", default="finetune/our_pgm")
    ap.add_argument("--out", default="finetune/weights")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--oversample", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-frac", type=float, default=0.25)
    ap.add_argument("--data", default="finetune/mnist_data")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # --- our 6/8, stratified train/val split (held out for honest eval) ------
    by = load_our(args.our_pgm_dir)
    tr_x, tr_y, va_x, va_y = [], [], [], []
    for lab, arrs in by.items():
        idx = np.random.permutation(len(arrs)); nval = int(len(arrs)*args.val_frac)
        for j, k in enumerate(idx):
            (va_x if j < nval else tr_x).append(arrs[k])
            (va_y if j < nval else tr_y).append(lab)
    tr_x = torch.tensor(np.stack(tr_x)).unsqueeze(1); tr_y = torch.tensor(tr_y)
    va_x = torch.tensor(np.stack(va_x)).unsqueeze(1).to(dev); va_y = torch.tensor(va_y)
    full = {lab: torch.tensor(np.stack(a)).unsqueeze(1).to(dev) for lab, a in by.items()}
    print(f"[v2] our split: train {len(tr_y)} / val {len(va_y)} "
          f"(val 6={int((va_y==6).sum())}, 8={int((va_y==8).sum())})")

    # --- MNIST (keeps all digits incl. 1/3/5) + oversampled our-train --------
    mnist = datasets.MNIST(args.data, train=True, download=True,
                           transform=transforms.ToTensor(),
                           target_transform=lambda y: torch.tensor(y, dtype=torch.long))
    our_tr = TensorDataset(tr_x, tr_y)
    loader = DataLoader(ConcatDataset([mnist] + [our_tr]*args.oversample),
                        batch_size=128, shuffle=True)

    # --- model: init original, freeze conv1/conv2 ----------------------------
    model = LeNet().to(dev); load_original(model, args.bin_dir)
    for p in list(model.conv1.parameters()) + list(model.conv2.parameters()):
        p.requires_grad = False
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    sample = [(read_pgm(os.path.join(args.bin_dir, f"{n}_28x28.pgm")), l)
              for n, l in [("one",1),("three",3),("five",5)]]
    def ev(x, y):
        model.eval()
        with torch.no_grad(): return (model(x).argmax(1).cpu() == y).float().mean().item()
    def ev_full(lab):
        model.eval()
        with torch.no_grad():
            pred = model(full[lab]).argmax(1).cpu()
        return int((pred == lab).sum()), len(pred)
    def gate_135():
        model.eval()
        with torch.no_grad():
            return all(int(model(torch.tensor(im)[None,None].to(dev)).argmax(1)) == l
                       for im, l in sample)

    # baseline (original) on the SAME val split, for honest A/B
    base_v6, base_v8 = ev(va_x[va_y==6], va_y[va_y==6]), ev(va_x[va_y==8], va_y[va_y==8])
    print(f"[v2] ORIG  val 6={base_v6:.3f} 8={base_v8:.3f} | "
          f"full 6={ev_full(6)} 8={ev_full(8)} | gate135={gate_135()}")

    best = None
    for ep in range(args.epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); F.cross_entropy(model(x), y).backward(); opt.step()
        v6 = ev(va_x[va_y==6], va_y[va_y==6]); v8 = ev(va_x[va_y==8], va_y[va_y==8])
        f6c, f6n = ev_full(6); f8c, f8n = ev_full(8); g = gate_135()
        print(f"[v2] ep{ep+1:2d} val 6={v6:.3f} 8={v8:.3f} | "
              f"full 6={f6c}/{f6n} 8={f8c}/{f8n} | gate135={g}")
        # early-stop bookkeeping: best = max val-6 subject to no val-8 drop + gate
        score = (g and v8 >= base_v8 - 1e-9, v6, v8)
        if best is None or score > best[0]:
            best = (score, {k: v.detach().clone() for k, v in model.state_dict().items()},
                    (f6c, f6n, f8c, f8n))

    # restore best snapshot
    model.load_state_dict(best[1])
    f6c, f6n, f8c, f8n = best[2]; tot = f6c + f8c; totn = f6n + f8n
    g = gate_135()
    print(f"\n[v2] BEST  full 6={f6c}/{f6n} 8={f8c}/{f8n}  total={tot}/{totn} | gate135={g}")

    # ship gate (docs): 8>=67/69 AND total>133/139 AND 1/3/5 gate
    ship = g and f8c >= 67 and tot > 133
    if ship:
        export_bins(model, args.out)
        print(f"[v2] ADOPT ✓ -> wrote 8 .bin to {args.out} (total {tot}/{totn} > 133, 8 ok, gate ok)")
    else:
        print(f"[v2] REJECT ✗ keep ORIGINAL (need 8>=67 & total>133 & gate; "
              f"got 8={f8c} total={tot} gate={g}). No .bin written.")


if __name__ == "__main__":
    main()
