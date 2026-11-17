"""Network definitions and training loop of models A (GRU) and B (convolution + GRU).

Output heads: dead logit d and own-possession logit o per perspective; l = (dH + dA) / 2 (symmetric),
z = (oH - oA) / 2 (antisymmetric). Losses on l (graded frames) and z (frames with the ball alive).
Training needs the external datasets and their preprocessing, which are not in this repository."""
import json, os, resource, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

CFG = dict(arch="gru", fs="rd", h=128, n=2, drop=0.1, L=512, steps=4000, bs=32, lr=2e-3, wd=1e-2,
           seed=0, nb=8, rep=2, heads=4, k=5, pff=1, draws=3, sw=0.0, swap=1, ls=0.0, dw=1.0)
H_, A_, D_ = 0, 1, 2
resource.setrlimit(resource.RLIMIT_NOFILE, (resource.getrlimit(resource.RLIMIT_NOFILE)[1],) * 2)


def parse(argv):
    c = dict(CFG)
    for a in argv:
        k, v = a.split("=")
        c[k] = type(CFG[k])(v) if k in CFG else v
    return c


# ---------------------------------------------------------------- data
class Data:
    def __init__(self, cache, cfg):
        self.cache, self.cfg = cache, cfg
        self.keys = json.load(open(os.path.join(cache, "keys.json")))
        self.pub = self.keys[:10]
        self.pff = self.keys[10:]
        self.meta = {k: json.load(open(os.path.join(cache, k, "meta.json"))) for k in self.keys}
        an = self.meta[self.keys[0]]["an"]
        self.na = len(an)

    def arr(self, key, k, name):
        return np.load(os.path.join(self.cache, key, f"k{k}_{name}.npy"), mmap_mode="r")

    def item(self, key, k):
        fs = self.cfg["fs"]
        parts = {}
        if fs in ("rd", "both"):
            parts["Xa"], parts["Xs"] = self.arr(key, k, "Xa"), self.arr(key, k, "Xs")
        if fs in ("ku", "both"):
            parts["FH"], parts["FA"] = self.arr(key, k, "FH"), self.arr(key, k, "FA")
        return parts, self.arr(key, k, "y"), self.arr(key, k, "per")


def stats(data, keys):
    """per-column normalisation: Xa / rms, Xs (x - mean) / std, ku as is (already scaled)."""
    acc = {"Xa": [], "Xs": []}
    for key in keys[::3]:
        p, _, _ = data.item(key, 0)
        for n in acc:
            if n in p:
                acc[n].append(np.asarray(p[n][::20], np.float32))
    out = {}
    if acc["Xa"]:
        Xa = np.concatenate(acc["Xa"])
        out["sa"] = 1.0 / np.maximum(np.sqrt(np.nanmean(Xa ** 2, 0)), 1e-6)
        Xs = np.concatenate(acc["Xs"])
        out["ms"] = np.nanmean(Xs, 0)
        out["ss"] = 1.0 / np.maximum(np.nanstd(Xs, 0), 1e-6)
    return {k: np.nan_to_num(v).astype(np.float32) for k, v in out.items()}


def to_inputs(parts, sl, st, dev):
    """-> (XH, XA) float32 tensors (n, F) for rows sl."""
    xs_h, xs_a = [], []
    if "Xa" in parts:
        Xa = torch.as_tensor(np.asarray(parts["Xa"][sl], np.float32), device=dev) * st["sa"]
        Xs = (torch.as_tensor(np.asarray(parts["Xs"][sl], np.float32), device=dev) - st["ms"]) * st["ss"]
        Xa, Xs = (torch.nan_to_num(x, 0.0, 0.0, 0.0).clamp(-5, 5) for x in (Xa, Xs))
        xs_h += [Xa, Xs]
        xs_a += [-Xa, Xs]
    if "FH" in parts:
        xs_h.append(torch.as_tensor(np.asarray(parts["FH"][sl], np.float32), device=dev).clamp(-10, 10))
        xs_a.append(torch.as_tensor(np.asarray(parts["FA"][sl], np.float32), device=dev).clamp(-10, 10))
    return torch.cat(xs_h, 1), torch.cat(xs_a, 1)


# ---------------------------------------------------------------- models
class TCNBlock(nn.Module):
    def __init__(self, w, d, k, drop):
        super().__init__()
        self.c1 = nn.Conv1d(w, w, k, padding=d * (k - 1) // 2, dilation=d)
        self.c2 = nn.Conv1d(w, w, 1)
        self.n = nn.GroupNorm(8, w)
        self.dr = nn.Dropout(drop)

    def forward(self, x):
        return x + self.c2(self.dr(Fn.gelu(self.n(self.c1(x)))))


class Net(nn.Module):
    def __init__(self, F, c):
        super().__init__()
        h, a = c["h"], c["arch"]
        self.a = a
        self.inp = nn.Sequential(nn.Linear(F, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(c["drop"]))
        if a in ("gru", "lstm", "cgru", "clstm"):
            if a.startswith("c"):
                self.conv = nn.Sequential(*[TCNBlock(h, 1, c["k"], c["drop"]) for _ in range(2)])
            R = nn.GRU if "gru" in a else nn.LSTM
            self.rnn = R(h, h // 2, num_layers=c["n"], batch_first=True, bidirectional=True,
                         dropout=c["drop"] if c["n"] > 1 else 0.0)
        elif a == "tcn":
            self.blocks = nn.Sequential(*[TCNBlock(h, 2 ** i, 3, c["drop"]) for _ in range(c["rep"]) for i in range(c["nb"])])
        elif a == "tfm":
            self.down = nn.Conv1d(h, h, 4, stride=2, padding=1)
            self.pos = nn.Parameter(torch.zeros(1, c["L"] // 2, h))
            layer = nn.TransformerEncoderLayer(h, c["heads"], 2 * h, c["drop"], batch_first=True, norm_first=True,
                                               activation="gelu")
            self.enc = nn.TransformerEncoder(layer, c["n"])
            self.up = nn.ConvTranspose1d(h, h, 4, stride=2, padding=1)
        self.out = nn.Linear(h, 3)       # dead, own, switch-to-own (aux)

    def forward(self, x):                # (B, T, F) -> (B, T, 3)
        x = self.inp(x)
        if self.a in ("gru", "lstm", "cgru", "clstm"):
            if self.a.startswith("c"):
                x = self.conv(x.transpose(1, 2)).transpose(1, 2)
            x, _ = self.rnn(x)
        elif self.a == "tcn":
            x = self.blocks(x.transpose(1, 2)).transpose(1, 2)
        elif self.a == "tfm":
            T = x.shape[1]
            y = self.down(x.transpose(1, 2)).transpose(1, 2)
            y = self.enc(y + self.pos[:, :y.shape[1]])
            x = x + self.up(y.transpose(1, 2)).transpose(1, 2)[:, :T]
        return self.out(x)


def combine(net, XH, XA):
    """XH, XA (B, T, F) -> l, z, swH, swA (B, T)"""
    o = net(torch.cat([XH, XA], 0)).float()
    B = XH.shape[0]
    oh, oa = o[:B], o[B:]
    return (oh[..., 0] + oa[..., 0]) / 2, (oh[..., 1] - oa[..., 1]) / 2, oh[..., 2], oa[..., 2]


def switch_to(y):
    """frames within +-0.5 s (6 frames at 12.5 Hz) of a direct switch to H (resp. A)."""
    out = []
    for code in (H_, A_):
        s = np.zeros(len(y), np.float32)
        ch = np.flatnonzero((y[1:] == code) & (y[:-1] == 1 - code)) + 1
        for c in ch:
            s[max(c - 6, 0):c + 7] = 1
        out.append(s)
    return out


# ---------------------------------------------------------------- train / infer
def train_fold(data, cfg, train_keys, st, dev, log):
    rng = np.random.default_rng(cfg["seed"] * 100 + 7)
    torch.manual_seed(cfg["seed"])
    items = [(key, k) for key in train_keys for k in range(cfg["draws"])]
    loaded = [data.item(*it) for it in items]
    sws = [switch_to(np.asarray(y)) if cfg["sw"] > 0 else None for _, y, _ in loaded]
    lens = np.array([len(y) for _, y, _ in loaded], float)
    p = lens / lens.sum()
    F = to_inputs(loaded[0][0], slice(0, 2), st, dev)[0].shape[1]
    net = Net(F, cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=cfg["steps"], pct_start=0.1)
    L, t0, run = cfg["L"], time.time(), []
    for step in range(1, cfg["steps"] + 1):
        net.train()
        xh, xa, ys, ss = [], [], [], []
        for i in rng.choice(len(loaded), cfg["bs"], p=p):
            parts, y, per = loaded[i]
            a = int(rng.integers(0, len(y) - L))
            sl = slice(a, a + L)
            h_, a_ = to_inputs(parts, sl, st, dev)
            yy = np.where(np.isin(per[sl], (1, 2)), y[sl], -1).astype(np.int64)
            xh.append(h_), xa.append(a_), ys.append(yy)
            if sws[i] is not None:
                ss.append(np.stack([sws[i][0][sl], sws[i][1][sl]]))
        XH, XA = torch.stack(xh), torch.stack(xa)
        y = torch.as_tensor(np.stack(ys), device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, z, sh, sa = combine(net, XH, XA)
        sc = (y >= 0) & (y <= 2)
        al = (y == H_) | (y == A_)
        ls = cfg["ls"]
        td = (y[sc] == D_).float() * (1 - 2 * ls) + ls
        th = (y[al] == H_).float() * (1 - 2 * ls) + ls
        loss = cfg["dw"] * Fn.binary_cross_entropy_with_logits(l[sc], td) + Fn.binary_cross_entropy_with_logits(z[al], th)
        if ss:
            S = torch.as_tensor(np.stack(ss), device=dev)
            loss = loss + cfg["sw"] * (Fn.binary_cross_entropy_with_logits(sh[al], S[:, 0][al], pos_weight=torch.tensor(10.0, device=dev))
                                       + Fn.binary_cross_entropy_with_logits(sa[al], S[:, 1][al], pos_weight=torch.tensor(10.0, device=dev)))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sch.step()
        run.append(loss.item())
        if step % 500 == 0:
            log(f"[{time.time() - t0:5.0f}s] step {step} loss {np.mean(run):.4f}")
            run = []
    return net


@torch.no_grad()
def infer(net, parts, n, st, dev, L):
    """sliding windows L, stride L/2, keep the centre half -> l, z (n,) float32 at 12.5 Hz"""
    net.eval()
    l_out, z_out = np.zeros(n, np.float32), np.zeros(n, np.float32)
    if n <= L:
        starts = [0]
    else:
        starts = list(range(0, n - L, L // 2)) + [n - L]
    for s in starts:
        e = min(s + L, n)
        h_, a_ = to_inputs(parts, slice(s, e), st, dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l, z, _, _ = combine(net, h_[None], a_[None])
        lo = s if s == 0 else s + L // 4
        hi = e if e == n else e - L // 4
        l_out[lo:hi] = l[0, lo - s:hi - s].float().cpu().numpy()
        z_out[lo:hi] = z[0, lo - s:hi - s].float().cpu().numpy()
    return l_out, z_out


def main():
    cache, outdir, tag = sys.argv[1:4]
    cfg = parse(sys.argv[4:])
    od = os.path.join(outdir, tag)
    os.makedirs(od, exist_ok=True)
    json.dump(cfg, open(os.path.join(od, "cfg.json"), "w"))
    logf = open(os.path.join(od, "log"), "a")

    def log(s):
        print(s, flush=True)
        logf.write(s + "\n"), logf.flush()
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = "cuda"
    data = Data(cache, cfg)
    folds = [[k for i, k in enumerate(data.pub) if i % 3 == f] for f in range(3)]
    oof = {}
    t0 = time.time()
    for f in range(3):
        tr = [k for k in data.pub if k not in folds[f]] + (data.pff if cfg["pff"] else [])
        st_np = stats(data, tr)
        np.savez(os.path.join(od, f"norm{f}.npz"), **st_np)
        st = {k: torch.as_tensor(v, device=dev) for k, v in st_np.items()}
        log(f"fold {f} train {len(tr)} matches")
        net = train_fold(data, cfg, tr, st, dev, log)
        torch.save(net.state_dict(), os.path.join(od, f"f{f}.pt"))
        for key in folds[f]:
            parts, y, _ = data.item(key, 0)
            l, z = infer(net, parts, len(y), st, dev, cfg["L"])
            oof[key.split("_", 1)[1] + "|l"], oof[key.split("_", 1)[1] + "|z"] = l, z
        del net
        torch.cuda.empty_cache()
    np.savez(os.path.join(od, "oof.npz"), **oof)
    log(f"done train {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
