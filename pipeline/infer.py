"""Run the 21 saved networks on one match and return the three per-frame probability sources.

Each source is an (n_frames, 3) float32 array of H/A/D probabilities, NaN outside the two halves.
Nothing here reads a label. The windowing and resampling helpers are taken unchanged from the
script that produced the submitted predictions.
"""
import json
from pathlib import Path

import numpy as np
import torch

from members.kashiwagi import feats
from members.kashiwagi.train import seq_train
from members.oota.solo import seq

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights"
FOLDS, SEEDS = (0, 1, 2), (0, 1, 2)
DEVICE = torch.device("cpu")


def stable_logs(names_a, Xa, names_s, Xs):
    """Recompute the two log features in float64 so they round the same on every platform."""
    g = Xa[:, names_a.index("gap")].astype(float)
    Xa[:, names_a.index("gap_log")] = np.sign(g) * np.log1p(np.abs(g))
    Xs[:, names_s.index("dmin_log")] = np.log1p(Xs[:, names_s.index("dmin")].astype(float))
    return Xa, Xs


def _windows_kashiwagi(net, parts, n, st, window):
    """Overlapping 512-step windows; keep the middle half of each."""
    l_out, z_out = np.zeros(n, np.float32), np.zeros(n, np.float32)
    starts = [0] if n <= window else list(range(0, n - window, window // 2)) + [n - window]
    with torch.no_grad():
        for s in starts:
            e = min(s + window, n)
            xh, xa = seq_train.to_inputs(parts, slice(s, e), st, DEVICE)
            l, z, _, _ = seq_train.combine(net, xh[None], xa[None])
            lo = s if s == 0 else s + window // 4
            hi = e if e == n else e - window // 4
            l_out[lo:hi] = l[0, lo - s:hi - s].float().cpu().numpy()
            z_out[lo:hi] = z[0, lo - s:hi - s].float().cpu().numpy()
    return l_out, z_out


def _interpolate_by_period(dead, home, frame_index, period_model, period_full):
    """Bring 12.5 Hz logits to the native 25 Hz grid without mixing the two halves."""
    dead, home = np.asarray(dead, np.float64), np.asarray(home, np.float64)
    target = np.arange(len(period_full), dtype=np.int64)
    dead_full = np.zeros(len(period_full), np.float32)
    home_full = np.zeros(len(period_full), np.float32)
    for half in (1, 2):
        src = np.flatnonzero(period_model == half)
        dst = np.flatnonzero(period_full == half)
        if len(dst):
            dead_full[dst] = np.interp(target[dst], frame_index[src], dead[src]).astype(np.float32)
            home_full[dst] = np.interp(target[dst], frame_index[src], home[src]).astype(np.float32)
    return dead_full, home_full


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -60, 60)))


def kashiwagi_members(match, arch, log=print):
    """Nine networks (3 folds x 3 seeds) of one architecture ('gru' or 'cgru') -> list of H/A/D arrays."""
    names_a, Xa, names_s, Xs = feats.features(match)
    Xa, Xs = stable_logs(names_a, Xa, names_s, Xs)
    parts = {"Xa": Xa[::2], "Xs": Xs[::2]}
    n = len(parts["Xa"])
    period = np.asarray(match.period, np.int8)
    index = np.arange(n, dtype=np.int64) * 2
    out = []
    for fold in FOLDS:
        for seed in SEEDS:
            run = WEIGHTS / f"kashiwagi-{arch}-fold{fold}-seed{seed}"
            cfg = dict(seq_train.CFG)
            cfg.update(json.loads((run / "config.json").read_text()))
            norm = np.load(run / "normalization.npz", allow_pickle=False)
            st = {k: torch.as_tensor(norm[k].copy(), device=DEVICE) for k in norm.files}
            width = seq_train.to_inputs(parts, slice(0, 2), st, DEVICE)[0].shape[1]
            net = seq_train.Net(width, cfg).to(DEVICE)
            net.load_state_dict(torch.load(run / "step-04000.pt", map_location=DEVICE, weights_only=True))
            net.eval()
            l, z = _windows_kashiwagi(net, parts, n, st, cfg["L"])
            l, z = _interpolate_by_period(l, z, index, period[index], period)
            p_dead, p_home = _sigmoid(l), _sigmoid(z)
            out.append(np.column_stack(((1 - p_dead) * p_home, (1 - p_dead) * (1 - p_home), p_dead)).astype(np.float32))
            log(f"  {run.name}: {width} input features")
    return out


def _windows_oota(net, X, mu, sd, perm, sign, batch=64):
    """Overlapping windows, averaged with the team-swapped copy of the input."""
    Z = seq.normalise(np.asarray(X), mu, sd)
    n, W = len(Z), seq.WIN
    half = W // 2
    starts = list(range(0, max(n - W, 0) + 1, half))
    if starts[-1] + W < n:
        starts.append(n - W)
    if n < W:
        starts = [0]
    Zs = Z[:, perm] * sign
    acc = np.zeros((n, 3), np.float64)
    cnt = np.zeros(n, np.float64)
    with torch.no_grad():
        for b in range(0, len(starts), batch):
            ss = starts[b:b + batch]
            x0 = torch.from_numpy(np.stack([Z[s:s + W] for s in ss])).to(DEVICE)
            x1 = torch.from_numpy(np.stack([Zs[s:s + W] for s in ss])).to(DEVICE)
            p0 = torch.softmax(net(x0).float(), -1).cpu().numpy()
            p1 = torch.softmax(net(x1).float(), -1)[..., [1, 0, 2]].cpu().numpy()
            p = (p0 + p1) / 2
            for i, s in enumerate(ss):
                L = p.shape[1]
                a = 0 if s == 0 else L // 4
                e = L if s + L >= n else L - L // 4
                acc[s + a:s + e] += p[i, a:e]
                cnt[s + a:s + e] += 1
    return np.log(np.clip(acc / np.maximum(cnt, 1)[:, None], 1e-6, 1.0)).astype(np.float32)


def oota_members(match, log=print):
    """Three networks (one per fold) of the third model -> list of H/A/D arrays."""
    names, X = seq.seq_features(match)
    out = []
    for fold in FOLDS:
        run = WEIGHTS / f"oota-fold{fold}-seed0"
        norm = np.load(run / "normalization.npz", allow_pickle=False)
        swap = np.load(run / "swap.npz", allow_pickle=False)
        net = seq.build_net(len(names), hidden=128).to(DEVICE)
        net.load_state_dict(torch.load(run / "step-03000.pt", map_location=DEVICE, weights_only=True))
        net.eval()
        logp = _windows_oota(net, X, norm["mu"], norm["sd"], swap["perm"], swap["sign"])
        p = np.repeat(np.exp(logp), 2, axis=0)
        if len(p) < match.n_frames:
            p = np.concatenate([p, np.repeat(p[-1:], match.n_frames - len(p), axis=0)])
        out.append(p[:match.n_frames].astype(np.float32))
        log(f"  {run.name}: {len(names)} input features")
    return out


def mean_members(members, period):
    """Equal mean over networks; NaN outside the two halves."""
    active = np.isin(period, (1, 2))
    out = np.full((len(period), 3), np.nan, np.float32)
    out[active] = np.mean(np.stack(members)[:, active], axis=0, dtype=np.float64).astype(np.float32)
    return out


def sources(match, log=print):
    """The three probability sources the ensemble mixes."""
    period = np.asarray(match.period, np.int8)
    return {
        "gru4000_s012": mean_members(kashiwagi_members(match, "gru", log), period),
        "cgru4000_s012": mean_members(kashiwagi_members(match, "cgru", log), period),
        "oota3000": mean_members(oota_members(match, log), period),
    }
