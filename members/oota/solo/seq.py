"""oota_seq：フレームごとの特徴量の並びを、畳み込み＋双方向 GRU に入れて H/A/D を出す系列モデル。

- 入力：`features.frame_features` の値（時間の窓の平均は足さない。前後は網が見る）を STEP フレームごと（12.5 Hz）に
  まとめたもの。学習用の試合の平均と標準偏差で割り、欠けは 0、±5 で切る
- 網：線形 → 畳み込み（幅 5）→ 双方向 GRU（2 層）→ H/A/D の 3 つの logit
- ホームとアウェイの入れ替え：特徴量の列の入れ替えと符号（`swap_map` が実際の試合から求める）で作り、
  学習では半分の確率で入れ替え、予測では元の向きと入れ替えた向きの確率を平均する
- 予測は長さ WIN の窓を半分ずつ重ね、各窓の真ん中を使う
"""
import numpy as np

from members.oota.solo import features

STEP = 2                                  # まとめるフレームの数（25 fps で 12.5 Hz）
WIN = 512                                 # 窓の長さ [刻み]（約 41 秒）
DROP = ("tsec",)                          # 試合の中の時刻そのものは入れない


def seq_features(m, swap=False):
    """-> (名前, (N, F) float32)。N = n_frames // STEP。"""
    names, F = features.frame_features(m, swap)
    keep = [k for k, n in enumerate(names) if n not in DROP]
    T = len(F) // STEP * STEP
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        B = np.nanmean(F[:T][:, keep].reshape(-1, STEP, len(keep)), 1)
    return [names[k] for k in keep], B.astype(np.float32)


def seq_labels(truth):
    return np.asarray(truth[:len(truth) // STEP * STEP:STEP], np.int8)


def swap_map(m, tol=1e-3):
    """入れ替えた試合の特徴量の各列が、元のどの列の何倍（±1）かを求める。-> (perm, sign)。
    G[:, k] == sign[k] * F[:, perm[k]] が成り立つ（合わない列があれば止める）。"""
    names, F = seq_features(m, False)
    _, G = seq_features(m, True)
    rows = np.linspace(0, len(F) - 1, min(len(F), 20000)).astype(int)
    F, G = F[rows].astype(np.float64), G[rows].astype(np.float64)
    perm, sign = np.zeros(len(names), np.int64), np.ones(len(names), np.float32)
    for k in range(len(names)):
        best = (np.inf, k, 1.0)
        for j in range(len(names)):
            for s in (1.0, -1.0):
                d = np.abs(G[:, k] - s * F[:, j])
                ok = np.isfinite(d)
                if ok.sum() < 100 or (np.isfinite(G[:, k]) != np.isfinite(F[:, j])).mean() > 0.001:
                    continue
                e = d[ok].mean() / (np.abs(G[ok, k]).mean() + 1e-6)
                if e < best[0] - 1e-9 or (abs(e - best[0]) <= 1e-9 and j == k):
                    best = (e, j, s)
        if best[0] > tol:
            raise ValueError(f"入れ替えで対応する列が見つからない：{names[k]}（誤差 {best[0]:.4f}）")
        perm[k], sign[k] = best[1], best[2]
    return perm, sign


def norm_stats(arrays):
    """学習用の試合の特徴量（(N, F) のリスト）から、列ごとの平均と標準偏差（1〜99% で切ってから）。"""
    X = np.concatenate([a[::7] for a in arrays]).astype(np.float64)
    lo, hi = np.nanquantile(X, 0.01, 0), np.nanquantile(X, 0.99, 0)
    X = np.clip(X, lo, hi)
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0)
    return mu.astype(np.float32), np.where(sd > 1e-6, sd, 1.0).astype(np.float32)


def normalise(X, mu, sd):
    Z = (X - mu) / sd
    return np.clip(np.nan_to_num(Z, nan=0.0, posinf=5.0, neginf=-5.0), -5, 5).astype(np.float32)


def build_net(n_feat, hidden=128, layers=2, drop=0.1):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Sequential(nn.Linear(n_feat, hidden), nn.LayerNorm(hidden), nn.GELU())
            self.conv = nn.Conv1d(hidden, hidden, 5, padding=2)
            self.act = nn.GELU()
            self.gru = nn.GRU(hidden, hidden, layers, batch_first=True, bidirectional=True,
                              dropout=drop if layers > 1 else 0.0)
            self.out = nn.Linear(2 * hidden, 3)

        def forward(self, x):                                   # (B, T, F)
            h = self.inp(x)
            h = h + self.act(self.conv(h.transpose(1, 2))).transpose(1, 2)
            return self.out(self.gru(h)[0])                     # (B, T, 3)：H, A, D

    return Net()


def predict_logp(net, Z, perm, sign, batch=64):
    """正規化済みの (N, F) -> (N, 3) の対数確率（元の向きと入れ替えた向きの平均）。"""
    import torch
    n = len(Z)
    half = WIN // 2
    starts = list(range(0, max(n - WIN, 0) + 1, half))
    if starts[-1] + WIN < n:
        starts.append(n - WIN)
    if n < WIN:
        starts = [0]
    Zs = Z[:, perm] * sign                                      # 入れ替えた向き
    acc = np.zeros((n, 3), np.float64)
    cnt = np.zeros(n)
    net.eval()
    with torch.no_grad():
        for b in range(0, len(starts), batch):
            ss = starts[b:b + batch]
            x0 = torch.from_numpy(np.stack([Z[s:s + WIN] for s in ss]))
            x1 = torch.from_numpy(np.stack([Zs[s:s + WIN] for s in ss]))
            p0 = torch.softmax(net(x0), -1).numpy()
            p1 = torch.softmax(net(x1), -1).numpy()[..., [1, 0, 2]]
            p = (p0 + p1) / 2
            for i, s in enumerate(ss):
                L = p.shape[1]
                a = 0 if s == 0 else L // 4                     # 端の窓は端まで、他は真ん中の半分
                e = L if s + L >= n else L - L // 4
                acc[s + a:s + e] += p[i, a:e]
                cnt[s + a:s + e] += 1
    return np.log(np.clip(acc / np.maximum(cnt, 1)[:, None], 1e-6, 1))


class Ensemble:
    """学習済みの 3 つの網（`seq_train.py`）。公開の試合はその試合を見ていない網で、それ以外は 3 つの平均で予測する。"""

    def __init__(self, folder, cost=4.0):
        import json
        import torch
        from pathlib import Path
        folder = Path(folder)
        if not (folder / "f0.pt").exists():
            raise FileNotFoundError(f"{folder} に重みが無い。members/oota/solo/seq_train.py で学ぶ（PFF の試合が要る）")
        sw = np.load(folder / "swap.npz")
        self.perm, self.sign, self.cost, self.nets = sw["perm"], sw["sign"], cost, []
        for k in range(3):
            cfg = json.loads((folder / f"cfg{k}.json").read_text())
            z = np.load(folder / f"norm{k}.npz")
            net = build_net(len(cfg["names"]), cfg["hidden"])
            net.load_state_dict(torch.load(folder / f"f{k}.pt", map_location="cpu"))
            self.nets.append((net, z["mu"], z["sd"], set(cfg["held"]), cfg["names"]))

    def proba(self, match):
        """-> (N, 3) の H/A/D の確率（12.5 Hz）。"""
        names, X = seq_features(match)
        key = f"{match.source}_{match.id}"
        use = [n for n in self.nets if key in n[3]] or self.nets
        assert names == use[0][4], "特徴量の並びが学習のときと違う"
        return np.mean([np.exp(predict_logp(net, normalise(X, mu, sd), self.perm, self.sign))
                        for net, mu, sd, _, _ in use], 0)

    def predict(self, match):
        from members.oota.solo import decode
        from members.oota.solo.model import CODES
        states = decode.viterbi(np.log(np.clip(self.proba(match), 1e-6, 1)), decode.costs(self.cost, self.cost, self.cost))
        out = np.repeat(CODES[states], STEP)
        n = match.n_frames
        return np.r_[out, np.full(max(n - len(out), 0), out[-1], np.int8)][:n].astype(np.int8)
