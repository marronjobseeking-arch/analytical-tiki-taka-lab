"""Dead-ball features from player motion and the ball track (used as model inputs). Assumes 25 Hz."""
import warnings

import numpy as np

from possession.match import AWAY, HOME

FPS = 25                     # ponytail: 窓はフレーム数で固定。25 Hz 以外のデータが来たら fps から作る
PLAYER = ["v_mean_10s", "v_mean_3s", "still_mean_s", "slow_share", "v_team_min"]
SNAP = 2 * FPS               # 境目を寄せる範囲 ±2 秒
OUT_MARGIN = 0.5             # ライン越え：ラインの外 0.5 m
MOVE, SPIKE = 2.0, 40.0      # 動き出し：2〜40 m/s（40 より速いのは跳び）
STILL = FPS // 2             # 動き出しの前に 0.5 秒止まっている


def _roll(a, w, f=np.nanmean):
    """中心の窓 w フレームの統計（NaN を除く）。"""
    pad = np.pad(a.astype(float), (w // 2, w - 1 - w // 2), constant_values=np.nan)
    return f(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)


def _run(mask):
    """各フレームで終わる、True の続いたフレーム数。"""
    c = np.cumsum(mask)
    return c - np.maximum.accumulate(np.where(mask, 0, c))


def stretches(mask):
    """True のひと続き [(始まり, 終わり（含まない）)]。"""
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(e[::2], e[1::2]))


def _ball_speed(ball):
    return np.hypot(*np.diff(ball, axis=0, prepend=np.nan).T) * FPS


def features(m):
    """dead_v1b の手がかり {名前: (T,)}。全部 NaN の窓（ボールや選手の欠け）は普通にあるので、その警告は出さない。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return _features(m)


def _features(m):
    px, py = m.xy[..., 0], m.xy[..., 1]
    vp = np.hypot(np.diff(px, axis=0, prepend=np.nan), np.diff(py, axis=0, prepend=np.nan)) * FPS
    vp[vp > 12] = np.nan                                    # 12 m/s より速いのは枠の入れ替わり
    v_mean = np.nanmean(vp, 1)
    v_team = np.fmin(np.nanmean(vp[:, m.team == HOME], 1), np.nanmean(vp[:, m.team == AWAY], 1))
    slow = np.nanmean(np.where(np.isnan(vp), np.nan, vp < 1.0), 1)
    half_l, half_w = m.pitch[0] / 2, m.pitch[1] / 2
    out_m = np.maximum.reduce([np.abs(m.ball[:, 0]) - half_l, np.abs(m.ball[:, 1]) - half_w, np.zeros(len(m.ball))])
    v_mean_1s = _roll(v_mean, FPS)
    return dict(
        v_mean_10s=_roll(v_mean, 10 * FPS), v_mean_3s=_roll(v_mean, 3 * FPS),
        still_mean_s=_run(v_mean_1s < 1.5) / FPS, slow_share=_roll(slow, FPS), v_team_min=_roll(v_team, FPS),
        ball_out_m=np.nan_to_num(out_m), ball_missing_log=np.log1p(_run(np.isnan(m.ball[:, 0])) / FPS),
        ball_jump=np.nan_to_num(_roll((_ball_speed(m.ball) > 50).astype(float), FPS, np.nanmax)),
    )


def player_design(f):
    X = np.column_stack([f[n] for n in PLAYER])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nan_to_num(np.nanmedian(X, 0))
    return np.where(np.isnan(X), med, X)    # 欠けは試合の中央値で埋める（個人のリポジトリと同じ）


class LogReg:
    """標準化 ＋ ロジスティック回帰（クラスの重みを釣り合わせ、L2 = 1。sklearn の既定 C=1 と同じ）。ニュートン法。"""

    def fit(self, X, y, iters=50):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd < 1e-9] = 1.0
        Z = np.column_stack([np.ones(len(X)), (X - self.mu) / self.sd])
        w = np.where(y, len(y) / (2 * max(y.sum(), 1)), len(y) / (2 * max((~y).sum(), 1)))
        reg = np.r_[0.0, np.ones(X.shape[1])]
        self.beta = np.zeros(Z.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-Z @ self.beta))
            step = np.linalg.solve((Z * (w * p * (1 - p))[:, None]).T @ Z + np.diag(reg + 1e-9),
                                   Z.T @ (w * (p - y)) + reg * self.beta)
            self.beta -= step
            if np.abs(step).max() < 1e-8:
                break
        return self

    def logit(self, X):
        return self.beta[0] + ((X - self.mu) / self.sd) @ self.beta[1:]


def stack_design(f, v1):
    p = np.clip(1 / (1 + np.exp(-v1.logit(player_design(f)))), 1e-6, 1 - 1e-6)
    return np.column_stack([np.log(p / (1 - p)), f["ball_out_m"], f["ball_missing_log"], f["ball_jump"]])


def smooth(p, on=0.6, off=0.4, min_dead=2 * FPS, min_alive=FPS):
    """デッドの確率 -> デッドか（bool）。1 秒の平均、ヒステリシス、1 秒未満のインプレーを埋め、2 秒未満のデッドを捨てる。"""
    p = _roll(p, FPS)
    d = np.zeros(len(p), bool)
    s = False
    for i, v in enumerate(p):
        s = v > (off if s else on)
        d[i] = s
    for val, mn in ((False, min_alive), (True, min_dead)):
        for a, b in stretches(d == val):
            if b - a < mn and a > 0 and b < len(d):
                d[a:b] = not val
    return d


def kicks(ball):
    """ボールが 0.5 秒止まった（2 m/s 以下）後に動き出す（2〜40 m/s）フレーム。"""
    v = _ball_speed(ball)
    prev = np.r_[0, _run(np.nan_to_num(v, nan=np.inf) <= MOVE)[:-1]]
    return (v > MOVE) & (v < SPIKE) & (prev >= STILL)


def crossings(ball, pitch):
    """見えているボールが、ピッチの中（＋余白）から外へ出るフレーム。"""
    out = (np.abs(ball[:, 0]) > pitch[0] / 2 + OUT_MARGIN) | (np.abs(ball[:, 1]) > pitch[1] / 2 + OUT_MARGIN)
    inside = ~np.isnan(ball[:, 0]) & ~out
    return out & np.r_[False, inside[:-1]]


def _first(flag, lo, hi):
    lo, hi = max(lo, 0), max(hi, 0)
    w = np.flatnonzero(flag[lo:hi])
    return lo + w[0] if len(w) else None


def refine(dead, ball, pitch):
    """デッドの始まりをライン越えへ、終わりを動き出しへ寄せる（どちらも ±2 秒、隣のデッドの中へは入らない）。"""
    cr, kk = crossings(ball, pitch), kicks(ball)
    st = stretches(dead)
    out = np.zeros_like(dead)
    for i, (a, b) in enumerate(st):
        lo_a = st[i - 1][1] if i else 0
        hi_b = st[i + 1][0] if i + 1 < len(st) else len(dead)
        s = _first(cr, max(a - SNAP, lo_a), a + SNAP + 1)
        a2 = s if s is not None and s < b else a
        k = _first(kk, max(b - SNAP, a2 + 1), min(b + SNAP + 1, hi_b))
        out[a2:k if k is not None else b] = True
    return out


class DeadV1b:
    def fit(self, clean, degraded, every=5):
        """clean・degraded：[(特徴の dict, デッドか (T,) bool, 採点する (T,) bool)]。v1 は clean、重ねる方は degraded で学習。"""
        def stack(rows, design):
            return (np.vstack([design(f)[ok][::every] for f, _, ok in rows]),
                    np.concatenate([y[ok][::every] for _, y, ok in rows]))
        self.v1 = LogReg().fit(*stack(clean, player_design))
        self.v1b = LogReg().fit(*stack(degraded, lambda f: stack_design(f, self.v1)))
        return self

    def prob(self, f):
        return 1 / (1 + np.exp(-self.v1b.logit(stack_design(f, self.v1))))

    def detect(self, m, f=None):
        return refine(smooth(self.prob(features(m) if f is None else f)), m.ball, m.pitch)
