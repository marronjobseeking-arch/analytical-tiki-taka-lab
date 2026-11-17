"""フレームごとの手がかり（kashiwagi_allfeat、kashiwagi_nn）。入力は `Match` だけ（正解を使わない）。

前半・後半を別々に作る（窓が半をまたがない）。半の外は NaN。H と A を入れ替えると符号が反転する列（反対称、`Xa`）と、
変わらない列（対称、`Xs`）に分ける。欠け（ボールが見えない、選手がいない）は NaN のまま返す。

攻める向き：半ごとに、A の x の平均 − H の x の平均の符号（H が負の x の側にいれば +x へ攻める）。各チームの
「自分の攻める向きの座標」を o と書く（H は x·向き、A は −x·向き）。

| まとまり | 反対称（H − A、または H の攻める向きの量） | 対称 |
|---|---|---|
| ボール | x（H の攻める向き）、その速度、最後・次に見えた x | 見えるか、前後 1・5 秒で見える割合、最後に見えてから・次に見えるまでの時間、速さ・加速（ならしたものも）、\\|y\\|、タッチライン・ゴールライン・ペナルティマーク・中央への距離、ピッチの外への距離（最後・次に見えた値も） |
| 距離 | 最も近い選手の距離の差 dA − dH（±10 m で切る）、その log、近い（1・2・3・5 m）の差、3・5・10 m 以内の人数の差。0.5〜8 秒のならし、前後 1・3 秒の平均とその差、最後・次に見えたときの差 | 近い方の距離とその log、両チームの人数の和 |
| 並び・動き | 重心の o、縦・横の広がり、最後尾（10%）・前線（90%）、最も後ろの選手（GK）、まとまり、相手陣の割合、平均の速さ、攻める向きの速さ、遅い選手の割合。1・4・10 秒のならしと、前後 2 秒の差 | 同じ量の両チームの和、全体の遅い選手の割合とならし |
| 場面 | ペナルティエリアの人数の差 | 半の始まりからの時間、ペナルティエリアの人数の多い方、ボールから 9.15 m 以内の人数、最後のライン越えからの時間、次の動き出しまでの時間、dead_v1 の手がかり（`dead.features`） |
"""
import warnings

import numpy as np

from possession.match import AWAY, HOME

from members.kashiwagi import dead

GAP_MAX = 10.0
NEAR_R = (1.0, 2.0, 3.0, 5.0)
COUNT_R = (3.0, 5.0, 10.0)
ROLL_S = (0.5, 1.0, 2.0, 4.0, 8.0)
SIDE_S = (1.0, 3.0)
SHAPE_ROLL_S = (1.0, 4.0, 10.0)
TIME_MAX_S = 60.0
SPEED_LAG_S = 0.4
SPEED_MAX = 12.0             # これより速いのは枠の入れ替わり
BOX_X, BOX_Y = 16.5, 20.16   # ペナルティエリア
SPOT = 11.0


def roll(x, w):
    """中心の窓 w フレームの平均（NaN を除く。端は詰める。全部 NaN なら NaN）。"""
    x = np.asarray(x, float)
    if w <= 1:
        return x
    ok = ~np.isnan(x)
    c = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
    k = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w // 2), np.minimum(len(x), i + w - w // 2)
    n = k[hi] - k[lo]
    return np.where(n > 0, (c[hi] - c[lo]) / np.maximum(n, 1), np.nan)


def side(x, w, future):
    """直後 w フレーム（future）か直前 w フレーム（今のフレームを含まない）の平均（NaN を除く）。"""
    x = np.asarray(x, float)
    ok = ~np.isnan(x)
    c = np.r_[0.0, np.cumsum(np.where(ok, x, 0.0))]
    k = np.r_[0, np.cumsum(ok)]
    i = np.arange(len(x))
    lo, hi = (np.minimum(len(x), i + 1), np.minimum(len(x), i + 1 + w)) if future else (np.maximum(0, i - w), i)
    n = k[hi] - k[lo]
    return np.where(n > 0, (c[hi] - c[lo]) / np.maximum(n, 1), np.nan)


def last_idx(ok):
    """各フレームまでで最後に ok だった添字（無ければ −1）。"""
    return np.maximum.accumulate(np.where(ok, np.arange(len(ok)), -1))


def next_idx(ok):
    """各フレームから先で最初に ok の添字（無ければ len）。"""
    n = len(ok)
    return np.minimum.accumulate(np.where(ok, np.arange(n), n)[::-1])[::-1]


def _at(v, idx):
    ok = (idx >= 0) & (idx < len(v))
    return np.where(ok, v[np.clip(idx, 0, len(v) - 1)], np.nan)


def _since_until(ok, fps):
    i = np.arange(len(ok))
    li, ni = last_idx(ok), next_idx(ok)
    since = np.where(li >= 0, np.minimum((i - li) / fps, TIME_MAX_S), TIME_MAX_S)
    until = np.where(ni < len(ok), np.minimum((ni - i) / fps, TIME_MAX_S), TIME_MAX_S)
    return li, ni, since, until


def _nanq(v, q):
    """(T, S) の行ごとの分位点（NaN を除く、最も近い順位）。行が全部 NaN なら NaN。"""
    s = np.sort(v, axis=1)                          # NaN は後ろへ
    n = (~np.isnan(v)).sum(1)
    k = np.clip(np.round(q * (n - 1)).astype(int), 0, v.shape[1] - 1)
    return np.where(n > 0, s[np.arange(len(v)), k], np.nan)


def _team(xy, d, fps, sign):
    """1 チームの並びと動き（o は自分の攻める向きの座標）。-> {名前: (T,)}"""
    o, y = sign * d * xy[..., 0], xy[..., 1]
    lag = max(1, int(round(SPEED_LAG_S * fps)))
    dxy = np.full_like(xy, np.nan)
    dxy[lag:] = (xy[lag:] - xy[:-lag]) / (lag / fps)
    sp = np.hypot(dxy[..., 0], dxy[..., 1])
    bad = ~(sp <= SPEED_MAX)
    sp[bad] = np.nan
    vo = np.where(bad, np.nan, sign * d * dxy[..., 0])
    cx, cy = np.nanmean(xy[..., 0], 1), np.nanmean(y, 1)
    return dict(
        cen=np.nanmean(o, 1), depth=np.nanstd(o, 1), width=np.nanstd(y, 1),
        last=_nanq(o, 0.1), front=_nanq(o, 0.9), gk=np.nanmin(o, 1),
        compact=np.nanmean(np.hypot(xy[..., 0] - cx[:, None], y - cy[:, None]), 1),
        opp_half=np.nanmean(np.where(np.isnan(o), np.nan, o > 0), 1),
        speed=np.nanmean(sp, 1), v_fwd=np.nanmean(vo, 1),
        slow=np.nanmean(np.where(np.isnan(sp), np.nan, sp < 1.0), 1),
        n=(~np.isnan(o)).sum(1).astype(float),
    )


def _half(xy, team, ball, fps, pitch):
    """1 つの半の手がかり -> (反対称の dict, 対称の dict)。"""
    T = len(ball)
    f = lambda s: max(1, int(round(s * fps)))
    L, W = pitch[0] / 2, pitch[1] / 2
    h, a = team == HOME, team == AWAY
    mh, ma = np.nanmean(xy[:, h, 0]) if h.any() else np.nan, np.nanmean(xy[:, a, 0]) if a.any() else np.nan
    d = 1.0 if not np.isfinite(ma - mh) or ma == mh else float(np.sign(ma - mh))
    A, S = {}, {}
    bx, by = ball[:, 0], ball[:, 1]
    vis = ~np.isnan(bx)
    visf = vis.astype(float)

    # ボール
    S["vis"] = visf
    for w in (1, 5):
        S[f"vis_m{w}"] = roll(visf, f(w))
    li, ni, S["since"], S["until"] = _since_until(vis, fps)
    v = np.full((T, 2), np.nan)
    v[1:] = np.diff(ball, axis=0) * fps
    sp = np.hypot(v[:, 0], v[:, 1])
    S["b_speed"] = sp
    S["b_speed_m0.5"] = roll(sp, f(0.5))
    S["b_speed_m2"] = roll(sp, f(2))
    acc = np.r_[np.nan, np.abs(np.diff(S["b_speed_m0.5"])) * fps]
    S["b_acc"] = acc
    S["b_acc_m1"] = roll(acc, f(1))
    A["b_vx"] = d * roll(v[:, 0], f(0.5))
    bxo = d * bx
    A["b_x"] = bxo
    A["b_x_last"], A["b_x_next"] = _at(bxo, li), _at(bxo, ni)
    ay = np.abs(by)
    out = np.maximum.reduce([np.abs(bx) - L, ay - W, np.zeros(T)])
    out[~vis] = np.nan
    S["b_absy"], S["b_absy_last"], S["b_absy_next"] = ay, _at(ay, li), _at(ay, ni)
    S["b_touch"], S["b_goal"] = W - ay, L - np.abs(bx)
    S["b_spot"] = np.hypot(L - SPOT - np.abs(bx), by)
    S["b_centre"] = np.hypot(bx, by)
    S["b_out"], S["b_out_last"], S["b_out_next"] = out, _at(out, li), _at(out, ni)

    # 距離
    dist = np.hypot(xy[..., 0] - bx[:, None], xy[..., 1] - by[:, None])
    dh = np.nanmin(np.where(h, dist, np.inf), 1)
    da = np.nanmin(np.where(a, dist, np.inf), 1)
    dh[~vis | np.isinf(dh)] = np.nan
    da[~vis | np.isinf(da)] = np.nan
    dh, da = np.where(vis & np.isnan(dh), 50.0, dh), np.where(vis & np.isnan(da), 50.0, da)   # チームが 1 人も見えない
    gap = np.clip(da - dh, -GAP_MAX, GAP_MAX)
    A["gap"] = gap
    A["gap_log"] = np.sign(gap) * np.log1p(np.abs(gap))
    for r in NEAR_R:
        A[f"near{r:g}"] = np.where(vis, (dh < r).astype(float) - (da < r), np.nan)
    for r in COUNT_R:
        nh = np.where(vis, ((dist < r) & h).sum(1), np.nan)
        na = np.where(vis, ((dist < r) & a).sum(1), np.nan)
        A[f"cnt{r:g}"], S[f"cnt{r:g}_all"] = nh - na, nh + na
    S["cnt9.15_all"] = np.where(vis, (dist < 9.15).sum(1), np.nan)
    S["dmin"] = np.fmin(dh, da)
    S["dmin_log"] = np.log1p(S["dmin"])
    for nm in ("gap", "near2"):
        x = A[nm]
        for w in ROLL_S:
            A[f"{nm}_m{w:g}"] = roll(x, f(w))
        for w in SIDE_S:
            p, n = side(x, f(w), False), side(x, f(w), True)
            A[f"{nm}_past{w:g}"], A[f"{nm}_next{w:g}"], A[f"{nm}_d{w:g}"] = p, n, n - p
        A[f"{nm}_last"], A[f"{nm}_nextseen"] = _at(x, li), _at(x, ni)

    # 並び・動き
    th, ta = _team(xy[:, h], d, fps, 1.0), _team(xy[:, a], d, fps, -1.0)
    for k in th:
        A[f"t_{k}"] = th[k] - ta[k]
        S[f"t_{k}_sum"] = th[k] + ta[k]
    for k in ("cen", "front", "last", "v_fwd", "speed", "opp_half", "compact"):
        x = A[f"t_{k}"]
        for w in SHAPE_ROLL_S:
            A[f"t_{k}_m{w:g}"] = roll(x, f(w))
        A[f"t_{k}_d2"] = side(x, f(2), True) - side(x, f(2), False)
    for k in ("speed", "slow"):
        for w in SHAPE_ROLL_S:
            S[f"t_{k}_sum_m{w:g}"] = roll(S[f"t_{k}_sum"], f(w))
        S[f"t_{k}_sum_d2"] = side(S[f"t_{k}_sum"], f(2), True) - side(S[f"t_{k}_sum"], f(2), False)

    # 場面
    S["t_half"] = np.arange(T) / fps / 60.0
    x, ya = d * xy[..., 0], np.abs(xy[..., 1])
    in_box = ya < BOX_Y
    box_h = (in_box & (x > L - BOX_X)).sum(1).astype(float)          # H の攻める側のエリア
    box_a = (in_box & (x < -L + BOX_X)).sum(1).astype(float)
    A["box"], S["box_max"] = box_h - box_a, np.maximum(box_h, box_a)
    A["box_m2"], S["box_max_m2"] = roll(A["box"], f(2)), roll(S["box_max"], f(2))
    _, _, S["since_cross"], _ = _since_until(dead.crossings(ball, pitch), fps)
    _, _, _, S["until_kick"] = _since_until(dead.kicks(ball), fps)
    return A, S


def features(match):
    """-> (反対称の名前, Xa (T, Fa), 対称の名前, Xs (T, Fs))。float32、欠けと半の外は NaN。"""
    T = match.n_frames
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        dv = dead.features(match)
        parts = []
        for p in (1, 2):
            idx = np.flatnonzero(match.period == p)
            if len(idx):
                s, e = idx[0], idx[-1] + 1
                parts.append((s, e, *_half(match.xy[s:e], match.team, match.ball[s:e], match.fps, match.pitch)))
    if not parts:
        raise ValueError(f"{match.id}: 前半も後半も無い")
    an, sn = list(parts[0][2]), list(parts[0][3]) + [f"dv1_{k}" for k in dv]
    Xa = np.full((T, len(an)), np.nan, np.float32)
    Xs = np.full((T, len(sn)), np.nan, np.float32)
    for s, e, A, S in parts:
        Xa[s:e] = np.column_stack([A[k] for k in an])
        Xs[s:e, :len(S)] = np.column_stack([S[k] for k in S])
        Xs[s:e, len(S):] = np.column_stack([dv[k][s:e] for k in dv])
    return an, Xa, sn, Xs
