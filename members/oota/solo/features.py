"""oota_solo の特徴量。試合（Match）だけから作る（他の人の部品は使わない）。

25 fps のフレームごとの値を作り、BIN フレーム（0.2 秒）ごとにまとめ、前後の窓の平均を足す。
swap=True はホームとアウェイを入れ替えた試合の特徴量（学習の水増しと、予測の対称化に使う）。
"""
import numpy as np

from possession.match import AWAY, HOME

BIN = 5                                   # まとめるフレームの数（25 fps で 0.2 秒）
WINDOWS = (2, 5, 15, 40)                  # 前後の窓の長さ [まとめた後の刻み]：0.4、1、3、8 秒


def _ffill_idx(valid):
    """各位置について、そこまでで最後に valid だった位置（無ければ -1）。"""
    idx = np.where(valid, np.arange(len(valid)), -1)
    return np.maximum.accumulate(idx)


def _gaps(vis):
    """-> (最後に見えてからのフレーム数, 次に見えるまでのフレーム数)。見えていれば 0。端は大きな値。"""
    n = len(vis)
    last = _ffill_idx(vis)
    nxt = n - 1 - _ffill_idx(vis[::-1])[::-1]
    since = np.where(last >= 0, np.arange(n) - last, 10 ** 6)
    until = np.where(nxt < n, nxt - np.arange(n), 10 ** 6)
    return since.astype(np.float32), until.astype(np.float32)


def _interp(ball, vis):
    out = ball.copy()
    if vis.any():
        t = np.arange(len(ball))
        for k in range(2):
            out[:, k] = np.interp(t, t[vis], ball[vis, k])
    return out


def _smooth(x, k):
    """幅 2k+1 の移動平均（NaN は無視）。"""
    w = np.ones(2 * k + 1)
    ok = ~np.isnan(x)
    num = np.convolve(np.where(ok, x, 0.0), w, "same")
    den = np.convolve(ok.astype(float), w, "same")
    return np.where(den > 0, num / np.maximum(den, 1), np.nan)


def _team_stats(d, xy, spd, ball_v, pvel, sel):
    """チーム sel の選手について、ボールとの距離などをまとめる。d は (T, S)。"""
    T = len(d)
    out = {}
    if sel.sum() == 0:
        z = np.full(T, np.nan, np.float32)
        return {k: z for k in ("d1", "d2", "d3", "n2", "n5", "n10", "spd1", "rel1", "cx", "cy", "sx", "sy", "mspd",
                               "still", "n", "xmin", "xmax")}
    dd = np.where(np.isnan(d[:, sel]), np.inf, d[:, sel])
    order = np.argsort(dd, axis=1)[:, :3]
    srt = np.take_along_axis(dd, order, axis=1)
    srt = np.where(np.isinf(srt), np.nan, srt)
    for i in range(3):
        out[f"d{i + 1}"] = srt[:, i] if srt.shape[1] > i else np.full(T, np.nan)
    for r in (2, 5, 10):
        out[f"n{r}"] = (dd < r).sum(1).astype(np.float32)
    i1 = order[:, 0]
    rows = np.arange(T)
    s = spd[:, sel]
    out["spd1"] = s[rows, i1]
    pv = pvel[:, sel][rows, i1]                              # 最も近い選手の速度
    out["rel1"] = np.hypot(pv[:, 0] - ball_v[:, 0], pv[:, 1] - ball_v[:, 1])
    x, y = xy[:, sel, 0], xy[:, sel, 1]
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out["cx"], out["cy"] = np.nanmean(x, 1), np.nanmean(y, 1)
            out["sx"], out["sy"] = np.nanstd(x, 1), np.nanstd(y, 1)
            out["mspd"] = np.nanmean(s, 1)
            out["still"] = np.nanmean(np.where(np.isnan(s), np.nan, s < 1.0), 1)
            out["xmin"], out["xmax"] = np.nanmin(x, 1), np.nanmax(x, 1)
    out["n"] = (~np.isnan(x)).sum(1).astype(np.float32)
    return out


def frame_features(m, swap=False):
    """25 fps のフレームごとの特徴量 -> (名前のリスト, (T, F) float32)。"""
    fps = m.fps
    T = m.n_frames
    team = m.team.copy()
    if swap:
        team = np.where(team == HOME, AWAY, np.where(team == AWAY, HOME, team))
    xy = m.xy.astype(np.float64)
    L, W = m.pitch
    ball = m.ball.astype(np.float64)
    vis = ~np.isnan(ball[:, 0])
    since, until = _gaps(vis)
    bi = _interp(ball, vis)
    # ボールの速度と加速度（見えているフレームの差分。欠けをまたぐ所は補間の値）
    k = max(1, int(round(0.08 * fps)))
    bs = np.stack([_smooth(bi[:, 0], k), _smooth(bi[:, 1], k)], 1)
    bv = np.gradient(bs, axis=0) * fps
    ba = np.gradient(bv, axis=0) * fps
    speed = np.hypot(bv[:, 0], bv[:, 1])
    acc = np.hypot(ba[:, 0], ba[:, 1])
    head = np.arctan2(bv[:, 1], bv[:, 0])
    lag = max(1, int(round(0.2 * fps)))
    dh = np.zeros(T)
    dh[lag:] = np.abs(np.angle(np.exp(1j * (head[lag:] - head[:-lag]))))
    dh = dh * (speed > 1.0)
    dspeed = np.zeros(T)
    dspeed[lag:] = speed[lag:] - speed[:-lag]
    # 選手の速度（枠が別の人に替わる跳びは欠けにする）
    pvel = np.full_like(xy, np.nan)
    pvel[1:] = (xy[1:] - xy[:-1]) * fps
    pspd = np.hypot(pvel[..., 0], pvel[..., 1])
    bad = pspd > 13.0
    pspd[bad] = np.nan
    pvel[bad] = np.nan
    d = np.hypot(xy[..., 0] - bi[:, None, 0], xy[..., 1] - bi[:, None, 1])
    st = {"h": _team_stats(d, xy, pspd, bv, pvel, team == HOME), "a": _team_stats(d, xy, pspd, bv, pvel, team == AWAY)}
    # 攻める向き：前後半ごとに、ホームの重心がアウェイの重心より −x 側ならホームは +x へ攻める
    sgn = np.ones(T)
    for p in np.unique(m.period):
        idx = m.period == p
        diff = np.nanmean(st["a"]["cx"][idx] - st["h"]["cx"][idx]) if idx.any() else 0.0
        sgn[idx] = 1.0 if not np.isfinite(diff) or diff >= 0 else -1.0
    # 前後半の始まりからの秒数
    tsec = np.zeros(T)
    for p in (1, 2):
        idx = np.flatnonzero(m.period == p)
        if len(idx):
            tsec[idx] = (idx - idx[0]) / fps
    f = {}
    f["vis"] = vis.astype(float)
    f["since"] = np.minimum(since / fps, 120)
    f["until"] = np.minimum(until / fps, 120)
    f["gap"] = np.minimum((since + until) / fps, 240)
    f["bx"] = bi[:, 0] * sgn                                  # ホームが攻める向きを + にそろえた x
    f["by_abs"] = np.abs(bi[:, 1])
    f["bx_abs"] = np.abs(bi[:, 0])
    f["out"] = np.maximum(np.abs(bi[:, 0]) - L / 2, np.abs(bi[:, 1]) - W / 2)
    f["dcorner"] = np.hypot(np.abs(bi[:, 0]) - L / 2, np.abs(bi[:, 1]) - W / 2)
    f["dcentre"] = np.hypot(bi[:, 0], bi[:, 1])
    f["speed"] = speed
    f["acc"] = np.minimum(acc, 200)
    f["dhead"] = dh
    f["dspeed"] = dspeed
    f["vx"] = bv[:, 0] * sgn
    f["tsec"] = np.minimum(tsec, 3600)
    f["t_early"] = np.minimum(tsec, 30)
    for t_ in ("h", "a"):
        s = st[t_]
        for kk in ("d1", "d2", "d3", "n2", "n5", "n10", "spd1", "rel1", "sx", "sy", "mspd", "still", "n"):
            f[f"{t_}_{kk}"] = s[kk]
        f[f"{t_}_cx"] = s["cx"] * sgn
        f[f"{t_}_bcx"] = (bi[:, 0] - s["cx"]) * sgn           # ボールが重心よりどれだけ前（ホームの攻める向き）か
        f[f"{t_}_bcy"] = np.abs(bi[:, 1] - s["cy"])
        cv = np.gradient(_smooth(s["cx"], int(fps // 2)), axis=0) * fps * sgn
        f[f"{t_}_cvx"] = cv
        # 最終ライン（GK を除くため 2 番目ではなく端から。粗い近似）
        f[f"{t_}_back"] = np.where(sgn > 0, s["xmin"], -s["xmax"]) if t_ == "h" else np.where(sgn > 0, -s["xmax"], s["xmin"])
    f["d1_diff"] = f["h_d1"] - f["a_d1"]
    f["d1_min"] = np.fmin(f["h_d1"], f["a_d1"])
    f["d2_diff"] = f["h_d2"] - f["a_d2"]
    f["n5_diff"] = f["h_n5"] - f["a_n5"]
    f["cx_diff"] = f["h_cx"] - f["a_cx"]
    f["sx_diff"] = f["h_sx"] - f["a_sx"]
    f["sy_diff"] = f["h_sy"] - f["a_sy"]
    f["cvx_sum"] = f["h_cvx"] + f["a_cvx"]
    f["mspd_all"] = (f["h_mspd"] + f["a_mspd"]) / 2
    f["still_all"] = (f["h_still"] + f["a_still"]) / 2
    names = list(f)
    return names, np.stack([f[n] for n in names], 1).astype(np.float32)


CONTEXT = ("vis", "gap", "out", "speed", "acc", "dhead", "vx", "bx", "by_abs", "d1_diff", "d1_min", "h_d1", "a_d1",
           "n5_diff", "h_n2", "a_n2", "cx_diff", "sx_diff", "sy_diff", "cvx_sum", "h_cvx", "a_cvx", "mspd_all",
           "still_all", "h_rel1", "a_rel1", "h_bcx", "a_bcx")


def _bin(x, how="mean"):
    T = len(x) // BIN * BIN
    r = x[:T].reshape(-1, BIN, *x.shape[1:])
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(r, 1) if how == "mean" else (np.nanmax(r, 1) if how == "max" else np.nanmin(r, 1))


def _roll(c, w):
    """c は NaN を 0 にした累積和（先頭に 0）と数。-> 過去 w 刻み（今を含む）の平均、未来 w 刻み（今を除く）の平均。"""
    cs, cn = c
    n = len(cs) - 1
    i = np.arange(n)
    a0, a1 = np.maximum(i + 1 - w, 0), i + 1
    b0, b1 = np.minimum(i + 1, n), np.minimum(i + 1 + w, n)
    with np.errstate(all="ignore"):
        past = (cs[a1] - cs[a0]) / np.maximum(cn[a1] - cn[a0], 1)
        fut = (cs[b1] - cs[b0]) / np.maximum(cn[b1] - cn[b0], 1)
    past = np.where(cn[a1] - cn[a0] > 0, past, np.nan)
    fut = np.where(cn[b1] - cn[b0] > 0, fut, np.nan)
    return past, fut


def features(m, swap=False):
    """まとめた後の特徴量 -> (名前のリスト, (N, F) float32)。N = n_frames // BIN。"""
    names, F = frame_features(m, swap)
    B = _bin(F)
    cols, out_names = [B], list(names)
    for extra, how in (("acc", "max"), ("dhead", "max"), ("d1_min", "min"), ("h_d1", "min"), ("a_d1", "min")):
        cols.append(_bin(F[:, names.index(extra)], how)[:, None])
        out_names.append(f"{extra}_{how}")
    for cname in CONTEXT:
        x = B[:, names.index(cname)].astype(np.float64)
        ok = ~np.isnan(x)
        c = (np.r_[0, np.cumsum(np.where(ok, x, 0))], np.r_[0, np.cumsum(ok)])
        for w in WINDOWS:
            p, q = _roll(c, w)
            cols += [p[:, None], q[:, None], (q - p)[:, None]]
            out_names += [f"{cname}_p{w}", f"{cname}_f{w}", f"{cname}_d{w}"]
    return out_names, np.concatenate(cols, 1).astype(np.float32)


def bin_labels(truth):
    """正解をまとめる：各刻みで最も多い符号。"""
    T = len(truth) // BIN * BIN
    r = truth[:T].reshape(-1, BIN)
    cnt = np.stack([(r == c).sum(1) for c in range(4)], 1)
    return cnt.argmax(1).astype(np.int8)
