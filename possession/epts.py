"""FIFA/EPTS 形式（メタデータの XML と、生のトラッキングのテキスト）を読む。

本番の ATD（BAR-ATH_tactical_atd.xml / .txt）と、Metrica のサンプルの 3 試合目が同じ形式。
生のテキストは 1 行 1 フレームで、`<フレーム番号>:<枠>;<枠>;...:<ボール>`。枠の中の値の並び
（x, y だけか、速さも入るか）と枠の並びは、メタデータの DataFormatSpecification が決めていて、
フレームの範囲ごとに別の指定がありうる（交代で枠の並びが変わるなど）。ここではその指定どおりに読む。

座標はピッチの 0〜1。左上が (0, 0)。共通の形式（m、中央が原点）への直しは呼ぶ側で行う。
"""
import xml.etree.ElementTree as ET

import numpy as np


def _tag(e):
    return e.tag.rsplit("}", 1)[-1]


def _find_all(root, name):
    return [e for e in root.iter() if _tag(e) == name]


def read_meta(path):
    """-> dict(fps, pitch, teams [(id, name)], players {player_id: team_id}, specs, params)

    specs は [(start, end, [(player_id, [channel, ...]), ...], [ball_channel, ...])]。
    teams はファイルに出てくる順（FIFA/EPTS の慣例で、最初がホーム）。
    """
    root = ET.parse(path).getroot()
    fps = float(_find_all(root, "FrameRate")[0].text)
    fs = _find_all(root, "FieldSize")
    pitch = (float(_find_all(fs[0], "Width")[0].text), float(_find_all(fs[0], "Height")[0].text)) if fs else None
    teams = []
    for t in _find_all(root, "Team"):
        names = [c.text for c in t if _tag(c) == "Name"]
        teams.append((t.get("id"), names[0] if names else None))
    players = {p.get("id"): p.get("teamId") for p in _find_all(root, "Player")}
    channel = {c.get("id"): (c.get("playerId"), c.get("channelId")) for c in _find_all(root, "PlayerChannel")}
    params = {}
    for p in _find_all(root, "ProviderParameter"):
        k = [c.text for c in p if _tag(c) == "Name"]
        v = [c.text for c in p if _tag(c) == "Value"]
        if k:
            params[k[0]] = v[0] if v else None
    specs = []
    for s in _find_all(root, "DataFormatSpecification"):
        groups = []
        for g in [c for c in s if _tag(c) == "SplitRegister"][0]:
            refs = [channel[r.get("playerChannelId")] for r in g if _tag(r) == "PlayerChannelRef"]
            if refs:
                groups.append((refs[0][0], [ch for _, ch in refs]))
        ball = [r.get("channelId") for r in s.iter() if _tag(r) == "BallChannelRef"]
        specs.append((int(s.get("startFrame")), int(s.get("endFrame")), groups, ball))
    return dict(fps=fps, pitch=pitch, teams=teams, players=players, specs=specs, params=params)


def read_raw(raw_path, meta):
    """-> (frame (T,), player_ids [S], xy (T, S, 2), ball (T, 2))。値は 0〜1 のまま、欠けは NaN。"""
    pids = []
    for _, _, groups, _ in meta["specs"]:
        for pid, _ in groups:
            if pid not in pids:
                pids.append(pid)
    col = {pid: j for j, pid in enumerate(pids)}
    plans = []
    for start, end, groups, ball in meta["specs"]:
        plans.append((start, end, [(col[pid], chs.index("x"), chs.index("y")) for pid, chs in groups],
                      (ball.index("x"), ball.index("y")) if ball else None))
    frames, xys, balls = [], [], []
    plan = plans[0]
    with open(raw_path) as f:
        for line in f:
            head, players, ball = line.rstrip("\n").split(":")
            n = int(head)
            if not plan[0] <= n <= plan[1]:
                plan = next((p for p in plans if p[0] <= n <= p[1]), None)
                if plan is None:
                    continue
            xy = np.full((len(pids), 2), np.nan, np.float32)
            for (j, ix, iy), group in zip(plan[2], players.split(";")):
                v = group.split(",")
                if len(v) > max(ix, iy):                 # 欠けは "NaN,NaN,..." のことも "NaN" だけのこともある
                    xy[j, 0], xy[j, 1] = float(v[ix]), float(v[iy])
            b = ball.split(",")
            frames.append(n)
            xys.append(xy)
            bi = plan[3]
            balls.append((float(b[bi[0]]), float(b[bi[1]])) if bi and len(b) > max(bi) else (np.nan, np.nan))
    return np.array(frames), pids, np.stack(xys), np.array(balls, np.float32)


def to_metres(xy, pitch):
    """0〜1（左上が原点）を m（中央が原点）に直す。y は下向きのまま（向きは推定する側が扱う）。"""
    out = np.empty_like(xy)
    out[..., 0] = (xy[..., 0] - 0.5) * pitch[0]
    out[..., 1] = (xy[..., 1] - 0.5) * pitch[1]
    return out
