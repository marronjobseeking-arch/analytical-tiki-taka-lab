"""Convert the provided automatic tracking (FIFA/EPTS format) into the common `Match` format.

- Array index = tactical-video frame = submission row (0 to 199318). The tracking's frame ids
  start at 1, so 1 is subtracted. Frames without a record are NaN.
- `period` is 1 or 2 inside the graded blocks of `games.half_video_blocks` and 0 elsewhere.
- A slot is a track fragment, not a person. Its team comes from the metadata; the first team is home.
- The pitch size comes from `games.py`.
"""
import numpy as np

import games
from . import epts
from .match import AWAY, HOME, Match, UNKNOWN

GAME = games.GAME


def build(meta_path, raw_path, n_frames, blocks, pitch, match_id=GAME):
    """Build a Match from the two tracking files. blocks is [(half, first frame, last frame)] in video frames."""
    meta = epts.read_meta(meta_path)
    frame_id, pids, xy01, ball01 = epts.read_raw(raw_path, meta)
    vf = frame_id - 1
    if vf.min() < 0 or vf.max() >= n_frames:
        raise ValueError(f"tracking frames {vf.min()}..{vf.max()} fall outside the video range 0..{n_frames - 1}")
    xy = np.full((n_frames, len(pids), 2), np.nan, np.float32)
    ball = np.full((n_frames, 2), np.nan, np.float32)
    xy[vf] = epts.to_metres(xy01, pitch)
    ball[vf] = epts.to_metres(ball01, pitch)
    period = np.zeros(n_frames, np.int8)
    for half, a, b in blocks:
        period[a:b + 1] = half
    home = meta["teams"][0][0]
    team = np.array([HOME if meta["players"].get(p) == home else AWAY if p in meta["players"] else UNKNOWN
                     for p in pids], np.int8)
    m_meta = dict(teams=meta["teams"], tracks=pids, n_records=int(len(vf)))
    return Match(id=match_id, source="atd", fps=meta["fps"], period=period, xy=xy, team=team, ball=ball,
                 pitch=tuple(pitch), meta=m_meta)

