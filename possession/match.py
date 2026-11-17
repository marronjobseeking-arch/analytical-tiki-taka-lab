"""The common match format. Every loader converts its data into this form.

- Coordinates are in metres with the pitch centre at the origin; x runs goal line to goal line, y across the pitch.
- Attack direction is not normalised; the feature code estimates it.
- A slot is a track fragment, not a person. Only its team is known.
- Missing values are NaN. Labels (H/A/D/X) are never part of a Match.
"""
from dataclasses import dataclass, field

import numpy as np

from evaluation import possession_metrics as pm

H, A, D, X = pm.H, pm.A, pm.D, pm.X
HOME, AWAY, UNKNOWN = 0, 1, -1


@dataclass
class Match:
    id: str
    source: str                   # "atd" for the provided tracking
    fps: float
    period: np.ndarray            # (T,) int8: 1 = first half, 2 = second half, 0 = outside the halves
    xy: np.ndarray                # (T, S, 2) float32: position per slot
    team: np.ndarray              # (S,) int8。HOME / AWAY / UNKNOWN
    ball: np.ndarray              # (T, 2) float32
    pitch: tuple                  # (length, width) in metres
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        T, S = self.xy.shape[:2]
        assert self.xy.shape == (T, S, 2), self.xy.shape
        assert self.period.shape == (T,) and self.ball.shape == (T, 2), (self.period.shape, self.ball.shape)
        assert self.team.shape == (S,), self.team.shape

    @property
    def n_frames(self):
        return len(self.period)

    def save(self, path):
        np.savez_compressed(path, id=self.id, source=self.source, fps=self.fps, period=self.period,
                            xy=self.xy, team=self.team, ball=self.ball, pitch=np.asarray(self.pitch),
                            meta=np.asarray(repr(self.meta)))

    @classmethod
    def load(cls, path):
        import ast
        z = np.load(path, allow_pickle=False)
        return cls(id=str(z["id"]), source=str(z["source"]), fps=float(z["fps"]), period=z["period"],
                   xy=z["xy"], team=z["team"], ball=z["ball"], pitch=tuple(z["pitch"].tolist()),
                   meta=ast.literal_eval(str(z["meta"])))

