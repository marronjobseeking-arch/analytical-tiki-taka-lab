"""Regenerate the submission file from the organisers' data package.

    uv run python -m pipeline.run --team "<team name>"

Steps: convert the tracking, run the 21 networks, mix and decode, write and validate the CSV.
Everything written goes under out/, which is not committed. No label is read at any step.
"""
import argparse
import hashlib
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import games
from evaluation import possession_metrics as pm
from members.higuchi.team_lb.ens_20261116 import core
from possession import load_atd
from possession import paths as P
from possession.match import Match

from . import infer

GAME = "BAR-ATH"
# Fixed ensemble weights: dead-ball probability from the two 140-feature models, team probability
# weighted towards the 62-feature model.
MIXING = {
    "family": "fixed_mean",
    "mode": "factorized_mean",
    "dead_weights": {"gru4000_s012": 0.5, "cgru4000_s012": 0.5, "oota3000": 0.0},
    "ha_weights": {"gru4000_s012": 0.25, "cgru4000_s012": 0.25, "oota3000": 0.5},
    "weights": {"gru4000_s012": 1 / 3, "cgru4000_s012": 1 / 3, "oota3000": 1 / 3},
}
SWITCH_PENALTY = 16.0
# SHA-256 of the label array we submitted (core.sha256_array). The run reports whether it matches.
SUBMITTED_LABELS_SHA256 = "20b6d591b9c801ca093edf31ba56cdecfb58f29c41c43fd66a210c35265f0ef3"


def convert_tracking():
    """Organisers' automatic tracking -> the common Match format (saved under out/match/)."""
    P.OUT_MATCH.mkdir(parents=True, exist_ok=True)
    path = P.OUT_MATCH / f"{GAME}.npz"
    # Only the two tracking files are read. Smart Tagging is not loaded.
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], GAME)
    match = load_atd.build(meta_path, raw_path, games.source_frames(game=GAME), games.half_video_blocks(GAME),
                           games.GAMES[GAME]["pitch"], GAME)
    match.save(path)
    return Match.load(path)


def ensemble(match, sources):
    """Fixed mixing of the three sources, then the three-state Viterbi decoder."""
    period = np.asarray(match.period, np.int8)
    active = np.isin(period, (1, 2))
    arrays = {}
    for name, proba in sources.items():
        safe = np.asarray(proba, dtype=np.float64).copy()
        safe[~active] = 1.0 / 3.0
        arrays[name] = safe
    probability = core.combine_fixed_mean(arrays, MIXING, output_dtype="float64")
    return core.viterbi_decode(np.asarray(probability, np.float64), period, period == 0, SWITCH_PENALTY)


def write_csv(codes, path):
    labels = np.asarray(pm.LABELS)[np.asarray(codes, np.int64)]
    data = ("frame,label\n" + "".join(f"{i},{lab}\n" for i, lab in enumerate(labels.tolist()))).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m pipeline.run", description=__doc__.splitlines()[0])
    parser.add_argument("--team", required=True, help="team name; the CSV is written as <team>.csv")
    parser.add_argument("--out", help="output folder (default: out/match/submission)")
    args = parser.parse_args(argv)
    started = time.time()

    print("1/4 converting the tracking")
    match = convert_tracking()
    print("2/4 running the 21 networks on CPU (about 10 minutes)")
    sources = infer.sources(match)
    print("3/4 mixing and decoding")
    labels = np.asarray(ensemble(match, sources))
    print("4/4 writing and validating the CSV")
    out = (Path(args.out) if args.out else P.SUBMISSIONS) / f"{args.team}.csv"
    sha = write_csv(labels, out)
    check = subprocess.run([sys.executable, str(P.ROOT / "evaluation" / "validate_submission.py"), str(out),
                            "--rows", str(games.GAMES[GAME]["n_video_frames"])], capture_output=True, text=True, cwd=P.ROOT)
    print((check.stdout + check.stderr).rstrip())
    same = core.sha256_array(labels) == SUBMITTED_LABELS_SHA256
    print(f"wrote {out} ({len(labels)} rows, sha256 {sha})")
    print(f"identical to the submitted labels: {'yes' if same else 'NO'}")
    print(f"done in {time.time() - started:.0f} s")
    return 0 if check.returncode == 0 and same else 1


if __name__ == "__main__":
    sys.exit(main())
