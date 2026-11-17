"""Paths, defined in one place.

- Competition data: `games.DATA_DIR` (default ./data, override with MTAH_DATA_DIR)
- Outputs: ./out (not committed; everything in it is derived from competition data)
"""
from pathlib import Path

import games

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(games.DATA_DIR)
OUT = ROOT / "out"
OUT_MATCH = OUT / "match"
SUBMISSIONS = OUT_MATCH / "submission"
