# Analytical Tiki-Taka Lab: ball possession for BAR-ATH

Our submission to More than a Hack 2027, Phase 1. This repository regenerates the submitted
`frame,label` file from the organisers' data package with one command. The method is described
in the technical document submitted alongside it.

## Reproduce the submission

You need Python 3.12 with [`uv`](https://docs.astral.sh/uv/) and the organisers' data package
unpacked into `data/` (or `MTAH_DATA_DIR` pointing at it). Only the tactical tracking files are read.

```bash
uv sync
uv run python -m pipeline.run --team Analytical_Tiki-Taka_Lab
```

The run takes about 10 minutes on a laptop CPU and needs no GPU. It writes
`out/match/submission/Analytical_Tiki-Taka_Lab.csv` (199,319 rows), runs the organisers'
`validate_submission.py` on it, and ends with

```
identical to the submitted labels: yes
```

That line compares a SHA-256 of the regenerated label array with the one we submitted. The
command exits non-zero if the labels differ or the validator fails.

## What the command does

| Step | Input | Code |
|---|---|---|
| 1. Convert the tracking | `data/tactical/BAR-ATH_tactical_atd.{txt,xml}` | `possession/load_atd.py` |
| 2. Run 21 networks | converted tracking, `weights/` | `pipeline/infer.py` |
| 3. Mix and decode | the three probability sources | `pipeline/run.py`, `members/higuchi/team_lb/ens_20261116/core.py` |

No step reads a possession label, and the repository contains none.

## Models and weights

`weights/` holds 21 checkpoints (30 MB) with the normalisation statistics each one needs.

| Source | Networks | Features | Definition |
|---|---|---|---|
| GRU | 9 (3 folds x 3 seeds) | 140 | `members/kashiwagi/train/seq_train.py`, `members/kashiwagi/feats.py` |
| Convolution + GRU | 9 (3 folds x 3 seeds) | 140 | same files |
| Convolution + bidirectional GRU | 3 (one per fold) | 62 | `members/oota/solo/seq.py`, `members/oota/solo/features.py` |

All networks were trained only on external matches: 7 DFL open-data matches, 3 Metrica sample
matches and 64 PFF FC World Cup 2022 matches. `weights/SHA256SUMS` lists every file.

This repository runs inference from the saved weights. It does not retrain them: training needs
the three external datasets and our preprocessing of them, which are not included here. The
network definitions and the training loop of the first two models are in the files above.

## Numerical note

Network outputs differ in the seventh decimal place between machines and library versions. The
decoded labels did not change in our tests (training on GPUs, reproduction on a CPU with torch 2.15). The versions we tested are pinned in `uv.lock`.

## Layout

```
pipeline/      the single entry point and the inference code
weights/       the 21 checkpoints
possession/    tracking converter and the common match format
members/       model, feature and ensemble code, kept at its original paths
report/        the technical document (PDF) and its HTML source
evaluation/    organisers' metric and validator, unchanged
games.py       organisers' game registry, unchanged
data/          empty; unpack the organisers' data package here
```

Our code is under the MIT License (`LICENSE`). `games.py` and `evaluation/` are the organisers'
files, copied unchanged from `metrica-sports/more-than-a-hack-2027`; `NOTICE` carries their licence.
