# BirdCLEF 2026 - Experiment Summary

This file replaces the raw Kaggle notebook dump that previously lived in
`notebooks/`. The full exported notebooks were useful locally, but too noisy for
a public portfolio repo. The technical value is preserved through the reports,
curated CSV files and reference scripts.

## Public Result

| Item | Value |
| --- | ---: |
| Public score | `0.95087` |
| Public rank | `574 / 4095` |
| Best observed private score | `0.94216` |
| Total submissions | `79` |
| Notebook artifacts summarized | `84` |

## Solution Arc

1. **Perch-first baseline**: use Perch v2 logits and embeddings as the main
   acoustic representation for 60-second soundscapes.
2. **Window-level reasoning**: split each soundscape into `12 x 5s` windows to
   preserve local calls while keeping sequence context.
3. **Ecological calibration**: add site, hour and taxonomy priors, with grouped
   validation to reduce obvious leakage.
4. **Temporal heads**: add ProtoSSM/ResidualSSM-style correction layers over
   Perch features.
5. **Model diversity**: test EfficientNet SED and pseudo-label variants as
   ensemble members.
6. **Endgame blending**: use TTA, smoothing, rank blending and gated blending to
   push public score, then document the public/private fragility.

## What Stayed in Git

- `reports/research_report.md`: full narrative and postmortem.
- `data/submissions.csv`: score progression.
- `data/notebook_inventory.csv`: searchable inventory of removed notebook
  artifacts and detected techniques.
- `src/train.py`: reference temporal-head training entrypoint.
- `src/infer.py`: fold averaging, prior fusion and rank-blend inference.

## What Was Removed

- Raw `.ipynb` exports and `kernel-metadata.json` files.
- Local Kaggle pulls and heavyweight model/data artifacts.
- Any file that only duplicated the same experiment name without adding a
  readable explanation.

The result is a repo that shows the competition solution instead of making the
reviewer browse dozens of near-duplicate notebooks.
