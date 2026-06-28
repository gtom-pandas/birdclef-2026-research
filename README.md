# BirdCLEF 2026 Research

Portfolio-ready research archive for my Kaggle BirdCLEF 2026 work. The repo is
kept intentionally compact: no raw notebook dump, no model weights, no Kaggle
datasets. It keeps the solution narrative, curated metadata and executable
reference scripts.

## What This Repo Shows

This repository is not only a dump of Kaggle notebooks. It is a cleaned research
artifact that explains the solution path, the score progression, the modeling
choices and the reproducibility contract behind the final submissions.

The competition solution combined:

- **Perch v2 transfer learning**: logits and embeddings from Google's bird
  vocalization model used as the main acoustic representation.
- **Soundscape windowing**: each 60-second file processed as `12 x 5s` temporal
  windows to keep local calls while reasoning over the whole soundscape.
- **Temporal sequence modeling**: ProtoSSM/ResidualSSM-style correction layers
  over Perch features instead of independent window predictions only.
- **Ecological priors**: site, hour and taxonomy priors for calibration, with
  care around public/private leaderboard mismatch.
- **SED diversity**: EfficientNet sound event detection and pseudo-label
  experiments used as ensemble diversity when calibration was stable.
- **Competition ensembling**: TTA, smoothing, rank blending, gated blending and
  taxonomy-aware post-processing.

## Result Snapshot

- Competition: [BirdCLEF 2026](https://www.kaggle.com/competitions/birdclef-2026)
- Team: `T Soo`
- Kaggle user: `ttgrcgrc`
- Public leaderboard rank: `574 / 4095`
- Public score: `0.95087`
- Best observed private score in submissions: `0.94216`
- Final selected submission: `BirdCLEF+ 2026 | EoS.9`
- Total submissions: `79`

## Repository Map

- `reports/research_report.md` - full research narrative, bottlenecks, and lessons learned.
- `reports/portfolio_blurb.md` - short text blocks for portfolio integration.
- `docs/experiment_summary.md` - compact replacement for the removed notebook dump.
- `data/submissions.csv` - curated submission history.
- `data/leaderboard_summary.csv` - final public leaderboard row.
- `data/notebook_inventory.csv` - notebook inventory and detected technical themes.
- `src/artifact_summary.py` - utility script to summarize the curated CSV metadata.
- `src/train.py` - reproducible PyTorch training entrypoint for the compact
  temporal sequence head trained on precomputed acoustic features.
- `src/infer.py` - fold averaging, prior fusion and rank-blend inference script
  for Kaggle-style submission generation.

## Technical Themes

The work evolved from Perch-based audio inference and lightweight tabular/probe baselines into a stacked soundscape system:

- Perch v2 logits and embeddings as the main acoustic backbone.
- 60-second soundscape processing split into 12 windows of 5 seconds.
- Site/hour/taxonomy priors for ecological calibration.
- MLP probes and PCA features over Perch embeddings.
- ProtoSSM / LightProtoSSM temporal modeling.
- ResidualSSM second-pass correction.
- EfficientNet SED and pseudo-label experiments.
- TTA, smoothing, rank blending, gated blending, and taxonomy-aware post-processing.

## Reproduce Metadata Summary

```bash
python src/artifact_summary.py --root .
```

This script reads the curated CSV files only. Raw notebooks were removed from
GitHub because they made the repo harder to review without adding much signal.

## Scripted Pipeline

The original Kaggle runs relied on notebooks because the competition environment
requires attached datasets and offline artifacts. The scripts below expose the
same architecture in a cleaner form for review and reuse.

Train the temporal correction head from exported features:

```bash
python src/train.py \
  --features artifacts/birdclef_train_features.npz \
  --output-dir artifacts/birdclef_sequence_head \
  --epochs 8 \
  --folds 5
```

Run fold-averaged inference with optional ecological priors:

```bash
python src/infer.py \
  --features artifacts/birdclef_test_features.npz \
  --classes artifacts/classes.txt \
  --checkpoint artifacts/birdclef_sequence_head/bird_sequence_head_fold1.pt \
  --checkpoint artifacts/birdclef_sequence_head/bird_sequence_head_fold2.pt \
  --priors artifacts/site_hour_taxonomy_priors.csv \
  --output submission.csv
```

Expected feature contract:

- `X`: `float32`, shape `[n_soundscapes, n_windows, n_features]`
- `y`: `float32`, multi-label targets for training, shape `[n_soundscapes, n_classes]`
- `groups`: optional grouped-validation IDs, usually filename or recording group
- `row_id`: required for inference submission rows

The heavy feature extraction artifacts are intentionally not committed because
they depend on Kaggle datasets/model weights and can be large. The scripts make
the modeling layer explicit without leaking private data or bloating the repo.
