# BirdCLEF 2026 Research Report

Portfolio-ready research archive for my Kaggle BirdCLEF 2026 work.

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
- `notebooks/` - pulled Kaggle notebooks, cleaned of execution outputs.
- `data/submissions.csv` - curated submission history.
- `data/leaderboard_summary.csv` - final public leaderboard row.
- `data/notebook_inventory.csv` - notebook inventory and detected technical themes.
- `src/artifact_summary.py` - utility script to rebuild the local artifact summary.

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

This script expects the Kaggle-pulled notebooks and curated CSV files to already exist locally.
