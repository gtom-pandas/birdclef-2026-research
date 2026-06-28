# BirdCLEF 2026 Research Report

## Executive Summary

This repository documents my BirdCLEF 2026 Kaggle research track. The goal was multi-label soundscape classification for bird and wildlife acoustic events under Kaggle notebook constraints: offline execution, strict runtime limits, hidden test soundscapes, and dependency friction across TensorFlow, PyTorch, ONNX/TFLite, and audio processing libraries.

The final public result was `0.95087`, rank `574 / 4095` on the public leaderboard. Across my submissions, the best observed private score was `0.94216`, while the final EoS.9 public-maximizing submission scored `0.94138` private. That public/private gap became one of the main research lessons: late-stage rank blending improved the visible leaderboard but introduced fragility against the private split.

## Competition Context

BirdCLEF 2026 is a bioacoustic classification challenge. The task is to predict species and acoustic event probabilities across soundscape windows. The practical difficulty is not only recognizing bird vocalizations, but doing so across noisy field recordings, sparse positives, overlapping species, unmapped taxonomy cases, and domain shift between train, public test, and private test.

## Results

| Metric | Value |
| --- | ---: |
| Public rank | `574 / 4095` |
| Public score | `0.95087` |
| Best observed private score | `0.94216` |
| Final selected private score | `0.94138` |
| Submissions | `79` |
| Pulled Kaggle notebooks | `84` |

Key submission milestones:

| Date | Experiment | Public | Private |
| --- | --- | ---: | ---: |
| 2026-04-20 | Version 3 LGBM | `0.92948` | `0.92345` |
| 2026-04-21 | Two-Pass SSM + Advanced PP | `0.92746` | `0.92645` |
| 2026-05-01 | ProtoSSM + EfficientNet | `0.93515` | `0.93640` |
| 2026-05-02 | ONNX Perch Sequence + SED | `0.94307` | `0.93793` |
| 2026-05-18 | EoS.5 | `0.94903` | `0.94075` |
| 2026-05-27 | EoS.8 | `0.95035` | `0.94216` |
| 2026-06-03 | EoS.9 final | `0.95087` | `0.94138` |

## Method Evolution

### 1. Perch-First Baseline

The early system used Google's Bird Vocalization Classifier / Perch v2 as the main acoustic feature generator. Inference was organized around 60-second soundscapes split into `12` windows of `5` seconds. Perch logits and embeddings became the base representation for both direct predictions and downstream probes.

Important components:

- Perch v2 TensorFlow inference.
- Mapping between competition taxonomy and Perch labels.
- Proxy handling for unmapped classes.
- Basic smoothing and threshold calibration.
- LGBM and MLP probe experiments on embedding features.

### 2. Ecological Priors and OOF Discipline

The next layer added site/hour/taxonomy priors. This helped because acoustic activity is structured by location, time, and taxonomic group. To reduce leakage risk, the pipeline used grouped validation patterns such as `GroupKFold` by filename when estimating out-of-fold behavior.

The priors were useful, but also risky. Strong priors can boost public score while harming private generalization if the hidden split changes location/time composition.

### 3. Temporal Sequence Modeling

The project then moved from independent-window scoring to temporal modeling. The main direction was a lightweight SSM family:

- `LightProtoSSM`
- `ProtoSSMv2`
- `ResidualSSM`
- temporal shift TTA
- sequence features across the 12 windows

This was the core technical lift. The models used Perch embeddings/logits as input and learned a temporal correction over the soundscape, instead of treating each window as isolated.

### 4. SED and Pseudo-Label Experiments

I also explored EfficientNet-based SED models and pseudo-label variants. These experiments were useful as ensemble diversity, especially when they provided a different error profile than Perch-only systems.

The main lesson: SED additions helped when blended carefully, but their value depended heavily on calibration and runtime safety. A model that looks useful alone can still be a bad ensemble member if it is overconfident on rare classes.

### 5. End-of-Competition Ensembling

The last phase was a series of `EoS` notebooks with rank blending, gated blending, taxonomy smoothing, PCEN/gated variants, and submission-level fusion. This phase pushed public score from the mid `0.949` range to `0.95087`.

The best private score came from EoS.8/Gendaijin EoS6 (`0.94216`). The final EoS.9 submission improved public score but had lower private score (`0.94138`), suggesting public leaderboard overfitting or blend instability.

## Final Pipeline Shape

```text
soundscape audio
  -> 60s read / normalization
  -> 12 x 5s windows
  -> Perch v2 logits + embeddings
  -> taxonomy mapping and proxy classes
  -> site/hour/taxonomy prior fusion
  -> PCA + MLP class probes
  -> ProtoSSM temporal model
  -> ResidualSSM correction
  -> optional SED / pseudo-label model blend
  -> TTA + smoothing + rank/gated blend
  -> Kaggle submission.csv
```

## Bottlenecks

### Runtime and Offline Constraints

The Kaggle environment forced all inference to run offline with pre-attached datasets, wheels, and model artifacts. TensorFlow/Perch compatibility required careful wheel handling. Some notebooks had explicit dry-run behavior when hidden test soundscapes were not mounted.

### Dependency Friction

The stack mixed TensorFlow, PyTorch, ONNX/TFLite, librosa/soundfile, scikit-learn, and model-specific artifacts. The practical bottleneck was not only model quality, but making the full inference graph run reliably inside Kaggle.

### Public/Private Mismatch

The final public-best submission was not the private-best submission. This is the clearest evidence that late-stage blend optimization was partially overfit to the public leaderboard.

### Sparse and Noisy Labels

Many classes have few positives, ambiguous calls, overlapping species, and non-bird acoustic textures. This made threshold calibration and rare-class handling difficult.

### Taxonomy Mapping

Perch and BirdCLEF label spaces do not align perfectly. Unmapped classes required proxy strategies, genus-level matching, and taxonomy-aware fallback logic.

### Asset Management

The project produced many notebooks, forks, reruns, and submission variants. This repo exists partly to turn that competition trail into a coherent research artifact.

## What Worked

- Perch embeddings were a strong base signal.
- Temporal modeling improved the soundscape-level reasoning.
- Grouped OOF validation was necessary to avoid fooling myself.
- Site/hour/taxonomy priors helped when kept under control.
- Ensemble diversity mattered more than a single "best" model.
- Rank blending was useful, but only up to the point where it started chasing the public split.

## What I Would Improve Next

- Build a stricter experiment registry from day one.
- Track public/private risk by holding back a more adversarial validation split.
- Separate model diversity experiments from final blend experiments.
- Add automated blend audits that flag when public gain comes with private-risk patterns.
- Package inference into smaller reusable modules instead of large notebooks.

## Portfolio Angle

This project demonstrates practical competition ML under real constraints: audio representation learning, sequence modeling, model ensembling, fast iteration, Kaggle operational discipline, and honest postmortem analysis. The main story is not just the score; it is the progression from baseline inference to a layered, competition-grade soundscape system and the understanding of where leaderboard optimization becomes fragile.
