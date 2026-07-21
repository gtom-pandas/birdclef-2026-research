# BirdCLEF+ 2026 — EoS.8 Reproduction Archive

Evidence-backed archive of my best private BirdCLEF+ 2026 submission.

## Result

| Item | Verified value |
| --- | ---: |
| Team / Kaggle user | `T Soo` / `ttgrcgrc` |
| Best private submission | `BirdCLEF+ 2026 \| EoS.8`, version 1 |
| Public / private score | `0.95035` / **`0.94216`** |
| Submission reference | `53075232` |
| Final private rank | **947 / 4,094 teams** |
| Final selected EoS.9 private score | `0.94138` |

EoS.8 is the reference because it produced the highest private score in my
official submission history. EoS.9 scored higher on the public split but lower
on the private split.

## What the best submission actually was

EoS.8 was an integration and ensemble notebook, not a wholly original model
trained from scratch. The active final path was:

```text
Model_22 prediction (3%) ─┐
                          ├─ direct probability blend
Model_51 prediction (97%) ┘
          ↓
genus smoothing (alpha = 0.15)
          ↓
taxonomic-class smoothing (alpha = 0.05)
          ↓
submission.csv
```

- **Model_22** came from yukiZ's public Perch + ProtoSSM + ResidualSSM
  reproduction pipeline.
- **Model_51** came from Derek's public EoS.4 / rank-power experiment. Its
  internal active path combined a LightProtoSSM branch and a distilled SED
  branch, including on-the-fly lightweight head training and post-processing.
- My EoS.8 notebook selected the 3/97 top-level blend and added the v221
  taxonomy-smoothing stage.

See [SOURCES.md](SOURCES.md) for exact Kaggle slugs, versions, authors and
dependency metadata. This attribution is essential: the repository documents
my competition integration and experiment path without claiming authorship of
the upstream architectures.

## Repository contents

- `notebooks/birdclef-2026-eos-8.ipynb` — exact notebook downloaded from my
  Kaggle account; SHA-256 is recorded in `SOURCES.md`.
- `src/train.py` — cleaned extraction of the LightProtoSSM and ResidualSSM
  training stages executed inside Model_51.
- `src/infer.py` — exact score-defining EoS.8 final blend and taxonomy
  smoothing, with explicit validation.
- `tests/test_infer.py` — synthetic regression tests for blend, alignment and
  smoothing behavior.
- `data/submissions.csv` — curated official submission history.
- `data/notebook_inventory.csv` — experiment/notebook inventory.
- `paper/birdclef technical paper GRACI.pdf` — compiled research paper intended
  for the matching Zenodo/ORCID record.
- `paper/` — LaTeX source, bibliography and reproducible metadata figures.

## Reproduce the final ensemble

The two component CSVs are produced by the canonical notebook and are not
redistributed because they depend on Kaggle competition inputs and attached
third-party model assets.

```bash
python src/infer.py \
  --model-22 subm_22.csv \
  --model-51 subm_51.csv \
  --taxonomy /kaggle/input/competitions/birdclef-2026/taxonomy.csv \
  --sample-submission /kaggle/input/competitions/birdclef-2026/sample_submission.csv \
  --output submission.csv
```

The final combiner is deterministic. Given the original `subm_22.csv`,
`subm_51.csv` and taxonomy, it reproduces the EoS.8 ensemble logic.

## Train the lightweight sequence heads

The best notebook trained LightProtoSSM and ResidualSSM on cached Perch
embeddings during execution. It did **not** train Perch v2 or the distilled SED
backbone. Export the arrays described in `src/train.py`, then run:

```bash
python src/train.py \
  --features artifacts/eos8_model51_train_features.npz \
  --output-dir artifacts/eos8_heads
```

The canonical notebook remains the source of truth for raw-audio loading,
Perch/ONNX inference, MLP probes, prior construction, distilled SED inference
and creation of the component CSVs.

## Validation

```bash
python -m pytest -q
python -m py_compile src/train.py src/infer.py
python paper/make_figures.py
```

## Reproducibility boundary

Fully reproduced here:

- exact EoS.8 notebook and Kaggle metadata;
- official score history;
- score-defining 3/97 blend;
- genus and class smoothing;
- lightweight sequence-head architecture and training schedule;
- metadata EDA and paper figures.

Requires Kaggle-attached assets:

- BirdCLEF competition audio and labels;
- Perch v2 model and label metadata;
- ONNX Perch package;
- distilled SED checkpoints;
- caches/checkpoints attached by the upstream notebooks.

No local cross-validation result is presented as official unless it appears in
the archived notebook. No missing experiment detail is reconstructed by guess.
