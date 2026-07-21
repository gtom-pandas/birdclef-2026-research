# Sources and provenance

Collected from Kaggle on 2026-07-21 through the authenticated Kaggle CLI.

## Author submission

- Notebook: `ttgrcgrc/birdclef-2026-eos-8`
- Title: `BirdCLEF+ 2026 | EoS.8`
- Kaggle notebook ID: `120725221`
- Submission reference: `53075232`
- Public score: `0.95035`
- Private score: `0.94216`
- Canonical local copy: `notebooks/birdclef-2026-eos-8.ipynb`
- SHA-256: `B40A1D396B477BF9B9F28D64B688743749C0634537EC231CD8023E645591DC66`
- Original Kaggle container:
  `gcr.io/kaggle-images/python@sha256:e5452ce6268c2e8345cfe5141f31ca7ff47032aca46a7ea532bbb87481281d0c`

## Upstream Model_22

- Author: yukiZ (`hideyukizushi`)
- Notebook: `hideyukizushi/bird26-reproduce-perch-protossm-resssm-inf-train`
- Title: `Bird26|REPRODUCE|Perch+ProtoSSM+ResSSM|INF/TRAIN`
- Notebook ID: `114154690`
- Source version referenced by EoS.8: `309368265`
- Role in EoS.8: complete prediction table `subm_22.csv`, weight `0.03`
- Kaggle URL: <https://www.kaggle.com/code/hideyukizushi/bird26-reproduce-perch-protossm-resssm-inf-train>

## Upstream Model_51

- Author: Derek (`sunderekkiz`)
- Notebook: `sunderekkiz/birdclef-2026-exp019-eos4-rank-power-06`
- Title: `birdclef 2026 exp019 eos4 rank power 06`
- Notebook ID: `119604329`
- Source version referenced by EoS.8: `320064228`
- Role in EoS.8: complete prediction table `subm_51.csv`, weight `0.97`
- Kaggle URL: <https://www.kaggle.com/code/sunderekkiz/birdclef-2026-exp019-eos4-rank-power-06>

Model_51 itself is an ensemble/reproduction notebook built from public BirdCLEF
work. Its active internal path includes Perch v2 embeddings, LightProtoSSM,
ResidualSSM, MLP probes, ecological priors, a distilled EfficientNet SED model,
rank blending and post-processing. The original notebook and its own credits
remain authoritative for deeper provenance.

## Model and dataset dependencies recorded by Kaggle

- Competition: `birdclef-2026`
- Google model: `google/bird-vocalization-classifier/TensorFlow2/perch_v2_cpu/1`
- `tuckerarrants/bc2026-distilled-sed-public`
- `jaejohn/perch-meta`
- `tuckerarrants/perch-v2-no-dft-onnx`
- `rishikeshjani/perch-onnx-for-birdclef-2026`
- `hideyukizushi/sgkfk-202604041716`
- Kernel source: `ashok205/tf-wheels`
- Kernel source: `hideyukizushi/bird26-reprod-perch-proto-residualssm-train-s7177`

The raw model weights and competition data are not redistributed in this
repository. Their licenses and access rules remain those of their respective
Kaggle/model pages.

## Competition citation

Stefan Kahl, Tom Denton, Larissa Sugai, Liliana Piatti, Ryan Holbrook, Holger
Klinck, and Ashley Oldacre. *BirdCLEF+ 2026*. Kaggle, 2026.
<https://www.kaggle.com/competitions/birdclef-2026>
