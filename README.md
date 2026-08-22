# SpectraShift

**Sensor-adaptive foundation-model evaluation for evidence-backed hyperspectral mineral screening.**

SpectraShift is an independent, public-data research prototype that asks a commercially useful
question: how reliably and efficiently can a pretrained hyperspectral model be adapted to a new
sensor configuration, geography, and label budget, and how should its output be converted into
inspectable analyst targets?

The intended path is:

`hyperspectral cube -> quality control -> model adaptation -> calibrated mineral maps -> ranked target cards`

## Project snapshot

| Capability | Evidence |
| --- | --- |
| Real 285-band ingestion | Pinned, hash-verified OxHyperMinerals-MINI tiles |
| Leakage control | Published splits plus disjoint source-capture validation |
| Model comparison | Prototype SAM and PCA-logistic held-out baselines |
| Foundation adaptation | Pinned DOFA ViT-B frozen-token probe; benchmark configured, not yet reported |
| Decision quality | F1, AUPRC, IoU, ECE, Brier, risk-coverage, latency, and memory |
| Failure analysis | Preserved false-positive/false-negative spectra and threshold-edge flags |
| Analyst evidence | Cited vector retrieval over manifests, metrics, targets, and limitations |

## Why this exists

Hyperspectral mineral systems operate under difficult conditions: hundreds of bands, limited
verified labels, sensor-to-sensor wavelength differences, mixed pixels, spatial leakage risk, and
high costs when false confidence reaches an analyst. SpectraShift treats evaluation and evidence as
part of the model, not as an afterthought.

## Current status

This repository is being built in evidence-first stages.

- **Implemented and tested:** hyperspectral data contracts, wavelength adaptation, geographic
  split validation, Spectral Angle Mapper baseline, multilabel metrics, calibration, connected
  target extraction, target ranking, a deterministic synthetic demonstration, and a real-data
  OxHyperMinerals pilot runner with ENVI ingestion, source-group leakage detection, validation-only
  threshold selection, PCA-logistic and prototype-SAM baselines, prediction maps, and failure cases.
- **Foundation adapter implemented:** the pinned DOFA ViT-B checkpoint accepts the full 285-band
  EMIT wavelength grid, interpolates its position embedding to a resource-bounded token grid,
  freezes 111.2M backbone parameters, and fits three linear mineral heads. The public-data run is
  configured but no score is claimed yet.
- **Evidence retrieval implemented:** analyst questions retrieve cited Markdown sections and JSON
  pointers from immutable experiment bundles. Retrieval can explain a result but cannot change a
  prediction, threshold, rank, or evidence grade.
- **Public pilot observed:** prototype-SAM and PCA-logistic were run on a pinned, hash-verified,
  source-group-safe OxHyperMinerals-MINI pilot. The reviewed result and limitations are preserved
  in [`results/oxhyper-mini`](results/oxhyper-mini/README.md).
- **Not yet claimed as executed:** HyperFree and HyperSIGMA comparisons.
- **No benchmark result is reported until its run manifest and artifacts exist.**

The synthetic demo is a software test fixture. It is not geological evidence and must not be used
for mineral exploration decisions.

## Planned research comparison

| Family | Role |
| --- | --- |
| Spectral Angle Mapper | Physics-informed spectral baseline |
| PCA + SVM / Random Forest | Classical commercial baselines |
| DOFA ViT-B | Wavelength-aware frozen foundation-model features |
| HyperSegFormer | Task-specific hyperspectral transformer |
| HyperSIGMA ViT-B | Pretrained spectral-spatial foundation model |
| HyperFree | Channel-adaptive, promptable foundation model |

The primary experiments compare frozen features, parameter-efficient adaptation, and full
fine-tuning where compute allows. Splits are held out by source capture or geography, never by
random neighboring pixels.

## Quick start

The tested core requires only Python 3.11+ and NumPy.

```bash
python3 -m pip install -e .
spectrashift demo --output artifacts/demo
python3 -m unittest discover -s tests -v
```

The demo writes a run manifest, metrics, probability maps, and ranked target cards. Install model
and geospatial extras only when needed:

```bash
python3 -m pip install -e '.[ml,geo,dev]'
```

### Real-data pilot

The authors of OxHyperMinerals publish a small development subset with the same 285-band EMIT
format. The real-data path is explicit and does not redistribute their data:

```bash
python3 -m pip install -e '.[benchmark,geo]'
spectrashift download-oxhyper-mini --output data/external/OxHyperMinerals_MINI
spectrashift index-oxhyper \
  --dataset-root data/external/OxHyperMinerals_MINI \
  --output data/pilots/oxhyper-mini.json \
  --hash-files
spectrashift benchmark-oxhyper \
  --manifest data/pilots/oxhyper-mini.json \
  --dataset-root data/external/OxHyperMinerals_MINI \
  --model pca-logistic \
  --output artifacts/oxhyper-mini-pca-logistic
```

See [the real-data pilot protocol](docs/REAL_DATA_PILOT.md) and
[foundation-model adaptation protocol](docs/FOUNDATION_MODEL_ADAPTATION.md), plus the
[reviewed public-pilot result](results/oxhyper-mini/README.md). On one held-out MINI test tile,
prototype-SAM reached 0.199 macro F1 and 0.212 macro AUPRC; PCA-logistic reached 0.055 and 0.242.
These are pseudo-label pilot results, not field or deposit validation.

### Analyst evidence query

```bash
spectrashift query-evidence \
  --root results/oxhyper-mini \
  --question "Why is PCA-logistic not deployment ready?" \
  --top-k 3
```

The response contains ranked excerpts and citations such as `README.md#failure-analysis` and
`benchmark_summary.json#/decision`; it is extractive by design.

## Evidence boundaries

- Public EMIT mineral products and OxHyperMinerals labels are useful research evidence, but not a
  substitute for field verification or drill results.
- OxHyperMinerals uses algorithm-generated pseudo-labels. Agreement with those labels is not the
  same as verified geological truth.
- Detected surface alteration indicators do not prove a subsurface economic deposit.
- The language layer may retrieve and explain evidence; it must never manufacture or override the
  spectral model's result.

See [the evidence contract](docs/EVIDENCE_CONTRACT.md),
[experiment protocol](docs/EXPERIMENT_PROTOCOL.md), and
[architecture](docs/ARCHITECTURE.md).

## Data and model sources

- [NASA EMIT data resources](https://github.com/nasa/EMIT-Data-Resources)
- [OxHyperMinerals and HyperspectralViTs](https://github.com/previtus/HyperspectralViTs)
- [OxHyperMinerals-MINI](https://huggingface.co/datasets/previtus/OxHyperMinerals_MINI)
- [USGS Spectral Library Version 7](https://www.usgs.gov/data/usgs-spectral-library-version-7-data)
- [HyperSIGMA](https://github.com/WHU-Sigma/HyperSIGMA)
- [HyperFree](https://github.com/Jingtao-Li-CVer/HyperFree)

Third-party datasets, labels, code, and checkpoints retain their original terms. They are not
redistributed by this repository.

## Independence statement

SpectraShift is an independent portfolio and research project built from public sources. It is not
affiliated with, commissioned by, or representative of Esper Industries or any other commercial
hyperspectral provider.
