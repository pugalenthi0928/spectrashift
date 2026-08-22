# OxHyperMinerals-MINI public pilot

## Decision

Keep `prototype-sam` as the thresholded-map reference baseline for the next foundation-model
comparison. Keep `pca-logistic` as a ranking/calibration failure baseline, not as the candidate to
ship. The pilot completed the commercial decision loop—ingestion, leakage control, validation-only
thresholding, held-out evaluation, latency, calibration, target ranking, and failure preservation—
but is too small for a geological or deployment claim.

## Locked evidence

| Item | Value |
| --- | --- |
| Dataset | `previtus/OxHyperMinerals_MINI@0b58274` |
| Source commit | `74feebb24f19b62213cea91d6bbd8d3cfb6da6ec` |
| Selection | 4 train / 3 validation / 1 test tiles |
| Source groups | 1 train / 2 validation / 1 test; 5 crossing groups excluded |
| Valid pixels | 654,174 validation / 239,006 test |
| Validation support | 13,848 goethite / 29,931 hematite / 23,420 kaolinite |
| Artifact | [GitHub Actions run 32518705156](https://github.com/pugalenthi0928/spectrashift/actions/runs/32518705156) |
| Artifact SHA-256 | `86d7e8647f7ddc98d8f741f108127f61cd2ee5b1755dcb40bea35c1f5ebf2127` |
| Canonical manifest SHA-256 | `fdf851be0fe1af472286599c6a3bac9b39c25521135fc76ecb61e02185784343` |
| Manifest file SHA-256 | `cb787a80e168a8fa7469f8ad64e5ec4879e6e02ccb081f080906dc33546feef1` |
| Stable pilot definition SHA-256 | `21691337ef2290d197993a3dd907b7d6317c71cb7ddffe3a88c70b6aba7ff4f4` |

Every selected cube and label has a SHA-256 digest in `pilot_manifest.json`. The full CI artifact
also contains compressed prediction maps, ranked target cards, and preserved false-positive and
false-negative spectra. The compact manifests and metrics in this directory are the durable review
record. The canonical manifest digest hashes normalized JSON content; the file digest hashes the
preserved pretty-printed bytes.

## Held-out test result

| Model | Macro F1 | Macro AUPRC | Mean IoU | ECE | Brier | Test latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Prototype SAM | **0.199** | 0.212 | **0.119** | **0.164** | **0.184** | **79 ms/tile** |
| PCA + logistic | 0.055 | **0.242** | 0.029 | 0.487 | 0.487 | 182 ms/tile |

The test tile's class prevalence is 12.4% goethite, 41.6% hematite, and 5.4% kaolinite, versus
2.1%, 4.6%, and 3.6% across validation. This is a material source/geography shift.

| Mineral | SAM F1 / AUPRC | PCA-logistic F1 / AUPRC |
| --- | ---: | ---: |
| Goethite | **0.201** / **0.339** | 0.000 / 0.198 |
| Hematite | **0.390** / 0.266 | 0.083 / **0.495** |
| Kaolinite | 0.005 / 0.030 | **0.081** / **0.032** |

## Failure analysis

- PCA-logistic ranks hematite pixels well (0.495 AUPRC) but transfers a poor operating point:
  80.6% precision and only 4.4% recall. Its ECE and Brier score both near 0.49 show that the
  probabilities are not deployment-ready.
- PCA-logistic produces no goethite true positives at the validation-selected threshold.
- Both baselines are weak on kaolinite (AUPRC near 0.03). That is a model/data problem to investigate,
  not a number to hide.
- Hematite remains at the lower threshold-search boundary for both models; PCA-logistic kaolinite
  reaches the upper boundary. These flags remain in the machine-readable result.
- The pilot uses only one held-out test tile. The result demonstrates a reproducible mechanism and
  a useful failure diagnosis, not population-level generalization.

## Next experiment

Run a frozen HyperFree feature adapter on this exact `pilot_definition_sha256`, then compare
threshold-free AUPRC, calibrated F1, risk-coverage, peak memory, and latency. The hypothesis is that
a hyperspectral foundation model will preserve spectral-spatial ranking under the observed
source/geography shift better than either classical baseline.

OxHyperMinerals labels are EMIT L2B-derived experimental pseudo-labels. This result is
`public-benchmark-observed`; it is not field validation and does not indicate an economic deposit.
