# OxHyperMinerals-MINI public pilot

## Decision

Keep `prototype-sam` as the thresholded-map reference. The frozen DOFA probe is the stronger
representation baseline for hematite ranking, but not a candidate to ship: its validation-selected
goethite and hematite maps collapse to all-positive predictions on the held-out geography and its
probabilities are severely miscalibrated. The pilot completed the commercial decision loop—
ingestion, leakage control, validation-only thresholding, held-out evaluation, latency, calibration,
target ranking, and failure preservation—but is too small for a geological or deployment claim.

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

The foundation-model result used the identical pilot definition. It ran from source commit
`dbbf13024e693aa9faa24146ac18b36517c1e5f4` in
[GitHub Actions run 32570622426](https://github.com/pugalenthi0928/spectrashift/actions/runs/32570622426).
The workflow artifact is ID `9475232408` with SHA-256
`4c46768cf57b4f0558496fe3c919a045d9c1a5c5ecb93aaa5b7c27e9b176ffe9`.

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
| DOFA frozen probe | **0.273** | 0.225 | **0.182** | 0.776 | 0.758 | 603 ms/tile |

The test tile's class prevalence is 12.4% goethite, 41.6% hematite, and 5.4% kaolinite, versus
2.1%, 4.6%, and 3.6% across validation. This is a material source/geography shift.

| Mineral | SAM F1 / AUPRC | PCA-logistic F1 / AUPRC | DOFA F1 / AUPRC |
| --- | ---: | ---: | ---: |
| Goethite | 0.201 / **0.339** | 0.000 / 0.198 | **0.221** / 0.106 |
| Hematite | 0.390 / 0.266 | 0.083 / 0.495 | **0.588** / **0.533** |
| Kaolinite | 0.005 / 0.030 | **0.081** / 0.032 | 0.009 / **0.036** |

## Failure analysis

- PCA-logistic ranks hematite pixels well (0.495 AUPRC) but transfers a poor operating point:
  80.6% precision and only 4.4% recall. Its ECE and Brier score both near 0.49 show that the
  probabilities are not deployment-ready.
- DOFA produces the best hematite ranking (0.533 AUPRC), but its goethite and hematite maps predict
  every test pixel as positive. Their recall is 1.0 and their precision equals class prevalence;
  the headline macro F1 gain is therefore not evidence of a usable operating point.
- DOFA's 0.776 ECE and 0.758 Brier score are the worst of the three models. Pretraining provided a
  useful hematite representation, not calibrated confidence under geographic shift.
- PCA-logistic produces no goethite true positives at the validation-selected threshold.
- All three models are weak on kaolinite (AUPRC from 0.030 to 0.036). That is a model/data problem to
  investigate, not a number to hide.
- Hematite remains at the lower threshold-search boundary for both models; PCA-logistic kaolinite
  reaches the upper boundary. These flags remain in the machine-readable result.
- The pilot uses only one held-out test tile. The result demonstrates a reproducible mechanism and
  a useful failure diagnosis, not population-level generalization.

## Next experiment

Run a random-initialized DOFA control on this exact `pilot_definition_sha256` to isolate the value of
pretraining, then test higher input resolution and post-hoc calibration without touching the held-out
tile. HyperFree remains the next architecture comparison. Compare threshold-free AUPRC, calibrated
F1, risk-coverage, peak memory, and latency rather than selecting a winner from macro F1 alone.

OxHyperMinerals labels are EMIT L2B-derived experimental pseudo-labels. This result is
`public-benchmark-observed`; it is not field validation and does not indicate an economic deposit.
