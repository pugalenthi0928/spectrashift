# Experiment protocol

## Research question

How reliably and efficiently can pretrained hyperspectral representations be adapted to a new
sensor configuration, geography, and label budget for alteration-mineral screening?

## Primary dataset

OxHyperMinerals is the intended first public benchmark. It contains EMIT-derived hyperspectral
tiles using 285 bands and multi-label pseudo-masks for goethite, hematite, and kaolinite. Its labels
are generated from an existing mineral-mapping method and must be described as pseudo-ground truth.

## Split policy

- Preserve source capture and published train/validation/test boundaries.
- Hold out entire captures or geographic groups.
- Detect duplicated sample identifiers and overlapping source groups before training.
- Fit normalization, PCA, calibration, and thresholds using training or validation data only.
- Apply locked parameters to the test set exactly once per declared run.

## Comparison matrix

| Axis | Values |
| --- | --- |
| Model | SAM, SVM, Random Forest, HyperSegFormer, HyperSIGMA, HyperFree |
| Adaptation | zero/prompt, frozen probe, LoRA, full fine-tune where feasible |
| Label budget | 1%, 5%, 10%, 100% of the declared training split |
| Spectral input | full bands, selected bands, dropped bands, wavelength-shift stress |
| Representation | spectral, spatial, fused where the model supports it |

## Metrics

Primary: macro-F1, per-mineral F1, AUPRC, and mean IoU.

Decision-quality: calibrated precision and recall, Expected Calibration Error, Brier score, and
risk-coverage behavior under abstention.

Efficiency: trainable parameter count, peak memory, latency, throughput, and artifact size.

## Ablations

1. Random initialization versus pretrained weights.
2. Frozen features versus LoRA versus full fine-tuning.
3. Fixed channel projection versus wavelength-aware adaptation.
4. Full spectrum versus band subsets and simulated missing bands.
5. Spectral-only versus spatial-only versus fused features.
6. Calibration and abstention on versus off.

## Failure analysis

Review the highest-confidence false positives and false negatives; mixed pixels; bad-band and
quality-mask interactions; scene/geography breakdowns; model disagreement; band-shift sensitivity;
and cases rejected by the abstention policy. Each reviewed case must retain the source scene,
location, spectrum, label provenance, predictions, and model version.

## Success criteria

Success does not require a state-of-the-art score. The project succeeds when it provides a credible
answer to which adaptation method is best under stated label, sensor, compute, and reliability
constraints, including a clear account of failure modes and unsupported conclusions.

