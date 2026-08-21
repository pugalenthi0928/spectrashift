# SpectraShift project brief

## User

A geospatial or mineral analyst screening a large area before costly field investigation.

## Problem

Hyperspectral imagery contains useful surface-mineral evidence but varies by sensor, wavelength
configuration, geography, and quality. Labels are scarce, neighboring pixels create leakage risk,
and raw probability maps require substantial expert review.

## Product hypothesis

A sensor-aware foundation-model evaluation layer can reduce adaptation effort and analyst review by
combining low-label transfer, calibrated uncertainty, abstention, connected-region ranking, and
inspectable evidence.

## Output

The system produces alteration-mineral probability maps and ranked target cards. It is a screening
tool, not a deposit discovery claim or autonomous drilling recommendation.

## Commercial measures

- Time required to adapt to a new wavelength grid.
- Labels and trainable parameters required to reach a useful operating point.
- Precision at the analyst's review capacity.
- Fraction of pixels safely abstained without losing high-value recall.
- Inference time and memory per scene.
- Analyst time required to understand and reject a target.

