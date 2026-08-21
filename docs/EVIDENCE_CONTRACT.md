# Evidence contract

SpectraShift separates a mechanism that runs from a scientific claim that is supported.

## Claim grades

| Grade | Meaning |
| --- | --- |
| `mechanism-tested` | Unit or integration tests show the software mechanism behaves as specified. |
| `synthetic-demonstration` | A deterministic synthetic fixture exercises the end-to-end path. |
| `public-benchmark-observed` | A run on a named public split produced a preserved manifest and artifacts. |
| `externally-validated` | A result was tested on a separately sourced sensor, geography, or verified label set. |
| `field-validated` | Qualified domain experts linked the output to field or laboratory evidence. |

Only the first two grades exist at repository initialization. Later grades require preserved run
artifacts and must never be inferred from functioning code alone.

## Required result bundle

Every reported model result must include:

1. Dataset and label provenance.
2. Source-scene or geographic split manifest.
3. Wavelength selection, resampling, masking, and normalization steps.
4. Model source, checkpoint identifier, license, and code revision.
5. Configuration, seed, environment, and hardware.
6. Primary and secondary metrics with denominators.
7. Per-class results, calibration, and uncertainty.
8. Failure cases and known limitations.
9. Exact output hashes where practical.

## Prohibited claims

- A mineral indicator is not proof of an orebody.
- Agreement with algorithm-generated labels is not field validation.
- A random pixel split is not evidence of geographic generalization.
- A probability is not calibrated confidence unless calibration was measured.
- A retrieved document does not validate a model prediction.
- Synthetic demo outputs are never scientific results.

## Language-model boundary

Retrieval may expose model cards, target cards, spectral-library metadata, experiment manifests,
and limitations. A language model may summarize those records with citations. It may not change a
pixel label, raise a target score, invent missing evidence, or present an unsupported geological
interpretation.

