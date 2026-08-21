# Real-data pilot protocol

## Decision this pilot supports

The pilot asks whether an evidence-preserving pipeline can turn real 285-band EMIT tiles into
reproducible mineral-screening metrics and inspectable analyst targets without pixel leakage. It is
the gate before integrating a frozen or adapted hyperspectral foundation model.

## Dataset and scope

The first run uses the authors' public
[OxHyperMinerals-MINI](https://huggingface.co/datasets/previtus/OxHyperMinerals_MINI) development
subset. Their official
[HyperspectralViTs examples](https://github.com/previtus/HyperspectralViTs/blob/main/bash/demos_data_explore.sh)
identify this subset as the small preview for OxHyperMinerals.

The MINI repository is 9.32 GB, so the download command first retrieves its split CSVs and then
materializes only the deterministic leakage-safe pilot tiles. The complete dataset is 372 GB and
contains 796 training, 198 validation, and 200 test tiles. Each
tile is 512 by 512 pixels with 285 EMIT bands from approximately 381 to 2493 nm. The three primary
labels are goethite, hematite, and kaolinite. Those labels are experimental products aggregated
from EMIT L2B constituents; they are pseudo-ground truth, not field observations.

## Locked procedure

1. Download the MINI repository at the pinned `0b58274` revision.
2. Discover only tiles containing `C`, `C.hdr`, and `minerals3ghk.tif`.
3. Read the MINI split files `train_minerals_10.csv`, `val_minerals_10.csv`, and
   `test_minerals_10.csv` (the full release uses the corresponding names without `_10`).
4. Infer the source capture from the tile identifier and remove any group crossing splits.
5. Select whole source groups deterministically and preserve file sizes and optional SHA-256 hashes.
6. Fit preprocessing and the model on training pixels only.
7. Select one decision threshold per mineral on validation pixels only.
8. Evaluate the locked model and thresholds on the test tiles once.
9. Preserve prediction maps, metrics, calibration, target cards, failure spectra, environment, and
   limitations in one result bundle.

Some MINI ENVI headers omit wavelength fields. The loader first checks the header, then numeric
raster band descriptions, and only for a generic 285-band EMIT raster uses the exact band centres
published in the authors' `HyperspectralViTs` loader. It fails on any other unresolved layout rather
than synthesizing an evenly spaced wavelength grid.

The MINI `minerals3ghk.tif` payloads are headerless raw arrays despite the filename extension. The
fallback decoder activates only when raster loading fails and the file exactly matches a
three-band, little-endian uint16, band-sequential array at the cube dimensions. It validates that
all values are binary before assigning the published goethite/hematite/kaolinite band order.

## Baselines

`prototype-sam` estimates one median positive spectrum per mineral from training data and applies
Spectral Angle Mapper to held-out scenes. It is a fast spectral reference baseline.

`pca-logistic` standardizes sampled training spectra, fits PCA on training data only, and trains a
class-balanced one-vs-rest logistic regression head. It is the primary commercial baseline because
it is inexpensive, inspectable, and produces probabilities that can be evaluated for calibration.

## Commands

```bash
python3 -m pip install -e '.[benchmark,geo]'

spectrashift download-oxhyper-mini \
  --output data/external/OxHyperMinerals_MINI

spectrashift index-oxhyper \
  --dataset-root data/external/OxHyperMinerals_MINI \
  --output data/pilots/oxhyper-mini.json \
  --train-groups 4 \
  --validation-groups 2 \
  --test-groups 2 \
  --tiles-per-group 2 \
  --hash-files

spectrashift benchmark-oxhyper \
  --manifest data/pilots/oxhyper-mini.json \
  --dataset-root data/external/OxHyperMinerals_MINI \
  --model pca-logistic \
  --output artifacts/oxhyper-mini-pca-logistic
```

If the MINI subset does not contain enough source groups for the requested counts, the indexer uses
the available leakage-safe groups and records both the requested and actual group/tile counts. If a
split has no safe group, it fails rather than silently falling back to a random pixel split.

The `oxhyper-mini-public-benchmark` GitHub Actions workflow runs a resource-bounded version of this
protocol on every benchmark-branch update. It uploads the hash-verified pilot manifest and both
baseline evidence bundles; it does not commit the downloaded dataset or generated predictions.

## Required review before reporting a score

- Verify the dataset revision, split records, file hashes, and class order.
- Inspect per-class support and confirm that every training class contains positives and negatives.
- Compare validation and test prevalence and flag a distribution shift.
- Review the highest-confidence false positives and false negatives with their spectra.
- Report macro and per-class F1, AUPRC, IoU, ECE, Brier score, risk-coverage, latency, and memory.
- Retain the pseudo-label and surface-versus-subsurface limitations beside every public result.

No result from this pilot should be called externally validated or field validated.
