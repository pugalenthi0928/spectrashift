# Foundation-model adaptation protocol

## Decision

Run a frozen DOFA ViT-B feature probe before HyperFree. DOFA is not a mineral-specific model, but
its published wavelength-conditioned patch embedder accepts arbitrary channel counts and its
pretraining includes hyperspectral Earth-observation data. Its clean PyTorch path also makes the
first comparison reproducible on commodity CI hardware. HyperFree remains a later promptable
hyperspectral comparison rather than an unexecuted claim.

## Locked source and weights

| Item | Value |
| --- | --- |
| Source | `zhu-xlab/DOFA` |
| Source revision | `0cfb7e1099f4d4c4022946ff7862c7cd7b8411b9` |
| Checkpoint | `earthflow/DOFA/DOFA_ViT_base_e100.pth` |
| Checkpoint revision | `7a5219e48d2f8848511b0fabea7920a8836bc480` |
| Checkpoint SHA-256 | `4720985e42b918ac0307009eb06121a3435d9bbce6fd95446f84824a538165b1` |
| Frozen parameters | 111,199,232 |

The adapter refuses a different Git revision or checkpoint digest. Third-party source and weights
remain outside this repository and retain their original terms.

## Adaptation path

1. Supply all 285 EMIT band centres to DOFA in micrometres; no evenly spaced wavelength grid is
   invented.
2. Fit one mean and standard deviation per band from sampled training pixels only.
3. Resize each 512 by 512 tile to a resource-bounded 112 by 112 input while retaining every band.
4. Interpolate the published 14 by 14 position embedding to a 7 by 7 token grid.
5. Freeze the 111.2M-parameter backbone and extract the final 768-dimensional token features.
6. Pool the pixel pseudo-labels to the same grid. A token is positive when at least 5% of its valid
   pixels carry the mineral label; the threshold gives 27 kaolinite-positive training tokens on the
   locked pilot, compared with only two at 25%.
7. Fit three class-balanced logistic heads: 2,307 trainable parameters in total.
8. Upsample token probabilities to the source tile, select thresholds on validation pixels only,
   and evaluate the held-out tile once.

This is frozen-feature adaptation, not full fine-tuning. The lower spatial resolution is an explicit
compute trade-off and must stay beside the result when compared with pixel-level SAM and PCA
baselines.

## Required evidence

The run must preserve the pinned data manifest, source and checkpoint identities, normalization
policy, trainable and frozen parameter counts, validation thresholds, held-out metrics, latency,
memory, prediction maps, target cards, and false-positive/false-negative spectra. Until that bundle
exists, the configuration remains `configured_not_executed`.

## Next ablations

- pretrained checkpoint versus random initialization;
- frozen final-layer tokens versus earlier layers;
- 112 versus 224 pixel inputs;
- frozen probe versus LoRA and full fine-tuning where GPU compute permits;
- complete spectrum versus dropped bands and wavelength-shift stress;
- 1%, 5%, 10%, and 25% positive patch-label pooling thresholds.

The next promptable comparison is HyperFree on the identical pilot definition. It should not replace
the locked baseline or DOFA evidence bundle.
