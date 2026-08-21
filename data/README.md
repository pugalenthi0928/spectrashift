# Data policy

No raw hyperspectral imagery, derived labels, credentials, or third-party model weights are stored
in this repository.

Expected local directories are `data/raw`, `data/interim`, and `data/processed`; all are ignored by
Git. Each external dataset must have a machine-readable local manifest containing its source URL or
DOI, retrieval date, license or terms, checksums where available, sensor, wavelength units, spatial
resolution, label provenance, and known limitations.

