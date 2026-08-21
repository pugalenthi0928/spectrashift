.PHONY: install install-benchmark test demo check oxhyper-index oxhyper-benchmark

install:
	python3 -m pip install -e .

install-benchmark:
	python3 -m pip install -e '.[benchmark,geo]'

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 -m spectrashift demo --output artifacts/demo

check:
	PYTHONPATH=src python3 scripts/check_environment.py

oxhyper-index:
	PYTHONPATH=src python3 -m spectrashift index-oxhyper \
		--dataset-root data/external/OxHyperMinerals_MINI \
		--output data/pilots/oxhyper-mini.json

oxhyper-benchmark:
	PYTHONPATH=src python3 -m spectrashift benchmark-oxhyper \
		--manifest data/pilots/oxhyper-mini.json \
		--dataset-root data/external/OxHyperMinerals_MINI \
		--model pca-logistic \
		--output artifacts/oxhyper-mini-pca-logistic
