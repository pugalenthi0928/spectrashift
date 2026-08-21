.PHONY: install test demo check

install:
	python3 -m pip install -e .

test:
	PYTHONPATH=src python3 -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python3 -m spectrashift demo --output artifacts/demo

check:
	PYTHONPATH=src python3 scripts/check_environment.py

