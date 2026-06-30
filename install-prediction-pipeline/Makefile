.PHONY: install data eda train test serve docker clean

PY ?= python
ARTIFACTS ?= artifacts

install:
	$(PY) -m pip install -r requirements.txt

# Generate a schema-faithful synthetic sample (the real data is private).
data:
	$(PY) scripts/generate_synthetic_data.py --rows 50000 --out data/sample.parquet

eda:
	$(PY) scripts/eda.py --data data/sample.parquet

train:
	PYTHONPATH=src $(PY) -m ipm.train --config config/config.yaml

test:
	PYTHONPATH=src $(PY) -m pytest

# Run the REST inference service on the trained artifacts.
serve:
	PYTHONPATH=src IPM_ARTIFACTS=$(ARTIFACTS) \
		$(PY) -m uvicorn ipm.serve:app --host 0.0.0.0 --port 8080

docker:
	docker build -t ipm-serve .

clean:
	rm -rf artifacts mlruns data/sample.parquet __pycache__ .pytest_cache
