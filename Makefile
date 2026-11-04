.PHONY: setup data features funnel segment propensity rank nba experiment app-data app test lint smoke all clean

PYTHON := .venv/Scripts/python

setup:
	uv pip install -e .

data:
	$(PYTHON) -m custintel ingest

features:
	$(PYTHON) -m custintel features

funnel:
	$(PYTHON) -m custintel funnel

segment:
	$(PYTHON) -m custintel segment

propensity:
	$(PYTHON) -m custintel propensity

rank:
	$(PYTHON) -m custintel rank

nba:
	$(PYTHON) -m custintel nba

experiment:
	$(PYTHON) -m custintel experiment

app-data:
	$(PYTHON) -m custintel app-data

app:
	$(PYTHON) -m streamlit run app/app.py

test:
	$(PYTHON) -m pytest tests/

lint:
	ruff check .
	ruff format --check .

smoke:
	$(PYTHON) -m custintel all --smoke
	$(PYTHON) -m pytest tests/

all: data features funnel segment propensity rank nba experiment app-data

clean:
	rm -rf data/processed/*.parquet data/processed/features/*.parquet data/processed/duckdb_tmp
