# Convenience targets. Everything here is a one-liner you could type yourself;
# the point is that CI and a new contributor run the same commands.

PY ?= python

.PHONY: help install sample analysis setlist web test lint check clean serve

help:
	@echo "install   install runtime + dev dependencies"
	@echo "sample    regenerate the synthetic sample data"
	@echo "analysis  run the pipeline, render figures, write REPORT.md"
	@echo "setlist   build a peak-time set and explain it"
	@echo "web       re-export the dashboard's JSON payloads"
	@echo "serve     serve the dashboard at http://localhost:8000"
	@echo "test      run the test suite"
	@echo "lint      run ruff"
	@echo "check     lint + test + a full pipeline run (what CI does)"

install:
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt

sample:
	$(PY) src/generate_sample.py

analysis:
	$(PY) -m src.run_analysis

setlist:
	$(PY) -m src.setlist --arc peak --minutes 90 --explain

web:
	$(PY) -m src.export_web

serve: web
	@echo "http://localhost:8000"
	@cd web && $(PY) -m http.server 8000

test:
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests

check: lint test
	$(PY) -m src.run_analysis --no-figures

clean:
	rm -rf .pytest_cache **/__pycache__ .ruff_cache
