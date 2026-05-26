# Development targets — always use .venv (see scripts/ensure_venv.sh).
SHELL := /bin/bash
ROOT := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
VENV := $(ROOT).venv
PYTHON := $(VENV)/bin/python
DEV := $(ROOT)scripts/dev

.PHONY: venv install test format format-check lint cli-help

venv:
	@$(DEV) true

install: venv
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -e ".[dev]"

test: install
	@$(PYTHON) -m pytest tests/ -v

format: install
	@$(PYTHON) -m black --target-version py310 edge_train/ tests/

format-check: install
	@$(PYTHON) -m black --check --target-version py310 edge_train/ tests/

lint: format-check test

cli-help: install
	@$(PYTHON) -c "from edge_train.cli import main; main(['--help'])"
