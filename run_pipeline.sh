#!/usr/bin/env bash
set -e
export PYTHONPATH=src
python src/download.py
python src/clean.py
python -m pytest -q
python src/train.py
python src/analysis.py
