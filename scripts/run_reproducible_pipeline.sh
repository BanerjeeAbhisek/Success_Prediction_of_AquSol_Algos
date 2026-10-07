#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

python -m pip install -r requirements-lock.txt
python -m pip install -e . --no-deps

aquasol-build-master
aquasol-build-features
aquasol-build-splits

aquasol-run-baselines \
  --strategies random scaffold low_similarity \
  --repeats 1 2 3 4 5 \
  --models dummy ridge random_forest extra_trees hist_gradient_boosting svr mlp \
           elastic_net knn xgboost ngboost esol

aquasol-run-chemprop --design within --append-canonical
aquasol-build-meta-dataset

aquasol-build-source-holdouts
aquasol-run-source-holdouts \
  --models dummy ridge random_forest extra_trees hist_gradient_boosting svr mlp

aquasol-run-source-holdouts \
  --models elastic_net knn xgboost ngboost esol \
  --append

aquasol-run-chemprop --design source --append-canonical

aquasol-build-meta-dataset
aquasol-run-meta-model
aquasol-build-reproducibility
pytest
