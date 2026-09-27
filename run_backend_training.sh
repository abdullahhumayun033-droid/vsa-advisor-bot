#!/usr/bin/env bash

# Rebuild the VSA Advisor Bot training and evaluation evidence from the project
# dataset. This script prepares the environment, trains the LR and RF models,
# then runs the robustness and explainability evaluation scripts.
set -euo pipefail
cd "$(dirname "$0")"
./setup_environment.sh
echo "Starting backend training for screenshots..."
echo "This trains LR+VSA and RF+VSA separately, then compares them."
.venv/bin/python backend_training/train_models.py --data data/bars.csv --lookback 20 --max-bars-after-cab 36 --outcome-horizon 36 --rf-trees 80 --rf-depth 8 --rf-leaf 80 --seed 42
.venv/bin/python backend_training/evaluate_robustness.py
.venv/bin/python backend_training/evaluate_explainability.py
echo "Backend training complete. See results/training_log.txt, model comparison, robustness and XAI evidence files."
