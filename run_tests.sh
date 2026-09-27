#!/usr/bin/env bash

# Run the project's automated unittest suite and save the complete console
# output as results/automated_test_log.txt for reproducible evaluation evidence.
set -euo pipefail
cd "$(dirname "$0")"
./setup_environment.sh
mkdir -p results
GLOG_minloglevel=2 .venv/bin/python -m unittest discover -s tests -v 2>&1 | tee results/automated_test_log.txt
