#!/usr/bin/env bash

# Start the VSA Advisor Bot live worker after preparing the project environment.
# Any additional command-line arguments are forwarded to scripts.run_live_advisor.
set -euo pipefail
cd "$(dirname "$0")"
./setup_environment.sh
exec .venv/bin/python -m scripts.run_live_advisor "$@"
