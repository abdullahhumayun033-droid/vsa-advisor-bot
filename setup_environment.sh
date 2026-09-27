#!/usr/bin/env bash

# Prepare the project's local Python environment. The script creates the
# virtual environment when needed and installs the pinned dependencies only
# when the expected runtime packages or scikit-learn version are unavailable.
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x .venv/bin/python ]]; then
  echo "Creating portable project virtual environment..."
  rm -rf .venv
  python3 -m venv .venv
fi

if ! .venv/bin/python - <<'PY' >/dev/null 2>&1
import joblib, numpy, pandas, plotly, sklearn, streamlit
assert sklearn.__version__ == "1.9.0"
PY
then
  echo "Installing pinned project dependencies..."
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi

echo "Environment ready: $(.venv/bin/python --version)"
