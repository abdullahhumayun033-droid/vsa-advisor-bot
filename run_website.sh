#!/usr/bin/env bash

# Start the VSA Advisor Bot Streamlit website on the local machine after
# preparing the project environment. File watching is disabled for a stable
# demonstration/runtime session.
set -euo pipefail
cd "$(dirname "$0")"
./setup_environment.sh
exec .venv/bin/python -m streamlit run app.py --server.port=8501 --server.address=127.0.0.1 --server.fileWatcherType=none
