#!/usr/bin/env bash

# Copy the latest VSA Advisor Bot signal file into MetaTrader 5 Common/Files
# on the project's macOS/Wine setup. This helper does not generate, score or
# modify the recommendation; it only transfers the already-produced
# outputs/ml_signal.csv file to the location read by the MT5 integration.
set -e
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MT5_COMMON="$HOME/Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/users/user/AppData/Roaming/MetaQuotes/Terminal/Common/Files"
mkdir -p "$MT5_COMMON"
cp "$PROJECT_DIR/outputs/ml_signal.csv" "$MT5_COMMON/ml_signal.csv"
echo "MT5 signal updated: $MT5_COMMON/ml_signal.csv"
