"""Integration-evidence helpers for the VSA Advisor Bot.

This module provides small shared utilities used to keep the Python and MT5
parts of the project aligned and auditable. It creates the stable signal
identifier used across the live advisor, journal and MT5 bridge, and exposes an
integration manifest describing the main deployed boundaries between offline
model training, live completed-bar inference and the MT5 runtime.

The manifest is descriptive rather than executable trading logic. It records
which files and runtime components are involved, confirms the expected model
and signal paths, and makes clear that historical labelled VSA data is used for
offline LR/RF training while live MT5 M5 bars are used only to construct the
feature vector for inference through the saved scikit-learn model.
"""


from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping


def make_signal_id(signal: Mapping[str, Any]) -> str:
    """Create the same human-auditable signal identity used by Python and MT5."""
    timestamp = re.sub(r"[^0-9]", "", str(signal.get("signal_time", "")))
    direction = str(signal.get("direction", "UNKNOWN")).upper().strip()
    scenario = str(signal.get("scenario_id", "0")).strip()
    return f"{timestamp}-{direction}-S{scenario}"


def integration_manifest(
    *,
    bars_path: str | Path,
    model_path: str | Path,
    project_signal_path: str | Path,
    bridge_signal_path: str | Path,
) -> dict[str, Any]:
    """Describe the deployed boundaries so the website and evidence log agree."""
    bars = Path(bars_path)
    model = Path(model_path)
    project_signal = Path(project_signal_path)
    bridge_signal = Path(bridge_signal_path)
    return {
        "training_data_role": "Offline labelled VSA examples used to fit LR and RF; not reread for each live decision.",
        "live_data_role": "Completed MT5 M5 bars used only to construct a new setup feature vector for inference.",
        "vsa_runtime": "Python modules/vsa_strategy.py::scan_vsa_scenarios",
        "ml_runtime": "Python saved scikit-learn pipeline::predict_proba",
        "mt5_runtime": "MQL5 bar exporter + visible VSA indicator + validated signal bridge",
        "bars_path": str(bars),
        "bars_file_exists": bars.exists(),
        "model_path": str(model),
        "model_file_exists": model.exists(),
        "project_signal_path": str(project_signal),
        "bridge_signal_path": str(bridge_signal),
    }
