"""Export untouched-test VSA/ML signals for MT5 Strategy Tester replay.

This script converts the Logistic Regression predictions from the project's
held-out test period into the CSV format used by the MT5 Strategy Tester replay.
It reads the saved LR test predictions and the stored LR recommendation
threshold, preserves the formal VSA setup direction, scenario and trade levels,
and records whether each setup would be accepted by the LR filter.

The exported replay is evaluation evidence rather than live model inference.
No model is trained or rescored in this script; it reuses predictions that were
already generated from the untouched Python test split. The resulting
vsa_lr_test_signals.csv file allows MT5 execution behaviour to be compared
against the same formal VSA setups evaluated in Python.

An accompanying audit file records the replay period, number of formal VSA
setups, LR threshold, accepted candidates and important benchmark assumptions,
including that MT5 enters on the first tradable tick after the completed signal
candle and that historical news filtering is excluded unless equivalent
verified event data is supplied.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MT5 = ROOT / "mt5"


def main() -> None:
    predictions = pd.read_csv(RESULTS / "lr_test_predictions.csv")
    metadata = json.loads((ROOT / "models" / "model_metadata.json").read_text())
    threshold = float(metadata["model_thresholds"]["LR_VSA_BOT"])

    replay = predictions.rename(columns={
        "time": "signal_time",
        "probability_valid_trade": "lr_probability",
    })[[
        "signal_time", "direction", "scenario_id", "lr_probability", "risk_percent",
        "entry_price", "stop_price", "target_price", "runner_target_price",
    ]].copy()
    replay["lr_trade_candidate"] = (replay["lr_probability"] >= threshold).astype(int)
    replay = replay.sort_values("signal_time").reset_index(drop=True)
    output = MT5 / "vsa_lr_test_signals.csv"
    replay.to_csv(output, index=False, date_format="%Y-%m-%d %H:%M:%S")

    audit = {
        "purpose": "Untouched-test VSA setup replay for the MT5 Strategy Tester",
        "source": "results/lr_test_predictions.csv",
        "output": str(output.relative_to(ROOT)),
        "test_start": str(replay.iloc[0]["signal_time"]),
        "test_end": str(replay.iloc[-1]["signal_time"]),
        "formal_vsa_setups": int(len(replay)),
        "lr_threshold": threshold,
        "lr_accepted_setups": int(replay["lr_trade_candidate"].sum()),
        "vsa_only_setups": int(len(replay)),
        "signal_rows_match_python_test_predictions": True,
        "execution_note": "The tester opens at the first tradable tick after a completed signal candle; stops and 1R/4R targets are recalculated from that market entry.",
        "news_filter_note": "Historical news filtering is intentionally excluded from every MT5 benchmark unless equivalent verified event data is supplied.",
    }
    (RESULTS / "mt5_replay_signal_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
