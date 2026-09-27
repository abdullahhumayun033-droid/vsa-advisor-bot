"""Generate a deliberately blocked MT5 integration-test signal.

This utility creates a controlled WATCH_ONLY signal for validating the
Python-to-MT5 bridge without producing an executable trade recommendation. It
loads the latest project signal only as a schema/template, updates the timestamp,
and explicitly marks the final advisor decision as blocked by a synthetic
high-impact news condition.

The underlying ML field is retained as TRADE_CANDIDATE so the integration test
can demonstrate the distinction between the raw model recommendation and the
final safety-gated decision. The resulting signal is written both to a project
evidence file and to the MT5 Common/Files bridge location.

This script is intended only for safe integration testing. It does not train or
run the ML model, does not execute the VSA scanner, and does not represent a real
market or economic-calendar event.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "outputs" / "ml_signal.csv"
EVIDENCE_OUTPUT = PROJECT / "outputs" / "mt5_safe_integration_test_signal.csv"
MT5_COMMON_OUTPUT = Path.home() / (
    "Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/users/user/"
    "AppData/Roaming/MetaQuotes/Terminal/Common/Files/ml_signal.csv"
)


def main() -> None:
    signal = pd.read_csv(SOURCE).tail(1).copy()
    for column in [
        "signal_time", "advisor_output", "ml_advisor_output", "news_status",
        "news_event", "news_event_time", "news_reason",
    ]:
        signal[column] = signal[column].astype("object")
    signal.loc[:, "signal_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signal.loc[:, "advisor_output"] = "WATCH_ONLY"
    signal.loc[:, "ml_advisor_output"] = "TRADE_CANDIDATE"
    signal.loc[:, "news_status"] = "HIGH_IMPACT_USD_BLOCK_TEST"
    signal.loc[:, "news_blocked"] = True
    signal.loc[:, "news_event"] = "Controlled integration test"
    signal.loc[:, "news_event_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    signal.loc[:, "news_reason"] = "Safe test: final trade decision intentionally blocked."
    EVIDENCE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MT5_COMMON_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    signal.to_csv(EVIDENCE_OUTPUT, index=False)
    signal.to_csv(MT5_COMMON_OUTPUT, index=False)
    print(f"Safe WATCH_ONLY test signal written at {signal.iloc[0]['signal_time']}")
    print(f"Evidence copy: {EVIDENCE_OUTPUT}")
    print(f"MT5 copy: {MT5_COMMON_OUTPUT}")


if __name__ == "__main__":
    main()
