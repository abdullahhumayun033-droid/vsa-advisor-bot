"""Controlled live-pipeline replay evaluation for the VSA Advisor Bot.

This script evaluates the integration of the project's deployed advisory
components by replaying a known formal VSA setup through the same
run_live_cycle() function used by the live worker. Historical completed XAUUSD
M5 bars are written to an isolated temporary replay environment so the test
does not modify the real MT5 bridge or live project signal.

The replay checks that the saved LR model can score the formal setup, that the
selected recommendation threshold is applied, that a clear verified calendar
preserves the underlying ML candidate, and that a controlled high-impact USD
event can downgrade the final recommendation to WATCH_ONLY without changing
the original ML decision.

The evaluation also checks duplicate-signal behaviour, creation of the local
Logistic Regression explanation, and the fixed signal-file contract expected
by the MT5 bridge. Replay signals and a JSON integration summary are written to
the results directory as evidence, and the script raises an error when any
required integration check fails.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.live_pipeline import PROJECT_ROOT, run_live_cycle
from modules.ml_pipeline import read_mt5_bars


RESULTS = PROJECT_ROOT / "results"


def _calendar_provider(event_time: pd.Timestamp):
    def provider(**_kwargs):
        return pd.DataFrame([{
            "event_time": event_time,
            "currency": "USD",
            "event": "Controlled high-impact USD replay event",
            "impact": "High",
            "source": "Controlled verified replay calendar",
        }]), True, "Controlled verified replay calendar"
    return provider


def main() -> None:
    setup_evidence = pd.read_csv(RESULTS / "vsa_setup_training_dataset.csv")
    latest_setup_time = pd.to_datetime(setup_evidence["time"]).max()
    bars = read_mt5_bars(PROJECT_ROOT / "data" / "bars.csv")
    replay = bars[bars["time"] <= latest_setup_time].tail(2500).copy()
    if replay.empty or replay.iloc[-1]["time"] != latest_setup_time:
        raise AssertionError("Replay window does not end on the selected formal VSA setup candle.")

    RESULTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as folder:
        temp = Path(folder)
        bars_path = temp / "replay_bars.csv"
        replay.to_csv(bars_path, index=False)

        clear_project = temp / "clear_project_signal.csv"
        clear_bridge = temp / "clear_bridge_signal.csv"
        clear_status_path = temp / "clear_status.json"
        clear_status = run_live_cycle(
            bars_path,
            clear_project,
            clear_bridge,
            clear_status_path,
            broker_utc_offset_hours=3,
            calendar_fetcher=_calendar_provider(latest_setup_time + pd.Timedelta(days=1)),
        )
        clear_signal = pd.read_csv(clear_project)
        clear_signal.to_csv(RESULTS / "live_replay_clear_signal.csv", index=False)
        pd.DataFrame(clear_status.get("lr_local_explanation", [])).to_csv(
            RESULTS / "live_replay_lr_explanation.csv", index=False
        )

        duplicate_status = run_live_cycle(
            bars_path,
            clear_project,
            clear_bridge,
            temp / "duplicate_status.json",
            broker_utc_offset_hours=3,
            calendar_fetcher=_calendar_provider(latest_setup_time + pd.Timedelta(days=1)),
        )

        blocked_project = temp / "blocked_project_signal.csv"
        blocked_bridge = temp / "blocked_bridge_signal.csv"
        blocked_status = run_live_cycle(
            bars_path,
            blocked_project,
            blocked_bridge,
            temp / "blocked_status.json",
            broker_utc_offset_hours=3,
            calendar_fetcher=_calendar_provider(latest_setup_time),
        )
        blocked_signal = pd.read_csv(blocked_project)
        blocked_signal.to_csv(RESULTS / "live_replay_news_blocked_signal.csv", index=False)

    clear_row = clear_signal.iloc[0]
    blocked_row = blocked_signal.iloc[0]
    checks = {
        "selected_model_is_lr": clear_row["selected_model"] == "LR_VSA_BOT",
        "new_setup_scored": clear_status.get("state") == "NEW_SETUP_SCORED",
        "probability_above_selected_threshold": float(clear_row["ml_probability_valid"]) >= float(clear_row["recommendation_threshold"]),
        "clear_calendar_keeps_ml_candidate": clear_row["advisor_output"] == "TRADE_CANDIDATE",
        "duplicate_setup_suppressed": duplicate_status.get("state") == "DUPLICATE_SETUP_SKIPPED",
        "news_gate_preserves_raw_ml_decision": blocked_row["ml_advisor_output"] == "TRADE_CANDIDATE",
        "news_gate_changes_final_decision": blocked_row["advisor_output"] == "WATCH_ONLY",
        "news_gate_records_block": bool(blocked_row["news_blocked"]),
        "local_lr_explanation_created": len(clear_status.get("lr_local_explanation", [])) == 8,
        "live_signal_contract_has_29_columns": len(clear_signal.columns) == 29,
    }
    summary = {
        "evaluation": "Controlled completed-bar replay; no live MT5 signal was modified",
        "replayed_setup_time": str(latest_setup_time),
        "selected_model": str(clear_row["selected_model"]),
        "scenario_id": int(clear_row["scenario_id"]),
        "direction": str(clear_row["direction"]),
        "lr_probability_valid": float(clear_row["ml_probability_valid"]),
        "selected_threshold": float(clear_row["recommendation_threshold"]),
        "clear_news_final_decision": str(clear_row["advisor_output"]),
        "blocked_news_raw_ml_decision": str(blocked_row["ml_advisor_output"]),
        "blocked_news_final_decision": str(blocked_row["advisor_output"]),
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    (RESULTS / "live_replay_integration_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if not summary["all_checks_passed"]:
        raise AssertionError("One or more controlled live-replay checks failed.")


if __name__ == "__main__":
    main()
