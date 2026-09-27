"""Command-line live worker for the VSA Advisor Bot.

This script runs the project's live advisory loop around the shared
modules.live_pipeline implementation. It watches the completed XAUUSD M5 bar
export produced by MT5, determines the broker UTC offset, and repeatedly calls
run_live_cycle() to process the latest completed-bar state.

The worker does not implement the VSA strategy or retrain the machine-learning
model itself. Those responsibilities remain in the shared strategy, ML and live
pipeline modules. This script mainly supplies runtime paths, polling behaviour
and broker-time configuration, then reports the resulting live-cycle state to
the terminal.

By default it keeps polling for new completed-bar information at a configurable
interval, while the --once option allows a single cycle to be processed for
testing or diagnostics. Signal output is written both to the project output
location and, when configured, to the MT5 Common/Files bridge path used by the
live integration.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from modules.live_pipeline import DEFAULT_MT5_COMMON, PROJECT_ROOT, read_broker_offset_from_status, run_live_cycle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Watch completed XAUUSD M5 bars, score new formal VSA setups with LR, and update MT5."
    )
    parser.add_argument("--bars", default=str(DEFAULT_MT5_COMMON / "xauusd_m5_bars.csv"))
    parser.add_argument("--project-output", default=str(PROJECT_ROOT / "outputs" / "ml_signal.csv"))
    parser.add_argument("--bridge-output", default=str(DEFAULT_MT5_COMMON / "ml_signal.csv"))
    parser.add_argument("--status", default=str(PROJECT_ROOT / "outputs" / "live_advisor_status.json"))
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--recent-bars", type=int, default=2500)
    parser.add_argument(
        "--broker-utc-offset",
        type=float,
        default=None,
        help=("Broker server offset from UTC. If omitted, the worker reads the offset "
              "automatically from vsa_mt5_connection_status.csv written by the MT5 exporter EA."),
    )
    parser.add_argument("--once", action="store_true", help="Process one cycle and exit.")
    args = parser.parse_args()

    broker_offset = args.broker_utc_offset
    if broker_offset is None:
        broker_offset = read_broker_offset_from_status(DEFAULT_MT5_COMMON)
        if broker_offset is None:
            # Weekend/no-tick MT5 sessions can leave TimeCurrent stale and therefore
            # produce an impossible heartbeat offset. Use the project's broker fallback
            # instead of aborting the worker; an explicit CLI value still overrides it.
            broker_offset = 3.0
            print(
                "MT5 broker UTC offset is unavailable/invalid; using project fallback UTC+3. "
                "Pass --broker-utc-offset to override.",
                flush=True,
            )
        else:
            print(f"Auto-detected broker UTC offset: {broker_offset:+g} hours", flush=True)

    while True:
        result = run_live_cycle(
            bars_path=Path(args.bars),
            project_signal_path=Path(args.project_output),
            bridge_signal_path=Path(args.bridge_output),
            status_path=Path(args.status),
            broker_utc_offset_hours=float(broker_offset),
            recent_bars=args.recent_bars,
        )
        print(json.dumps({
            "cycle_time": result.get("cycle_time"),
            "state": result.get("state"),
            "latest_closed_bar": result.get("latest_closed_bar", ""),
            "latest_setup_time": result.get("latest_setup_time", ""),
            "selected_model": result.get("selected_model", ""),
            "output_written": result.get("output_written", False),
            "message": result.get("message", ""),
        }, default=str), flush=True)
        if args.once:
            break
        time.sleep(max(args.poll_seconds, 5))


if __name__ == "__main__":
    main()
