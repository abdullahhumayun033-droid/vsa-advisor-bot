"""Live MT5 inference and signal-integration pipeline for the VSA Advisor Bot.

This module coordinates one live advisory cycle from completed XAUUSD M5
candles through to the signal files consumed by the wider application and MT5
bridge. It locates the MetaTrader 5 Common/Files directory, reads the exporter
heartbeat and broker-time information, checks that the completed-bar feed is
current, and distinguishes stale/disconnected data from the normal weekly
XAUUSD market closure.

When the feed is valid, the pipeline loads the already-fitted LR_VSA_BOT model
and its saved metadata, runs the formal VSA Scenario 1/2/3 scanner on recent
completed bars, and uses the fitted model for probability inference. No model
training or retraining takes place during this live cycle. The resulting ML
decision is then reassessed against the current news-risk state before the
final signal is written atomically to the project output and Python-to-MT5
bridge files.

The module also records integration evidence, stable signal identifiers and
the Logistic Regression local explanation used for transparency. Safety checks
are deliberately fail-closed: disconnected or stale MT5 data, invalid model
configuration, unavailable inputs or other processing errors do not silently
produce a new trade recommendation.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from modules.ml_pipeline import (
    explain_lr_setup,
    load_model_metadata,
    load_selected_model,
    predict_latest_setup_from_bars,
    read_mt5_bars,
    save_ml_signal_csv,
)
from modules.news_calendar import fetch_faireconomy_calendar
from modules.news_risk import apply_news_risk
from modules.integration_evidence import integration_manifest, make_signal_id


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def discover_mt5_common() -> Path:
    """Locate MetaTrader 5 Common/Files on macOS/Windows/Wine.

    An explicit VSA_MT5_COMMON environment variable wins. Otherwise the function
    checks the standard MetaQuotes macOS Wine bundle, native Windows location and
    common Wine layouts. If nothing exists yet, it returns the standard macOS path
    used by the official MetaTrader 5 package so the UI can explain what is missing.
    """
    explicit = os.environ.get("VSA_MT5_COMMON", "").strip()
    if explicit:
        return Path(explicit).expanduser()

    home = Path.home()
    candidates = [
        home / "Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/users/user/AppData/Roaming/MetaQuotes/Terminal/Common/Files",
        home / "AppData/Roaming/MetaQuotes/Terminal/Common/Files",
        home / ".wine/drive_c/users/user/AppData/Roaming/MetaQuotes/Terminal/Common/Files",
    ]
    users_root = home / "Library/Application Support/net.metaquotes.wine.metatrader5/drive_c/users"
    if users_root.exists():
        candidates.extend(users_root.glob("*/AppData/Roaming/MetaQuotes/Terminal/Common/Files"))
    wine_users = home / ".wine/drive_c/users"
    if wine_users.exists():
        candidates.extend(wine_users.glob("*/AppData/Roaming/MetaQuotes/Terminal/Common/Files"))

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_MT5_COMMON = discover_mt5_common()


def read_broker_offset_from_status(common_dir: str | Path = DEFAULT_MT5_COMMON) -> float | None:
    """Read broker UTC offset written by the MT5 bar-exporter EA, if available."""
    status_path = Path(common_dir) / "vsa_mt5_connection_status.csv"
    if not status_path.exists():
        return None
    try:
        frame = pd.read_csv(status_path)
        if frame.empty or "broker_utc_offset_hours" not in frame.columns:
            return None
        value = float(frame.iloc[-1]["broker_utc_offset_hours"])
        if -12.0 <= value <= 14.0:
            return value
    except Exception:
        return None
    return None


def read_mt5_terminal_status(common_dir: str | Path = DEFAULT_MT5_COMMON) -> dict[str, Any]:
    """Read the bar-exporter heartbeat and broker connection flag.

    The status file is refreshed by the exporter timer even when no new M5 candle has
    closed, so it distinguishes an open-but-disconnected MT5 terminal from an merely
    quiet market. A missing file remains "unknown" so isolated tests and historical
    replay are not made dependent on a live terminal.
    """
    status_path = Path(common_dir) / "vsa_mt5_connection_status.csv"
    if not status_path.exists():
        return {"available": False, "connected": None, "age_seconds": None}
    try:
        frame = pd.read_csv(status_path)
        if frame.empty:
            return {"available": False, "connected": None, "age_seconds": None}
        row = frame.iloc[-1].to_dict()
        raw = str(row.get("terminal_connected", "0")).strip().lower()
        connected = raw in {"1", "true", "1.0", "yes"}
        age_seconds = max(0.0, datetime.now().timestamp() - status_path.stat().st_mtime)
        return {
            "available": True,
            "connected": connected,
            "age_seconds": age_seconds,
            "server_time": row.get("server_time", ""),
            "gmt_time": row.get("gmt_time", ""),
            "broker_utc_offset_hours": row.get("broker_utc_offset_hours"),
        }
    except Exception as exc:
        return {"available": False, "connected": None, "age_seconds": None, "error": str(exc)}


def _atomic_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temp.replace(path)


def _existing_signal_time(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        frame = pd.read_csv(path)
        return str(frame.iloc[-1].get("signal_time", "")) if not frame.empty else ""
    except Exception:
        return ""



def _xauusd_weekly_market_closed(now_utc: datetime) -> bool:
    """Return True during the normal XAUUSD weekly weekend closure.

    This guard is used only by the feed-freshness check so a Friday close is not
    incorrectly labelled stale on Saturday/Sunday. It does not alter VSA, ML,
    news-risk or trade-decision logic. Times are intentionally conservative.
    """
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    current = now_utc.astimezone(timezone.utc)
    wd = current.weekday()  # Monday=0 ... Sunday=6
    return (wd == 5) or (wd == 4 and current.hour >= 22) or (wd == 6 and current.hour < 22)

def assess_live_bar_freshness(
    latest_closed_bar: Any,
    broker_utc_offset_hours: float = 0.0,
    *,
    now_utc: datetime | None = None,
    maximum_lag_bars: int = 2,
    timeframe_minutes: int = 5,
) -> dict[str, Any]:
    """Compare the latest exported bar with the current broker clock.

    MT5 bar timestamps represent the opening time of each M5 candle. At 10:06 broker
    time, for example, the most recent *completed* candle should have opened at 10:00.
    The check therefore compares the file's final bar with the expected open time of
    the latest completed bar, rather than only comparing a setup with other rows in the
    same potentially stale CSV.
    """
    latest = pd.to_datetime(latest_closed_bar, errors="coerce")
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    broker_now = now.astimezone(timezone(timedelta(hours=float(broker_utc_offset_hours)))).replace(tzinfo=None)
    broker_now_ts = pd.Timestamp(broker_now)
    floor = broker_now_ts.floor(f"{int(timeframe_minutes)}min")
    expected_latest_open = floor - pd.Timedelta(minutes=int(timeframe_minutes))

    if pd.isna(latest):
        return {
            "feed_state": "DISCONNECTED",
            "feed_fresh": False,
            "broker_now": str(broker_now_ts),
            "expected_latest_closed_bar": str(expected_latest_open),
            "feed_lag_minutes": None,
            "message": "The latest completed-bar timestamp could not be read.",
        }

    # A Friday close is expected to remain the latest bar throughout the weekend.
    # Treat that condition explicitly instead of reporting a many-hour stale feed.
    if _xauusd_weekly_market_closed(now):
        return {
            "feed_state": "MARKET_CLOSED",
            "feed_fresh": False,
            "market_closed": True,
            "broker_now": str(broker_now_ts),
            "expected_latest_closed_bar": str(latest),
            "feed_lag_minutes": None,
            "message": (
                "XAUUSD is in the normal weekly market-closed period. "
                "The final completed broker candle from the previous session is retained; "
                "freshness checking resumes automatically when the weekly session reopens."
            ),
        }

    lag = expected_latest_open - latest
    lag_minutes = float(lag.total_seconds() / 60.0)
    # A timestamp slightly ahead of the calculated expectation can occur around clock
    # boundaries or broker offsets; it is not treated as stale unless it is implausible.
    future_tolerance = pd.Timedelta(minutes=int(timeframe_minutes))
    stale_limit = pd.Timedelta(minutes=max(int(maximum_lag_bars), 0) * int(timeframe_minutes))
    if lag < -future_tolerance:
        state = "CLOCK_MISMATCH"
        fresh = False
        message = "The exported bar is ahead of the expected broker-clock completed bar. Check broker UTC offset/time settings."
    elif lag > stale_limit:
        state = "STALE"
        fresh = False
        message = f"The completed-bar export is stale by approximately {max(lag_minutes, 0):.1f} minutes."
    else:
        state = "LIVE"
        fresh = True
        message = "The completed-bar export is within the allowed live freshness window."

    return {
        "feed_state": state,
        "feed_fresh": fresh,
        "broker_now": str(broker_now_ts),
        "expected_latest_closed_bar": str(expected_latest_open),
        "feed_lag_minutes": lag_minutes,
        "message": message,
    }


def run_live_cycle(
    bars_path: str | Path,
    project_signal_path: str | Path,
    bridge_signal_path: str | Path,
    status_path: str | Path,
    broker_utc_offset_hours: float = 0.0,
    recent_bars: int = 2_500,
    calendar_fetcher: Callable[..., tuple[pd.DataFrame, bool, str]] = fetch_faireconomy_calendar,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Process one completed-bar cycle and fail closed when the live feed is not current."""
    bars_path = Path(bars_path)
    project_signal_path = Path(project_signal_path)
    bridge_signal_path = Path(bridge_signal_path)
    status_path = Path(status_path)
    cycle_now = now_utc or datetime.now(timezone.utc)
    if cycle_now.tzinfo is None:
        cycle_now = cycle_now.replace(tzinfo=timezone.utc)
    status: dict[str, Any] = {
        "cycle_time": cycle_now.astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "cycle_time_utc": cycle_now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "bars_path": str(bars_path),
        "project_signal_path": str(project_signal_path),
        "bridge_signal_path": str(bridge_signal_path),
        "broker_utc_offset_hours": float(broker_utc_offset_hours),
        "output_written": False,
    }
    status.update(integration_manifest(
        bars_path=bars_path,
        model_path=PROJECT_ROOT / "models" / "selected_model.joblib",
        project_signal_path=project_signal_path,
        bridge_signal_path=bridge_signal_path,
    ))
    status.update({
        "ml_inference_executed": False,
        "vsa_scanner_executed": False,
        "retraining_executed": False,
        "feed_state": "DISCONNECTED",
        "feed_fresh": False,
    })

    try:
        terminal = read_mt5_terminal_status(bars_path.parent)
        status["mt5_terminal_status_available"] = terminal.get("available", False)
        status["mt5_terminal_connected"] = terminal.get("connected")
        status["mt5_terminal_heartbeat_age_seconds"] = terminal.get("age_seconds")
        if terminal.get("available") and terminal.get("connected") is False:
            status.update({
                "feed_state": "DISCONNECTED",
                "feed_fresh": False,
                "state": "MT5_TERMINAL_DISCONNECTED",
                "message": "MetaTrader 5 is open/exporter is present, but the terminal is not connected to the broker server.",
            })
            _atomic_json(status, status_path)
            return status
        if terminal.get("available") and terminal.get("age_seconds") is not None and float(terminal["age_seconds"]) > 45:
            status.update({
                "feed_state": "DISCONNECTED",
                "feed_fresh": False,
                "state": "MT5_EXPORTER_HEARTBEAT_STALE",
                "message": "The MT5 exporter heartbeat is stale. Confirm the exporter EA is attached and Algo Trading is enabled.",
            })
            _atomic_json(status, status_path)
            return status

        metadata = load_model_metadata()
        selected = str(metadata.get("selected_model", ""))
        status["selected_model"] = selected
        if selected != "LR_VSA_BOT":
            raise ValueError(
                f"Live worker is locked to LR_VSA_BOT for transparent deployment; metadata selects {selected or 'nothing'}."
            )
        model = load_selected_model()
        bars = read_mt5_bars(bars_path, tail_rows=max(int(recent_bars), 250))
        status["bars_read"] = int(len(bars))
        status["latest_closed_bar"] = str(bars.iloc[-1]["time"]) if not bars.empty else ""

        freshness = assess_live_bar_freshness(
            status["latest_closed_bar"],
            broker_utc_offset_hours=broker_utc_offset_hours,
            now_utc=cycle_now,
        )
        status.update(freshness)
        if freshness.get("feed_state") == "MARKET_CLOSED":
            status.update({
                "state": "MARKET_CLOSED",
                "message": freshness["message"],
            })
            _atomic_json(status, status_path)
            return status

        if not freshness["feed_fresh"]:
            status.update({
                "state": "STALE_OR_DISCONNECTED_FEED",
                "message": freshness["message"],
            })
            _atomic_json(status, status_path)
            return status

        status["vsa_scanner_executed"] = True
        signal, setup_row = predict_latest_setup_from_bars(
            bars,
            model=model,
            metadata=metadata,
            require_recent_closed_bar=True,
            maximum_age_bars=1,
        )
        status["ml_inference_executed"] = True
        status["latest_setup_time"] = signal["signal_time"]
        status["signal_id"] = make_signal_id(signal)

        # Safety is intentionally reassessed on every cycle, even if this is the same
        # VSA setup. A newly approaching news event must be able to downgrade the signal.
        calendar, verified, calendar_source = calendar_fetcher(
            broker_utc_offset_hours=broker_utc_offset_hours,
            fallback_path=PROJECT_ROOT / "data" / "high_impact_news_fallback.csv",
        )
        signal = apply_news_risk(signal, calendar, verified_live=verified)
        explanation = explain_lr_setup(model, setup_row, top_n=8)

        previous_time = _existing_signal_time(project_signal_path)
        duplicate = previous_time == str(signal["signal_time"])

        # Rewriting the same stable signal ID refreshes safety state without creating a
        # logically new recommendation. The MT5 bridge's own duplicate-ID protection
        # remains responsible for suppressing repeated notifications.
        save_ml_signal_csv(signal, project_signal_path)
        if bridge_signal_path.resolve() != project_signal_path.resolve():
            save_ml_signal_csv(signal, bridge_signal_path)

        status.update({
            "state": "DUPLICATE_SETUP_REFRESHED" if duplicate else "NEW_SETUP_SCORED",
            "output_written": True,
            "new_signal": not duplicate,
            "direction": signal["direction"],
            "scenario_id": signal["scenario_id"],
            "lr_probability_valid": signal["ml_probability_valid"],
            "lr_threshold": signal["recommendation_threshold"],
            "ml_decision": signal["ml_advisor_output"],
            "final_decision": signal["advisor_output"],
            "news_calendar_verified": bool(verified),
            "news_calendar_source": calendar_source,
            "news_status": signal["news_status"],
            "news_reason": signal.get("news_reason", ""),
            "signal_id": make_signal_id(signal),
            "model_artifact_used": str(PROJECT_ROOT / "models" / "selected_model.joblib"),
            "model_method_used": "predict_proba",
            "feature_count_used": int(len(setup_row.index.intersection(metadata.get("feature_columns", [])))),
            "lr_local_explanation": explanation.to_dict(orient="records"),
        })
        if duplicate:
            status["message"] = "The existing setup was rescored for current safety/news state; its stable signal ID prevents a duplicate alert."
    except FileNotFoundError as error:
        status.update({
            "state": "STALE_OR_DISCONNECTED_FEED",
            "feed_state": "DISCONNECTED",
            "feed_fresh": False,
            "message": f"Live MT5 completed-bar file is unavailable: {error}",
        })
    except ValueError as error:
        message = str(error)
        status.update({
            "state": "NO_NEW_VSA_SETUP" if message.startswith("NO_NEW_VSA_SETUP:") else "SAFE_ERROR",
            "message": message,
        })
    except Exception as error:
        status.update({"state": "SAFE_ERROR", "message": f"{type(error).__name__}: {error}"})

    _atomic_json(status, status_path)
    return status
