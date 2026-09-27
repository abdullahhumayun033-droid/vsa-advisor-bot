"""Trade-journal persistence and reflection support for the VSA Advisor Bot.

This module manages the user-facing trade journal used to record and review
advisor-generated setups and manually entered demo/practice trades. It stores
signal details, model probability and threshold information, entry/stop/target
levels, the user's own decision, realised R result, notes and lessons learned
in a persistent CSV file.

Advisor-generated entries are keyed by the project's stable signal identifier
so an existing setup can be updated without creating unnecessary duplicate
journal rows. The module also supports later reflection updates and separate
manual journal entries that are clearly marked as user-authored records.

Journal data is intentionally isolated from the trading and machine-learning
pipeline. It is not used to train or retrain the models, does not affect live
model inference, and is not used by MT5 execution. Its purpose is educational:
to help the user review decisions, outcomes and lessons from VSA Advisor Bot
signals and their own practice trades.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from modules.integration_evidence import make_signal_id

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOURNAL_PATH = PROJECT_ROOT / "user_data" / "trade_journal.csv"

JOURNAL_COLUMNS = [
    "journal_id",
    "saved_at_utc",
    "signal_id",
    "signal_time",
    "source_mode",
    "selected_model",
    "scenario_id",
    "direction",
    "advisor_output",
    "model_probability",
    "recommendation_threshold",
    "entry_price",
    "stop_price",
    "target_1r",
    "target_4r",
    "risk_percent",
    "user_decision",
    "result_r",
    "notes",
    "lesson_learned",
]


def read_journal(path: str | Path = DEFAULT_JOURNAL_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    try:
        frame = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    for column in JOURNAL_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    # Journal fields intentionally mix text and an optional numeric result. Keeping
    # object dtype avoids pandas 3.x rejecting a later blank unresolved result.
    return frame[JOURNAL_COLUMNS].astype(object).copy()


def _atomic_write(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(path)


def upsert_journal_entry(
    signal: dict[str, Any],
    source_mode: str,
    user_decision: str,
    notes: str = "",
    lesson_learned: str = "",
    result_r: float | None = None,
    path: str | Path = DEFAULT_JOURNAL_PATH,
    saved_at_utc: datetime | None = None,
) -> pd.DataFrame:
    """Create or update one learning-journal row per stable signal ID.

    Journal data is deliberately isolated from training artefacts. It is a user reflection
    record only and is never consumed by the ML training or inference pipeline.
    """
    path = Path(path)
    signal_id = make_signal_id(signal)
    if not signal_id or signal_id.startswith("UNKNOWN"):
        raise ValueError("A valid signal timestamp, direction and scenario are required before journalling.")

    now = saved_at_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    def _value(name: str, default: Any = "") -> Any:
        value = signal.get(name, default)
        return default if pd.isna(value) else value

    row = {
        "journal_id": signal_id,
        "saved_at_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "signal_id": signal_id,
        "signal_time": str(_value("signal_time")),
        "source_mode": str(source_mode),
        "selected_model": str(_value("selected_model")),
        "scenario_id": int(float(_value("scenario_id", 0))),
        "direction": str(_value("direction")),
        "advisor_output": str(_value("advisor_output", "WATCH_ONLY")),
        "model_probability": float(_value("ml_probability_valid", 0.0)),
        "recommendation_threshold": float(_value("recommendation_threshold", 0.0)),
        "entry_price": float(_value("entry_price", 0.0)),
        "stop_price": float(_value("stop_price", 0.0)),
        "target_1r": float(_value("target_price", 0.0)),
        "target_4r": float(_value("runner_target_price", _value("target_price", 0.0))),
        "risk_percent": float(_value("risk_percent", 0.0)),
        "user_decision": str(user_decision),
        "result_r": None if result_r is None else float(result_r),
        "notes": str(notes).strip(),
        "lesson_learned": str(lesson_learned).strip(),
    }

    frame = read_journal(path)
    existing = frame.index[frame["signal_id"].astype(str) == signal_id].tolist()
    if existing:
        idx = existing[-1]
        # Preserve the original save time so the row represents one continuing reflection.
        row["saved_at_utc"] = str(frame.loc[idx, "saved_at_utc"] or row["saved_at_utc"])
        for column in JOURNAL_COLUMNS:
            frame.loc[idx, column] = row[column]
    else:
        new_row = pd.DataFrame([row], columns=JOURNAL_COLUMNS).astype(object)
        frame = new_row if frame.empty else pd.concat([frame, new_row], ignore_index=True)

    _atomic_write(frame[JOURNAL_COLUMNS], path)
    return frame[JOURNAL_COLUMNS].copy()


def update_journal_reflection(
    signal_id: str,
    *,
    user_decision: str | None = None,
    result_r: float | None = None,
    clear_result: bool = False,
    notes: str | None = None,
    lesson_learned: str | None = None,
    path: str | Path = DEFAULT_JOURNAL_PATH,
) -> pd.DataFrame:
    path = Path(path)
    frame = read_journal(path)
    matches = frame.index[frame["signal_id"].astype(str) == str(signal_id)].tolist()
    if not matches:
        raise ValueError(f"Journal entry not found for signal {signal_id}.")
    idx = matches[-1]
    if user_decision is not None:
        frame.loc[idx, "user_decision"] = str(user_decision)
    if clear_result:
        frame.loc[idx, "result_r"] = None
    elif result_r is not None:
        frame.loc[idx, "result_r"] = float(result_r)
    if notes is not None:
        frame.loc[idx, "notes"] = str(notes).strip()
    if lesson_learned is not None:
        frame.loc[idx, "lesson_learned"] = str(lesson_learned).strip()
    _atomic_write(frame[JOURNAL_COLUMNS], path)
    return frame[JOURNAL_COLUMNS].copy()


def add_manual_journal_entry(
    *,
    trade_time: str,
    direction: str,
    scenario_id: int = 0,
    entry_price: float | None = None,
    stop_price: float | None = None,
    target_1r: float | None = None,
    target_4r: float | None = None,
    risk_percent: float | None = None,
    user_decision: str = "Demo-traded",
    result_r: float | None = None,
    notes: str = "",
    lesson_learned: str = "",
    path: str | Path = DEFAULT_JOURNAL_PATH,
    saved_at_utc: datetime | None = None,
) -> pd.DataFrame:
    """Add a user-authored trade/reflection that is not tied to a model signal.

    Manual journal rows are deliberately marked as MANUAL and are never consumed by
    training, inference or MT5 execution. This lets a learner keep one journal for both
    advisor-generated examples and their own demo/practice trades.
    """
    path = Path(path)
    direction = str(direction).strip().upper()
    if direction not in {"BUY", "SELL"}:
        raise ValueError("Direction must be BUY or SELL.")
    stamp = pd.to_datetime(trade_time, errors="coerce")
    if pd.isna(stamp):
        raise ValueError("A valid trade date/time is required.")
    now = saved_at_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    signal_id = f"MANUAL-{stamp.strftime('%Y%m%d%H%M%S')}-{direction}-S{int(scenario_id)}"

    def _num(value: float | None):
        return None if value is None else float(value)

    row = {
        "journal_id": signal_id,
        "saved_at_utc": now.astimezone(timezone.utc).isoformat(timespec="seconds"),
        "signal_id": signal_id,
        "signal_time": stamp.isoformat(sep=" "),
        "source_mode": "Manual journal entry",
        "selected_model": "",
        "scenario_id": int(scenario_id),
        "direction": direction,
        "advisor_output": "USER_LOG",
        "model_probability": None,
        "recommendation_threshold": None,
        "entry_price": _num(entry_price),
        "stop_price": _num(stop_price),
        "target_1r": _num(target_1r),
        "target_4r": _num(target_4r),
        "risk_percent": _num(risk_percent),
        "user_decision": str(user_decision),
        "result_r": _num(result_r),
        "notes": str(notes).strip(),
        "lesson_learned": str(lesson_learned).strip(),
    }
    frame = read_journal(path)
    if signal_id in set(frame["signal_id"].astype(str)):
        suffix = now.strftime("%H%M%S")
        row["journal_id"] = row["signal_id"] = f"{signal_id}-{suffix}"
    new_row = pd.DataFrame([row], columns=JOURNAL_COLUMNS).astype(object)
    frame = new_row if frame.empty else pd.concat([frame, new_row], ignore_index=True)
    _atomic_write(frame[JOURNAL_COLUMNS], path)
    return frame[JOURNAL_COLUMNS].copy()
