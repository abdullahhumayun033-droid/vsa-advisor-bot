"""Machine-learning and inference pipeline for the VSA Advisor Bot.

This module connects the rule-based VSA strategy to the project's supervised
machine-learning workflow. It reads and prepares XAUUSD M5 market data, builds
model features from formal Scenario 1, 2 and 3 VSA setups, creates chronological
and purged temporal splits, and provides the shared data structures used by the
offline Logistic Regression (LR) and Random Forest (RF) training/evaluation
workflow.

For deployment, this module does not retrain a model when the Streamlit
application starts or when new MT5 bars arrive. It loads the already-fitted
selected model and its saved metadata from the models directory. The deployed
advisor requires the selected LR_VSA_BOT model and uses predict_proba() to
estimate the probability that a formal VSA candidate is a VALID_TRADE. That
probability is compared with the stored recommendation threshold to produce a
TRADE_CANDIDATE or WATCH_ONLY ML decision before the separate news-risk layer
is applied.

The module also supports recent-bar VSA inference, the fixed CSV signal contract
used by the Python-to-MT5 bridge, and intrinsic Logistic Regression explanations.
The explanation audit reconstructs LR probabilities from the fitted scaler,
coefficients and intercept to verify model faithfulness; these contributions
explain the fitted model calculation rather than claiming causal market effects.
"""

from __future__ import annotations

import json
from collections import deque
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn

from modules.vsa_strategy import VSAConfig, scan_vsa_scenarios
from modules.news_risk import apply_news_risk

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
OUTPUTS_DIR = Path("outputs")
DEFAULT_DATA_FILE = DATA_DIR / "bars.csv"

SIGNAL_COLUMNS = [
    "signal_time", "cab_time", "symbol", "timeframe", "selected_model",
    "direction", "scenario_id", "ml_probability_valid",
    "recommendation_threshold", "advisor_output", "entry_price", "stop_price",
    "target_price", "relative_spread", "relative_volume", "close_location",
    "body_ratio", "had_opposite_sweep", "breakout_strength", "risk_percent",
    "runner_target_price", "ml_advisor_output", "news_window_before_minutes",
    "news_window_after_minutes", "news_event", "news_event_time", "news_status",
    "news_blocked", "news_reason",
]

MODEL_FEATURES = [
    "scenario_id",
    "direction_num",
    "relative_spread",
    "relative_volume",
    "close_location",
    "body_ratio",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "cab_range",
    "cab_relative_volume",
    "entry_to_stop_distance",
    "risk_reward_distance",
    "bars_after_cab",
    "had_opposite_sweep",
    "breakout_strength",
    "return_3",
    "return_6",
    "return_12",
    "volatility_12",
    "volatility_48",
    "hour",
    "day_of_week",
    "london_session",
    "newyork_session",
]

LABELS = [0, 1]
LABEL_NAMES = {0: "AVOID_OR_FAILED", 1: "VALID_TRADE"}


def ensure_dirs() -> None:
    for folder in [DATA_DIR, MODELS_DIR, RESULTS_DIR, OUTPUTS_DIR]:
        folder.mkdir(parents=True, exist_ok=True)


def read_mt5_bars(path: str | Path, tail_rows: int | None = None) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    source: str | Path = path
    if tail_rows:
        with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
            header = stream.readline()
            recent_lines = deque(stream, maxlen=max(int(tail_rows), 1))
        source = StringIO(header + "".join(recent_lines))
    try:
        raw = pd.read_csv(source, sep="\t")
        if raw.shape[1] <= 1:
            if hasattr(source, "seek"):
                source.seek(0)
            raw = pd.read_csv(source)
    except Exception:
        if hasattr(source, "seek"):
            source.seek(0)
        raw = pd.read_csv(source)
    raw.columns = [str(c).strip().strip("<>").lower().replace(" ", "_") for c in raw.columns]
    if "date" in raw.columns and "time" in raw.columns:
        dt = raw["date"].astype(str) + " " + raw["time"].astype(str)
        raw["time"] = pd.to_datetime(dt, format="%Y.%m.%d %H:%M:%S", errors="coerce")
    elif "datetime" in raw.columns:
        raw["time"] = pd.to_datetime(raw["datetime"], errors="coerce")
    else:
        raw["time"] = pd.to_datetime(raw["time"], errors="coerce")
    rename = {
        "open": "open", "high": "high", "low": "low", "close": "close",
        "tickvol": "tick_volume", "tick_volume": "tick_volume",
        "vol": "real_volume", "volume": "real_volume", "spread": "broker_spread_points",
    }
    raw = raw.rename(columns=rename)
    required = ["time", "open", "high", "low", "close", "tick_volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing required columns in bars file: {missing}")
    if "real_volume" not in raw.columns:
        raw["real_volume"] = 0
    if "broker_spread_points" not in raw.columns:
        raw["broker_spread_points"] = 0
    out = raw[["time", "open", "high", "low", "close", "tick_volume", "real_volume", "broker_spread_points"]].copy()
    for col in ["open", "high", "low", "close", "tick_volume", "real_volume", "broker_spread_points"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["time", "open", "high", "low", "close", "tick_volume"])
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return out


def add_candle_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    d = df.copy()
    d["bar_range"] = d["high"] - d["low"]
    d["body"] = d["close"] - d["open"]
    d["body_abs"] = d["body"].abs()
    d["upper_wick"] = d["high"] - d[["open", "close"]].max(axis=1)
    d["lower_wick"] = d[["open", "close"]].min(axis=1) - d["low"]
    d["avg_range"] = d["bar_range"].rolling(lookback).mean().shift(1)
    d["avg_volume"] = d["tick_volume"].rolling(lookback).mean().shift(1)
    d["relative_spread"] = d["bar_range"] / d["avg_range"].replace(0, np.nan)
    d["relative_volume"] = d["tick_volume"] / d["avg_volume"].replace(0, np.nan)
    d["close_location"] = (d["close"] - d["low"]) / d["bar_range"].replace(0, np.nan)
    d["body_ratio"] = d["body_abs"] / d["bar_range"].replace(0, np.nan)
    d["upper_wick_ratio"] = d["upper_wick"] / d["bar_range"].replace(0, np.nan)
    d["lower_wick_ratio"] = d["lower_wick"] / d["bar_range"].replace(0, np.nan)
    d["return_3"] = d["close"].pct_change(3)
    d["return_6"] = d["close"].pct_change(6)
    d["return_12"] = d["close"].pct_change(12)
    d["return_1"] = d["close"].pct_change(1)
    d["volatility_12"] = d["return_1"].rolling(12).std().shift(1)
    d["volatility_48"] = d["return_1"].rolling(48).std().shift(1)
    d["hour"] = d["time"].dt.hour
    d["day_of_week"] = d["time"].dt.dayofweek
    d["london_session"] = d["hour"].between(7, 15).astype(int)
    d["newyork_session"] = d["hour"].between(13, 21).astype(int)
    d["is_cab"] = ((d["relative_spread"] >= 1.10) & (d["relative_volume"] >= 1.25)).astype(int)
    return d


def first_touch_outcome(
    highs: np.ndarray,
    lows: np.ndarray,
    direction: str,
    target_price: float,
    stop_price: float,
) -> str:
    if len(highs) == 0:
        return "EXPIRED"
    if direction == "BUY":
        for h, l in zip(highs, lows):
            if l <= stop_price:
                return "LOSS"
            if h >= target_price:
                return "WIN"
    else:
        for h, l in zip(highs, lows):
            if h >= stop_price:
                return "LOSS"
            if l <= target_price:
                return "WIN"
    return "EXPIRED"


def build_vsa_setup_dataset(
    data_path: str | Path = DEFAULT_DATA_FILE,
    lookback: int = 20,
    max_bars_after_cab: int = 36,
    outcome_horizon: int = 36,
    reward_risk: float = 1.2,
    min_risk_points: float = 0.5,
    cooldown_bars: int = 6,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Create ML rows from VSA strategy setups, then label each setup by TP/SL outcome.

    VSA is the strategy and feature-engineering layer. LR/RF learn from *VSA setup outcomes*,
    not from arbitrary raw candles. A row appears only when the VSA scanner finds a candidate
    BUY/SELL setup around a recent Climactic Action Bar (CAB).
    """
    raw = read_mt5_bars(data_path)
    d = add_candle_features(raw, lookback=lookback)
    d = d.dropna().reset_index(drop=True)

    strategy_config = VSAConfig(
        volume_lookback=30,
        cab_volume_ratio=1.5,
        max_bars_after_cab=max_bars_after_cab,
        outcome_horizon=outcome_horizon,
        min_risk_points=min_risk_points,
    )
    formal_setups = scan_vsa_scenarios(d, strategy_config)
    if formal_setups.empty:
        raise ValueError("No formal VSA Scenario 1/2/3 setups were generated with the configured thresholds.")
    formal_setups = formal_setups.replace([np.inf, -np.inf], np.nan).dropna(subset=MODEL_FEATURES + ["target"])
    formal_setups = formal_setups.sort_values("time").reset_index(drop=True)
    n = len(formal_setups)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    meta = {
        "rows_raw": int(len(raw)), "rows_after_candle_features": int(len(d)),
        "vsa_setup_rows": int(n), "period_start": str(raw["time"].min()),
        "period_end": str(raw["time"].max()), "timeframe": "M5", "lookback": int(lookback),
        "cab_volume_lookback": strategy_config.volume_lookback,
        "cab_minimum_volume_ratio": strategy_config.cab_volume_ratio,
        "momentum_minimum_body_ratio": strategy_config.momentum_body_ratio,
        "momentum_minimum_range_ratio": strategy_config.momentum_range_ratio,
        "trend_definition": f"Close relative to EMA{strategy_config.trend_ema_period} with five-bar EMA slope",
        "max_bars_after_cab": int(max_bars_after_cab),
        "outcome_horizon_candles": int(outcome_horizon),
        "outcome_horizon_minutes": int(outcome_horizon * 5),
        "exit_management": "Close 70% at 1R; move the remaining 30% stop to breakeven; exit runner at 4R or breakeven.",
        "maximum_managed_result_r": 1.9,
        "train_rows": int(train_end), "validation_rows": int(val_end - train_end), "test_rows": int(n - val_end),
        "train_start": str(formal_setups.iloc[0]["time"]), "train_end": str(formal_setups.iloc[train_end - 1]["time"]),
        "validation_start": str(formal_setups.iloc[train_end]["time"]), "validation_end": str(formal_setups.iloc[val_end - 1]["time"]),
        "test_start": str(formal_setups.iloc[val_end]["time"]), "test_end": str(formal_setups.iloc[-1]["time"]),
        "wins": int((formal_setups["target"] == 1).sum()), "non_wins": int((formal_setups["target"] == 0).sum()),
        "scenario_1_risk_percent": 0.5, "scenario_2_risk_percent": 1.0, "scenario_3_risk_percent": 1.0,
        "label_definition": "VALID_TRADE means price reached the 1R first target before the protective stop. Then 70% is realised at 1R and the remaining 30% either exits at breakeven or reaches 4R. Same-bar ambiguity is scored conservatively as stop first.",
        "final_decision_design": "The formal mirrored VSA state machine generates Scenario 1/2/3 candidates. LR and RF learn only from those setup outcomes; the selected model filters future VSA candidates.",
    }
    return formal_setups, meta

    setups: List[Dict[str, Any]] = []
    last_cab_idx = None
    last_cab = None
    opposite_low_swept = False
    opposite_high_swept = False
    last_signal_idx = -10_000

    for i, row in d.iterrows():
        if row["is_cab"] == 1:
            last_cab_idx = i
            last_cab = row
            opposite_low_swept = False
            opposite_high_swept = False
            continue

        if last_cab is None or last_cab_idx is None:
            continue
        bars_after = i - last_cab_idx
        if bars_after < 1 or bars_after > max_bars_after_cab:
            continue

        cab_high = float(last_cab["high"])
        cab_low = float(last_cab["low"])
        cab_range = max(cab_high - cab_low, min_risk_points)
        cab_relative_volume = float(last_cab["relative_volume"])

        if row["low"] < cab_low:
            opposite_low_swept = True
        if row["high"] > cab_high:
            opposite_high_swept = True

        # Candidate setup is a close beyond the CAB trigger with candle confirmation.
        buy_breakout = (row["close"] > cab_high) and (row["close_location"] >= 0.55)
        sell_breakout = (row["close"] < cab_low) and (row["close_location"] <= 0.45)
        if not (buy_breakout or sell_breakout):
            continue
        if i - last_signal_idx < cooldown_bars:
            continue

        direction = "BUY" if buy_breakout else "SELL"
        direction_num = 1 if direction == "BUY" else -1
        had_opposite_sweep = int(opposite_low_swept if direction == "BUY" else opposite_high_swept)
        breakout_strength = abs(float(row["close"] - (cab_high if direction == "BUY" else cab_low))) / cab_range

        if had_opposite_sweep and breakout_strength >= 0.10:
            scenario_id = 3
        elif had_opposite_sweep:
            scenario_id = 2
        else:
            scenario_id = 1

        entry = float(row["close"])
        if direction == "BUY":
            stop = min(cab_low, float(row["low"]))
            risk = max(entry - stop, min_risk_points)
            target = entry + reward_risk * risk
        else:
            stop = max(cab_high, float(row["high"]))
            risk = max(stop - entry, min_risk_points)
            target = entry - reward_risk * risk

        future = d.iloc[i + 1: i + 1 + outcome_horizon]
        outcome = first_touch_outcome(
            future["high"].to_numpy(),
            future["low"].to_numpy(),
            direction,
            target,
            stop,
        )
        target_label = 1 if outcome == "WIN" else 0

        setups.append({
            "time": row["time"],
            "cab_time": last_cab["time"],
            "direction": direction,
            "direction_num": direction_num,
            "scenario_id": scenario_id,
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "outcome": outcome,
            "target": target_label,
            "relative_spread": float(row["relative_spread"]),
            "relative_volume": float(row["relative_volume"]),
            "close_location": float(row["close_location"]),
            "body_ratio": float(row["body_ratio"]),
            "upper_wick_ratio": float(row["upper_wick_ratio"]),
            "lower_wick_ratio": float(row["lower_wick_ratio"]),
            "cab_range": float(cab_range),
            "cab_relative_volume": cab_relative_volume,
            "entry_to_stop_distance": float(risk),
            "risk_reward_distance": float(abs(target - entry)),
            "bars_after_cab": int(bars_after),
            "had_opposite_sweep": int(had_opposite_sweep),
            "breakout_strength": float(breakout_strength),
            "return_3": float(row["return_3"]),
            "return_6": float(row["return_6"]),
            "return_12": float(row["return_12"]),
            "volatility_12": float(row["volatility_12"]),
            "volatility_48": float(row["volatility_48"]),
            "hour": int(row["hour"]),
            "day_of_week": int(row["day_of_week"]),
            "london_session": int(row["london_session"]),
            "newyork_session": int(row["newyork_session"]),
        })
        last_signal_idx = i

    setup_df = pd.DataFrame(setups)
    if setup_df.empty:
        raise ValueError("No VSA setups were generated. Loosen CAB/scenario thresholds.")
    setup_df = setup_df.replace([np.inf, -np.inf], np.nan).dropna(subset=MODEL_FEATURES + ["target"])
    setup_df = setup_df.sort_values("time").reset_index(drop=True)

    n = len(setup_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    meta = {
        "rows_raw": int(len(raw)),
        "rows_after_candle_features": int(len(d)),
        "vsa_setup_rows": int(n),
        "period_start": str(raw["time"].min()),
        "period_end": str(raw["time"].max()),
        "timeframe": "M5",
        "lookback": int(lookback),
        "max_bars_after_cab": int(max_bars_after_cab),
        "outcome_horizon_candles": int(outcome_horizon),
        "outcome_horizon_minutes": int(outcome_horizon * 5),
        "reward_risk": float(reward_risk),
        "train_rows": int(train_end),
        "validation_rows": int(val_end - train_end),
        "test_rows": int(n - val_end),
        "train_start": str(setup_df.iloc[0]["time"]),
        "train_end": str(setup_df.iloc[train_end - 1]["time"]),
        "validation_start": str(setup_df.iloc[train_end]["time"]),
        "validation_end": str(setup_df.iloc[val_end - 1]["time"]),
        "test_start": str(setup_df.iloc[val_end]["time"]),
        "test_end": str(setup_df.iloc[-1]["time"]),
        "wins": int((setup_df["target"] == 1).sum()),
        "non_wins": int((setup_df["target"] == 0).sum()),
        "label_definition": "A VSA setup is labelled VALID_TRADE if its target is hit before stop-loss within the future M5 window; otherwise AVOID_OR_FAILED.",
        "final_decision_design": "VSA creates candidate setups and features. LR and RF are trained separately on VSA setup outcomes. The selected model filters future VSA setups.",
    }
    return setup_df, meta


def chronological_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def purged_chronological_split(
    df: pd.DataFrame,
    outcome_horizon_candles: int = 36,
    timeframe_minutes: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Create 70/15/15 temporal splits with an embargo at both boundaries.

    Labels inspect future candles. Removing setup rows immediately before the next
    split prevents a training label from using price action belonging to validation,
    and likewise prevents validation labels from reaching into the final test period.
    """
    ordered = df.sort_values("time").reset_index(drop=True).copy()
    ordered["time"] = pd.to_datetime(ordered["time"])
    n = len(ordered)
    train_cut = int(n * 0.70)
    validation_cut = int(n * 0.85)
    validation_start = ordered.iloc[train_cut]["time"]
    test_start = ordered.iloc[validation_cut]["time"]
    embargo = pd.Timedelta(minutes=int(outcome_horizon_candles) * int(timeframe_minutes))

    train = ordered.iloc[:train_cut]
    train = train[train["time"] < validation_start - embargo].copy()
    validation = ordered.iloc[train_cut:validation_cut]
    validation = validation[validation["time"] < test_start - embargo].copy()
    test = ordered.iloc[validation_cut:].copy()
    audit = {
        "split_method": "purged_chronological_70_15_15",
        "embargo_minutes": int(embargo.total_seconds() // 60),
        "train_rows_after_purge": int(len(train)),
        "validation_rows_after_purge": int(len(validation)),
        "test_rows": int(len(test)),
        "purged_train_rows": int(train_cut - len(train)),
        "purged_validation_rows": int((validation_cut - train_cut) - len(validation)),
    }
    return train, validation, test, audit


def load_model_metadata() -> Dict[str, Any]:
    path = MODELS_DIR / "model_metadata.json"
    if path.exists():
        return json.loads(path.read_text())
    return {}


def load_selected_model():
    path = MODELS_DIR / "selected_model.joblib"
    if not path.exists():
        raise FileNotFoundError("models/selected_model.joblib not found. Run backend training first.")
    metadata = load_model_metadata()
    trained_version = str(metadata.get("runtime_versions", {}).get("scikit_learn", ""))
    if trained_version and trained_version != sklearn.__version__:
        raise RuntimeError(
            "Saved-model scikit-learn version mismatch: "
            f"trained with {trained_version}, running with {sklearn.__version__}. "
            "Install the pinned requirements or rerun backend training before inference."
        )
    return joblib.load(path)


def vsa_setups_from_bars(
    bars: pd.DataFrame,
    metadata: Dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Detect formal VSA candidates in already-loaded bars without training or labels."""
    meta = metadata or load_model_metadata()
    featured = add_candle_features(bars, lookback=int(meta.get("lookback", 20)))
    featured = featured.dropna().reset_index(drop=True)
    config = VSAConfig(
        volume_lookback=int(meta.get("cab_volume_lookback", 30)),
        cab_volume_ratio=float(meta.get("cab_minimum_volume_ratio", 1.5)),
        momentum_body_ratio=float(meta.get("momentum_minimum_body_ratio", 0.60)),
        momentum_range_ratio=float(meta.get("momentum_minimum_range_ratio", 1.20)),
        max_bars_after_cab=int(meta.get("max_bars_after_cab", 36)),
        outcome_horizon=int(meta.get("outcome_horizon_candles", 36)),
    )
    return scan_vsa_scenarios(featured, config)


def predict_vsa_setup_row(
    row: pd.Series,
    model=None,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Use the saved supervised model to estimate whether one VSA setup reaches 1R."""
    meta = metadata or load_model_metadata()
    selected_model = str(meta.get("selected_model", ""))
    if selected_model != "LR_VSA_BOT":
        raise ValueError(
            f"Live deployment requires the examiner-aligned LR_VSA_BOT; metadata selects {selected_model or 'nothing'}."
        )
    fitted_model = model if model is not None else load_selected_model()
    X = row[MODEL_FEATURES].to_frame().T.astype(float)
    prob_valid = float(fitted_model.predict_proba(X)[0][list(fitted_model.classes_).index(1)])
    threshold = float(meta.get("recommendation_threshold", 0.55))
    advice = "TRADE_CANDIDATE" if prob_valid >= threshold else "WATCH_ONLY"
    return {
        "signal_time": str(row["time"]),
        "cab_time": str(row["cab_time"]),
        "symbol": "XAUUSD",
        "timeframe": "M5",
        "selected_model": selected_model,
        "direction": str(row["direction"]),
        "scenario_id": int(row["scenario_id"]),
        "ml_probability_valid": prob_valid,
        "recommendation_threshold": threshold,
        "advisor_output": advice,
        "entry_price": float(row["entry_price"]),
        "stop_price": float(row["stop_price"]),
        "target_price": float(row["target_price"]),
        "relative_spread": float(row["relative_spread"]),
        "relative_volume": float(row["relative_volume"]),
        "close_location": float(row["close_location"]),
        "body_ratio": float(row["body_ratio"]),
        "had_opposite_sweep": int(row["had_opposite_sweep"]),
        "breakout_strength": float(row["breakout_strength"]),
        "risk_percent": float(row.get("risk_percent", 0.0)),
        "runner_target_price": float(row.get("runner_target_price", row["target_price"])),
    }


def predict_latest_setup_from_bars(
    bars: pd.DataFrame,
    model=None,
    metadata: Dict[str, Any] | None = None,
    require_recent_closed_bar: bool = False,
    maximum_age_bars: int = 1,
) -> tuple[Dict[str, Any], pd.Series]:
    """Score the newest formal setup and optionally require it to be on a recent closed bar."""
    if bars.empty:
        raise ValueError("No M5 bars were supplied for live inference.")
    meta = metadata or load_model_metadata()
    setups = vsa_setups_from_bars(bars, meta)
    if setups.empty:
        raise ValueError("No formal VSA Scenario 1/2/3 setup exists in the supplied M5 window.")
    row = setups.sort_values("time").iloc[-1]
    if require_recent_closed_bar:
        ordered_times = pd.Series(pd.to_datetime(bars["time"], errors="coerce")).dropna().sort_values().drop_duplicates()
        allowed = ordered_times.tail(max(int(maximum_age_bars) + 1, 1))
        if pd.to_datetime(row["time"]) not in set(allowed.tolist()):
            raise ValueError(
                f"NO_NEW_VSA_SETUP: latest formal setup is {row['time']}; latest closed M5 bar is {ordered_times.iloc[-1]}."
            )
    return predict_vsa_setup_row(row, model=model, metadata=meta), row


def latest_vsa_setup_prediction(
    data_path: str | Path = DEFAULT_DATA_FILE,
    recent_bars: int = 2_000,
) -> Dict[str, Any]:
    """Score the newest VSA setup without rebuilding the full training dataset.

    The model was trained on the complete historical file. Live inference only needs
    enough recent candles to calculate rolling features, find a CAB, and identify a
    subsequent breakout. Limiting inference to a recent window keeps the Streamlit
    advisor responsive while preserving the training workflow.
    """
    meta = load_model_metadata()
    model = load_selected_model()
    recent = read_mt5_bars(data_path, tail_rows=max(int(recent_bars), 250))
    out, _ = predict_latest_setup_from_bars(recent, model=model, metadata=meta)
    return apply_news_risk(out, events=None, verified_live=False)


def save_ml_signal_csv(signal: Dict[str, Any], path: str | Path = OUTPUTS_DIR / "ml_signal.csv") -> Path:
    """Atomically save the fixed 29-column bridge contract expected by the MT5 EA."""
    ensure_dirs()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    defaults: Dict[str, Any] = {column: "" for column in SIGNAL_COLUMNS}
    defaults.update({
        "news_blocked": True,
        "advisor_output": "WATCH_ONLY",
        "ml_advisor_output": signal.get("advisor_output", "WATCH_ONLY"),
        "news_status": "CALENDAR_UNVERIFIED",
    })
    defaults.update(signal)
    temp_path = path.with_name(path.name + ".tmp")
    pd.DataFrame([[defaults[column] for column in SIGNAL_COLUMNS]], columns=SIGNAL_COLUMNS).to_csv(temp_path, index=False)
    temp_path.replace(path)
    return path


def explain_lr_setup(model, row: pd.Series, top_n: int = 8) -> pd.DataFrame:
    """Return faithful local LR contributions after applying the fitted scaler."""
    if not hasattr(model, "named_steps") or "scaler" not in model.named_steps or "model" not in model.named_steps:
        raise ValueError("LR explanation requires the fitted scaler + LogisticRegression pipeline.")
    X = row[MODEL_FEATURES].to_frame().T.astype(float)
    scaled = model.named_steps["scaler"].transform(X)[0]
    coefficients = model.named_steps["model"].coef_[0]
    contributions = scaled * coefficients
    explanation = pd.DataFrame({
        "feature": MODEL_FEATURES,
        "standardised_value": scaled,
        "lr_coefficient": coefficients,
        "log_odds_contribution": contributions,
    })
    explanation["effect"] = np.where(
        explanation["log_odds_contribution"] >= 0,
        "supports VALID_TRADE",
        "supports WATCH_ONLY",
    )
    return explanation.reindex(explanation["log_odds_contribution"].abs().sort_values(ascending=False).index).head(top_n).reset_index(drop=True)


def audit_lr_explanations(model, frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    """Verify that LR feature contributions exactly reconstruct deployed probabilities.

    This is a model-faithfulness check, not a claim that the features cause market
    outcomes.  All fitted StandardScaler transformations, all LR coefficients and
    the fitted intercept are included in the reconstruction.
    """
    if not hasattr(model, "named_steps") or "scaler" not in model.named_steps or "model" not in model.named_steps:
        raise ValueError("LR faithfulness audit requires the fitted scaler + LogisticRegression pipeline.")
    if frame.empty:
        raise ValueError("LR faithfulness audit requires at least one row.")

    X = frame[MODEL_FEATURES].astype(float)
    scaled = model.named_steps["scaler"].transform(X)
    lr_model = model.named_steps["model"]
    coefficients = lr_model.coef_[0]
    intercept = float(lr_model.intercept_[0])
    contributions = scaled * coefficients
    reconstructed_logit = intercept + contributions.sum(axis=1)
    reconstructed_probability = 1.0 / (1.0 + np.exp(-np.clip(reconstructed_logit, -709, 709)))
    positive_index = list(lr_model.classes_).index(1)
    model_probability = model.predict_proba(X)[:, positive_index]
    absolute_error = np.abs(model_probability - reconstructed_probability)

    positive_index_per_row = np.argmax(contributions, axis=1)
    negative_index_per_row = np.argmin(contributions, axis=1)
    audit = pd.DataFrame({
        "row": np.arange(len(frame), dtype=int),
        "signal_time": frame["time"].astype(str).to_numpy() if "time" in frame.columns else "",
        "model_probability": model_probability,
        "reconstructed_probability": reconstructed_probability,
        "absolute_error": absolute_error,
        "top_supporting_feature": [MODEL_FEATURES[i] for i in positive_index_per_row],
        "top_supporting_contribution": contributions[np.arange(len(frame)), positive_index_per_row],
        "top_opposing_feature": [MODEL_FEATURES[i] for i in negative_index_per_row],
        "top_opposing_contribution": contributions[np.arange(len(frame)), negative_index_per_row],
    })
    tolerance = 1e-12
    summary = {
        "method": "Intrinsic Logistic Regression contribution reconstruction",
        "rows_audited": int(len(frame)),
        "features_per_explanation": int(len(MODEL_FEATURES)),
        "intercept_included": True,
        "mean_absolute_probability_error": float(absolute_error.mean()),
        "maximum_absolute_probability_error": float(absolute_error.max()),
        "faithful_within_tolerance": bool(np.all(absolute_error <= tolerance)),
        "tolerance": tolerance,
        "interpretation_boundary": "Faithful to the fitted LR calculation; not evidence of causal market effects.",
    }
    return summary, audit


def lr_local_explanation(row: pd.Series, coefficients: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    if coefficients.empty:
        return pd.DataFrame()
    coef = coefficients.set_index("feature")["coefficient_valid_trade"]
    vals = row[coef.index].astype(float)
    contrib = (vals * coef).sort_values(key=lambda s: s.abs(), ascending=False).head(top_n)
    return pd.DataFrame({"feature": contrib.index, "local_contribution": contrib.values})
