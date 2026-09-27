"""Canonical Volume Spread Analysis strategy for the VSA Advisor Bot.

This module defines the rule-based VSA setup logic used to identify Climactic
Action Bars (CABs) and track the project's mirrored BUY and SELL Scenario 1,
Scenario 2 and Scenario 3 state transitions. It also applies the
scenario-specific stop placement and risk geometry, calculates the 1R partial
target and 4R runner target, and labels historical setups using the managed
70%/30% exit approach.

The resulting setup records include the VSA, market, risk and outcome fields
needed by the wider project, allowing the historical training/evaluation
pipeline and the advisor workflow to use the same underlying strategy
definition rather than maintaining separate versions of the trading rules.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class VSAConfig:
    volume_lookback: int = 30
    cab_volume_ratio: float = 1.5
    trend_ema_period: int = 50
    momentum_body_ratio: float = 0.60
    momentum_range_ratio: float = 1.20
    max_bars_after_cab: int = 36
    outcome_horizon: int = 36
    first_target_r: float = 1.0
    runner_target_r: float = 4.0
    first_close_fraction: float = 0.70
    runner_fraction: float = 0.30
    min_risk_points: float = 0.50


def _first_touch(highs, lows, direction: str, target: float, stop: float) -> str:
    for high, low in zip(highs, lows):
        if direction == "BUY":
            if low <= stop:
                return "LOSS"
            if high >= target:
                return "WIN"
        else:
            if high >= stop:
                return "LOSS"
            if low <= target:
                return "WIN"
    return "EXPIRED"


def managed_exit_outcome(
    highs, lows, direction: str, entry: float, stop: float,
    first_target: float, runner_target: float,
) -> tuple[str, float]:
    """Evaluate 70% at 1R, then 30% at 4R with stop moved to breakeven."""
    first_target_hit = False
    for high, low in zip(highs, lows):
        if direction == "BUY":
            if not first_target_hit:
                if low <= stop:
                    return "FULL_LOSS", -1.0
                if high >= first_target:
                    first_target_hit = True
                    if high >= runner_target:
                        return "FULL_TARGET", 1.9
            else:
                if low <= entry:
                    return "PARTIAL_WIN_BREAKEVEN", 0.7
                if high >= runner_target:
                    return "FULL_TARGET", 1.9
        else:
            if not first_target_hit:
                if high >= stop:
                    return "FULL_LOSS", -1.0
                if low <= first_target:
                    first_target_hit = True
                    if low <= runner_target:
                        return "FULL_TARGET", 1.9
            else:
                if high >= entry:
                    return "PARTIAL_WIN_BREAKEVEN", 0.7
                if low <= runner_target:
                    return "FULL_TARGET", 1.9
    if first_target_hit:
        return "PARTIAL_WIN_OPEN_AT_HORIZON", 0.7
    return "EXPIRED_BEFORE_TARGET", 0.0


def scenario_stop(active: dict, cab_high: float, cab_low: float, direction: str, scenario: int) -> float:
    if scenario == 1:
        return float(cab_low if direction == "BUY" else cab_high)
    if scenario == 2:
        return float(active["sweep_stop"])
    return float(active["post_break_extreme"])


def prepare_strategy_features(df: pd.DataFrame, config: VSAConfig) -> pd.DataFrame:
    d = df.copy()
    d["strategy_avg_volume"] = d["tick_volume"].rolling(config.volume_lookback).mean().shift(1)
    d["strategy_volume_ratio"] = d["tick_volume"] / d["strategy_avg_volume"].replace(0, np.nan)
    d["trend_ema"] = d["close"].ewm(span=config.trend_ema_period, adjust=False).mean()
    d["trend_ema_slope"] = d["trend_ema"].diff(5)
    d["downtrend"] = (d["close"] < d["trend_ema"]) & (d["trend_ema_slope"] < 0)
    d["uptrend"] = (d["close"] > d["trend_ema"]) & (d["trend_ema_slope"] > 0)
    d["bull_momentum"] = (
        (d["close"] > d["open"])
        & (d["body_ratio"] >= config.momentum_body_ratio)
        & (d["relative_spread"] >= config.momentum_range_ratio)
    )
    d["bear_momentum"] = (
        (d["close"] < d["open"])
        & (d["body_ratio"] >= config.momentum_body_ratio)
        & (d["relative_spread"] >= config.momentum_range_ratio)
    )
    d["buy_cab"] = (
        (d["close"] < d["open"])
        & d["downtrend"]
        & (d["strategy_volume_ratio"] >= config.cab_volume_ratio)
    )
    d["sell_cab"] = (
        (d["close"] > d["open"])
        & d["uptrend"]
        & (d["strategy_volume_ratio"] >= config.cab_volume_ratio)
    )
    return d


def scan_vsa_scenarios(featured: pd.DataFrame, config: VSAConfig) -> pd.DataFrame:
    """Detect the user's three VSA scenarios using mirrored BUY/SELL logic."""
    d = prepare_strategy_features(featured, config).dropna().reset_index(drop=True)
    setups: list[dict] = []
    active: dict | None = None

    for i, row in d.iterrows():
        if active is None:
            if bool(row["buy_cab"]) or bool(row["sell_cab"]):
                direction = "BUY" if bool(row["buy_cab"]) else "SELL"
                active = {
                    "idx": i, "row": row, "direction": direction,
                    "first_break": False, "reentered": False,
                    "swept": False, "broken": False,
                    "sweep_stop": None, "post_break_extreme": None,
                    "extreme_low": float(row["low"]), "extreme_high": float(row["high"]),
                }
            continue

        bars_after = i - active["idx"]
        if bars_after > config.max_bars_after_cab:
            active = None
            continue

        cab = active["row"]
        cab_high, cab_low = float(cab["high"]), float(cab["low"])
        active["extreme_low"] = min(active["extreme_low"], float(row["low"]))
        active["extreme_high"] = max(active["extreme_high"], float(row["high"]))
        lower_volume = float(row["tick_volume"]) < float(cab["tick_volume"])

        if active["direction"] == "BUY":
            if active["broken"]:
                active["post_break_extreme"] = min(active["post_break_extreme"], float(row["low"]))
            elif float(row["close"]) < cab_low:
                active["broken"] = True
                if active["post_break_extreme"] is None:
                    active["post_break_extreme"] = float(row["low"])
                else:
                    active["post_break_extreme"] = min(active["post_break_extreme"], float(row["low"]))
            elif float(row["low"]) < cab_low and float(row["close"]) >= cab_low:
                active["swept"] = True
                active["sweep_stop"] = float(row["low"])
            valid_breakout = float(row["close"]) > cab_high and bool(row["bull_momentum"]) and lower_volume
            inside = float(row["close"]) <= cab_high and float(row["close"]) >= cab_low
        else:
            if active["broken"]:
                active["post_break_extreme"] = max(active["post_break_extreme"], float(row["high"]))
            elif float(row["close"]) > cab_high:
                active["broken"] = True
                if active["post_break_extreme"] is None:
                    active["post_break_extreme"] = float(row["high"])
                else:
                    active["post_break_extreme"] = max(active["post_break_extreme"], float(row["high"]))
            elif float(row["high"]) > cab_high and float(row["close"]) <= cab_high:
                active["swept"] = True
                active["sweep_stop"] = float(row["high"])
            valid_breakout = float(row["close"]) < cab_low and bool(row["bear_momentum"]) and lower_volume
            inside = float(row["close"]) >= cab_low and float(row["close"]) <= cab_high

        scenario = None
        if valid_breakout and active["broken"]:
            scenario = 3
        elif valid_breakout and active["swept"]:
            scenario = 2
        elif valid_breakout and active["first_break"] and active["reentered"]:
            scenario = 1
        elif valid_breakout:
            active["first_break"] = True
            continue
        elif active["first_break"] and inside and not active["swept"] and not active["broken"]:
            active["reentered"] = True

        if scenario is None:
            continue

        direction = active["direction"]
        entry = float(row["close"])
        stop = scenario_stop(active, cab_high, cab_low, direction, scenario)
        if direction == "BUY":
            risk = max(entry - stop, config.min_risk_points)
            first_target = entry + config.first_target_r * risk
            runner_target = entry + config.runner_target_r * risk
        else:
            risk = max(stop - entry, config.min_risk_points)
            first_target = entry - config.first_target_r * risk
            runner_target = entry - config.runner_target_r * risk
        future = d.iloc[i + 1:i + 1 + config.outcome_horizon]
        outcome, result_r = managed_exit_outcome(
            future["high"], future["low"], direction, entry, stop, first_target, runner_target
        )

        setups.append({
            "time": row["time"], "cab_time": cab["time"], "direction": direction,
            "direction_num": 1 if direction == "BUY" else -1,
            "scenario_id": scenario, "risk_percent": 0.5 if scenario == 1 else 1.0,
            "entry_price": entry, "stop_price": stop, "target_price": first_target,
            "runner_target_price": runner_target, "result_r": result_r,
            "broker_spread_points": float(row.get("broker_spread_points", 0.0)),
            "spread_cost_r": float(row.get("broker_spread_points", 0.0) * 0.01 / risk),
            "outcome": outcome, "target": int(result_r > 0),
            "relative_spread": float(row["relative_spread"]),
            "relative_volume": float(row["relative_volume"]),
            "close_location": float(row["close_location"]), "body_ratio": float(row["body_ratio"]),
            "upper_wick_ratio": float(row["upper_wick_ratio"]), "lower_wick_ratio": float(row["lower_wick_ratio"]),
            "cab_range": float(cab_high - cab_low), "cab_relative_volume": float(cab["relative_volume"]),
            "entry_to_stop_distance": float(risk), "risk_reward_distance": float(abs(first_target - entry)),
            "bars_after_cab": int(bars_after), "had_opposite_sweep": int(active["swept"]),
            "breakout_strength": float(abs(entry - (cab_high if direction == "BUY" else cab_low)) / max(cab_high - cab_low, config.min_risk_points)),
            "return_3": float(row["return_3"]), "return_6": float(row["return_6"]),
            "return_12": float(row["return_12"]), "volatility_12": float(row["volatility_12"]),
            "volatility_48": float(row["volatility_48"]), "hour": int(row["hour"]),
            "day_of_week": int(row["day_of_week"]), "london_session": int(row["london_session"]),
            "newyork_session": int(row["newyork_session"]),
        })
        active = None

    return pd.DataFrame(setups)
