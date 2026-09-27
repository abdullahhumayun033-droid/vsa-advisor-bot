"""Economic-news safety gate for VSA Advisor Bot recommendations.

This module applies the project's high-impact USD news-risk rules after a VSA
setup has been scored by the machine-learning model. It checks that the
economic calendar is verified and actually covers the signal timestamp, then
searches the configured time window around the signal for high-impact USD
events.

The logic is deliberately fail-safe. If the live calendar is unavailable,
unverified, outside its coverage period, or the signal time cannot be validated,
the final advisor output is downgraded to WATCH_ONLY. A verified high-impact
USD event within the configured window also blocks a new demo trade. When the
calendar is verified, covers the signal time and contains no relevant event in
the risk window, the underlying ML recommendation is left unchanged.

This module does not alter the VSA scenario rules or retrain/rescore the machine-
learning model; it acts as a separate safety layer over the recommendation
produced by the VSA and ML pipeline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def _calendar_covers_signal(calendar: pd.DataFrame, signal_time: pd.Timestamp) -> tuple[bool, str, str]:
    """Check whether a verified calendar actually spans the signal timestamp."""
    attr_start = calendar.attrs.get("coverage_start")
    attr_end = calendar.attrs.get("coverage_end")
    start = pd.to_datetime(attr_start, errors="coerce") if attr_start else pd.NaT
    end = pd.to_datetime(attr_end, errors="coerce") if attr_end else pd.NaT

    # Hand-built/test frames do not carry feed metadata. Infer a conservative coverage
    # envelope from event timestamps so a calendar from an unrelated date cannot clear risk.
    if pd.isna(start) or pd.isna(end):
        event_times = pd.to_datetime(calendar.get("event_time"), errors="coerce").dropna()
        if event_times.empty:
            return False, "", ""
        padding = pd.Timedelta(hours=48)
        start = event_times.min() - padding
        end = event_times.max() + padding

    return bool(start <= signal_time < end), str(start), str(end)


def apply_news_risk(
    signal: dict[str, Any],
    events: pd.DataFrame | None,
    verified_live: bool,
    minutes_before: int = 30,
    minutes_after: int = 30,
) -> dict[str, Any]:
    """Apply a fail-safe high-impact USD news gate to an ML/VSA signal."""
    out = dict(signal)
    out["ml_advisor_output"] = out.get("ml_advisor_output", out.get("advisor_output", "WATCH_ONLY"))
    out["advisor_output"] = out["ml_advisor_output"]
    out["news_window_before_minutes"] = int(minutes_before)
    out["news_window_after_minutes"] = int(minutes_after)
    out["news_event"] = ""
    out["news_event_time"] = ""
    out["news_calendar_coverage_start"] = ""
    out["news_calendar_coverage_end"] = ""

    if not verified_live or events is None or events.empty:
        out["news_status"] = "CALENDAR_UNVERIFIED"
        out["news_blocked"] = True
        out["news_reason"] = "A verified live USD calendar was unavailable; fail-safe blocks new demo trades."
        out["advisor_output"] = "WATCH_ONLY"
        return out

    signal_time = pd.to_datetime(out.get("signal_time"), errors="coerce")
    if pd.isna(signal_time):
        out["news_status"] = "INVALID_SIGNAL_TIME"
        out["news_blocked"] = True
        out["news_reason"] = "Signal time could not be compared with the economic calendar."
        out["advisor_output"] = "WATCH_ONLY"
        return out

    calendar = events.copy()
    # DataFrame.attrs is not guaranteed to survive every pandas operation, so keep it explicitly.
    calendar.attrs = dict(events.attrs)
    calendar["event_time"] = pd.to_datetime(calendar["event_time"], errors="coerce")
    calendar = calendar.dropna(subset=["event_time"])

    covered, coverage_start, coverage_end = _calendar_covers_signal(calendar, signal_time)
    out["news_calendar_coverage_start"] = coverage_start
    out["news_calendar_coverage_end"] = coverage_end
    if not covered:
        out["news_status"] = "CALENDAR_OUT_OF_COVERAGE"
        out["news_blocked"] = True
        out["news_reason"] = "The verified calendar does not cover the signal time; fail-safe blocks new demo trades."
        out["advisor_output"] = "WATCH_ONLY"
        return out

    high_usd = calendar[
        calendar["currency"].astype(str).str.contains("USD|US", case=False, na=False)
        & calendar["impact"].astype(str).str.contains("High", case=False, na=False)
    ].copy()

    window_start = signal_time - pd.Timedelta(minutes=minutes_before)
    window_end = signal_time + pd.Timedelta(minutes=minutes_after)
    nearby = high_usd[high_usd["event_time"].between(window_start, window_end)].copy()
    if nearby.empty:
        out["news_status"] = "CLEAR"
        out["news_blocked"] = False
        out["news_reason"] = "No verified high-impact USD event falls inside the configured risk window."
        return out

    event = nearby.iloc[(nearby["event_time"] - signal_time).abs().argmin()]
    out["news_status"] = "HIGH_IMPACT_USD_BLOCK"
    out["news_blocked"] = True
    out["news_event"] = str(event.get("event", "High-impact USD event"))
    out["news_event_time"] = str(event["event_time"])
    out["news_reason"] = "A high-impact USD event is within the configured risk window."
    out["advisor_output"] = "WATCH_ONLY"
    return out
