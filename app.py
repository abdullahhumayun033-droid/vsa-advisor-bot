"""Main Streamlit interface for the VSA Advisor Bot.

This file provides the user-facing application and technical evidence interface
for the project. It brings together the shared VSA, machine-learning, news-risk,
journal and MT5 integration modules and presents their outputs through a
beginner-focused Streamlit website.

The application does not define a second copy of the core VSA trading strategy
or train the deployed machine-learning model when the website opens. Backend
training is performed separately, while the interface loads saved model
metadata and inference artifacts, displays historical evaluation evidence, and
calls the shared live pipeline when the user requests a live completed-bar
update. Historical demonstration mode is kept separate from live MT5 mode and
does not overwrite the MT5 bridge signal.

The main user experience includes the Live Advisor, verified-news awareness,
risk and safety guidance, model explanations, learning material, glossary,
trade journal, feedback form and MT5 connection guidance. Technical audit pages
also expose backend training evidence, LR/RF comparison, XAI faithfulness,
testing, dataset information, robustness results and end-to-end MT5 proof for
assessment and debugging.

Visual charts and interface formatting in this file are presentation layers over
the existing completed-candle data and saved strategy/model outputs. They are
not separate implementations of the VSA state machine. Live recommendations
continue to depend on the shared pipeline's feed-freshness checks, saved-model
inference and fail-closed news/safety controls before a final signal is exposed
to the user or MT5 bridge.
"""

from __future__ import annotations

import html
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception:
    px = None
    go = None
    make_subplots = None

from modules.ml_pipeline import (
    DEFAULT_DATA_FILE,
    MODEL_FEATURES,
    MODELS_DIR,
    RESULTS_DIR,
    OUTPUTS_DIR,
    build_vsa_setup_dataset,
    latest_vsa_setup_prediction,
    load_model_metadata,
    read_mt5_bars,
    save_ml_signal_csv,
)
from modules.news_risk import apply_news_risk
from modules.news_calendar import fetch_faireconomy_calendar as fetch_verified_calendar
from modules.integration_evidence import make_signal_id
from modules.live_pipeline import DEFAULT_MT5_COMMON, PROJECT_ROOT, discover_mt5_common, read_broker_offset_from_status, run_live_cycle
from modules.journal import DEFAULT_JOURNAL_PATH, add_manual_journal_entry, read_journal, upsert_journal_entry, update_journal_reflection

st.set_page_config(page_title="VSA Advisor Bot", page_icon="🪙", layout="wide")

st.markdown(
    """
    <style>
    /* ================================================================
       VISUAL THEME ONLY — Obsidian + Champagne Gold
       No strategy, model, signal, risk, journal or MT5 logic is changed.
       ================================================================ */
    :root {
        --vsa-gold: #D4AF37;
        --vsa-gold-light: #F4D77B;
        --vsa-gold-dark: #9A7420;
        --vsa-ink: #11151C;
        --vsa-ink-2: #1A202B;
        --vsa-ivory: #FBF8F0;
        --vsa-card: rgba(255,255,255,0.94);
        --vsa-border: rgba(154,116,32,0.24);
        --vsa-shadow: 0 12px 30px rgba(18,20,24,0.08);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 82% 8%, rgba(212,175,55,0.10), transparent 28rem),
            linear-gradient(180deg, #FFFEFB 0%, #F7F3E9 100%);
        color: var(--vsa-ink);
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0.82);
        backdrop-filter: blur(12px);
        border-bottom: 1px solid rgba(212,175,55,0.16);
    }

    .block-container {
        padding-top: 1.45rem;
        padding-bottom: 3rem;
        max-width: 1500px;
    }

    /* ---------- Sidebar / navigation ---------- */
    section[data-testid="stSidebar"] {
        background:
            radial-gradient(circle at 20% 0%, rgba(212,175,55,0.15), transparent 18rem),
            linear-gradient(180deg, #0E1117 0%, #151A23 58%, #10131A 100%);
        border-right: 1px solid rgba(212,175,55,0.28);
        box-shadow: 10px 0 34px rgba(10,12,16,0.10);
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.2rem;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #FFF8DF !important;
        letter-spacing: -0.02em;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span {
        color: #E8E1D2;
    }

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
        color: #AFA99E !important;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.28rem;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label {
        margin: 0.10rem 0;
        padding: 0.52rem 0.70rem;
        min-height: 2.35rem;
        border: 1px solid rgba(212,175,55,0.14);
        border-radius: 12px;
        background: rgba(255,255,255,0.035);
        cursor: pointer;
        transition: transform 160ms ease, border-color 160ms ease, background 160ms ease, box-shadow 160ms ease;
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {
        transform: translateX(5px);
        border-color: rgba(244,215,123,0.72);
        background: linear-gradient(90deg, rgba(212,175,55,0.16), rgba(255,255,255,0.06));
        box-shadow: 0 7px 18px rgba(0,0,0,0.18);
    }

    section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {
        border-color: rgba(244,215,123,0.88);
        background: linear-gradient(90deg, rgba(212,175,55,0.24), rgba(212,175,55,0.08));
        box-shadow: inset 3px 0 0 #D4AF37, 0 7px 22px rgba(0,0,0,0.15);
    }

    section[data-testid="stSidebar"] input[type="radio"],
    section[data-testid="stSidebar"] input[type="checkbox"] {
        accent-color: var(--vsa-gold);
    }

    section[data-testid="stSidebar"] [data-testid="stAlert"] {
        background: linear-gradient(135deg, rgba(212,175,55,0.16), rgba(244,215,123,0.08));
        border: 1px solid rgba(244,215,123,0.42);
        color: #FFF8DF;
    }

    /* ---------- Headings ---------- */
    h1 {
        color: #16191F;
        letter-spacing: -0.035em;
        font-weight: 800 !important;
    }

    h1::after {
        content: "";
        display: block;
        width: 74px;
        height: 4px;
        margin-top: 0.42rem;
        border-radius: 99px;
        background: linear-gradient(90deg, #9A7420, #F4D77B);
    }

    h2, h3 {
        color: #1B1F27;
        letter-spacing: -0.018em;
    }

    /* ---------- Metric / information cards ---------- */
    div[data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        border: 1px solid var(--vsa-border);
        border-radius: 16px;
        padding: 0.82rem 1rem;
        background: linear-gradient(145deg, rgba(255,255,255,0.98), rgba(251,248,240,0.94));
        box-shadow: var(--vsa-shadow);
        transition: transform 170ms ease, box-shadow 170ms ease, border-color 170ms ease;
    }

    div[data-testid="stMetric"]::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: linear-gradient(180deg, #F4D77B, #9A7420);
    }

    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(212,175,55,0.48);
        box-shadow: 0 16px 35px rgba(18,20,24,0.12);
    }

    div[data-testid="stMetricLabel"] {
        color: #6B6355;
        font-weight: 650;
    }

    div[data-testid="stMetricValue"] {
        color: #171A21;
        font-weight: 780;
        font-size: clamp(1.05rem, 1.55vw, 1.75rem) !important;
        line-height: 1.12 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere;
    }

    div[data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere;
    }

    /* ---------- Responsive live-status cards ---------- */
    .live-status-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(165px, 1fr));
        gap: 0.78rem;
        width: 100%;
        margin: 0.35rem 0 0.85rem 0;
    }

    .live-status-card {
        position: relative;
        min-width: 0;
        min-height: 92px;
        padding: 0.80rem 0.95rem 0.76rem 1rem;
        border: 1px solid var(--vsa-border);
        border-radius: 16px;
        background: linear-gradient(145deg, rgba(255,255,255,0.98), rgba(251,248,240,0.94));
        box-shadow: var(--vsa-shadow);
        overflow: hidden;
    }

    .live-status-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 4px;
        background: linear-gradient(180deg, #F4D77B, #9A7420);
    }

    .live-status-label {
        color: #6B6355;
        font-size: 0.76rem;
        font-weight: 700;
        margin-bottom: 0.28rem;
    }

    .live-status-value {
        color: #171A21;
        font-size: clamp(1.08rem, 1.55vw, 1.75rem);
        line-height: 1.08;
        font-weight: 820;
        letter-spacing: -0.02em;
        white-space: normal;
        overflow-wrap: anywhere;
        word-break: normal;
    }

    .live-status-sub {
        color: #7A746A;
        font-size: 0.69rem;
        line-height: 1.25;
        margin-top: 0.36rem;
    }

    @media (max-width: 760px) {
        .live-status-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .live-status-card { min-height: 84px; }
    }

    @media (max-width: 480px) {
        .live-status-grid { grid-template-columns: 1fr; }
    }

    /* ---------- Buttons ---------- */
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {
        border: 1px solid #A47C22 !important;
        border-radius: 12px !important;
        color: #17130A !important;
        font-weight: 750 !important;
        background: linear-gradient(135deg, #F7DE8A 0%, #D4AF37 52%, #BE922B 100%) !important;
        box-shadow: 0 7px 18px rgba(154,116,32,0.20);
        transition: transform 150ms ease, box-shadow 150ms ease, filter 150ms ease !important;
    }

    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {
        transform: translateY(-2px);
        filter: brightness(1.06);
        box-shadow: 0 11px 24px rgba(154,116,32,0.30);
        border-color: #7E5D17 !important;
    }

    div[data-testid="stButton"] > button:active,
    div[data-testid="stDownloadButton"] > button:active {
        transform: translateY(0);
        box-shadow: 0 4px 10px rgba(154,116,32,0.22);
    }

    /* ---------- Tabs ---------- */
    button[data-baseweb="tab"] {
        border-radius: 11px 11px 0 0;
        padding-left: 1.1rem !important;
        padding-right: 1.1rem !important;
        transition: background 150ms ease, color 150ms ease;
    }

    button[data-baseweb="tab"]:hover {
        background: rgba(212,175,55,0.10);
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #8A661A !important;
        font-weight: 750;
        background: rgba(212,175,55,0.12);
    }

    /* ---------- Inputs / expanders ---------- */
    [data-baseweb="input"] > div,
    [data-baseweb="select"] > div,
    textarea {
        border-radius: 11px !important;
    }

    [data-baseweb="input"] > div:focus-within,
    [data-baseweb="select"] > div:focus-within,
    textarea:focus {
        border-color: #C79A2C !important;
        box-shadow: 0 0 0 1px rgba(212,175,55,0.28) !important;
    }

    details[data-testid="stExpander"] {
        border: 1px solid rgba(154,116,32,0.22) !important;
        border-radius: 13px !important;
        background: rgba(255,255,255,0.72);
        box-shadow: 0 7px 18px rgba(18,20,24,0.04);
    }

    /* ---------- Alerts / status strips ---------- */
    div[data-testid="stAlert"] {
        border-radius: 13px;
        border-left-width: 4px;
        box-shadow: 0 5px 16px rgba(18,20,24,0.05);
    }

    /* ---------- Data tables ---------- */
    [data-testid="stDataFrame"] {
        border: 1px solid rgba(154,116,32,0.18);
        border-radius: 13px;
        overflow: hidden;
        box-shadow: 0 8px 20px rgba(18,20,24,0.05);
    }

    /* ---------- Plotly container polish ---------- */
    [data-testid="stPlotlyChart"] {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 14px 34px rgba(11,14,19,0.14);
        border: 1px solid rgba(212,175,55,0.22);
    }

    /* ---------- Forex Factory-style red-folder calendar ---------- */
    .ff-source-strip {
        display:flex; align-items:center; justify-content:space-between; gap:1rem;
        padding:0.9rem 1rem; margin:0.35rem 0 1rem 0; border-radius:14px;
        border:1px solid rgba(154,116,32,0.22);
        background:linear-gradient(135deg, rgba(17,21,28,0.97), rgba(31,37,48,0.96));
        color:#F8F0D7; box-shadow:0 10px 26px rgba(12,14,18,0.12);
    }
    .ff-source-strip strong { color:#F4D77B; }
    .ff-week-label {
        margin:0.4rem 0 0.75rem 0; color:#6D5420; font-weight:800; letter-spacing:0.01em;
    }
    .ff-day-card {
        min-height:172px; height:100%; padding:0.85rem 0.85rem 0.75rem; border-radius:15px;
        background:linear-gradient(155deg, rgba(255,255,255,0.99), rgba(249,245,235,0.96));
        border:1px solid rgba(154,116,32,0.22); box-shadow:0 8px 20px rgba(18,20,24,0.06);
        transition:transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
    }
    .ff-day-card:hover {
        transform:translateY(-2px); border-color:rgba(212,175,55,0.52);
        box-shadow:0 13px 27px rgba(18,20,24,0.10);
    }
    .ff-day-card.today {
        border:2px solid #D4AF37; background:linear-gradient(155deg, #FFFDF6, #FFF3CB);
    }
    .ff-day-card.past { opacity:0.62; }
    .ff-day-head { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:0.55rem; }
    .ff-day-name { font-size:0.77rem; text-transform:uppercase; letter-spacing:0.08em; color:#7E6C49; font-weight:800; }
    .ff-day-date { font-size:1.25rem; color:#151922; font-weight:850; line-height:1.1; }
    .ff-today-pill {
        display:inline-block; padding:0.12rem 0.42rem; border-radius:999px; background:#D4AF37; color:#17130A;
        font-size:0.65rem; font-weight:850; letter-spacing:0.04em;
    }
    .ff-event {
        margin-top:0.46rem; padding:0.52rem 0.55rem; border-radius:10px;
        border-left:4px solid #C62828; background:#FFF0F0; color:#2C1717;
    }
    .ff-event-time { font-size:0.72rem; font-weight:850; color:#A4161A; }
    .ff-event-title { font-size:0.82rem; font-weight:780; margin-top:0.08rem; line-height:1.25; }
    .ff-event-meta { font-size:0.68rem; color:#6E5D5D; margin-top:0.14rem; }
    .ff-none {
        margin-top:0.7rem; padding:0.6rem; border-radius:10px; background:#F4F5F7; color:#777;
        font-size:0.75rem; text-align:center;
    }
    .red-folder-badge {
        display:inline-flex; align-items:center; gap:0.35rem; padding:0.22rem 0.55rem; border-radius:999px;
        background:#B71C1C; color:white; font-size:0.72rem; font-weight:850; box-shadow:0 4px 10px rgba(183,28,28,0.18);
    }
    .ff-red-folder {
        display:inline-block; width:18px; height:13px; border-radius:2px; background:#D9272E;
        position:relative; margin-right:0.36rem; top:2px; box-shadow:inset 0 -2px 0 rgba(0,0,0,0.12);
    }
    .ff-red-folder:before {
        content:""; position:absolute; left:1px; top:-4px; width:8px; height:5px; border-radius:2px 2px 0 0;
        background:#EF4D52;
    }

    /* ---------- Scrollbar ---------- */
    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.04); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #CBA43B, #8F6B1B);
        border-radius: 10px;
        border: 2px solid transparent;
        background-clip: padding-box;
    }

    @media (max-width: 900px) {
        .block-container {padding-top: 1rem;}
        section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover { transform: none; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=60, show_spinner=False)
def _read_csv_cached(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def read_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return _read_csv_cached(str(p), p.stat().st_mtime)


@st.cache_data(ttl=60, show_spinner=False)
def _read_json_cached(path_str: str, mtime: float) -> dict[str, Any]:
    return json.loads(Path(path_str).read_text())


def read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return _read_json_cached(str(p), p.stat().st_mtime)


def metric_fmt(x, digits=3):
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def style_high_impact(row):
    impact = str(row.get("impact", "")).lower()
    if "high" in impact:
        return ["background-color: #ffd6d6; color: #8b0000; font-weight: 700"] * len(row)
    return [""] * len(row)


@st.cache_data(ttl=300, show_spinner=False)
def fetch_forex_factory_calendar(broker_utc_offset: float = 0.0) -> tuple[pd.DataFrame, bool, str]:
    calendar, verified, source = fetch_verified_calendar(
        broker_utc_offset_hours=broker_utc_offset,
        fallback_path=Path("data/high_impact_news_fallback.csv"),
    )
    return calendar, bool(verified), source



def _normalise_news_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise the calendar for display without changing the trading news-gate logic."""
    if df is None or df.empty:
        return pd.DataFrame(columns=[
            "event_time", "event_date", "time_label", "currency", "event", "impact",
            "actual", "forecast", "previous", "source",
        ])
    out = df.copy()
    for col in ["event_time", "event_date", "time_label", "currency", "event", "impact", "actual", "forecast", "previous", "source"]:
        if col not in out.columns:
            out[col] = ""
    out["event_time"] = pd.to_datetime(out["event_time"], errors="coerce")
    out["event_date"] = pd.to_datetime(out["event_date"], errors="coerce")
    missing_date = out["event_date"].isna() & out["event_time"].notna()
    out.loc[missing_date, "event_date"] = out.loc[missing_date, "event_time"].dt.normalize()
    out["event_date"] = out["event_date"].dt.normalize()
    missing_label = out["time_label"].astype(str).str.strip().isin({"", "nan", "NaT"}) & out["event_time"].notna()
    out.loc[missing_label, "time_label"] = out.loc[missing_label, "event_time"].dt.strftime("%H:%M")
    return out.sort_values(["event_date", "event_time", "time_label"], na_position="last").reset_index(drop=True)


def _high_impact_usd_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Return exactly the USD High-impact events (Forex Factory 'red-folder' equivalent)."""
    cal = _normalise_news_calendar(df)
    if cal.empty:
        return cal
    currency = cal["currency"].astype(str).str.strip().str.upper()
    impact = cal["impact"].astype(str).str.strip().str.lower()
    return cal[(currency == "USD") & impact.str.contains("high", na=False)].copy()


def _broker_now(offset_hours: float) -> pd.Timestamp:
    fixed = timezone(timedelta(hours=float(offset_hours)))
    return pd.Timestamp(datetime.now(timezone.utc).astimezone(fixed).replace(tzinfo=None))


def _clean_event_value(value: Any) -> str:
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "nat"} else text


def _news_event_card(row: pd.Series) -> str:
    time_label = html.escape(_clean_event_value(row.get("time_label")) or "Time TBA")
    title = html.escape(_clean_event_value(row.get("event")) or "High-impact USD event")
    meta = []
    for label, key in (("Actual", "actual"), ("Forecast", "forecast"), ("Previous", "previous")):
        value = _clean_event_value(row.get(key))
        if value:
            meta.append(f"{label}: {html.escape(value)}")
    meta_html = " • ".join(meta) if meta else "Forecast details not yet published"
    return (
        '<div class="ff-event">'
        f'<div class="ff-event-time"><span class="ff-red-folder"></span>{time_label} · USD · HIGH IMPACT</div>'
        f'<div class="ff-event-title">{title}</div>'
        f'<div class="ff-event-meta">{meta_html}</div>'
        '</div>'
    )


def _render_news_week_calendar(events: pd.DataFrame, week_start: pd.Timestamp, today: pd.Timestamp, title: str) -> None:
    week_start = pd.Timestamp(week_start).normalize()
    week_end = week_start + pd.Timedelta(days=6)
    st.markdown(
        f'<div class="ff-week-label">{html.escape(title)} · {week_start.strftime("%d %b")} – {week_end.strftime("%d %b %Y")}</div>',
        unsafe_allow_html=True,
    )
    days = [week_start + pd.Timedelta(days=i) for i in range(7)]
    for start in (0, 4):
        subset = days[start:start + 4]
        cols = st.columns(len(subset))
        for col, day in zip(cols, subset):
            day_events = events[events["event_date"] == day]
            state_class = "today" if day == today else ("past" if day < today else "")
            today_badge = '<span class="ff-today-pill">TODAY</span>' if day == today else ""
            if day_events.empty:
                body = '<div class="ff-none">No high-impact USD events</div>'
            else:
                body = "".join(_news_event_card(row) for _, row in day_events.iterrows())
            card = (
                f'<div class="ff-day-card {state_class}">'
                '<div class="ff-day-head">'
                f'<div><div class="ff-day-name">{day.strftime("%A")}</div>'
                f'<div class="ff-day-date">{day.strftime("%d %b")}</div></div>{today_badge}'
                '</div>'
                f'{body}</div>'
            )
            with col:
                st.markdown(card, unsafe_allow_html=True)


def _news_table(events: pd.DataFrame, now_broker: pd.Timestamp) -> pd.DataFrame:
    rows = []
    for _, row in events.iterrows():
        event_time = pd.to_datetime(row.get("event_time"), errors="coerce")
        day = pd.to_datetime(row.get("event_date"), errors="coerce")
        if pd.notna(event_time):
            status = "Past" if event_time < now_broker else "Upcoming"
        elif pd.notna(day):
            status = "Past" if day < now_broker.normalize() else "Upcoming"
        else:
            status = "Upcoming"
        rows.append({
            "Date": day.strftime("%a %d %b") if pd.notna(day) else "",
            "Time": _clean_event_value(row.get("time_label")) or "TBA",
            "Currency": "USD",
            "Impact": "🔴 High",
            "Event": _clean_event_value(row.get("event")),
            "Actual": _clean_event_value(row.get("actual")),
            "Forecast": _clean_event_value(row.get("forecast")),
            "Previous": _clean_event_value(row.get("previous")),
            "Status": status,
        })
    return pd.DataFrame(rows)


def sidebar():
    st.sidebar.title("VSA Advisor Bot")
    st.sidebar.caption("Hybrid VSA + ML Financial Advisor Bot")
    st.sidebar.caption("Final Submission • VSA + ML Advisor v9")
    meta = load_model_metadata()
    if meta:
        st.sidebar.success(f"Selected advisor: {meta.get('selected_model')}")
        st.sidebar.caption(f"Data: {str(meta.get('period_start'))[:10]} to {str(meta.get('period_end'))[:10]}")
        st.sidebar.caption("Backend-trained model loaded for user recommendations.")
    else:
        st.sidebar.error("No model metadata found. Run backend training first.")
    novice_pages = [
        "Live Advisor",
        "News Risk Filter",
        "How the Bot Learns",
        "Why This Advice?",
        "Risk & Safety",
        "Learning Hub",
        "Trade Journal",
        "MT5 Connection",
        "About the Bot",
        "Feedback",
    ]
    show_audit = st.sidebar.checkbox(
        "Show technical audit pages",
        value=False,
        help="Training tables, test evidence and engineering details are hidden by default because they are mainly for assessment and model review.",
    )
    pages = novice_pages
    if show_audit:
        pages += [
            "Technical: Backend Training",
            "Technical: LR vs RF",
            "Technical: XAI Audit",
            "Technical: Testing",
            "Technical: Algorithm",
            "Technical: Dataset",
            "Technical: Feature Table",
            "Technical: Live MT5 Proof",
        ]
    return st.sidebar.radio("Pages", pages)


def page_home():
    st.title("📈 VSA Advisor Bot")
    st.subheader("Financial Advisor Bot using VSA strategy outcomes and machine learning")
    st.warning("Educational demo only. This is not personal financial advice and does not guarantee profit.")
    meta = load_model_metadata()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Market", "XAUUSD")
    c2.metric("Timeframe", "M5")
    c3.metric("Bars", f"{meta.get('rows_raw','n/a')}")
    c4.metric("VSA setups", f"{meta.get('vsa_setup_rows','n/a')}")
    st.markdown("""
    The project combines a defined VSA strategy with machine-learning evaluation. The VSA scanner creates candidate setups,
    each historical setup is labelled by whether target or stop-loss was reached first, and two separate advisor models are trained:
    **LR + VSA** and **RF + VSA**. The user-facing website then loads the selected trained model and presents the recommendation with evidence.
    """)
    st.markdown("### Architecture")
    st.dataframe(pd.DataFrame([
        ["2-year M5 data", "MT5 XAUUSD candles from bars.csv"],
        ["VSA strategy scanner", "Finds CAB-based BUY/SELL candidate setups"],
        ["Outcome labelling", "Labels setups as VALID_TRADE if TP is hit before SL"],
        ["Backend training", "Trains LR+VSA and RF+VSA separately"],
        ["Model comparison", "Compares validation/test metrics and selects final advisor"],
        ["Website", "Shows trained-model recommendation, evidence, news risk, learning hub and feedback form"],
        ["MT5 bridge", "Exports ml_signal.csv so an EA can read alerts/demo-trade signals"],
    ], columns=["Part", "Purpose"]), width="stretch")


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _worker_age_seconds(status: dict[str, Any]) -> float | None:
    stamp = pd.to_datetime(status.get("cycle_time_utc"), errors="coerce", utc=True)
    if pd.isna(stamp):
        return None
    return max(0.0, (pd.Timestamp.now(tz="UTC") - stamp).total_seconds())


def _safe_broker_offset(value: Any) -> float | None:
    """Return a Streamlit-safe broker UTC offset, or None when the heartbeat value is invalid."""
    try:
        offset = float(value)
    except (TypeError, ValueError):
        return None

    if pd.isna(offset) or offset < -12.0 or offset > 14.0:
        return None
    return offset


def _mt5_terminal_snapshot(common_dir: Path = DEFAULT_MT5_COMMON) -> dict[str, Any]:
    """Read the exporter heartbeat and broker-connection flag for the UI.

    The exporter can report an invalid UTC offset when the broker clock is stale
    (for example when no fresh quote has updated TimeCurrent). Invalid values are
    rejected here so they can never crash Streamlit number inputs or silently
    influence the news/calendar timing.
    """
    path = Path(common_dir) / "vsa_mt5_connection_status.csv"
    if not path.exists():
        return {
            "state": "WAITING",
            "connected": False,
            "age_seconds": None,
            "broker_offset": None,
            "raw_broker_offset": None,
        }

    try:
        frame = pd.read_csv(path)
        if frame.empty:
            return {
                "state": "WAITING",
                "connected": False,
                "age_seconds": None,
                "broker_offset": None,
                "raw_broker_offset": None,
            }

        row = frame.iloc[-1].to_dict()
        age = max(0.0, datetime.now().timestamp() - path.stat().st_mtime)
        raw = str(row.get("terminal_connected", "0")).strip().lower()
        connected = raw in {"1", "true", "1.0", "yes"}

        if age > 45:
            state = "STALE HEARTBEAT"
        elif connected:
            state = "CONNECTED"
        else:
            state = "DISCONNECTED"

        raw_offset = row.get("broker_utc_offset_hours")
        safe_offset = _safe_broker_offset(raw_offset)

        return {
            "state": state,
            "connected": bool(connected and age <= 45),
            "age_seconds": age,
            "broker_offset": safe_offset,
            "raw_broker_offset": raw_offset,
            "server_time": row.get("server_time", ""),
            "gmt_time": row.get("gmt_time", ""),
        }
    except Exception:
        return {
            "state": "UNREADABLE",
            "connected": False,
            "age_seconds": None,
            "broker_offset": None,
            "raw_broker_offset": None,
        }


def _render_advisor_chart(
    bars: pd.DataFrame,
    signal: dict[str, Any],
    title: str,
    *,
    show_signal_overlay: bool = True,
    feed_badge: str | None = None,
) -> None:
    """Render a TradingView-style completed-candle chart for novice users.

    The visual is intentionally presentation-only: it uses the same completed bars
    already supplied to the pipeline and never re-runs or changes the VSA logic.
    """
    if bars.empty:
        st.info("No candle data are available for the chart.")
        return

    frame = bars.copy()
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    for column in ["open", "high", "low", "close", "tick_volume"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").drop_duplicates("time")
    if frame.empty:
        st.info("The candle timestamps or prices could not be read for charting.")
        return

    signal_time = pd.to_datetime(signal.get("signal_time"), errors="coerce") if show_signal_overlay else pd.NaT
    if pd.notna(signal_time):
        distances = (frame["time"] - signal_time).abs()
        nearest_pos = int(distances.to_numpy().argmin())
        lo = max(0, nearest_pos - 72)
        hi = min(len(frame), nearest_pos + 25)
        window = frame.iloc[lo:hi].copy()
    else:
        window = frame.tail(90).copy()

    if go is None or make_subplots is None:
        st.line_chart(window.set_index("time")["close"], height=440)
        st.caption("Plotly is unavailable, so a simplified close-price chart is shown.")
        return

    latest = window.iloc[-1]
    latest_close = float(latest["close"])
    latest_time = latest["time"]
    candle_span = max(float(window["high"].max() - window["low"].min()), max(abs(latest_close) * 0.0015, 0.5))
    y_pad = candle_span * 0.12
    y_min = float(window["low"].min()) - y_pad
    y_max = float(window["high"].max()) + y_pad

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.025,
        row_heights=[0.80, 0.20],
    )

    # TradingView-style candle colours.
    up_colour = "#26A69A"
    down_colour = "#EF5350"
    grid_colour = "#242B38"
    text_colour = "#D1D4DC"
    bg_colour = "#0B0E11"

    fig.add_trace(
        go.Candlestick(
            x=window["time"],
            open=window["open"],
            high=window["high"],
            low=window["low"],
            close=window["close"],
            name="XAUUSD M5",
            increasing_line_color=up_colour,
            increasing_fillcolor=up_colour,
            decreasing_line_color=down_colour,
            decreasing_fillcolor=down_colour,
            whiskerwidth=0.35,
            hovertext=[
                f"{t:%d %b %Y %H:%M}<br>O {o:.2f}  H {h:.2f}<br>L {l:.2f}  C {c:.2f}"
                for t, o, h, l, c in zip(window["time"], window["open"], window["high"], window["low"], window["close"])
            ],
            hoverinfo="text",
        ),
        row=1,
        col=1,
    )

    if "tick_volume" in window.columns and window["tick_volume"].notna().any():
        volume_colours = [up_colour if c >= o else down_colour for o, c in zip(window["open"], window["close"])]
        fig.add_trace(
            go.Bar(
                x=window["time"],
                y=window["tick_volume"],
                name="Tick volume",
                marker_color=volume_colours,
                opacity=0.48,
                hovertemplate="Tick volume %{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Latest completed close. This is not a live tick quote.
    fig.add_hline(
        y=latest_close,
        line_width=1,
        line_dash="dot",
        line_color="#787B86",
        row=1,
        col=1,
    )
    fig.add_annotation(
        x=window["time"].iloc[-1],
        y=latest_close,
        text=f"  Last close {latest_close:.2f}",
        showarrow=False,
        xanchor="left",
        font=dict(color="#B2B5BE", size=11),
        bgcolor="rgba(43,46,57,0.85)",
        bordercolor="#787B86",
        row=1,
        col=1,
    )

    level_palette = {
        "entry_price": ("ENTRY", "#2962FF", "solid"),
        "stop_price": ("STOP", "#F23645", "dash"),
        "target_price": ("1R", "#00C853", "dot"),
        "runner_target_price": ("4R", "#AB47BC", "dot"),
    }
    offscreen_levels: list[str] = []
    if show_signal_overlay:
        for key, (label, colour, dash) in level_palette.items():
            value = pd.to_numeric(pd.Series([signal.get(key)]), errors="coerce").iloc[0]
            if pd.isna(value):
                continue
            value = float(value)
            if y_min <= value <= y_max:
                fig.add_hline(y=value, line_dash=dash, line_color=colour, line_width=1.7, row=1, col=1)
                fig.add_annotation(
                    x=window["time"].iloc[-1],
                    y=value,
                    text=f"  {label} {value:.2f}",
                    showarrow=False,
                    xanchor="left",
                    font=dict(color="#FFFFFF", size=11),
                    bgcolor=colour,
                    bordercolor=colour,
                    row=1,
                    col=1,
                )
            else:
                arrow = "↑" if value > y_max else "↓"
                offscreen_levels.append(f"{label} {value:.2f} {arrow}")

        cab_time = pd.to_datetime(signal.get("cab_time"), errors="coerce")
        if pd.notna(cab_time):
            cab_match = frame.loc[frame["time"] == cab_time]
            if cab_match.empty:
                tolerance = pd.Timedelta(minutes=3)
                cab_match = frame.loc[(frame["time"] - cab_time).abs() <= tolerance]
            if not cab_match.empty:
                cab = cab_match.iloc[-1]
                x1 = signal_time if pd.notna(signal_time) else window["time"].iloc[-1]
                fig.add_shape(
                    type="rect",
                    x0=cab["time"],
                    x1=x1,
                    y0=float(cab["low"]),
                    y1=float(cab["high"]),
                    fillcolor="rgba(255, 193, 7, 0.13)",
                    line=dict(color="rgba(255, 193, 7, 0.80)", width=1),
                    row=1,
                    col=1,
                )
                fig.add_annotation(
                    x=cab["time"],
                    y=min(float(cab["high"]), y_max),
                    text="CAB",
                    showarrow=False,
                    xanchor="left",
                    yanchor="bottom",
                    font=dict(color="#FFC107", size=11),
                    row=1,
                    col=1,
                )

        if pd.notna(signal_time):
            entry = pd.to_numeric(pd.Series([signal.get("entry_price")]), errors="coerce").iloc[0]
            if pd.notna(entry) and y_min <= float(entry) <= y_max:
                is_buy = str(signal.get("direction", "")).upper() == "BUY"
                fig.add_trace(
                    go.Scatter(
                        x=[signal_time],
                        y=[float(entry)],
                        mode="markers+text",
                        marker=dict(size=15, symbol="triangle-up" if is_buy else "triangle-down", color="#FFFFFF", line=dict(color="#2962FF", width=2)),
                        text=[f"S{signal.get('scenario_id', '')} {str(signal.get('direction', '')).upper()}"],
                        textposition="top center" if is_buy else "bottom center",
                        textfont=dict(color="#FFFFFF", size=11),
                        name="Setup",
                        hovertemplate="Scenario %{text}<br>Entry %{y:.2f}<extra></extra>",
                    ),
                    row=1,
                    col=1,
                )

    badge = feed_badge or "COMPLETED CANDLES"
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.985,
        text=f"XAUUSD · M5 · {badge}",
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font=dict(size=12, color="#B2B5BE"),
        bgcolor="rgba(19,23,34,0.84)",
        bordercolor="#2A2E39",
        borderwidth=1,
        borderpad=5,
    )

    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=19, color="#F0F3FA")),
        height=690,
        template="plotly_dark",
        paper_bgcolor=bg_colour,
        plot_bgcolor=bg_colour,
        margin=dict(l=12, r=78, t=62, b=15),
        showlegend=False,
        hovermode="closest",
        dragmode="pan",
        bargap=0.10,
        font=dict(color=text_colour),
        hoverlabel=dict(bgcolor="#1E222D", bordercolor="#363A45", font_color="#FFFFFF"),
    )
    fig.update_xaxes(
        rangeslider_visible=False,
        showgrid=True,
        gridcolor=grid_colour,
        zeroline=False,
        showspikes=True,
        spikecolor="#787B86",
        spikethickness=1,
        spikedash="dot",
        spikesnap="cursor",
        showline=False,
        tickfont=dict(color="#B2B5BE"),
        row=1,
        col=1,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_colour,
        zeroline=False,
        tickfont=dict(color="#B2B5BE"),
        title_text="Completed M5 candles",
        row=2,
        col=1,
    )
    fig.update_yaxes(
        range=[y_min, y_max],
        side="right",
        showgrid=True,
        gridcolor=grid_colour,
        zeroline=False,
        tickformat=".2f",
        tickfont=dict(color="#B2B5BE"),
        fixedrange=False,
        row=1,
        col=1,
    )
    fig.update_yaxes(
        side="right",
        showgrid=False,
        zeroline=False,
        tickfont=dict(color="#787B86", size=10),
        title_text="Volume",
        row=2,
        col=1,
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "displayModeBar": True,
            "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
        },
    )

    guide_cols = st.columns(4)
    guide_cols[0].markdown("**🟩/🟥 Candles**  \nGreen closed higher; red closed lower.")
    guide_cols[1].markdown("**🟨 CAB zone**  \nThe VSA climactic-action area.")
    guide_cols[2].markdown("**🔵 Entry / 🔴 Stop**  \nPlanned decision and protection levels.")
    guide_cols[3].markdown("**🟢 1R / 🟣 4R**  \nStrategy targets; not guaranteed outcomes.")
    st.caption(
        f"Latest completed candle: {latest_time:%d %b %Y %H:%M} · last close: {latest_close:.2f}. "
        "Use the mouse wheel to zoom and drag to pan. This is a completed-candle decision view, not a streaming tick chart."
    )
    if offscreen_levels:
        st.info("Targets outside the visible candle range are kept out of the price scale so candles stay readable: " + " · ".join(offscreen_levels))

def _journal_current_signal(signal: dict[str, Any], source_mode: str) -> None:
    signal_id = make_signal_id(signal)
    with st.expander("📓 Add this advice to my learning journal", expanded=False):
        st.caption(
            "The journal is for reflection only. It never places a trade, changes the model or retrains on your notes. "
            "Leave the result blank until you genuinely know it."
        )
        with st.form(f"journal_signal_{signal_id}_{source_mode}"):
            decision = st.selectbox("What did you do?", ["Watched", "Skipped", "Demo-traded"], key=f"decision_{signal_id}_{source_mode}")
            result_text = st.text_input("Optional realised result in R (leave blank if unresolved)", value="")
            notes = st.text_area("Notes", placeholder="What did you notice about the setup, news or risk?")
            lesson = st.text_area("Lesson learned", placeholder="What would you do differently next time?")
            submitted = st.form_submit_button("Save / update journal entry")
        if submitted:
            result_r = None
            if result_text.strip():
                try:
                    result_r = float(result_text.strip())
                except ValueError:
                    st.error("Result R must be a number such as 0.7, -1 or 1.9, or left blank.")
                    return
            try:
                upsert_journal_entry(
                    signal,
                    source_mode=source_mode,
                    user_decision=decision,
                    notes=notes,
                    lesson_learned=lesson,
                    result_r=result_r,
                )
                st.success(f"Saved {signal_id} to the Trade Journal.")
            except Exception as exc:
                st.error(f"Could not save the journal entry: {exc}")


def page_recommendation():
    st.title("📡 Live ML Advisor")
    meta = load_model_metadata()
    if not meta:
        st.error("Model metadata was not found. Run backend training first.")
        return

    st.write(
        "The VSA scanner identifies a formal completed-candle setup first. The already-trained Logistic Regression model then scores "
        "that setup, after which freshness and news controls can still force WATCH ONLY."
    )
    st.info("VSA creates the candidate; the saved LR model scores it; safety gates control whether the candidate can remain actionable in the demo.")

    mt5_snapshot = _mt5_terminal_snapshot()
    detected_offset = mt5_snapshot.get("broker_offset")
    raw_detected_offset = mt5_snapshot.get("raw_broker_offset")
    with st.expander("Connection settings", expanded=False):
        if detected_offset is not None:
            st.success(f"Broker UTC offset auto-detected from MT5: UTC{float(detected_offset):+g}")
        elif raw_detected_offset not in (None, "", "nan"):
            st.caption(
                "MT5 broker-time auto-detection is temporarily unavailable. "
                "The project fallback UTC+3 is being used; change it below only if your broker uses a different offset."
            )
        else:
            st.caption(
                "No broker offset heartbeat has been detected yet. "
                "The fallback below is used only until the exporter is connected."
            )
        broker_offset = st.number_input(
            "Broker server UTC offset",
            min_value=-12.0,
            max_value=14.0,
            value=float(detected_offset) if detected_offset is not None else 3.0,
            step=1.0,
            help="Normally auto-detected from the MT5 exporter. Change this only when deliberately overriding the broker clock.",
        )
    data_source = st.radio(
        "Recommendation data source",
        ["Live MT5 completed bars", "Saved historical project bars (demonstration only)"],
        horizontal=True,
        help="Historical demonstration mode never writes to the MT5 bridge. Live mode is only labelled LIVE when the feed and worker are current.",
    )

    source_mode = "LIVE"
    live_status: dict[str, Any] = {}
    signal_df = pd.DataFrame()
    bars_for_chart = pd.DataFrame()
    is_current = False

    if data_source == "Live MT5 completed bars":
        refresh = st.button("Refresh live feed + safety state", help="Runs one live scoring cycle using the saved LR model.")
        if refresh:
            with st.spinner("Checking the MT5 export, current clock, VSA setup, LR model and news gate..."):
                live_status = run_live_cycle(
                    bars_path=DEFAULT_MT5_COMMON / "xauusd_m5_bars.csv",
                    project_signal_path=OUTPUTS_DIR / "ml_signal.csv",
                    bridge_signal_path=DEFAULT_MT5_COMMON / "ml_signal.csv",
                    status_path=OUTPUTS_DIR / "live_advisor_status.json",
                    broker_utc_offset_hours=float(broker_offset),
                )
                _read_csv_cached.clear()
                _read_json_cached.clear()
        else:
            live_status = read_json(OUTPUTS_DIR / "live_advisor_status.json")

        signal_df = read_csv(OUTPUTS_DIR / "ml_signal.csv")
        worker_age = _worker_age_seconds(live_status)
        worker_connected = worker_age is not None and worker_age <= 90
        feed_state = str(live_status.get("feed_state", "DISCONNECTED"))
        state = str(live_status.get("state", "NO STATUS"))
        latest_saved_time = str(signal_df.iloc[-1].get("signal_time", "")) if not signal_df.empty else ""
        status_setup_time = str(live_status.get("latest_setup_time", ""))
        mt5_snapshot = _mt5_terminal_snapshot()
        terminal_connected = bool(mt5_snapshot.get("connected"))
        market_closed = feed_state == "MARKET_CLOSED"
        live_feed_ready = terminal_connected and worker_connected and feed_state == "LIVE"
        is_current = (
            live_feed_ready
            and state in {"NEW_SETUP_SCORED", "DUPLICATE_SETUP_REFRESHED"}
            and bool(status_setup_time)
            and latest_saved_time == status_setup_time
        )
        source_mode = (
            "LIVE" if is_current
            else ("LIVE_NO_SETUP" if live_feed_ready else ("MARKET_CLOSED" if market_closed else "EXPIRED_LIVE"))
        )

        age = mt5_snapshot.get("age_seconds")
        mode_display = "MARKET CLOSED" if market_closed else ("LIVE FEED" if live_feed_ready else "NOT LIVE")
        terminal_display = str(mt5_snapshot.get("state", "WAITING")).replace("_", " ")
        feed_display = str(feed_state).replace("_", " ")
        worker_display = "CONNECTED" if worker_connected else "DISCONNECTED"
        signal_display = (
            "WAITING FOR REOPEN" if state == "MARKET_CLOSED"
            else ("NO ACTIVE SETUP" if state == "NO_NEW_VSA_SETUP" else state.replace("_", " "))
        )
        exporter_sub = f"Exporter heartbeat {float(age):.0f}s ago" if age is not None else "No exporter heartbeat"
        worker_sub = "No recent worker heartbeat" if worker_age is None else f"Last cycle {worker_age:.0f}s ago"

        cards = [
            ("Mode", mode_display, "Weekly session state" if market_closed else "Advisor operating state"),
            ("MT5 terminal", terminal_display, exporter_sub),
            ("M5 feed", feed_display, "Last completed session retained" if market_closed else "Completed-bar freshness"),
            ("Python worker", worker_display, worker_sub),
            ("Signal state", signal_display, "No live setup can form while closed" if market_closed else "Current VSA setup state"),
        ]
        cards_html = "".join(
            '<div class="live-status-card">'
            f'<div class="live-status-label">{html.escape(str(label))}</div>'
            f'<div class="live-status-value">{html.escape(str(value))}</div>'
            f'<div class="live-status-sub">{html.escape(str(sub))}</div>'
            '</div>'
            for label, value, sub in cards
        )
        st.markdown(f'<div class="live-status-grid">{cards_html}</div>', unsafe_allow_html=True)
        lag = live_status.get("feed_lag_minutes")
        if lag is not None:
            st.caption(
                f"Latest completed candle: {live_status.get('latest_closed_bar', 'n/a')} · "
                f"expected latest completed candle: {live_status.get('expected_latest_closed_bar', 'n/a')} · "
                f"feed lag: {float(lag):.1f} min"
            )

        if not terminal_connected:
            st.error(
                "MT5 TERMINAL DISCONNECTED: the exporter heartbeat does not show an active broker session. "
                "Reconnect the account inside MetaTrader 5 first; the Python worker cannot make cached August candles live."
            )
        elif not worker_connected:
            st.error("The live worker has no current heartbeat. Any saved recommendation below is treated as previous/expired, not as live advice.")
        elif market_closed:
            st.info(
                "MARKET CLOSED: MT5 and the Python worker are connected, but XAUUSD is in the normal weekly closed period. "
                "The final completed candle from Friday is retained and freshness checks resume automatically when the market reopens."
            )
        elif feed_state != "LIVE":
            st.error(f"Live feed state: {feed_state}. {live_status.get('message', '')}")
        elif state == "NO_NEW_VSA_SETUP":
            st.info("The MT5 feed is current, but no new formal VSA Scenario 1/2/3 setup is confirmed on the latest completed bars.")
        elif state in {"SAFE_ERROR", "STALE_OR_DISCONNECTED_FEED"}:
            st.error(live_status.get("message", "The live cycle failed safely."))
        elif is_current:
            st.success("LIVE: feed, worker and displayed signal are aligned to the current completed-bar cycle.")

        try:
            bars_for_chart = read_mt5_bars(DEFAULT_MT5_COMMON / "xauusd_m5_bars.csv", tail_rows=800)
        except Exception:
            bars_for_chart = pd.DataFrame()
    else:
        source_mode = "HISTORICAL_DEMO"
        st.warning(
            "HISTORICAL DEMO: this scores the frozen project data for reproducible demonstration only. "
            "It is not a live quote and it is not written to the MT5 bridge."
        )
        try:
            signal = latest_vsa_setup_prediction(DEFAULT_DATA_FILE)
            signal_df = pd.DataFrame([signal])
            bars_for_chart = read_mt5_bars(DEFAULT_DATA_FILE, tail_rows=1_000)
        except Exception as exc:
            st.error(f"Could not load the historical demonstration: {exc}")
            return
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Mode", "HISTORICAL DEMO")
        h2.metric("Market", "XAUUSD")
        h3.metric("Timeframe", "M5")
        h4.metric("Bridge write", "DISABLED")

    if signal_df.empty:
        st.warning("No saved signal is available. In live mode this is normal until a new formal VSA setup is scored.")
        if not bars_for_chart.empty:
            st.markdown("### Current completed candles")
            _render_advisor_chart(bars_for_chart, {}, "XAUUSD M5 completed candles")
        return

    signal = signal_df.tail(1).iloc[0].to_dict()
    signal_id = make_signal_id(signal)
    prob = pd.to_numeric(pd.Series([signal.get("ml_probability_valid", signal.get("confidence", 0))]), errors="coerce").iloc[0]
    prob_float = float(prob) if pd.notna(prob) else 0.0
    advisor_output = str(signal.get("advisor_output", signal.get("signal", "WATCH_ONLY")))
    direction = str(signal.get("direction", "n/a"))
    selected_model = str(signal.get("selected_model", meta.get("selected_model", "n/a")))

    # Put the market context first. A live feed can be healthy even when there is no
    # current VSA setup, so feed status must not be tied to signal availability.
    st.markdown("### Market view")
    if data_source == "Live MT5 completed bars":
        market_closed = feed_state == "MARKET_CLOSED"
        live_feed_ready = terminal_connected and worker_connected and feed_state == "LIVE"
        if market_closed:
            chart_title = "XAUUSD M5 — last completed session candles"
            chart_badge = "MARKET CLOSED"
        else:
            chart_title = "XAUUSD M5 — live completed-candle view" if live_feed_ready else "XAUUSD M5 — last candles received from MT5"
            chart_badge = "LIVE FEED" if live_feed_ready else "STALE / DISCONNECTED FEED"
        _render_advisor_chart(
            bars_for_chart,
            signal if is_current else {},
            chart_title,
            show_signal_overlay=is_current,
            feed_badge=chart_badge,
        )
        if market_closed:
            st.caption(
                "The market is closed, so this chart intentionally shows the final completed candles from the previous session. "
                "No previous signal is presented as current advice."
            )
        elif live_feed_ready and not is_current:
            st.caption("The feed is live. No current VSA setup is overlaid because Scenario 1, 2 or 3 has not just confirmed. The previous saved signal remains audit-only.")
        elif not live_feed_ready:
            st.caption("The previous saved signal is intentionally not drawn over this chart because the live feed is not currently verified.")
    else:
        _render_advisor_chart(
            bars_for_chart,
            signal,
            "XAUUSD M5 — historical demonstration setup",
            show_signal_overlay=True,
            feed_badge="HISTORICAL DEMO",
        )
    st.caption("The chart uses the same completed candles supplied to the project pipeline. It is a decision-context visual, not a second VSA implementation.")

    expired_live = data_source == "Live MT5 completed bars" and not is_current
    details_context = st.expander("Previous signal (expired) — audit only", expanded=False) if expired_live else st.container()
    with details_context:
        if expired_live:
            st.warning("This is the last saved signal, not a current recommendation. It is retained only for audit, learning and journal review.")

        model_col, decision_col, direction_col, probability_col = st.columns(4)
        model_col.metric("Selected model", selected_model)
        decision_col.metric("Advisor output", advisor_output)
        direction_col.metric("Direction", direction)
        probability_col.metric("LR probability", f"{prob_float:.1%}")
        st.caption(f"Signal ID: {signal_id} · Scenario {signal.get('scenario_id', 'n/a')} · Signal candle: {signal.get('signal_time', 'n/a')}")

        ml_output = str(signal.get("ml_advisor_output", advisor_output))
        news_blocked = _as_bool(signal.get("news_blocked", True))
        if advisor_output == "TRADE_CANDIDATE" and not news_blocked:
            st.success(f"The trained LR model accepts this {direction} VSA setup and the safety/news gate is clear for the configured window.")
        elif ml_output == "TRADE_CANDIDATE" and news_blocked:
            st.warning("The LR model accepted this VSA setup, but the news/safety layer downgraded the final decision to WATCH ONLY.")
        else:
            st.info("The LR estimate is below the frozen decision threshold, so the setup remains WATCH ONLY.")

        st.markdown("### Proposed plan")
        plan_cols = st.columns(5)
        plan_cols[0].metric("Entry", metric_fmt(signal.get("entry_price", "n/a"), 2))
        plan_cols[1].metric("Stop-loss", metric_fmt(signal.get("stop_price", "n/a"), 2))
        plan_cols[2].metric("1R target", metric_fmt(signal.get("target_price", "n/a"), 2))
        plan_cols[3].metric("4R target", metric_fmt(signal.get("runner_target_price", "n/a"), 2))
        plan_cols[4].metric("Max strategy risk", f"{metric_fmt(signal.get('risk_percent', 0), 1)}%")
        st.caption("These levels are educational strategy outputs. They are not a promise of execution quality or future profit.")

        news_status = str(signal.get("news_status", "NOT_CHECKED"))
        if news_blocked:
            st.warning(f"News gate: {news_status}. {signal.get('news_reason', '')}")
        else:
            st.success(f"News gate: {news_status}. No verified high-impact USD event is inside the configured risk window.")

        st.markdown("### Why this decision")
        explanation = pd.DataFrame([
            ["VSA setup", f"Scenario {signal.get('scenario_id', 'n/a')} {direction} candidate from the formal state machine"],
            ["ML check", f"{selected_model}: {prob_float:.1%} versus frozen threshold {float(signal.get('recommendation_threshold', 0) or 0):.1%}"],
            ["Safety decision", advisor_output],
            ["News status", news_status],
            ["Source", ("Previous live MT5 signal — expired" if expired_live else source_mode.replace("_", " "))],
        ], columns=["Layer", "Evidence"])
        st.dataframe(explanation, width="stretch", hide_index=True)

        local_explanation = live_status.get("lr_local_explanation", []) if live_status else []
        if local_explanation and str(live_status.get("latest_setup_time")) == str(signal.get("signal_time")):
            with st.expander("Top LR feature contributions"):
                st.caption("These are exact fitted Logistic Regression contributions after the saved StandardScaler; they are associations, not causal claims.")
                st.dataframe(pd.DataFrame(local_explanation), width="stretch", hide_index=True)

        if data_source == "Live MT5 completed bars":
            with st.expander("MT5 bridge record / technical audit"):
                st.dataframe(signal_df.tail(1), width="stretch", hide_index=True)
                st.caption("Only live mode writes the fixed 29-field ml_signal.csv bridge record. Historical demo mode never overwrites it.")

        _journal_current_signal(signal, source_mode)

    st.warning("Educational and demo use only. Historical model evidence does not guarantee future performance.")

def page_training_evidence():
    st.title("🧪 Backend Training Evidence")
    st.info("Backend training summary for implementation evidence. Training is run from backend_training/train_models.py and saved as model/result files.")
    meta = load_model_metadata()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw M5 bars", f"{meta.get('rows_raw','n/a')}")
    c2.metric("VSA setup examples", f"{meta.get('vsa_setup_rows','n/a')}")
    c3.metric("Train / Val / Test", f"{meta.get('train_rows','n/a')} / {meta.get('validation_rows','n/a')} / {meta.get('test_rows','n/a')}")
    c4.metric("Selected", meta.get("selected_model", "n/a"))
    st.markdown("### Training log")
    log_path = RESULTS_DIR / "training_log.txt"
    if log_path.exists():
        st.code(log_path.read_text(), language="text")
    else:
        st.warning("training_log.txt not found. Run ./run_backend_training.sh")
    st.markdown("### Training metadata")
    st.json(meta)


def page_model_comparison():
    st.title("📊 LR vs RF Comparison")
    st.write("Both models are trained on the same VSA setup dataset. LR is easier to explain; RF is included as a separate comparison model.")
    comp = read_csv(RESULTS_DIR / "model_comparison.csv")
    if comp.empty:
        st.warning("No comparison results found. Run backend training first.")
        return
    st.dataframe(comp, width="stretch")
    if px is not None:
        val = comp[comp["split"] == "validation"]
        st.plotly_chart(px.bar(val, x="model", y=["accuracy", "f1", "roc_auc"], barmode="group", title="Validation comparison"), width="stretch")
        test = comp[comp["split"] == "test"]
        st.plotly_chart(px.bar(test, x="model", y=["accuracy", "f1", "roc_auc"], barmode="group", title="Unseen test comparison"), width="stretch")
    meta = load_model_metadata()
    st.success(f"Selected final advisor model: {meta.get('selected_model')} — {meta.get('selection_reason')}")
    st.markdown("### Advice-level test evaluation")
    st.dataframe(read_csv(RESULTS_DIR / "test_trading_evaluation.csv"), width="stretch")
    st.caption("The VSA-only baseline takes every formal setup. ML models filter the same setups. Net results include the recorded broker spread estimate.")
    st.markdown("### Robustness checks")
    st.write("Walk-forward evaluation uses four expanding historical windows. Mixed or negative windows are retained as limitations rather than removed.")
    st.dataframe(read_csv(RESULTS_DIR / "walk_forward_evaluation.csv"), width="stretch")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Feature ablation")
        st.dataframe(read_csv(RESULTS_DIR / "feature_ablation.csv"), width="stretch")
    with col_b:
        st.markdown("#### Transaction-cost sensitivity")
        st.dataframe(read_csv(RESULTS_DIR / "transaction_cost_sensitivity.csv"), width="stretch")
    st.markdown("### Smaller core feature-set validation")
    st.write(
        "The 10-feature VSA core and the full 24-feature context set are compared on the same untouched test period, "
        "across four future walk-forward folds, and with 1,000 bootstrap resamples. Wide intervals and negative folds are retained."
    )
    st.dataframe(read_csv(RESULTS_DIR / "feature_set_walk_forward.csv"), width="stretch")
    st.dataframe(read_csv(RESULTS_DIR / "feature_ablation_bootstrap.csv"), width="stretch")
    st.markdown("### Additional model-family benchmarks")
    st.caption("Decision Tree and Histogram Gradient Boosting provide context; they do not alter the teacher-required separate LR/RF selection process.")
    st.dataframe(read_csv(RESULTS_DIR / "additional_model_benchmarks.csv"), width="stretch")
    st.dataframe(read_csv(RESULTS_DIR / "model_family_justification.csv"), width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### LR coefficients")
        st.dataframe(read_csv(RESULTS_DIR / "lr_coefficients.csv").head(12), width="stretch")
    with col2:
        st.markdown("### RF feature importance")
        st.dataframe(read_csv(RESULTS_DIR / "rf_feature_importance.csv").head(12), width="stretch")


def page_xai():
    st.title("XAI and Explainability")
    st.write(
        "The deployed advisor uses Logistic Regression because its calculation can be explained directly. "
        "This page separates model XAI from the VSA narrative and from safety advice."
    )

    distinction = pd.DataFrame([
        ["Global model interpretation", "Fitted LR coefficients", "Shows the overall direction and strength of each standardised feature weight."],
        ["Local model explanation (XAI)", "Standardised value × fitted LR coefficient", "Shows how every feature changes the log-odds for one prediction."],
        ["Domain explanation", "CAB and Scenario 1/2/3 rules", "Explains why the rule-based VSA scanner created a candidate; this is not model XAI."],
        ["Action and safety advice", "Threshold, news gate and risk controls", "Explains what the prototype permits; this is not evidence about the model's internal calculation."],
    ], columns=["Explanation layer", "Method", "Meaning"])
    st.dataframe(distinction, width="stretch")

    summary = read_json(RESULTS_DIR / "xai_faithfulness_summary.json")
    if summary:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Unseen rows audited", summary.get("rows_audited", "n/a"))
        c2.metric("Features per explanation", summary.get("features_per_explanation", "n/a"))
        c3.metric("Maximum reconstruction error", f"{float(summary.get('maximum_absolute_probability_error', 0)):.2e}")
        c4.metric("Faithfulness check", "PASS" if summary.get("faithful_within_tolerance") else "FAIL")
        st.success(
            "Every audited probability is reconstructed from the fitted scaler, all LR feature contributions and the intercept "
            "within the stated numerical tolerance."
        )
    else:
        st.warning("Formal XAI evidence is missing. Run backend_training/evaluate_explainability.py.")

    left, right = st.columns(2)
    with left:
        st.markdown("### Global LR interpretation")
        st.dataframe(read_csv(RESULTS_DIR / "lr_coefficients.csv"), width="stretch")
    with right:
        st.markdown("### Local unseen-test audit")
        st.dataframe(read_csv(RESULTS_DIR / "xai_local_probability_audit.csv"), width="stretch")

    st.markdown("### Important interpretation limits")
    st.warning(
        "Faithfulness means the displayed contributions reproduce the trained LR calculation. It does not mean a feature causes "
        "a profitable trade, that the probability is certain, or that historical relationships will remain stable."
    )
    st.info(
        "Random Forest feature importance is shown only as global comparison evidence. The prototype does not label it as a complete "
        "local XAI explanation, which is one reason the transparent LR model is preferred for novice-facing deployment."
    )


def page_testing_evaluation():
    st.title("Testing & Evaluation")
    st.write(
        "This page separates software tests, machine-learning evaluation, strategy evaluation and safe MT5 integration evidence. "
        "Controlled replay files never overwrite the live MT5 bridge signal."
    )

    strategy = pd.DataFrame([
        ["Unit tests", "32 checks covering candle features, mirrored stops, managed exits, clock freshness, MT5 connection state, calendar coverage, journal persistence, CSV contract and LR explanation fidelity", "Automated unit/integration suite"],
        ["Leakage control", "Purged chronological 70/15/15 split with a 180-minute embargo", "model_metadata.json"],
        ["Model comparison", "LR and RF trained separately; threshold and selection use validation only", "model_comparison.csv"],
        ["Unseen evaluation", "Final metrics and trading results measured on the untouched test period", "test_trading_evaluation.csv"],
        ["Robustness", "Four expanding walk-forward folds, feature ablation and spread-cost sensitivity", "walk_forward_evaluation.csv"],
        ["Live failure modes", "Clock-based stale/disconnected rejection, no-new-setup handling, duplicate-ID safety refresh and news/calendar blocking", "live replay + MT5 evidence"],
    ], columns=["Test layer", "Method", "Evidence"])
    st.dataframe(strategy, width="stretch")

    summary = read_json(RESULTS_DIR / "live_replay_integration_summary.json")
    if summary:
        st.markdown("### Controlled completed-bar replay")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", summary.get("selected_model", "n/a"))
        c2.metric("Scenario", summary.get("scenario_id", "n/a"))
        c3.metric("LR probability", f"{float(summary.get('lr_probability_valid', 0)):.1%}")
        c4.metric("All checks", "PASS" if summary.get("all_checks_passed") else "FAIL")
        checks = pd.DataFrame([
            {"check": name.replace("_", " ").title(), "result": "PASS" if passed else "FAIL"}
            for name, passed in summary.get("checks", {}).items()
        ])
        st.dataframe(checks, width="stretch")

        clear_signal = read_csv(RESULTS_DIR / "live_replay_clear_signal.csv")
        blocked_signal = read_csv(RESULTS_DIR / "live_replay_news_blocked_signal.csv")
        if not clear_signal.empty and not blocked_signal.empty:
            comparison = pd.DataFrame([
                ["Verified clear calendar", clear_signal.iloc[0]["ml_advisor_output"], clear_signal.iloc[0]["advisor_output"], clear_signal.iloc[0]["news_status"]],
                ["High-impact USD event", blocked_signal.iloc[0]["ml_advisor_output"], blocked_signal.iloc[0]["advisor_output"], blocked_signal.iloc[0]["news_status"]],
            ], columns=["Replay condition", "Raw LR decision", "Final decision", "News status"])
            st.dataframe(comparison, width="stretch")

        st.markdown("### Explainable LR decision")
        st.caption("These are local standardised feature contributions from the same fitted LR pipeline used for prediction.")
        st.dataframe(read_csv(RESULTS_DIR / "live_replay_lr_explanation.csv"), width="stretch")

    st.markdown("### Automated test log")
    test_log = RESULTS_DIR / "automated_test_log.txt"
    if test_log.exists():
        st.code(test_log.read_text(), language="text")
    else:
        st.info("Run ./run_tests.sh to create the saved automated-test log.")

    st.markdown("### MT5 Expert Advisor benchmark")
    replay_audit = read_json(RESULTS_DIR / "mt5_replay_signal_audit.json")
    if replay_audit:
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Untouched VSA setups", replay_audit.get("formal_vsa_setups", "n/a"))
        b2.metric("LR candidates", replay_audit.get("lr_accepted_setups", "n/a"))
        b3.metric("LR threshold", replay_audit.get("lr_threshold", "n/a"))
        b4.metric("Replay parity", "PASS" if replay_audit.get("signal_rows_match_python_test_predictions") else "FAIL")
    st.write(
        "The completed tester comparison uses XAUUSD M5, 1 June–17 August 2026, 100% real ticks, a USD 10,000 "
        "deposit, 1:100 leverage and fixed 0.01-lot orders for all five systems. The untouched VSA/LR setup signals "
        "begin on 16 June inside that common tester window."
    )
    st.dataframe(read_csv(RESULTS_DIR / "mt5_ea_comparison.csv"), width="stretch")
    st.success(
        "Primary ablation result: the LR filter improved the identical VSA replay from $23.21 to $284.84 net profit, "
        "raised profit factor from 1.04 to 1.74 and reduced maximum equity drawdown from 1.69% to 1.31%."
    )
    st.caption(
        "ExpertMAPSAR produced the strongest contextual result. It uses different entry and exit logic, so the table "
        "does not show that VSA+LR is universally better than every EA."
    )

    st.warning(
        "Limitation retained for academic honesty: the latest held-out period is positive, but mean walk-forward net expectancy is negative. "
        "The prototype therefore supports educational/demo advice and does not establish stable future profitability."
    )


def page_algorithm():
    st.title("VSA + ML Algorithm")
    st.write(
        "This page gives a clean pseudocode view of the final advisor logic. It is useful for the implementation chapter and demo screenshots."
    )

    st.markdown("### Training pipeline pseudocode")
    st.code("""
INPUT: two_year_M5_bars.csv
OUTPUT: trained LR+VSA model, trained RF+VSA model, model comparison, selected advisor model

1. Load XAUUSD M5 OHLCV candles
2. Clean date, price and tick-volume fields
3. For each candle t:
      spread[t] = high[t] - low[t]
      body[t] = abs(close[t] - open[t])
      relative_volume[t] = volume[t] / rolling_mean(volume, 20)
      relative_spread[t] = spread[t] / rolling_mean(spread, 20)
      close_location[t] = (close[t] - low[t]) / spread[t]
4. Detect a BUY CAB when a bearish candle appears in a downtrend with
      tick volume >= 1.5 times the previous 30-candle average
   Mirror this condition for a SELL CAB in an uptrend
5. Mark the CAB high and low trigger lines
6. Detect the exact mirrored VSA state sequence:
      Scenario 1 = first breakout, return inside without opposite sweep,
                   then second momentum breakout on volume below CAB
      Scenario 2 = opposite trigger wick sweep without close beyond it,
                   then momentum breakout on volume below CAB
      Scenario 3 = candle close beyond the opposite trigger,
                   then momentum breakout on volume below CAB
7. Set scenario-specific stop and risk:
      S1 = CAB opposite trigger, 0.5% account risk
      S2 = sweep-candle extreme, 1% account risk
      S3 = post-break extreme, 1% account risk
8. Label future managed outcome:
      stop before 1R = -1R
      close 70% at 1R, move remaining stop to breakeven
      runner breakeven = +0.7R; runner reaches 4R = +1.9R
9. Chronologically split the labelled VSA setup dataset with a 180-minute embargo:
      first 70% = train
      next 15% = validation
      final 15% = unseen test
10. Train LR+VSA and RF+VSA separately on identical training data
11. Tune each probability threshold on validation data only
12. Select the highest validation-F1 model; never select using test results
13. Evaluate once on untouched test data against majority and VSA-only baselines
14. Run walk-forward, feature-ablation and transaction-cost checks
15. Save reproducible models, predictions, metrics and logs
""", language="text")

    st.markdown("### Live recommendation pseudocode")
    st.code("""
INPUT: latest M5 candles, selected_model.joblib, news calendar
OUTPUT: TRADE_CANDIDATE or WATCH_ONLY with BUY/SELL direction for MT5 bridge

1. Load latest M5 candles
2. Run VSA scanner on the latest completed candle sequence
3. If no valid VSA setup is present:
      return NO_NEW_VSA_SETUP and do not overwrite the existing signal CSV
4. Convert the latest VSA setup into model features
5. Load selected trained model
6. Predict probability that the setup is valid
7. Check high-impact USD news window
8. If probability >= threshold AND news is not blocked:
      return TRADE_CANDIDATE with BUY/SELL direction
   Else:
      return WATCH_ONLY with BUY/SELL direction and recorded reason
9. Write latest decision to outputs/ml_signal.csv
10. MT5 EA reads ml_signal.csv and sends alert or places demo trade if enabled
""", language="text")

    st.markdown("### Why this belongs in the report")
    st.write(
        "Add this pseudocode in Chapter 3 Design or Chapter 4 Implementation. It shows that VSA is not just a vague trading idea: it is converted into measurable features, labelled examples and a reproducible ML pipeline."
    )

    st.dataframe(pd.DataFrame([
        ["VSA", "Creates candidate setups and interpretable trading evidence"],
        ["LR", "Explainable model; coefficient-based comparison"],
        ["RF", "Non-linear comparison model; stronger benchmark but less transparent"],
        ["Chronological split", "Prevents random future leakage in time-series data"],
        ["MT5 CSV bridge", "Separates Python ML from MT5 alert/demo execution"],
    ], columns=["Component", "Reason in final design"]), width="stretch")


def page_vsa_dataset():
    st.title("🧾 VSA Setup Dataset")
    try:
        setup_df = read_csv(RESULTS_DIR / "vsa_setup_training_dataset.csv")
        if setup_df.empty:
            setup_df, _ = build_vsa_setup_dataset(DEFAULT_DATA_FILE)
    except Exception as e:
        st.error(str(e))
        return
    st.write("This is the dataset created by scanning the 2-year M5 bars with the VSA strategy. These rows are what LR and RF learn from.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Setup rows", len(setup_df))
    c2.metric("Valid/winning labels", int((setup_df["target"] == 1).sum()))
    c3.metric("Avoid/failed labels", int((setup_df["target"] == 0).sum()))
    st.dataframe(setup_df.tail(50), width="stretch")


def page_news():
    st.title("🗓️ USD Red-Folder News Calendar")

    terminal = _mt5_terminal_snapshot()
    detected_offset = terminal.get("broker_offset")
    raw_detected_offset = terminal.get("raw_broker_offset")
    default_offset = float(detected_offset) if detected_offset is not None else 3.0

    if detected_offset is None and raw_detected_offset not in (None, "", "nan"):
        st.caption(
            "MT5 broker-time auto-detection is temporarily unavailable, so the calendar is using the project's UTC+3 fallback. "
            "Change the value below only if your broker uses a different server offset."
        )

    top1, top2 = st.columns([1.3, 1.0])
    with top1:
        broker_offset = st.number_input(
            "Calendar time zone · broker UTC offset",
            min_value=-12.0,
            max_value=14.0,
            value=default_offset,
            step=1.0,
            help="If MT5 is connected, this starts from the broker offset detected by the exporter. Calendar times are converted into broker-server time.",
        )
    with top2:
        if st.button("↻ Refresh embedded news calendar", width="stretch"):
            fetch_forex_factory_calendar.clear()
            st.rerun()

    st.markdown(
        """
        <div class="ff-source-strip">
            <div><strong>USD HIGH-IMPACT ONLY</strong><br>
            The page filters the Forex Factory weekly calendar to the events traders commonly call <b>red-folder</b> USD news.</div>
            <div class="red-folder-badge">🔴 RED FOLDER</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df, verified, source = fetch_forex_factory_calendar(float(broker_offset))
    loaded_weeks = set(df.attrs.get("loaded_weeks", [])) if hasattr(df, "attrs") else set()
    missing_weeks = set(df.attrs.get("missing_weeks", [])) if hasattr(df, "attrs") else set()
    if verified:
        st.success("Live Forex Factory weekly export data are verified for the loaded calendar window.")
    else:
        fetch_error = df.attrs.get("fetch_error", "") if hasattr(df, "attrs") else ""
        missing_label = ", ".join(sorted(missing_weeks)) if missing_weeks else "export verification"
        st.warning(
            "The Forex Factory calendar is available for display, but one or more export checks are not fully verified. "
            "For safety, the trading news gate remains fail-closed until verified live coverage is available."
        )
        with st.expander("Technical calendar status", expanded=False):
            st.caption(f"Source: {source}")
            st.caption(f"Missing/unverified: {missing_label}")
            if fetch_error:
                st.code(fetch_error)

    if df.empty:
        st.error("The live Forex Factory calendar could not be verified. The app will not claim that there are zero high-impact events when calendar coverage is missing.")
        return

    calendar = _normalise_news_calendar(df)
    high = _high_impact_usd_calendar(calendar)
    now_broker = _broker_now(float(broker_offset))
    today = now_broker.normalize()
    current_week_start = today - pd.Timedelta(days=today.weekday())
    current_week_end = current_week_start + pd.Timedelta(days=6)
    next_week_start = current_week_start + pd.Timedelta(days=7)
    next_week_end = next_week_start + pd.Timedelta(days=6)

    coverage_start = df.attrs.get("coverage_start") if hasattr(df, "attrs") else None
    coverage_end = df.attrs.get("coverage_end") if hasattr(df, "attrs") else None
    retrieved = df.attrs.get("retrieved_at_utc") if hasattr(df, "attrs") else None
    info_parts = [f"Times shown in broker-server time UTC{float(broker_offset):+g}."]
    if coverage_start and coverage_end:
        info_parts.append(f"Feed coverage: {str(coverage_start)[:16]} to {str(coverage_end)[:16]}.")
    if retrieved:
        info_parts.append(f"Retrieved: {str(retrieved).replace('T', ' ')} UTC.")
    st.caption(" ".join(info_parts))

    if high.empty:
        if verified:
            st.info("Both weekly exports are verified and currently contain no USD high-impact / red-folder events. The calendar below is still shown so the empty days are explicit.")
        else:
            st.warning("No USD high-impact events are present in the data currently available, but one or more weekly exports are unverified. Missing coverage is not treated as 'no news'.")

    precise_upcoming = high[high["event_time"].notna() & (high["event_time"] >= now_broker)].copy()
    floating_upcoming = high[
        high["event_time"].isna()
        & high["event_date"].notna()
        & (high["event_date"] >= today)
    ].copy()
    upcoming_all = pd.concat([precise_upcoming, floating_upcoming], ignore_index=True).drop_duplicates(
        ["event_date", "time_label", "event"]
    ).sort_values(["event_date", "event_time", "time_label"], na_position="last")

    today_events = high[high["event_date"] == today].copy()
    remaining_this_week = upcoming_all[
        (upcoming_all["event_date"] >= today) & (upcoming_all["event_date"] <= current_week_end)
    ].copy()
    next_week_events = high[
        (high["event_date"] >= next_week_start) & (high["event_date"] <= next_week_end)
    ].copy()

    next_event_label = "Unverified" if not verified else "None scheduled"
    if not upcoming_all.empty:
        first = upcoming_all.iloc[0]
        first_day = pd.to_datetime(first.get("event_date"), errors="coerce")
        day_label = first_day.strftime("%a %d %b") if pd.notna(first_day) else ""
        next_event_label = f"{day_label} · {_clean_event_value(first.get('time_label')) or 'TBA'}"

    m1, m2, m3, m4 = st.columns(4)
    # Display parsed events even when the export verification is incomplete, but do
    # not present a zero count as authoritative when coverage itself is unverified.
    m1.metric("Red-folder events today", len(today_events) if (verified or not today_events.empty) else "—")
    m2.metric("Remaining this week", len(remaining_this_week) if (verified or not remaining_this_week.empty) else "—")
    m3.metric("Next week", len(next_week_events) if (verified or not next_week_events.empty) else "—")
    m4.metric("Next red-folder event", next_event_label)

    tab_today, tab_week, tab_next = st.tabs(["Today", "This week", "Next week"])

    with tab_today:
        st.markdown(f"### {today.strftime('%A, %d %B %Y')}")
        if today_events.empty:
            if verified:
                st.success("No high-impact USD / red-folder event is scheduled today.")
            else:
                st.info("No USD red-folder event is present in the currently loaded data for today; live export verification is incomplete.")
        else:
            for _, row in today_events.iterrows():
                st.markdown(_news_event_card(row), unsafe_allow_html=True)
        if not remaining_this_week.empty:
            st.caption(
                f"There are {len(remaining_this_week)} high-impact USD event(s) still scheduled from today through Sunday."
            )
        else:
            if verified:
                st.info("No further high-impact USD events are scheduled for the rest of this calendar week.")
            else:
                st.info("No further USD red-folder event is present in the loaded current-week data; verification remains incomplete.")

    with tab_week:
        _render_news_week_calendar(high, current_week_start, today, "Current Forex Factory week")
        if "this_week" in missing_weeks:
            st.error("Current-week export is not verified. Empty current-week days must not be interpreted as proof that no red-folder event exists.")
        elif remaining_this_week.empty:
            st.success("No high-impact USD / red-folder events are upcoming for the remainder of this verified calendar week.")
        else:
            st.warning(
                f"{len(remaining_this_week)} high-impact USD event(s) remain this week. "
                "The trading safety gate still applies its configured ±30-minute window around precise event times."
            )

    with tab_next:
        _render_news_week_calendar(high, next_week_start, today, "Next Forex Factory week")
        if "next_week" in missing_weeks:
            st.error("Next-week export is not verified. The app therefore will not label next week as having no red-folder events.")
        elif next_week_events.empty:
            st.success("No high-impact USD / red-folder events are listed in the verified next-week export.")
        else:
            st.info(f"{len(next_week_events)} high-impact USD event(s) are currently listed for next week.")

    st.markdown("### Upcoming USD high-impact events")
    upcoming_through_next_week = upcoming_all[
        upcoming_all["event_date"].notna() & (upcoming_all["event_date"] <= next_week_end)
    ].copy()
    if upcoming_through_next_week.empty:
        if verified:
            st.success("There are no upcoming USD high-impact / red-folder events through the end of next week.")
        else:
            st.info("No upcoming USD red-folder event was parsed from the currently loaded calendar; live export verification is incomplete.")
    else:
        table = _news_table(upcoming_through_next_week, now_broker)
        st.dataframe(table, width="stretch", hide_index=True)

    st.caption(
        "Source note: this page embeds the official Forex Factory / Fair Economy weekly export directly inside the application. "
        "Only USD events marked High impact are displayed. The calendar is awareness-only: it does not create a VSA setup, raise model probability or bypass the existing ±30-minute news-risk gate."
    )


def page_learning():
    st.title("🎓 Learning Hub")

    st.info(
        "This Learning Hub is designed for novice users. It explains the trading logic, "
        "the VSA strategy, the ML model, the news filter and the risk controls in beginner-friendly language."
    )

    st.markdown("## 1. Mini Lessons")

    with st.expander("Lesson 1: What is a candle?"):
        st.write(
            "A candle shows how price moved during one time period. In this project, the timeframe is M5, "
            "which means each candle represents five minutes of XAUUSD price movement."
        )
        st.markdown(
            """
            **A candle has four main values:**

            - **Open:** price at the start of the candle
            - **High:** highest price reached
            - **Low:** lowest price reached
            - **Close:** price at the end of the candle
            """
        )

    with st.expander("Lesson 2: What is tick volume?"):
        st.write(
            "Tick volume counts how many times the price changed during a candle. "
            "In retail gold/forex trading, true centralised exchange volume is not always available, "
            "so tick volume is used as a practical market-activity proxy in this prototype."
        )

    with st.expander("Lesson 3: What is Volume Spread Analysis?"):
        st.write(
            "Volume Spread Analysis, or VSA, compares candle size with volume. "
            "The idea is to look at the relationship between effort and result. "
            "A very large candle with high volume can show strong activity, but the direction and closing position "
            "help decide whether buyers or sellers are more likely in control."
        )

    with st.expander("Lesson 4: What is a CAB candle?"):
        st.write(
            "CAB means Climactic Action Bar. In this project, a CAB is a candle with unusually high volume "
            "and a wide spread compared with recent candles. It becomes the main zone used by the VSA strategy."
        )

    with st.expander("Lesson 5: How Scenario 1 works"):
        st.write(
            "Scenario 1 requires a first breakout of the CAB trigger, a return inside the CAB without sweeping the opposite trigger, "
            "and a second momentum breakout with volume below the CAB volume. It is a valid VSA candidate with 0.5% strategy risk; "
            "the ML model and news gate still decide whether the final output is TRADE_CANDIDATE or WATCH_ONLY."
        )

    with st.expander("Lesson 6: How Scenario 2 works"):
        st.write(
            "Scenario 2 is a confirmed candidate setup. It usually involves a sweep/reclaim style movement around the CAB zone. "
            "The bot still does not blindly trade it. The ML model must score the setup and news risk must be checked."
        )

    with st.expander("Lesson 7: How Scenario 3 works"):
        st.write(
            "Scenario 3 is another confirmed candidate setup. It is linked with breakout or continuation-style behaviour. "
            "The ML layer checks whether similar historical VSA Scenario 3 setups reached target before stop-loss."
        )

    with st.expander("Lesson 8: Why does the bot use ML with VSA?"):
        st.write(
            "The project does not use machine learning blindly. First, VSA detects possible trading setups. "
            "Then the machine learning model learns from two years of previous M5 VSA setup outcomes. "
            "This means the bot is trained to answer: which VSA setups historically worked better?"
        )

    with st.expander("Lesson 9: What does Logistic Regression do?"):
        st.write(
            "Logistic Regression is the explainable baseline/final candidate model. "
            "It gives weights to VSA features such as relative volume, candle spread, body ratio, close location, "
            "scenario type and direction. These weights help explain why the model supports or rejects a setup."
        )

    with st.expander("Lesson 10: What does Random Forest do?"):
        st.write(
            "Random Forest is trained as a comparison model. It can capture more complex non-linear relationships, "
            "but it is harder to explain to novice users. Both models are compared using the same validation criterion. "
            "LR won that stated criterion in this run, and its transparency is an additional deployment advantage rather than a hidden override."
        )

    with st.expander("Lesson 11: Why is news risk important for XAUUSD?"):
        st.write(
            "Gold can move sharply during high-impact USD news, such as CPI, NFP, interest-rate decisions and FOMC events. "
            "The bot highlights high-impact USD events in red and warns the user before taking signals during risky news periods."
        )

    with st.expander("Lesson 12: Why risk management matters more than signals"):
        st.write(
            "A trading signal can still lose. For that reason, the project uses risk controls such as confidence thresholds, "
            "stop-loss logic, position-size limits and news warnings. The bot is designed as a decision-support prototype, "
            "not a profit-guarantee system."
        )

    st.markdown("---")
    st.markdown("## 2. Blog: Why This Project Uses VSA + Machine Learning")

    st.markdown(
        """
        ### Why this bot does not use pure rule-based trading

        A basic trading bot can be built using fixed rules. For example, it can say: **if a VSA Scenario 2 setup appears, generate a BUY signal.**

        The problem is that financial markets are noisy. A setup that looks strong in one market condition may fail in another.
        This is why the final project improves the original VSA strategy by adding a machine-learning decision layer.

        ### The role of VSA

        VSA is still the foundation of the system. It gives the bot a trading idea based on price spread, volume and candle behaviour.
        Instead of looking at the chart randomly, the system first asks:

        **Has a meaningful VSA setup appeared?**

        If no setup appears, the bot does not force a trade. This is important because novice traders often overtrade when there is
        no clear structure.

        ### The role of machine learning

        After the VSA setup is detected, the ML model checks whether similar VSA setups worked in the two-year historical M5 dataset.
        The training data is not just raw candles. It is converted into VSA setup examples, including:

        - scenario type
        - direction
        - relative volume
        - candle spread
        - body ratio
        - close location
        - risk distance
        - time of day
        - future setup outcome

        The model is therefore trained to estimate the quality of a VSA setup, not simply to guess the next candle.

        ### Why LR and RF are compared

        Logistic Regression and Random Forest are trained separately on the same VSA setup dataset. Logistic Regression is easier to explain
        because its coefficients show which features increase or decrease the probability of a valid setup. Random Forest is useful as a
        comparison model because it can capture more complex patterns, but it is harder to explain.

        For a novice-facing financial advisor bot, accuracy alone is not enough. The final model must also be understandable and safe to present.
        This is why the project compares both models but also considers explainability before selecting the final advisor model.

        ### Why the website does not train the model

        The website is for non-technical users. A normal user should not train or modify the model. The model is trained in the backend using:

        `backend_training/train_models.py`

        After training, the website only loads the saved model and presents:

        - latest VSA setup evidence
        - model confidence
        - recommendation
        - risk warning
        - news filter status
        - MT5 signal export

        This matches the project brief because the final product is an advisor bot that the user can interact with once the training system has already produced a trained model.

        ### Why this is safer

        The final decision is not based on one isolated signal. It uses four checks:

        1. VSA setup detected
        2. ML model confidence checked
        3. news risk checked
        4. risk warning shown before MT5 alert or demo trade

        This makes the prototype more suitable for beginner education than a normal black-box trading bot.
        """
    )

    st.caption(
        "Source basis for this blog: Fidelity Learning Center explains technical analysis using price/volume/indicator evidence; "
        "Fidelity also discusses volume as important for chart-pattern analysis; CFA Institute describes quantitative active-investment "
        "as a process involving thesis definition, data cleaning, backtesting, evaluation, and risk/trading-cost controls."
    )

    st.markdown("### Source links used for this blog")
    st.markdown(
        """
        - [Fidelity: What is technical analysis?](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/what-is-technical-analysis)
        - [Fidelity: Why to consider stock volume](https://www.fidelity.com/viewpoints/active-investor/stock-volume)
        - [CFA Institute: Active Equity Investing Strategies](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/active-equity-investing-strategies)
        - [CFA Institute: Backtesting and Simulation](https://www.cfainstitute.org/insights/professional-learning/refresher-readings/2026/backtesting-and-simulation)
        """
    )

    st.markdown("---")
    st.markdown("## 3. Video Resources")

    st.write("These are learning links for users who want beginner-friendly explanations before using the prototype.")

    videos = [
        ("Candlestick basics for beginners", "https://www.youtube.com/results?search_query=candlestick+chart+basics+for+beginners"),
        ("Volume Spread Analysis basics", "https://www.youtube.com/results?search_query=volume+spread+analysis+for+beginners"),
        ("Trading risk management for beginners", "https://www.youtube.com/results?search_query=trading+risk+management+for+beginners"),
        ("MetaTrader 5 Expert Advisor basics", "https://www.youtube.com/results?search_query=metatrader+5+expert+advisor+beginner"),
        ("Backtesting trading strategies explained", "https://www.youtube.com/results?search_query=backtesting+trading+strategy+for+beginners"),
    ]

    for title, url in videos:
        st.markdown(f"- [{title}]({url})")

    st.markdown("---")
    st.markdown("## 4. Strategy Walkthrough")

    st.markdown(
        """
        **Final system flow:**

        1. A new M5 XAUUSD candle appears.
        2. The VSA scanner checks for CAB, sweep, breakout and scenario rules.
        3. If a VSA setup appears, the trained ML model scores it.
        4. The news calendar checks for high-impact USD risk.
        5. The advisor shows TRADE_CANDIDATE or WATCH_ONLY with BUY/SELL direction, probability and explanation.
        6. The system exports `ml_signal.csv`.
        7. MT5 reads the signal and can send alerts or place demo trades.
        """
    )

    st.markdown("---")
    st.markdown("## 5. Quick Quiz")

    q1 = st.radio(
        "What should a beginner do during high-impact USD news?",
        [
            "Increase risk because price moves fast",
            "Ignore the news completely",
            "Reduce or avoid new trades until conditions settle",
        ],
        index=None,
        key="quiz_news",
    )

    if q1 is not None:
        if q1 == "Reduce or avoid new trades until conditions settle":
            st.success("Correct. The project treats high-impact news as a risk condition.")
        else:
            st.warning("Not quite. High-impact news can create sharp and unpredictable movement in XAUUSD.")

    q2 = st.radio(
        "What is the role of VSA in this final project?",
        [
            "It randomly guesses price direction",
            "It creates explainable trading setup features for the ML model",
            "It replaces all machine learning",
        ],
        index=None,
        key="quiz_vsa",
    )

    if q2 is not None:
        if q2 == "It creates explainable trading setup features for the ML model":
            st.success("Correct. VSA provides the strategy and feature logic.")
        else:
            st.warning("Not quite. VSA is the strategy layer, and ML learns from VSA setup outcomes.")

    q3 = st.radio(
        "Where is the model trained?",
        [
            "Inside MT5",
            "Inside the user-facing website",
            "In the backend Python training script",
        ],
        index=None,
        key="quiz_training",
    )

    if q3 is not None:
        if q3 == "In the backend Python training script":
            st.success("Correct. The website loads the trained model after backend training has finished.")
        else:
            st.warning("Not quite. Training is backend-only; MT5 and the website use the saved model output.")


def page_faq_glossary():
    st.title("❓ FAQ and Trading Glossary")

    st.markdown("## Frequently Asked Questions")

    faqs = [
        (
            "Is this financial advice?",
            "No. This is an educational final-year project and decision-support prototype. It must not be used as real financial advice."
        ),
        (
            "Does the bot guarantee profit?",
            "No. The bot estimates the quality of a VSA setup using historical data. A historical pattern can still fail in live market conditions."
        ),
        (
            "What market does the bot focus on?",
            "The prototype focuses on XAUUSD, which means gold priced in US dollars."
        ),
        (
            "Why does the project use M5 data?",
            "M5 gives more examples than H1 because a new candle forms every five minutes. This gives the backend more VSA setups to train and test."
        ),
        (
            "Where is the model trained?",
            "The model is trained in the backend using the Python script backend_training/train_models.py. The website user does not train the model."
        ),
        (
            "Why should the website not train the model?",
            "The project brief expects the non-technical user to interact with the advisor bot once trained. Training is a backend development task, not a normal user task."
        ),
        (
            "How is the two-year CSV used?",
            "The CSV is loaded by the backend. The system scans historical M5 candles for VSA setups, labels their outcomes, then trains LR and RF on those VSA setup examples."
        ),
        (
            "What does VSA mean?",
            "VSA means Volume Spread Analysis. It compares candle spread, volume and closing position to understand effort versus result."
        ),
        (
            "What is a CAB candle?",
            "CAB means Climactic Action Bar. It is a wide-spread, high-volume candle that becomes an important VSA setup zone."
        ),
        (
            "What are Scenario 1, Scenario 2 and Scenario 3?",
            "They are three formal mirrored VSA candidate sequences. Scenario 1 uses 0.5% strategy risk, while Scenarios 2 and 3 use 1%; every candidate must still pass the trained ML threshold and news-risk gate."
        ),
        (
            "Is VSA replaced by machine learning?",
            "No. VSA remains the trading strategy and feature-engineering layer. Machine learning is trained on VSA setup outcomes."
        ),
        (
            "Why train LR and RF separately?",
            "They are different models. Training them separately allows fair comparison of their validation and test performance on the same VSA setup dataset."
        ),
        (
            "Which model is better?",
            "The better model is selected using validation/test evidence, but explainability also matters because the target users are beginners."
        ),
        (
            "Why is Logistic Regression easier to explain?",
            "Logistic Regression uses coefficients. These can show which VSA-derived features pushed the prediction towards or away from a trade."
        ),
        (
            "Why is Random Forest harder to explain?",
            "Random Forest combines many decision trees. It can model complex patterns, but its individual decision path is harder for beginners to understand."
        ),
        (
            "What does model confidence mean?",
            "Confidence is the model's estimated probability for the predicted class. It is not a guarantee that the trade will win."
        ),
        (
            "What does WATCH_ONLY mean?",
            "WATCH_ONLY means a VSA setup was scored but did not pass the ML threshold or was blocked by a safety condition such as unverified or high-impact news."
        ),
        (
            "How does the news filter work?",
            "The news page attempts to load an economic calendar and highlights high-impact USD events in red. If live data is unavailable, it uses a fallback CSV."
        ),
        (
            "Why are high-impact USD events important?",
            "XAUUSD can move sharply during USD events such as CPI, NFP, FOMC and interest-rate announcements."
        ),
        (
            "How does this connect to MT5?",
            "The trained model writes a signal to outputs/ml_signal.csv. The MT5 Expert Advisor reads this file and can send alerts or place demo trades."
        ),
        (
            "Does MT5 train the model?",
            "No. MT5 does not train the model. MT5 only reads the final signal and handles alerts or demo execution."
        ),
        (
            "Can the bot place real trades?",
            "The project should be demonstrated on demo mode only. Real-money execution is outside the safe scope of this university prototype."
        ),
        (
            "What is backtesting?",
            "Backtesting means testing a strategy on historical data to understand how it would have behaved. It is useful evidence, but it is not proof of future profit."
        ),
        (
            "What is chronological splitting?",
            "Chronological splitting means the model is trained on earlier data and tested on later data. This is more suitable for time-series trading data than random splitting."
        ),
        (
            "Why not use random train-test split?",
            "Random splitting can leak future market conditions into training. Chronological split better reflects the way a trading model would face unseen future data."
        ),
        (
            "What is the main limitation?",
            "The system is trained on historical data. Market behaviour can change, and past results do not guarantee future performance."
        ),
        (
            "What screenshots should be used in the report?",
            "Use screenshots of backend terminal training, VSA setup dataset, LR vs RF comparison, final advisor recommendation, news risk filter, MT5 CSV bridge and feedback form."
        ),
        (
            "Why is this suitable for novice users?",
            "The bot explains setup evidence, model confidence, news risk and risk controls instead of only showing a buy/sell signal."
        ),
        (
            "What should a user do if confidence is low?",
            "They should treat the setup as watch-only or avoid trading. Low confidence means the model has not found strong enough evidence from similar historical setups."
        ),
        (
            "What happens if live news cannot load?",
            "The app shows a fallback news CSV and clearly states that the live feed was unavailable. This avoids silently pretending that live news was checked."
        ),
    ]

    for q, a in faqs:
        with st.expander(q):
            st.write(a)

    st.markdown("---")
    st.markdown("## Glossary for Novice Traders")

    glossary = pd.DataFrame(
        [
            ["XAUUSD", "Gold priced in US dollars."],
            ["M5", "Five-minute timeframe. One candle forms every five minutes."],
            ["OHLC", "Open, High, Low and Close candle values."],
            ["Candle", "A chart block showing price movement during one period."],
            ["Open", "The price at the start of the candle."],
            ["High", "The highest price reached during the candle."],
            ["Low", "The lowest price reached during the candle."],
            ["Close", "The final price at the end of the candle."],
            ["Tick volume", "The number of price updates during a candle."],
            ["Spread / Range", "The distance between the candle high and low."],
            ["Body", "The distance between candle open and close."],
            ["Wick", "The thin part of the candle showing rejection beyond the body."],
            ["VSA", "Volume Spread Analysis, comparing spread and volume."],
            ["CAB", "Climactic Action Bar; a high-volume, wide-spread candle."],
            ["Scenario 1", "First breakout, return inside without opposite sweep, then a second lower-volume momentum breakout; 0.5% strategy risk."],
            ["Scenario 2", "Confirmed VSA candidate setup after sweep/reclaim behaviour."],
            ["Scenario 3", "Confirmed VSA candidate setup after breakout/continuation behaviour."],
            ["Sweep", "Price briefly breaks a level and then returns back through it."],
            ["Breakout", "Price closes beyond an important level."],
            ["Entry", "The price where a demo trade would start."],
            ["Stop-loss", "A safety exit level used to limit loss."],
            ["Take-profit", "A target level where profit is taken."],
            ["Risk", "The amount that could be lost if the trade fails."],
            ["Reward", "The amount that could be gained if the trade succeeds."],
            ["Risk/reward", "Comparison between possible loss and possible gain."],
            ["Win rate", "Percentage of trades that closed profitably."],
            ["Profit factor", "Gross profit divided by gross loss."],
            ["Drawdown", "A fall from a previous account peak."],
            ["Backtest", "Testing a strategy on historical data."],
            ["Validation data", "Data used to tune or compare models before final testing."],
            ["Test data", "Unseen data used for final evaluation."],
            ["Chronological split", "Train/validation/test split that keeps time order."],
            ["Logistic Regression", "Explainable ML model that uses coefficients."],
            ["Random Forest", "Tree-based ML model used as a comparison benchmark."],
            ["Feature", "An input variable used by the model, such as relative volume or candle spread."],
            ["Label", "The outcome the model learns to predict, such as valid setup or failed setup."],
            ["Confidence", "The model's estimated probability for its prediction."],
            ["WATCH_ONLY", "A scored setup that is not eligible for demo execution because ML evidence or a safety check did not pass."],
            ["News filter", "A safety check for high-impact market events."],
            ["Forex Factory", "Economic calendar source often used by traders to track news events."],
            ["MT5", "MetaTrader 5, the trading platform used for alerts/demo execution."],
            ["Expert Advisor", "An MT5 program that can automate alerts or demo trades."],
            ["CSV bridge", "A file-based connection where Python writes a signal and MT5 reads it."],
        ],
        columns=["Term", "Meaning"],
    )

    st.dataframe(glossary, width="stretch")

    st.download_button(
        "Download glossary as CSV",
        glossary.to_csv(index=False).encode("utf-8"),
        file_name="vsa_trading_glossary.csv",
        mime="text/csv",
    )


def page_trade_journal():
    st.title("📓 Trade Journal")
    st.write(
        "Keep a learning record of advisor signals and your own demo/practice trades. "
        "Journal entries are local reflections only: they never place an order, alter the VSA rules or retrain the ML model."
    )
    st.info("Self-reported journal results stay separate from the validated backtest and evaluation evidence.")

    tab_saved, tab_manual = st.tabs(["Journal history", "Add manual trade"])

    with tab_manual:
        st.markdown("### Record a demo/practice trade")
        st.caption("Use this when you want to journal a trade that was not saved directly from a VSA Advisor signal.")
        with st.form("manual_trade_journal_form"):
            c1, c2, c3 = st.columns(3)
            trade_date = c1.date_input("Trade date")
            trade_time = c2.time_input("Trade time")
            direction = c3.selectbox("Direction", ["BUY", "SELL"])
            scenario_label = st.selectbox("VSA scenario", ["Not linked to a VSA scenario", "Scenario 1", "Scenario 2", "Scenario 3"])
            p1, p2, p3, p4 = st.columns(4)
            entry_text = p1.text_input("Entry price", placeholder="optional")
            stop_text = p2.text_input("Stop-loss", placeholder="optional")
            tp1_text = p3.text_input("1R target", placeholder="optional")
            tp4_text = p4.text_input("4R target", placeholder="optional")
            r1, r2 = st.columns(2)
            risk_text = r1.text_input("Risk %", placeholder="optional")
            result_text = r2.text_input("Realised result in R", placeholder="leave blank if unresolved")
            decision = st.selectbox("What did you do?", ["Demo-traded", "Watched", "Skipped"])
            notes = st.text_area("Trade notes", placeholder="Why did you take/skip it? What did you observe?")
            lesson = st.text_area("Lesson learned", placeholder="What would you repeat or change next time?")
            manual_saved = st.form_submit_button("Save manual journal entry", type="primary")
        if manual_saved:
            try:
                def _optional_float(value: str):
                    return None if not value.strip() else float(value.strip())
                scenario_id = 0 if scenario_label.startswith("Not linked") else int(scenario_label[-1])
                stamp = f"{trade_date.isoformat()} {trade_time.strftime('%H:%M:%S')}"
                add_manual_journal_entry(
                    trade_time=stamp,
                    direction=direction,
                    scenario_id=scenario_id,
                    entry_price=_optional_float(entry_text),
                    stop_price=_optional_float(stop_text),
                    target_1r=_optional_float(tp1_text),
                    target_4r=_optional_float(tp4_text),
                    risk_percent=_optional_float(risk_text),
                    user_decision=decision,
                    result_r=_optional_float(result_text),
                    notes=notes,
                    lesson_learned=lesson,
                )
                st.success("Manual trade saved to your journal.")
            except ValueError as exc:
                st.error(f"Please check the numeric fields: {exc}")
            except Exception as exc:
                st.error(f"Could not save the manual journal entry: {exc}")

    with tab_saved:
        journal = read_journal()
        if journal.empty:
            st.warning("Your journal is empty. Save a recommendation from Live Advisor or use the Add manual trade tab.")
            return

        numeric_results = pd.to_numeric(journal["result_r"], errors="coerce")
        j1, j2, j3, j4 = st.columns(4)
        j1.metric("Journal entries", len(journal))
        j2.metric("Watched", int((journal["user_decision"].astype(str) == "Watched").sum()))
        j3.metric("Skipped", int((journal["user_decision"].astype(str) == "Skipped").sum()))
        j4.metric("Resolved demo results", int(numeric_results.notna().sum()))

        display_columns = [
            "signal_id", "signal_time", "source_mode", "scenario_id", "direction", "advisor_output",
            "model_probability", "risk_percent", "user_decision", "result_r", "notes", "lesson_learned",
        ]
        display = journal[display_columns].copy()
        display["model_probability"] = pd.to_numeric(display["model_probability"], errors="coerce").map(
            lambda x: f"{x:.1%}" if pd.notna(x) else ""
        )
        st.dataframe(display.sort_values("signal_time", ascending=False), width="stretch", hide_index=True)

        st.download_button(
            "Download my journal as CSV",
            journal.to_csv(index=False).encode("utf-8"),
            file_name="vsa_learning_trade_journal.csv",
            mime="text/csv",
        )

        st.markdown("### Update a journal reflection")
        selected_id = st.selectbox("Signal", journal["signal_id"].astype(str).tolist())
        selected = journal.loc[journal["signal_id"].astype(str) == selected_id].iloc[-1]
        decisions = ["Watched", "Skipped", "Demo-traded"]
        current_decision = str(selected.get("user_decision", "Watched"))
        decision_index = decisions.index(current_decision) if current_decision in decisions else 0
        existing_result = selected.get("result_r", "")
        result_default = "" if pd.isna(existing_result) else str(existing_result)
        if result_default.lower() == "nan":
            result_default = ""
        with st.form(f"journal_update_{selected_id}"):
            decision = st.selectbox("Decision", decisions, index=decision_index)
            result_text = st.text_input("Realised result in R", value=result_default, help="Leave blank while unresolved.")
            notes = st.text_area("Notes", value="" if pd.isna(selected.get("notes")) else str(selected.get("notes", "")))
            lesson = st.text_area(
                "Lesson learned",
                value="" if pd.isna(selected.get("lesson_learned")) else str(selected.get("lesson_learned", "")),
            )
            saved = st.form_submit_button("Update reflection")
        if saved:
            try:
                result_r = None
                clear_result = not result_text.strip()
                if result_text.strip():
                    result_r = float(result_text.strip())
                update_journal_reflection(
                    selected_id,
                    user_decision=decision,
                    result_r=result_r,
                    clear_result=clear_result,
                    notes=notes,
                    lesson_learned=lesson,
                )
                _read_csv_cached.clear()
                st.success("Journal reflection updated.")
            except ValueError as exc:
                st.error(str(exc))
            except Exception as exc:
                st.error(f"Could not update the journal: {exc}")
def page_feedback():
    st.title("📝 Project Feedback Form")
    st.write("This form asks opinions about the project idea and bot. It does not collect financial details and does not ask users to trade.")
    with st.form("feedback"):
        role = st.selectbox("What best describes you?", ["Novice trader", "Interested in trading", "Computer science student", "Finance/business student", "General user", "Other"])
        risk_clear = st.slider("The bot clearly explains risk and no-profit guarantee.", 1, 5, 4)
        useful = st.slider("This project idea would be useful for learning/demo trading.", 1, 5, 4)
        choose = st.slider("I would consider using this as an educational/demo tool, not real-money advice.", 1, 5, 4)
        confusing = st.text_area("What part seems confusing or risky?")
        improve = st.text_area("What should be improved before real users use it?")
        submitted = st.form_submit_button("Save feedback locally")
    if submitted:
        row = pd.DataFrame([{"time": datetime.now(), "role": role, "risk_clear": risk_clear, "useful": useful, "would_choose_demo": choose, "confusing": confusing, "improve": improve}])
        p = RESULTS_DIR / "project_feedback_local.csv"
        if p.exists():
            old = pd.read_csv(p)
            row = pd.concat([old, row], ignore_index=True)
        row.to_csv(p, index=False)
        st.success("Feedback saved locally for project evidence.")
    df = read_csv(RESULTS_DIR / "project_feedback_local.csv")
    if not df.empty:
        st.caption(f"{len(df)} local feedback response(s) are stored privately for the researcher. Individual responses are not displayed in the public interface.")


def page_feature_table():
    st.title("✅ Feature Table")
    features = pd.DataFrame([
        ["Backend ML training", "Completed", "train_models.py trains LR+VSA and RF+VSA separately"],
        ["Formal VSA state machine", "Completed", "Mirrored Scenario 1/2/3 rules and scenario-specific stops"],
        ["Managed-exit labelling", "Completed", "70% at 1R; 30% runner to 4R with breakeven stop"],
        ["Model comparison", "Completed", "Purged train/validation/test metrics saved separately"],
        ["Baselines and robustness", "Completed", "VSA-only, majority, walk-forward, ablation and cost sensitivity"],
        ["ML advisor", "Completed", "Website loads selected pre-trained model"],
        ["Live source-state verification", "Completed", "LIVE / HISTORICAL DEMO / expired states plus worker heartbeat and broker-clock candle freshness"],
        ["Candlestick decision chart", "Completed", "Same completed M5 candles with CAB zone, setup marker, entry, stop, 1R/4R and tick volume"],
        ["News decision gate", "Completed", "Verified high-impact USD risk blocks trades; unrelated calendar coverage or unavailable live data fails safe"],
        ["Learning Hub", "Completed", "Mini lessons, videos, walkthrough, quiz"],
        ["Learning trade journal", "Completed", "Persistent local signal snapshots, user decisions, notes and optional self-reported R results"],
        ["FAQ/glossary", "Completed", "Beginner trading terminology"],
        ["MT5 bridge", "Completed and runtime-tested", "CSV alerts, mobile notification, stale rejection and empty Trade evidence"],
        ["Live M5 exporter", "Completed and runtime-tested", "2,500 completed bars exported from a second XAUUSD M5 chart"],
        ["Live LR worker", "Completed", "New-setup-only scoring, duplicate suppression, news gate and local LR explanation"],
        ["Formal XAI evaluation", "Completed", "All LR feature contributions plus intercept reconstruct unseen probabilities within 1e-12"],
        ["Core-feature validation", "Completed", "10 versus 24 features on held-out, walk-forward and 1,000-resample bootstrap evidence"],
        ["Other model-family comparison", "Completed", "Decision Tree and Gradient Boosting context benchmarks plus documented inclusion/exclusion rationale"],
        ["MT5 EA comparison", "Completed", "Five saved 100%-real-tick HTML reports: VSA+LR, VSA-only, MACD, MAMA and MAPSAR under matched settings"],
    ], columns=["Feature", "Status", "Evidence"])
    st.dataframe(features, width="stretch")


def page_mt5():
    st.title("🔗 MT5 Connection")
    st.write(
        "Connect the advisor to the MetaTrader 5 terminal already logged in to your broker account. "
        "Your broker password is never entered into this website or Python project: MT5 keeps the account session, "
        "exports completed XAUUSD M5 candles, and reads the advisor CSV from its Common Files folder."
    )
    st.warning("Keep demo execution disabled while connecting and testing. The connection is sufficient for live candles, alerts and evidence without placing a trade.")

    common_dir = discover_mt5_common()
    bars_path = common_dir / "xauusd_m5_bars.csv"
    signal_path = common_dir / "ml_signal.csv"
    connection_path = common_dir / "vsa_mt5_connection_status.csv"
    raw_broker_offset = read_broker_offset_from_status(common_dir)
    broker_offset = _safe_broker_offset(raw_broker_offset)

    st.markdown("### Connection status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MT5 Common Files", "FOUND" if common_dir.exists() else "NOT FOUND")
    c2.metric("Live M5 export", "FOUND" if bars_path.exists() else "WAITING")
    c3.metric("Python → MT5 signal", "FOUND" if signal_path.exists() else "WAITING")
    c4.metric("Broker UTC offset", f"UTC{broker_offset:+g}" if broker_offset is not None else "WAITING")
    if broker_offset is None and raw_broker_offset is not None:
        st.warning(
            f"The exporter reported an invalid broker UTC offset ({raw_broker_offset}). "
            "The value is ignored until a valid offset is available."
        )
    st.caption("Detected Common Files location")
    st.code(str(common_dir), language="text")

    if connection_path.exists():
        connection = read_csv(connection_path)
        if not connection.empty:
            latest = connection.tail(1).copy()
            st.dataframe(latest, hide_index=True, width="stretch")
            connected_value = str(latest.iloc[-1].get("terminal_connected", "0")).strip().lower()
            if connected_value in {"1", "true", "1.0"}:
                st.success("MT5 terminal reports a broker connection and the exporter is writing connection evidence.")
            else:
                st.error(
                    "MT5 is open, but it is NOT connected to the broker server. In MT5 use File → Login to Trade Account, "
                    "select the correct broker server and wait until the bottom-right network indicator becomes active and traffic is non-zero."
                )
                st.info(
                    "A chart can still display old cached candles while MT5 is disconnected. The project deliberately treats those candles as stale instead of calling them live."
                )
    else:
        st.info("No MT5 connection-status file yet. Attach the bar exporter EA in MT5; this page will then detect the broker clock automatically.")

    if bars_path.exists():
        try:
            live_bars = read_mt5_bars(bars_path, tail_rows=5)
            if not live_bars.empty:
                st.markdown("#### Latest completed candles received from MT5")
                st.dataframe(live_bars.tail(5), hide_index=True, width="stretch")
        except Exception as exc:
            st.error(f"The MT5 candle file exists but could not be read: {exc}")

    st.markdown("### 1. Download / install the MT5 components")
    downloads = [
        ("M5 bar exporter source (.mq5)", PROJECT_ROOT / "mt5" / "VSA_M5_Bar_Exporter_EA.mq5", "text/plain"),
        ("Visual indicator source (.mq5)", PROJECT_ROOT / "mt5" / "VSA_Live_Visual_Indicator.mq5", "text/plain"),
        ("Visual indicator compiled (.ex5)", PROJECT_ROOT / "mt5" / "VSA_Live_Visual_Indicator.ex5", "application/octet-stream"),
        ("Signal bridge source (.mq5)", PROJECT_ROOT / "mt5" / "VSA_ML_Signal_Bridge_EA.mq5", "text/plain"),
        ("Signal bridge compiled (.ex5)", PROJECT_ROOT / "mt5" / "VSA_ML_Signal_Bridge_EA.ex5", "application/octet-stream"),
    ]
    download_columns = st.columns(2)
    for index, (label, file_path, mime_type) in enumerate(downloads):
        if file_path.exists():
            download_columns[index % 2].download_button(
                label=label, data=file_path.read_bytes(), file_name=file_path.name, mime=mime_type,
                key=f"download_{file_path.name}", width="stretch",
            )

    st.markdown("### 2. Connect MT5 to your broker")
    st.markdown(
        "1. Open **MetaTrader 5** and log in to your existing broker account inside MT5. Use a **demo account for project testing**.  "
        "\n2. Open **XAUUSD** and set the chart to **M5**. If your broker uses a suffix such as `XAUUSD.a`, that is supported.  "
        "\n3. Open **MetaEditor** from MT5. Put `VSA_M5_Bar_Exporter_EA.mq5` and `VSA_ML_Signal_Bridge_EA.mq5` in the Experts folder and compile them.  "
        "\n4. Open **two XAUUSD M5 charts**. Attach the **bar exporter** to the first and the **signal bridge** to the second. MT5 allows one EA per chart.  "
        "\n5. In the bridge inputs keep **EnableDemoTrading = false**. Desktop alerts may remain enabled.  "
        "\n6. Turn on MT5 **Algo Trading** so the EAs can run. With demo trading disabled, the bridge still reads and alerts but will not open an order."
    )

    st.markdown("### 3. Start the Python live worker")
    st.write("After the exporter has been attached for 10–20 seconds, return to Terminal in this project folder and run:")
    st.code("./run_live_advisor.sh", language="bash")
    st.caption(
        "This final build automatically reads the broker UTC offset from the exporter. "
        "You only need --broker-utc-offset if you intentionally want to override that detected value."
    )

    st.markdown("### 4. Verify the connection")
    checklist = pd.DataFrame([
        ["MT5 broker session", connection_path.exists(), "Exporter writes vsa_mt5_connection_status.csv"],
        ["Completed XAUUSD M5 candles", bars_path.exists(), "Exporter writes xauusd_m5_bars.csv"],
        ["Saved LR model", (MODELS_DIR / "selected_model.joblib").exists(), "Python inference artifact"],
        ["Signal returned to MT5", signal_path.exists(), "Python writes ml_signal.csv after a fresh VSA setup"],
    ], columns=["Check", "Detected", "Evidence"])
    st.dataframe(checklist, hide_index=True, width="stretch")
    st.info(
        "It is normal for ml_signal.csv to remain absent until a fresh Scenario 1/2/3 setup appears. "
        "The strongest proof that the feed itself is connected is a fresh connection-status file plus continuously updating completed M5 candles."
    )

    st.markdown("### Safety boundary")
    st.code(
        "MT5 account login stays inside MetaTrader 5\n"
        "MT5 exporter → completed M5 candles → Python VSA + saved LR model\n"
        "Python → ml_signal.csv → MT5 bridge\n"
        "EnableDemoTrading = false during connection/testing",
        language="text",
    )

def page_live_mt5_proof():
    st.title("🟢 Live MT5 End-to-End Proof")
    st.write(
        "This page exposes the complete deployed route. Historical bars train the saved model offline; "
        "new completed MT5 bars are used for VSA detection and model inference, not automatic retraining."
    )
    st.code("""OFFLINE: data/bars.csv → labelled VSA setups → LR/RF fit → selected_model.joblib
LIVE: MT5 CopyRates → xauusd_m5_bars.csv → Python VSA scanner → 24 features
      → selected_model.joblib.predict_proba → ml_signal.csv → MT5 bridge""", language="text")

    bars_path = DEFAULT_MT5_COMMON / "xauusd_m5_bars.csv"
    bridge_path = DEFAULT_MT5_COMMON / "ml_signal.csv"
    parity_path = DEFAULT_MT5_COMMON / "vsa_mql5_latest_audit.csv"
    status_path = OUTPUTS_DIR / "live_advisor_status.json"
    model_path = MODELS_DIR / "selected_model.joblib"
    metadata = load_model_metadata()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MT5 live bars file", "FOUND" if bars_path.exists() else "MISSING")
    c2.metric("Trained model artifact", "FOUND" if model_path.exists() else "MISSING")
    c3.metric("Selected model", metadata.get("selected_model", "n/a"))
    c4.metric("Deployed ML call", "predict_proba()")

    broker_offset = st.number_input(
        "Broker server UTC offset for live cycle",
        min_value=-12.0, max_value=14.0, value=0.0, step=1.0,
        key="live_proof_broker_offset",
    )
    if st.button("Run one verified live MT5 cycle", type="primary"):
        with st.spinner("Reading completed MT5 bars, running VSA, and applying the saved LR model..."):
            status = run_live_cycle(
                bars_path=bars_path,
                project_signal_path=OUTPUTS_DIR / "ml_signal.csv",
                bridge_signal_path=bridge_path,
                status_path=status_path,
                broker_utc_offset_hours=float(broker_offset),
            )
            _read_json_cached.clear()
            _read_csv_cached.clear()
        if status.get("state") == "NEW_SETUP_SCORED":
            st.success("A new live VSA setup was scored by the saved LR model and synchronized to MT5.")
        elif status.get("state") in {"NO_NEW_VSA_SETUP", "DUPLICATE_SETUP_SKIPPED"}:
            st.info(status.get("message", status.get("state")))
        else:
            st.error(status.get("message", status.get("state", "Live cycle failed safely.")))

    status = read_json(status_path)
    signal = read_csv(OUTPUTS_DIR / "ml_signal.csv")
    if status:
        st.markdown("### Verifiable live-cycle audit")
        proof = {
            "State": status.get("state"),
            "Bars source": status.get("bars_path"),
            "Completed bars read": status.get("bars_read", 0),
            "Latest closed bar": status.get("latest_closed_bar"),
            "VSA scanner executed": status.get("vsa_scanner_executed", False),
            "ML inference executed": status.get("ml_inference_executed", False),
            "Retraining executed": status.get("retraining_executed", False),
            "Model artifact": status.get("model_artifact_used", status.get("model_path")),
            "ML method": status.get("model_method_used", "predict_proba when a new setup exists"),
            "Signal ID": status.get("signal_id", "No fresh setup"),
        }
        st.dataframe(pd.DataFrame(proof.items(), columns=["Evidence", "Value"]), hide_index=True, width="stretch")
    if not signal.empty:
        latest = signal.iloc[-1].to_dict()
        st.markdown("### Same canonical signal consumed by website and MT5")
        st.caption(f"Shared signal ID: {make_signal_id(latest)}")
        visible = [
            "signal_time", "cab_time", "symbol", "timeframe", "selected_model", "direction",
            "scenario_id", "ml_probability_valid", "recommendation_threshold", "advisor_output",
            "entry_price", "stop_price", "target_price", "runner_target_price", "risk_percent",
            "news_status", "news_reason",
        ]
        st.dataframe(signal[[column for column in visible if column in signal.columns]], width="stretch")

    st.markdown("### Python ↔ MQL5 VSA parity evidence")
    parity = read_csv(parity_path)
    if parity.empty:
        st.warning(
            "No MQL5 parity audit is available yet. Compile and attach VSA_Live_Visual_Indicator.mq5 to the XAUUSD M5 chart; "
            "it writes vsa_mql5_latest_audit.csv from its independent rule mirror."
        )
    else:
        st.dataframe(parity, width="stretch")
        if not signal.empty:
            python_row = signal.iloc[-1]
            mql_row = parity.iloc[-1]
            comparisons = {
                "signal_time": str(python_row.get("signal_time", "")).replace("-", ".") == str(mql_row.get("signal_time", "")),
                "cab_time": str(python_row.get("cab_time", "")).replace("-", ".") == str(mql_row.get("cab_time", "")),
                "direction": str(python_row.get("direction", "")) == str(mql_row.get("direction", "")),
                "scenario_id": int(float(python_row.get("scenario_id", 0))) == int(float(mql_row.get("scenario_id", 0))),
            }
            parity_table = pd.DataFrame(
                [{"Field": field, "Match": passed} for field, passed in comparisons.items()]
            )
            st.dataframe(parity_table, hide_index=True, width="stretch")
            if all(comparisons.values()):
                st.success("PASS: Python and MQL5 identify the same latest CAB setup and VSA scenario.")
            else:
                st.error("FAIL: Python and MQL5 VSA evidence differs. Do not present or trade this signal until investigated.")

    st.markdown("### What changes and what does not")
    st.info(
        "Live MT5 candles affect the current feature vector and prediction. They do not silently modify LR coefficients, "
        "RF trees, or thresholds. Retraining remains a controlled offline operation followed by validation and versioned deployment."
    )


def page_how_bot_learns():
    st.title("🧠 How the Bot Learns")
    st.write("A short, non-technical explanation of how historical data becomes live advice.")
    meta = load_model_metadata()
    st.code("""Two years of completed XAUUSD M5 candles
        ↓
Find historical VSA Scenario 1/2/3 setups
        ↓
Describe each setup using market and candle measurements
        ↓
Label whether 1R was reached before the protective stop
        ↓
Train Logistic Regression and Random Forest separately
        ↓
Select the better required model using validation evidence
        ↓
Save the trained model
        ↓
Use it to score a new live MT5 setup""", language="text")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Historical M5 bars", f"{int(meta.get('rows_raw', 0)):,}" if meta.get("rows_raw") else "n/a")
    c2.metric("Labelled VSA setups", meta.get("vsa_setup_rows", "n/a"))
    c3.metric("Selected learner", meta.get("selected_model", "n/a"))
    c4.metric("Decision threshold", metric_fmt(meta.get("recommendation_threshold", "n/a"), 2))

    st.markdown("### What the model predicts")
    st.success(
        "For a newly confirmed VSA setup, the trained model estimates the probability that price reaches the 1R objective "
        "before the scenario's protective stop. It does not guarantee the next market movement."
    )
    st.markdown("### Why rules and machine learning are both present")
    st.write(
        "VSA defines when a meaningful candidate exists and where its protective stop belongs. Logistic Regression learns "
        "from historical successful and failed candidates and decides whether a new candidate is strong enough to advise."
    )
    st.table(pd.DataFrame([
        ["VSA scanner", "Does a formal Scenario 1, 2 or 3 candidate exist?"],
        ["Trained LR model", "How likely was this kind of setup to reach its objective historically?"],
        ["Risk policy", "What entry, stop, targets and maximum capital risk should be shown?"],
    ], columns=["Part", "Job"]))


def page_why_this_advice():
    st.title("🔎 Why This Advice?")
    signal = read_csv(OUTPUTS_DIR / "ml_signal.csv")
    status = read_json(OUTPUTS_DIR / "live_advisor_status.json")
    if signal.empty:
        st.warning("No saved model advice is available yet. Run the live advisor from the Live Advisor page.")
        return
    row = signal.iloc[-1].to_dict()
    probability = float(row.get("ml_probability_valid", 0.0))
    threshold = float(row.get("recommendation_threshold", 0.0))
    st.caption(f"Signal ID: {make_signal_id(row)}")
    c1, c2, c3 = st.columns(3)
    c1.metric("VSA candidate", f"Scenario {int(float(row.get('scenario_id', 0)))} {row.get('direction', '')}")
    c2.metric("Trained LR probability", f"{probability:.1%}")
    c3.metric("Required threshold", f"{threshold:.1%}")

    if probability >= threshold:
        st.success("The trained model probability passes the required threshold. The setup can proceed to safety checks.")
    else:
        st.warning("The VSA setup exists, but the trained model rejects it as insufficiently strong: WATCH ONLY.")

    st.markdown("### Two different explanations")
    st.table(pd.DataFrame([
        ["VSA explanation", "Why the candle sequence qualifies as Scenario 1, 2 or 3."],
        ["ML explanation", "Which measured feature values increased or reduced the trained LR probability."],
    ], columns=["Explanation", "Meaning"]))

    explanations = status.get("lr_local_explanation", []) if status else []
    if explanations and status.get("signal_id") == make_signal_id(row):
        explanation_frame = pd.DataFrame(explanations)
        supporting = explanation_frame[explanation_frame["log_odds_contribution"] >= 0].head(3)
        opposing = explanation_frame[explanation_frame["log_odds_contribution"] < 0].head(3)
        left, right = st.columns(2)
        with left:
            st.markdown("#### Strongest supporting factors")
            st.dataframe(supporting[["feature", "effect"]], hide_index=True, width="stretch")
        with right:
            st.markdown("#### Strongest opposing factors")
            st.dataframe(opposing[["feature", "effect"]], hide_index=True, width="stretch")
    else:
        st.info("A fresh live setup will show its strongest supporting and opposing LR factors here.")


def page_risk_safety():
    st.title("🛡️ Risk & Safety")
    signal = read_csv(OUTPUTS_DIR / "ml_signal.csv")
    latest = signal.iloc[-1].to_dict() if not signal.empty else {}
    default_risk = float(latest.get("risk_percent", 1.0) or 1.0)
    balance = st.number_input("Demo account balance", min_value=100.0, value=10_000.0, step=100.0)
    risk_percent = st.number_input(
        "Maximum capital risk (%)", min_value=0.1, max_value=1.0,
        value=min(max(default_risk, 0.1), 1.0), step=0.1,
    )
    risk_money = balance * risk_percent / 100.0
    c1, c2 = st.columns(2)
    c1.metric("Maximum money at risk", f"{risk_money:,.2f}")
    c2.metric("Capital remaining after full planned loss", f"{balance-risk_money:,.2f}")
    st.warning("A stop-loss limits the planned loss but cannot guarantee execution at the exact price during gaps or extreme volatility.")
    st.markdown("### Safety controls")
    st.write("- Completed candles only; the forming M5 candle is excluded.\n- Stale and duplicate signals are rejected.\n- High-impact USD news can force WATCH ONLY.\n- One-position protection is available.\n- Automatic demo trading is disabled by default.\n- Real-money automation is outside the submitted scope.")


def page_about_bot():
    st.title("ℹ️ About the Bot")
    st.write(
        "The VSA Advisor Bot is a learning and decision-support tool for novice XAUUSD traders. It combines a formal "
        "Volume Spread Analysis setup detector with a trained Logistic Regression probability model."
    )
    st.markdown("### What you receive")
    st.write("- BUY, SELL or WATCH ONLY advice.\n- Entry, protective stop, 1R and 4R targets.\n- Maximum strategy risk.\n- A trained-model probability.\n- A short VSA and ML explanation.\n- Matching MT5 desktop and configured mobile notifications.")
    st.markdown("### Important limitations")
    st.write(
        "The model learned from one XAUUSD M5 dataset and broker tick volume. Market conditions can change, transaction costs "
        "can reduce results, and a historical relationship is not a promise of future profit."
    )
    st.error("Educational and demo use only. This is not personal financial advice.")



page = sidebar()
if page == "Live Advisor": page_recommendation()
elif page == "News Risk Filter": page_news()
elif page == "How the Bot Learns": page_how_bot_learns()
elif page == "Why This Advice?": page_why_this_advice()
elif page == "Risk & Safety": page_risk_safety()
elif page == "Learning Hub": page_learning()
elif page == "Trade Journal": page_trade_journal()
elif page == "MT5 Connection": page_mt5()
elif page == "About the Bot": page_about_bot()
elif page == "Feedback": page_feedback()
elif page == "Technical: Backend Training": page_training_evidence()
elif page == "Technical: LR vs RF": page_model_comparison()
elif page == "Technical: XAI Audit": page_xai()
elif page == "Technical: Testing": page_testing_evaluation()
elif page == "Technical: Algorithm": page_algorithm()
elif page == "Technical: Dataset": page_vsa_dataset()
elif page == "Technical: Feature Table": page_feature_table()
elif page == "Technical: Live MT5 Proof": page_live_mt5_proof()
