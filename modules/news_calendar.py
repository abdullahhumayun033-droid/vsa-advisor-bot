"""Economic-calendar retrieval and verification support for the VSA Advisor Bot.

This module obtains the Forex Factory/Fair Economy economic calendar used by
the project's news-risk safety layer. It parses the official weekly JSON and
XML exports, converts event timestamps into broker-server time, combines the
current-week and next-week calendars, and records the coverage metadata needed
to determine whether a signal timestamp is genuinely protected by verified
calendar data.

The official JSON export is preferred, with XML used as the secondary official
format. A calendar is treated as verified live data only when both the current
and next weekly exports are successfully available through the official feed
formats. HTML calendar parsing, previously cached data and the project fallback
file may still be used for display purposes, but they are explicitly marked as
unverified so they cannot silently authorise a trade through the news gate.

The module also maintains a display cache and fetch diagnostics. These fallback
mechanisms are designed to preserve useful calendar information for the user
without weakening the fail-closed safety behaviour implemented by the separate
news-risk module.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import json
import xml.etree.ElementTree as ET
from html.parser import HTMLParser

import pandas as pd
import requests


JSON_URLS = (
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",
)

XML_URLS = (
    "https://nfs.faireconomy.media/ff_calendar_thisweek.xml",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.xml",
)

WEEK_FEEDS = (
    ("this_week", JSON_URLS[0], XML_URLS[0]),
    ("next_week", JSON_URLS[1], XML_URLS[1]),
)

CALENDAR_COLUMNS = [
    "event_time",
    "event_date",
    "time_label",
    "currency",
    "event",
    "impact",
    "actual",
    "forecast",
    "previous",
    "detail_url",
    "source",
]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DISPLAY_CACHE = PROJECT_ROOT / "user_data" / "forex_factory_calendar_cache.csv"
DEFAULT_DISPLAY_META = PROJECT_ROOT / "user_data" / "forex_factory_calendar_cache_meta.json"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0 Safari/537.36"
    ),
    "Accept": "application/json,text/xml,application/xml,text/plain,*/*",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def _parse_source_date(raw_date: str) -> datetime | None:
    for fmt in ("%m-%d-%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw_date, fmt)
        except ValueError:
            continue
    return None


def _frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CALENDAR_COLUMNS)
    frame = pd.DataFrame(rows)
    for col in CALENDAR_COLUMNS:
        if col not in frame.columns:
            frame[col] = ""
    frame = frame[CALENDAR_COLUMNS]
    frame["event_time"] = pd.to_datetime(frame["event_time"], errors="coerce")
    frame["event_date"] = pd.to_datetime(frame["event_date"], errors="coerce").dt.normalize()
    return frame.sort_values(["event_date", "event_time", "time_label"], na_position="last").reset_index(drop=True)


def parse_faireconomy_xml(xml_text: str, broker_utc_offset_hours: float = 0.0) -> pd.DataFrame:
    """Parse the Forex Factory/Fair Economy weekly XML export into broker-server time."""
    root = ET.fromstring(xml_text)
    broker_tz = timezone(timedelta(hours=float(broker_utc_offset_hours)))
    source_tz = ZoneInfo("America/New_York")
    rows: list[dict] = []

    for event in root.findall(".//event"):
        item = {child.tag.lower(): (child.text or "").strip() for child in event}
        raw_date = item.get("date", "").strip()
        raw_time = item.get("time", "").strip()
        source_date = _parse_source_date(raw_date)
        if source_date is None:
            continue

        parsed_clock = None
        combined = f"{raw_date} {raw_time}".strip()
        for fmt in (
            "%m-%d-%Y %I:%M%p",
            "%m-%d-%Y %I:%M %p",
            "%m/%d/%Y %I:%M%p",
            "%m/%d/%Y %I:%M %p",
        ):
            try:
                parsed_clock = datetime.strptime(combined, fmt)
                break
            except ValueError:
                continue

        if parsed_clock is not None:
            broker_time = parsed_clock.replace(tzinfo=source_tz).astimezone(broker_tz).replace(tzinfo=None)
            event_time: pd.Timestamp | pd.NaT = pd.Timestamp(broker_time)
            event_date = pd.Timestamp(broker_time.date())
            time_label = broker_time.strftime("%H:%M")
        else:
            source_noon = source_date.replace(hour=12, tzinfo=source_tz)
            broker_noon = source_noon.astimezone(broker_tz)
            event_time = pd.NaT
            event_date = pd.Timestamp(broker_noon.date())
            time_label = raw_time or "Time not specified"

        rows.append({
            "event_time": event_time,
            "event_date": event_date,
            "time_label": time_label,
            "currency": item.get("country", item.get("currency", "")),
            "event": item.get("title", item.get("event", "")),
            "impact": item.get("impact", ""),
            "actual": item.get("actual", ""),
            "forecast": item.get("forecast", ""),
            "previous": item.get("previous", ""),
            "detail_url": item.get("url", ""),
            "source": "Live Forex Factory/Faireconomy feed",
        })

    return _frame(rows)


def _parse_iso_date(value: str) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    # Fair Economy JSON uses ISO timestamps with an explicit offset. Python's
    # fromisoformat accepts both -0400 and -04:00 after this normalisation.
    if len(text) >= 5 and text[-5] in {"+", "-"} and text[-3] != ":":
        text = text[:-2] + ":" + text[-2:]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def parse_faireconomy_json(payload: str | bytes | list[dict] | dict, broker_utc_offset_hours: float = 0.0) -> pd.DataFrame:
    """Parse the official Forex Factory/Fair Economy JSON weekly export.

    JSON is preferred for the website because the export carries explicit timezone
    offsets and is less sensitive to XML/browser parsing differences.
    """
    if isinstance(payload, (str, bytes, bytearray)):
        data = json.loads(payload)
    else:
        data = payload
    if isinstance(data, dict):
        data = data.get("events", data.get("data", []))
    if not isinstance(data, list):
        return pd.DataFrame(columns=CALENDAR_COLUMNS)

    broker_tz = timezone(timedelta(hours=float(broker_utc_offset_hours)))
    rows: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        source_dt = _parse_iso_date(item.get("date", ""))
        if source_dt is None:
            continue
        if source_dt.tzinfo is None:
            # Historical export variants without an offset are New York calendar time.
            source_dt = source_dt.replace(tzinfo=ZoneInfo("America/New_York"))
        broker_dt = source_dt.astimezone(broker_tz).replace(tzinfo=None)
        rows.append({
            "event_time": pd.Timestamp(broker_dt),
            "event_date": pd.Timestamp(broker_dt.date()),
            "time_label": broker_dt.strftime("%H:%M"),
            "currency": item.get("country", item.get("currency", "")),
            "event": item.get("title", item.get("event", "")),
            "impact": item.get("impact", ""),
            "actual": item.get("actual", ""),
            "forecast": item.get("forecast", ""),
            "previous": item.get("previous", ""),
            "detail_url": item.get("url", ""),
            "source": "Live Forex Factory/Faireconomy feed",
        })
    return _frame(rows)


def _coverage_window_for_feed(
    broker_utc_offset_hours: float,
    now_utc: datetime | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the broker-time window represented by this-week + next-week feeds."""
    source_tz = ZoneInfo("America/New_York")
    broker_tz = timezone(timedelta(hours=float(broker_utc_offset_hours)))
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now_ny = now.astimezone(source_tz)
    monday_ny = (now_ny - timedelta(days=now_ny.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    end_ny = monday_ny + timedelta(days=14)
    start_broker = monday_ny.astimezone(broker_tz).replace(tzinfo=None)
    end_broker = end_ny.astimezone(broker_tz).replace(tzinfo=None)
    return pd.Timestamp(start_broker), pd.Timestamp(end_broker)


def _attach_live_metadata(calendar: pd.DataFrame, broker_utc_offset_hours: float) -> pd.DataFrame:
    coverage_start, coverage_end = _coverage_window_for_feed(broker_utc_offset_hours)
    calendar.attrs["coverage_start"] = str(coverage_start)
    calendar.attrs["coverage_end"] = str(coverage_end)
    calendar.attrs["retrieved_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    calendar.attrs["verified_live"] = True
    return calendar


def _write_display_cache(calendar: pd.DataFrame, cache_path: Path, meta_path: Path) -> None:
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        calendar.to_csv(cache_path, index=False)
        meta = {
            "retrieved_at_utc": calendar.attrs.get("retrieved_at_utc", ""),
            "coverage_start": calendar.attrs.get("coverage_start", ""),
            "coverage_end": calendar.attrs.get("coverage_end", ""),
            "source": "Last successful verified Forex Factory/Fair Economy fetch",
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception:
        # Cache persistence must never break the news safety path.
        pass


def _load_display_cache(cache_path: Path, meta_path: Path) -> pd.DataFrame:
    if not cache_path.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(cache_path)
        frame = _frame(frame.to_dict(orient="records"))
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        frame.attrs.update(meta)
        frame.attrs["verified_live"] = False
        return frame
    except Exception:
        return pd.DataFrame()


def _request_json(url: str, timeout_seconds: float) -> tuple[pd.DataFrame | None, int | None, str]:
    try:
        response = requests.get(url, headers=_BROWSER_HEADERS, timeout=float(timeout_seconds))
        status = int(response.status_code)
        if status != 200:
            return None, status, f"HTTP {status}"
        return response, status, ""
    except Exception as exc:
        return None, None, f"{type(exc).__name__}: {exc}"



class _ForexFactoryCalendarHTMLParser(HTMLParser):
    """Small dependency-free parser for Forex Factory calendar rows."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[dict] = []
        self._row: dict | None = None
        self._field: str | None = None
        self._field_depth = 0
        self._impact_red = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k: (v or "") for k, v in attrs}
        classes = set(attr.get("class", "").split())
        if tag == "tr" and ("calendar__row" in classes or "data-eventid" in attr):
            self._row = {"impact_red": False}
            self._field = None
            self._field_depth = 0
            self._impact_red = False
            return
        if self._row is None:
            return
        if self._field is not None:
            self._field_depth += 1
        mapping = {
            "calendar__date": "date",
            "calendar__time": "time",
            "calendar__currency": "currency",
            "calendar__impact": "impact",
            "calendar__event-title": "event",
            "calendar__event": "event",
            "calendar__actual": "actual",
            "calendar__forecast": "forecast",
            "calendar__previous": "previous",
        }
        for cls, field in mapping.items():
            if cls in classes:
                self._field = field
                self._field_depth = 0
                self._row.setdefault(field, "")
                break
        joined = " ".join(classes).lower()
        title = attr.get("title", "").lower()
        if "impact-red" in joined or "ff-impact-red" in joined or "high impact expected" in title:
            self._impact_red = True
            self._row["impact_red"] = True

    def handle_data(self, data: str) -> None:
        if self._row is None or self._field is None:
            return
        value = " ".join(data.split())
        if not value:
            return
        current = str(self._row.get(self._field, "")).strip()
        self._row[self._field] = (current + " " + value).strip()

    def handle_endtag(self, tag: str) -> None:
        if self._row is None:
            return
        if tag == "tr":
            self._row["impact_red"] = bool(self._row.get("impact_red") or self._impact_red)
            self.rows.append(self._row)
            self._row = None
            self._field = None
            self._field_depth = 0
            return
        if self._field is not None:
            if self._field_depth <= 0:
                self._field = None
            else:
                self._field_depth -= 1


def _parse_forex_factory_html_week(
    html_text: str,
    broker_utc_offset_hours: float,
    *,
    week: str,
    now_utc: datetime | None = None,
) -> pd.DataFrame:
    """Parse USD high-impact rows from the public Forex Factory calendar page.

    This HTML path is a display fallback for weeks where the weekly JSON/XML export
    is unavailable. It is deliberately NOT treated as verified trading-gate data.
    """
    parser = _ForexFactoryCalendarHTMLParser()
    parser.feed(html_text)
    rows: list[dict] = []
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    broker_tz = timezone(timedelta(hours=float(broker_utc_offset_hours)))
    current_date: datetime | None = None

    for item in parser.rows:
        date_text = str(item.get("date", "")).strip()
        if date_text:
            for fmt in ("%a %b %d %Y", "%b %d %Y"):
                try:
                    candidate = date_text if date_text.endswith(str(now.year)) else f"{date_text} {now.year}"
                    current_date = datetime.strptime(candidate, fmt)
                    if week == "next_week" and current_date.date() < (now.date() - timedelta(days=180)):
                        current_date = current_date.replace(year=current_date.year + 1)
                    break
                except ValueError:
                    continue
        if current_date is None:
            continue
        if str(item.get("currency", "")).strip().upper() != "USD":
            continue
        impact_text = str(item.get("impact", "")).strip().lower()
        if not (bool(item.get("impact_red")) or "high" in impact_text):
            continue
        event_name = str(item.get("event", "")).strip()
        if not event_name:
            continue

        raw_time = str(item.get("time", "")).strip()
        parsed_clock: datetime | None = None
        compact = raw_time.replace(" ", "").lower()
        for fmt in ("%I:%M%p", "%H:%M"):
            try:
                parsed_clock = datetime.strptime(compact.upper(), fmt)
                break
            except ValueError:
                continue

        if parsed_clock is not None:
            source_dt = current_date.replace(
                hour=parsed_clock.hour,
                minute=parsed_clock.minute,
                tzinfo=ZoneInfo("America/New_York"),
            )
            broker_dt = source_dt.astimezone(broker_tz).replace(tzinfo=None)
            event_time: pd.Timestamp | pd.NaT = pd.Timestamp(broker_dt)
            event_date = pd.Timestamp(broker_dt.date())
            time_label = broker_dt.strftime("%H:%M")
        else:
            event_time = pd.NaT
            event_date = pd.Timestamp(current_date.date())
            time_label = raw_time or "Time not specified"

        rows.append({
            "event_time": event_time,
            "event_date": event_date,
            "time_label": time_label,
            "currency": "USD",
            "event": event_name,
            "impact": "High",
            "actual": str(item.get("actual", "")).strip(),
            "forecast": str(item.get("forecast", "")).strip(),
            "previous": str(item.get("previous", "")).strip(),
            "detail_url": "",
            "source": "Forex Factory calendar webpage (display fallback)",
        })

    frame = _frame(rows)
    frame.attrs["feed_week"] = week
    frame.attrs["feed_format"] = "html"
    frame.attrs["display_only"] = True
    return frame


def _fetch_forex_factory_html_week(
    week_key: str,
    broker_utc_offset_hours: float,
    timeout_seconds: float,
) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    query = "this" if week_key == "this_week" else "next"
    url = f"https://www.forexfactory.com/calendar?week={query}"
    try:
        response = requests.get(url, headers=_BROWSER_HEADERS, timeout=float(timeout_seconds))
        if response.status_code != 200:
            return pd.DataFrame(columns=CALENDAR_COLUMNS), [f"{week_key} HTML HTTP {response.status_code}"]
        parsed = _parse_forex_factory_html_week(
            response.text,
            broker_utc_offset_hours,
            week=week_key,
        )
        if parsed.empty:
            errors.append(f"{week_key} HTML returned no parseable USD high-impact events")
        return parsed, errors
    except Exception as exc:
        return pd.DataFrame(columns=CALENDAR_COLUMNS), [f"{week_key} HTML {type(exc).__name__}: {exc}"]

def _fetch_one_week(
    week_key: str,
    json_url: str,
    xml_url: str,
    broker_utc_offset_hours: float,
    timeout_seconds: float,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Fetch one Forex Factory weekly export, preferring JSON and falling back to XML.

    Returning the week key separately lets callers distinguish a genuinely loaded week
    from a missing export. This prevents one successful week from being treated as a
    verified two-week calendar.
    """
    errors: list[str] = []

    try:
        response = requests.get(json_url, headers=_BROWSER_HEADERS, timeout=float(timeout_seconds))
        if response.status_code == 200:
            parsed = parse_faireconomy_json(response.text, broker_utc_offset_hours)
            if not parsed.empty:
                parsed.attrs["feed_week"] = week_key
                parsed.attrs["feed_format"] = "json"
                return parsed, "json", errors
            errors.append(f"{week_key} JSON returned no parseable events")
        else:
            errors.append(f"{week_key} JSON HTTP {response.status_code}")
    except Exception as exc:
        errors.append(f"{week_key} JSON {type(exc).__name__}: {exc}")

    try:
        response = requests.get(xml_url, headers=_BROWSER_HEADERS, timeout=float(timeout_seconds))
        if response.status_code == 200:
            parsed = parse_faireconomy_xml(response.text, broker_utc_offset_hours)
            if not parsed.empty:
                parsed.attrs["feed_week"] = week_key
                parsed.attrs["feed_format"] = "xml"
                return parsed, "xml", errors
            errors.append(f"{week_key} XML returned no parseable events")
        else:
            errors.append(f"{week_key} XML HTTP {response.status_code}")
    except Exception as exc:
        errors.append(f"{week_key} XML {type(exc).__name__}: {exc}")

    return pd.DataFrame(columns=CALENDAR_COLUMNS), "", errors


def fetch_faireconomy_calendar(
    broker_utc_offset_hours: float = 0.0,
    timeout_seconds: float = 10.0,
    fallback_path: str | Path | None = None,
    display_cache_path: str | Path | None = None,
    display_meta_path: str | Path | None = None,
) -> tuple[pd.DataFrame, bool, str]:
    """Fetch the current + next Forex Factory weekly calendar.

    Each week is verified independently. JSON is preferred and XML is used as an
    official-export fallback. The function returns verified=True only when *both*
    current-week and next-week exports were loaded. This prevents a successful
    current-week request from incorrectly authorising an empty next-week calendar.
    """
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    loaded_weeks: list[str] = []
    formats: dict[str, str] = {}

    for week_key, json_url, xml_url in WEEK_FEEDS:
        frame, fmt, week_errors = _fetch_one_week(
            week_key, json_url, xml_url, broker_utc_offset_hours, timeout_seconds
        )
        errors.extend(week_errors)
        if frame.empty:
            # Forex Factory currently exposes the public next-week page even when a
            # separate next-week JSON/XML export is unavailable. Parse that page for
            # display so novice users can still see upcoming red-folder USD events.
            html_frame, html_errors = _fetch_forex_factory_html_week(
                week_key, broker_utc_offset_hours, timeout_seconds
            )
            errors.extend(html_errors)
            if not html_frame.empty:
                frame = html_frame
                fmt = "html"

        if not frame.empty:
            frames.append(frame)
            loaded_weeks.append(week_key)
            formats[week_key] = fmt

    if frames:
        calendar = pd.concat(frames, ignore_index=True)
        dedupe_cols = [c for c in ["event_date", "time_label", "currency", "event"] if c in calendar.columns]
        calendar = calendar.drop_duplicates(dedupe_cols).sort_values(
            ["event_date", "event_time", "time_label"], na_position="last"
        ).reset_index(drop=True)
        calendar.attrs["loaded_weeks"] = loaded_weeks
        calendar.attrs["missing_weeks"] = [w for w, _, _ in WEEK_FEEDS if w not in loaded_weeks]
        calendar.attrs["feed_formats"] = formats
        calendar.attrs["fetch_error"] = "; ".join(errors)

        complete = (
            set(loaded_weeks) == {"this_week", "next_week"}
            and all(formats.get(w) in {"json", "xml"} for w in ("this_week", "next_week"))
        )
        if complete:
            calendar = _attach_live_metadata(calendar, broker_utc_offset_hours)
            cache_path = Path(display_cache_path) if display_cache_path else DEFAULT_DISPLAY_CACHE
            meta_path = Path(display_meta_path) if display_meta_path else DEFAULT_DISPLAY_META
            _write_display_cache(calendar, cache_path, meta_path)
            return calendar, True, "Live Forex Factory/Faireconomy weekly exports (current + next week verified)"

        calendar.attrs["verified_live"] = False
        return calendar, False, "Partial live Forex Factory calendar with display fallback (trading gate remains fail-closed)"

    cache_path = Path(display_cache_path) if display_cache_path else DEFAULT_DISPLAY_CACHE
    meta_path = Path(display_meta_path) if display_meta_path else DEFAULT_DISPLAY_META
    cached = _load_display_cache(cache_path, meta_path)
    if not cached.empty:
        reason = "; ".join(errors) if errors else "live export unavailable"
        cached.attrs["fetch_error"] = reason
        return cached, False, "Recent Forex Factory cache (display only; trading gate remains fail-closed)"

    if fallback_path is not None and Path(fallback_path).exists():
        fallback = pd.read_csv(fallback_path)
        fallback.attrs["verified_live"] = False
        fallback.attrs["fetch_error"] = "; ".join(errors)
        return fallback, False, "Project economic-calendar fallback (not current; display only)"

    empty = pd.DataFrame(columns=CALENDAR_COLUMNS)
    empty.attrs["fetch_error"] = "; ".join(errors)
    empty.attrs["loaded_weeks"] = []
    empty.attrs["missing_weeks"] = ["this_week", "next_week"]
    return empty, False, "No verified Forex Factory calendar available"
