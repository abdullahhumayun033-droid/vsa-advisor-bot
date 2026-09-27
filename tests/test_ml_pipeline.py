"""Automated tests for the VSA Advisor Bot pipeline and integration contracts.

This test module checks the main behaviours expected from the project's VSA,
machine-learning, news-risk, journal and MT5 integration layers. It covers
outcome labelling, scenario-specific stops, managed exits, feature calculation,
chronological splitting, saved-model inference, signal-file structure, calendar
handling, live-feed freshness, journal persistence and fail-closed live-cycle
behaviour.

The tests also verify important project integration contracts, including stable
signal identifiers, LR explanation faithfulness, MT5 replay parity and duplicate
signal safety refreshes. The module is validation code only; it does not define
the production trading logic itself.
"""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modules.ml_pipeline import (
    MODEL_FEATURES,
    add_candle_features,
    audit_lr_explanations,
    first_touch_outcome,
    purged_chronological_split,
    read_mt5_bars,
    load_selected_model,
    predict_vsa_setup_row,
    SIGNAL_COLUMNS,
    save_ml_signal_csv,
)
from modules.live_pipeline import assess_live_bar_freshness, read_broker_offset_from_status, read_mt5_terminal_status, run_live_cycle
from modules.news_calendar import fetch_faireconomy_calendar, parse_faireconomy_json, parse_faireconomy_xml
from modules.vsa_strategy import managed_exit_outcome, scenario_stop
from modules.news_risk import apply_news_risk
from modules.integration_evidence import make_signal_id
from modules.journal import add_manual_journal_entry, read_journal, upsert_journal_entry, update_journal_reflection


class OutcomeTests(unittest.TestCase):
    def test_buy_target_before_stop_is_win(self):
        result = first_touch_outcome(np.array([101.0, 103.0]), np.array([99.5, 100.0]), "BUY", 102.0, 98.0)
        self.assertEqual(result, "WIN")

    def test_sell_stop_before_target_is_loss(self):
        result = first_touch_outcome(np.array([101.5, 100.0]), np.array([99.5, 97.0]), "SELL", 98.0, 101.0)
        self.assertEqual(result, "LOSS")

    def test_same_bar_uses_conservative_stop_first_rule(self):
        result = first_touch_outcome(np.array([103.0]), np.array([97.0]), "BUY", 102.0, 98.0)
        self.assertEqual(result, "LOSS")

    def test_no_touch_expires(self):
        result = first_touch_outcome(np.array([101.0]), np.array([99.0]), "BUY", 103.0, 97.0)
        self.assertEqual(result, "EXPIRED")

    def test_scenario_specific_stop_rules_are_mirrored(self):
        active = {"sweep_stop": 97.5, "post_break_extreme": 96.0}
        self.assertEqual(scenario_stop(active, 105.0, 98.0, "BUY", 1), 98.0)
        self.assertEqual(scenario_stop(active, 105.0, 98.0, "BUY", 2), 97.5)
        self.assertEqual(scenario_stop(active, 105.0, 98.0, "BUY", 3), 96.0)
        sell_active = {"sweep_stop": 106.0, "post_break_extreme": 108.0}
        self.assertEqual(scenario_stop(sell_active, 105.0, 98.0, "SELL", 1), 105.0)
        self.assertEqual(scenario_stop(sell_active, 105.0, 98.0, "SELL", 2), 106.0)
        self.assertEqual(scenario_stop(sell_active, 105.0, 98.0, "SELL", 3), 108.0)

    def test_managed_exit_full_target_returns_1_9r(self):
        outcome, result_r = managed_exit_outcome([101.1, 104.1], [100.2, 100.5], "BUY", 100, 99, 101, 104)
        self.assertEqual(outcome, "FULL_TARGET")
        self.assertAlmostEqual(result_r, 1.9)

    def test_managed_exit_moves_runner_to_breakeven(self):
        outcome, result_r = managed_exit_outcome([101.2, 101.0], [100.2, 99.9], "BUY", 100, 99, 101, 104)
        self.assertEqual(outcome, "PARTIAL_WIN_BREAKEVEN")
        self.assertAlmostEqual(result_r, 0.7)

    def test_managed_exit_stop_before_first_target_loses_1r(self):
        outcome, result_r = managed_exit_outcome([100.5], [98.9], "BUY", 100, 99, 101, 104)
        self.assertEqual(outcome, "FULL_LOSS")
        self.assertAlmostEqual(result_r, -1.0)


class DataTests(unittest.TestCase):
    def test_feature_calculation_has_expected_relative_values(self):
        times = pd.date_range("2026-01-01", periods=25, freq="5min")
        bars = pd.DataFrame({
            "time": times, "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.5, "tick_volume": 100.0,
            "real_volume": 0, "broker_spread_points": 5,
        })
        featured = add_candle_features(bars, lookback=20)
        self.assertAlmostEqual(featured.iloc[20]["relative_spread"], 1.0)
        self.assertAlmostEqual(featured.iloc[20]["relative_volume"], 1.0)
        self.assertAlmostEqual(featured.iloc[20]["close_location"], 0.75)

    def test_mt5_reader_sorts_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bars.csv"
            path.write_text(
                "<DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\n"
                "2026.01.01\t00:05:00\t1\t2\t0\t1.5\t10\n"
                "2026.01.01\t00:00:00\t1\t2\t0\t1.5\t10\n"
                "2026.01.01\t00:05:00\t1\t2\t0\t1.5\t10\n"
            )
            result = read_mt5_bars(path)
            self.assertEqual(len(result), 2)
            self.assertTrue(result["time"].is_monotonic_increasing)


class SplitAndBridgeTests(unittest.TestCase):
    def test_trained_model_probability_controls_live_advice(self):
        class FixedProbabilityModel:
            classes_ = np.array([0, 1])

            def __init__(self, probability):
                self.probability = probability

            def predict_proba(self, frame):
                return np.array([[1.0 - self.probability, self.probability]])

        row_values = {feature: 0.0 for feature in MODEL_FEATURES}
        row_values.update({
            "time": "2026-08-18 10:35:00", "cab_time": "2026-08-18 10:00:00",
            "direction": "BUY", "scenario_id": 2, "entry_price": 3400.0,
            "stop_price": 3395.0, "target_price": 3405.0, "runner_target_price": 3420.0,
            "risk_percent": 1.0,
        })
        row = pd.Series(row_values, dtype=object)
        metadata = {"selected_model": "LR_VSA_BOT", "recommendation_threshold": 0.60}
        rejected = predict_vsa_setup_row(row, model=FixedProbabilityModel(0.40), metadata=metadata)
        accepted = predict_vsa_setup_row(row, model=FixedProbabilityModel(0.75), metadata=metadata)
        self.assertEqual(rejected["advisor_output"], "WATCH_ONLY")
        self.assertEqual(accepted["advisor_output"], "TRADE_CANDIDATE")

    def test_novice_pages_are_default_and_audits_are_optional(self):
        app_source = (Path(__file__).resolve().parents[1] / "app.py").read_text()
        self.assertIn('"Live Advisor"', app_source)
        self.assertIn('"How the Bot Learns"', app_source)
        self.assertIn('"Why This Advice?"', app_source)
        self.assertIn('"Show technical audit pages"', app_source)
        self.assertIn("value=False", app_source)

    def test_shared_signal_id_is_stable_and_human_auditable(self):
        signal = {"signal_time": "2026-08-18 10:35:00", "direction": "BUY", "scenario_id": 2}
        self.assertEqual(make_signal_id(signal), "20260818103500-BUY-S2")

    def test_mt5_sources_expose_vsa_and_ml_evidence(self):
        project_root = Path(__file__).resolve().parents[1]
        indicator = (project_root / "mt5" / "VSA_Live_Visual_Indicator.mq5").read_text()
        bridge = (project_root / "mt5" / "VSA_ML_Signal_Bridge_EA.mq5").read_text()
        for token in ["buy_cab", "sell_cab", "scenario = 3", "scenario = 2", "scenario = 1", "vsa_mql5_latest_audit.csv"]:
            self.assertIn(token, indicator)
        for token in ["scenario_id", "probability", "model_threshold", "SignalId", "DrawSignalEvidence"]:
            self.assertIn(token, bridge)

    def test_mt5_replay_file_matches_untouched_lr_predictions(self):
        project_root = Path(__file__).resolve().parents[1]
        replay = pd.read_csv(project_root / "mt5" / "vsa_lr_test_signals.csv")
        predictions = pd.read_csv(project_root / "results" / "lr_test_predictions.csv")
        self.assertEqual(len(replay), len(predictions))
        self.assertEqual(len(replay), 65)
        self.assertEqual(int(replay["lr_trade_candidate"].sum()), 45)
        np.testing.assert_allclose(replay["lr_probability"], predictions["probability_valid_trade"])
        self.assertEqual(replay.iloc[0]["signal_time"], predictions.iloc[0]["time"])

    def test_lr_explanation_reconstructs_probability_exactly(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.normal(size=(40, len(MODEL_FEATURES))), columns=MODEL_FEATURES)
        frame["time"] = pd.date_range("2026-01-01", periods=len(frame), freq="5min")
        target = (frame[MODEL_FEATURES[0]] + 0.5 * frame[MODEL_FEATURES[1]] > 0).astype(int)
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ])
        model.fit(frame[MODEL_FEATURES], target)
        summary, audit = audit_lr_explanations(model, frame)
        self.assertTrue(summary["faithful_within_tolerance"])
        self.assertLessEqual(summary["maximum_absolute_probability_error"], 1e-12)
        self.assertEqual(len(audit), len(frame))
        self.assertEqual(summary["features_per_explanation"], len(MODEL_FEATURES))

    def test_saved_model_version_mismatch_fails_before_inference(self):
        with tempfile.TemporaryDirectory() as folder:
            model_path = Path(folder) / "selected_model.joblib"
            model_path.write_bytes(b"test-placeholder")
            with patch("modules.ml_pipeline.MODELS_DIR", Path(folder)), patch(
                "modules.ml_pipeline.load_model_metadata",
                return_value={"runtime_versions": {"scikit_learn": "0.0-test"}},
            ):
                with self.assertRaisesRegex(RuntimeError, "version mismatch"):
                    load_selected_model()

    def test_purged_split_has_full_embargo(self):
        df = pd.DataFrame({
            "time": pd.date_range("2026-01-01", periods=1000, freq="5min"),
            "target": np.zeros(1000, dtype=int),
        })
        train, validation, test, audit = purged_chronological_split(df, outcome_horizon_candles=36)
        self.assertGreaterEqual(validation["time"].min() - train["time"].max(), pd.Timedelta(minutes=180))
        self.assertGreaterEqual(test["time"].min() - validation["time"].max(), pd.Timedelta(minutes=180))
        self.assertEqual(audit["embargo_minutes"], 180)

    def test_signal_csv_keeps_contract_columns(self):
        signal = {
            "signal_time": "2026-08-17 10:25:00", "cab_time": "2026-08-17 09:25:00",
            "symbol": "XAUUSD", "timeframe": "M5", "selected_model": "LR_VSA_BOT",
            "direction": "BUY", "scenario_id": 3, "ml_probability_valid": 0.61,
            "recommendation_threshold": 0.60, "advisor_output": "TRADE_CANDIDATE",
            "entry_price": 100.0, "stop_price": 99.0, "target_price": 101.2,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = save_ml_signal_csv(signal, Path(folder) / "signal.csv")
            result = pd.read_csv(path)
            self.assertEqual(list(result.columns), SIGNAL_COLUMNS)
            self.assertEqual(len(result.columns), 29)
            self.assertEqual(result.iloc[0]["symbol"], "XAUUSD")
            self.assertTrue(bool(result.iloc[0]["news_blocked"]))

    def test_news_gate_blocks_nearby_high_impact_usd_event(self):
        signal = {"signal_time": "2026-08-17 10:00:00", "advisor_output": "TRADE_CANDIDATE"}
        events = pd.DataFrame([{
            "event_time": "2026-08-17 10:20:00", "currency": "USD",
            "impact": "High", "event": "CPI",
        }])
        result = apply_news_risk(signal, events, verified_live=True)
        self.assertTrue(result["news_blocked"])
        self.assertEqual(result["advisor_output"], "WATCH_ONLY")

    def test_news_gate_fails_safe_when_calendar_unverified(self):
        signal = {"signal_time": "2026-08-17 10:00:00", "advisor_output": "TRADE_CANDIDATE"}
        result = apply_news_risk(signal, None, verified_live=False)
        self.assertTrue(result["news_blocked"])
        self.assertEqual(result["news_status"], "CALENDAR_UNVERIFIED")

    def test_calendar_parser_converts_new_york_time_to_broker_time(self):
        xml = """<weeklyevents><event><title>CPI</title><country>USD</country>
        <date>08-17-2026</date><time>8:30am</time><impact>High</impact></event></weeklyevents>"""
        result = parse_faireconomy_xml(xml, broker_utc_offset_hours=3)
        # New York is UTC-4 in August; a UTC+3 broker is seven hours ahead.
        self.assertEqual(str(result.iloc[0]["event_time"]), "2026-08-17 15:30:00")

    def test_calendar_json_parser_keeps_red_folder_usd_event(self):
        payload = [{
            "title": "Core PCE Price Index m/m",
            "country": "USD",
            "date": "2026-09-30T08:30:00-04:00",
            "impact": "High",
            "forecast": "0.3%",
            "previous": "0.2%",
        }]
        result = parse_faireconomy_json(payload, broker_utc_offset_hours=3)
        self.assertEqual(result.iloc[0]["event"], "Core PCE Price Index m/m")
        self.assertEqual(result.iloc[0]["impact"], "High")
        self.assertEqual(result.iloc[0]["currency"], "USD")
        self.assertEqual(str(result.iloc[0]["event_time"]), "2026-09-30 15:30:00")

    def test_calendar_json_parser_keeps_multiple_same_time_high_events(self):
        payload = [
            {"title": "Core PCE Price Index m/m", "country": "USD", "date": "2026-09-30T08:30:00-04:00", "impact": "High"},
            {"title": "Final GDP q/q", "country": "USD", "date": "2026-09-30T08:30:00-04:00", "impact": "High"},
        ]
        result = parse_faireconomy_json(payload, broker_utc_offset_hours=3)
        self.assertEqual(len(result), 2)
        self.assertEqual(set(result["event"].tolist()), {"Core PCE Price Index m/m", "Final GDP q/q"})

    def test_news_calendar_is_embedded_and_has_no_external_ff_button(self):
        app_text = (Path(__file__).resolve().parents[1] / "app.py").read_text(encoding="utf-8")
        self.assertIn("Refresh embedded news calendar", app_text)
        self.assertNotIn("Open Forex Factory calendar", app_text)

    def test_calendar_fetch_falls_back_to_xml_for_missing_next_week_json(self):
        class Response:
            def __init__(self, status_code, text=""):
                self.status_code = status_code
                self.text = text

        current_json = '[{"title":"Durable Goods Orders m/m","country":"USD","date":"2026-09-25T08:30:00-04:00","impact":"Low"}]'
        next_xml = """<weeklyevents><event><title>Core PCE Price Index m/m</title><country>USD</country>
        <date>09-30-2026</date><time>8:30am</time><impact>High</impact><forecast>0.3%</forecast><previous>0.2%</previous></event>
        <event><title>Final GDP q/q</title><country>USD</country><date>09-30-2026</date><time>8:30am</time><impact>High</impact></event></weeklyevents>"""
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "cache.csv"
            meta = Path(folder) / "cache.json"
            with patch("modules.news_calendar.requests.get", side_effect=[
                Response(200, current_json),
                Response(503, ""),
                Response(200, next_xml),
            ]):
                result, verified, source = fetch_faireconomy_calendar(
                    broker_utc_offset_hours=3,
                    display_cache_path=cache,
                    display_meta_path=meta,
                )
        self.assertTrue(verified)
        self.assertIn("current + next week verified", source)
        self.assertEqual(set(result.attrs.get("loaded_weeks", [])), {"this_week", "next_week"})
        high_usd = result[(result["currency"] == "USD") & (result["impact"] == "High")]
        self.assertEqual(set(high_usd["event"].tolist()), {"Core PCE Price Index m/m", "Final GDP q/q"})

    def test_calendar_fetch_does_not_verify_two_weeks_when_next_week_missing(self):
        class Response:
            def __init__(self, status_code, text=""):
                self.status_code = status_code
                self.text = text

        current_json = '[{"title":"Durable Goods Orders m/m","country":"USD","date":"2026-09-25T08:30:00-04:00","impact":"Low"}]'
        with tempfile.TemporaryDirectory() as folder:
            cache = Path(folder) / "cache.csv"
            meta = Path(folder) / "cache.json"
            with patch("modules.news_calendar.requests.get", side_effect=[
                Response(200, current_json),
                Response(503, ""),
                Response(503, ""),
            ]):
                result, verified, source = fetch_faireconomy_calendar(
                    broker_utc_offset_hours=3,
                    display_cache_path=cache,
                    display_meta_path=meta,
                )
        self.assertFalse(verified)
        self.assertIn("Partial live Forex Factory calendar", source)
        self.assertEqual(result.attrs.get("loaded_weeks"), ["this_week"])
        self.assertIn("next_week", result.attrs.get("missing_weeks", []))


    def test_clock_freshness_rejects_old_mt5_export(self):
        result = assess_live_bar_freshness(
            "2026-08-17 09:30:00",
            broker_utc_offset_hours=0,
            now_utc=datetime(2026, 8, 17, 10, 6, tzinfo=timezone.utc),
        )
        self.assertFalse(result["feed_fresh"])
        self.assertEqual(result["feed_state"], "STALE")

    def test_clock_freshness_accepts_expected_completed_bar(self):
        result = assess_live_bar_freshness(
            "2026-08-17 10:00:00",
            broker_utc_offset_hours=0,
            now_utc=datetime(2026, 8, 17, 10, 6, tzinfo=timezone.utc),
        )
        self.assertTrue(result["feed_fresh"])
        self.assertEqual(result["feed_state"], "LIVE")

    def test_news_gate_rejects_verified_but_unrelated_calendar_period(self):
        signal = {"signal_time": "2026-08-17 10:00:00", "advisor_output": "TRADE_CANDIDATE"}
        events = pd.DataFrame([{
            "event_time": "2026-09-20 10:00:00", "currency": "USD",
            "impact": "High", "event": "CPI",
        }])
        result = apply_news_risk(signal, events, verified_live=True)
        self.assertTrue(result["news_blocked"])
        self.assertEqual(result["news_status"], "CALENDAR_OUT_OF_COVERAGE")
        self.assertEqual(result["advisor_output"], "WATCH_ONLY")

    def test_journal_upsert_keeps_one_row_per_signal_and_blank_unresolved_result(self):
        signal = {
            "signal_time": "2026-08-17 10:00:00", "direction": "BUY", "scenario_id": 2,
            "selected_model": "LR_VSA_BOT", "advisor_output": "WATCH_ONLY",
            "ml_probability_valid": 0.62, "recommendation_threshold": 0.35,
            "entry_price": 3400.0, "stop_price": 3395.0, "target_price": 3405.0,
            "runner_target_price": 3420.0, "risk_percent": 1.0,
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "journal.csv"
            first = upsert_journal_entry(
                signal, "HISTORICAL_DEMO", "Watched", notes="first", path=path,
                saved_at_utc=datetime(2026, 8, 17, 10, 10, tzinfo=timezone.utc),
            )
            self.assertEqual(len(first), 1)
            self.assertTrue(pd.isna(pd.to_numeric(first.iloc[0]["result_r"], errors="coerce")))
            second = upsert_journal_entry(
                signal, "HISTORICAL_DEMO", "Demo-traded", notes="updated", result_r=0.7, path=path,
                saved_at_utc=datetime(2026, 8, 17, 11, 0, tzinfo=timezone.utc),
            )
            self.assertEqual(len(second), 1)
            self.assertEqual(second.iloc[0]["user_decision"], "Demo-traded")
            self.assertAlmostEqual(float(second.iloc[0]["result_r"]), 0.7)
            updated = update_journal_reflection(second.iloc[0]["signal_id"], clear_result=True, path=path)
            self.assertTrue(pd.isna(pd.to_numeric(updated.iloc[0]["result_r"], errors="coerce")))
            reread = read_journal(path)
            self.assertEqual(len(reread), 1)


    def test_manual_journal_entry_is_persistent_and_marked_manual(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "journal.csv"
            saved = add_manual_journal_entry(
                trade_time="2026-09-22 20:15:00",
                direction="BUY",
                scenario_id=0,
                entry_price=4378.8,
                stop_price=4276.4,
                user_decision="Demo-traded",
                notes="Manual practice trade",
                path=path,
            )
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.iloc[0]["source_mode"], "Manual journal entry")
            self.assertEqual(saved.iloc[0]["advisor_output"], "USER_LOG")
            reread = read_journal(path)
            self.assertEqual(len(reread), 1)
            self.assertEqual(reread.iloc[0]["direction"], "BUY")


    def test_mt5_terminal_status_detects_disconnected_exporter(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vsa_mt5_connection_status.csv"
            pd.DataFrame([{
                "server_time": "2026.09.22 21:30:00",
                "gmt_time": "2026.09.22 18:30:00",
                "broker_utc_offset_hours": 3.0,
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "terminal_connected": 0,
            }]).to_csv(path, index=False)
            status = read_mt5_terminal_status(folder)
            self.assertTrue(status["available"])
            self.assertFalse(status["connected"])

    def test_live_cycle_fails_closed_when_mt5_terminal_reports_disconnected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bars = root / "xauusd_m5_bars.csv"
            bars.write_text(
                "time,open,high,low,close,tick_volume\n"
                "2026-09-22 18:25:00,1,2,0,1.5,100\n"
            )
            pd.DataFrame([{
                "server_time": "2026.09.22 21:30:00",
                "gmt_time": "2026.09.22 18:30:00",
                "broker_utc_offset_hours": 3.0,
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "terminal_connected": 0,
            }]).to_csv(root / "vsa_mt5_connection_status.csv", index=False)
            result = run_live_cycle(
                bars, root / "signal.csv", root / "bridge.csv", root / "status.json",
                broker_utc_offset_hours=3.0,
                now_utc=datetime(2026, 9, 22, 18, 31, tzinfo=timezone.utc),
            )
            self.assertEqual(result["state"], "MT5_TERMINAL_DISCONNECTED")
            self.assertEqual(result["feed_state"], "DISCONNECTED")
            self.assertFalse(result["output_written"])

    def test_broker_offset_can_be_read_from_exporter_status(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "vsa_mt5_connection_status.csv"
            pd.DataFrame([{
                "server_time": "2026.09.22 17:00:00",
                "gmt_time": "2026.09.22 14:00:00",
                "broker_utc_offset_hours": 3.0,
                "symbol": "XAUUSD",
                "timeframe": "M5",
                "terminal_connected": 1,
            }]).to_csv(path, index=False)
            self.assertEqual(read_broker_offset_from_status(folder), 3.0)

    def test_duplicate_setup_rechecks_news_and_refreshes_safety_state(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bars = root / "bars.csv"
            bars.write_text(
                "time,open,high,low,close,tick_volume\n"
                "2026-08-17 10:00:00,1,2,0,1.5,100\n"
            )
            project_signal = root / "ml_signal.csv"
            bridge_signal = root / "bridge.csv"
            status = root / "status.json"
            pd.DataFrame([{"signal_time": "2026-08-17 10:00:00"}]).to_csv(project_signal, index=False)
            raw_signal = {
                "signal_time": "2026-08-17 10:00:00", "cab_time": "2026-08-17 09:30:00",
                "symbol": "XAUUSD", "timeframe": "M5", "selected_model": "LR_VSA_BOT",
                "direction": "BUY", "scenario_id": 2, "ml_probability_valid": 0.80,
                "recommendation_threshold": 0.35, "advisor_output": "TRADE_CANDIDATE",
                "entry_price": 100.0, "stop_price": 99.0, "target_price": 101.0,
                "runner_target_price": 104.0, "risk_percent": 1.0,
            }
            setup_row = pd.Series({"time": "2026-08-17 10:00:00"})
            events = pd.DataFrame([{
                "event_time": "2026-08-17 10:05:00", "currency": "USD",
                "impact": "High", "event": "CPI",
            }])

            def calendar_fetcher(**kwargs):
                return events, True, "test verified calendar"

            with patch(
                "modules.live_pipeline.load_model_metadata",
                return_value={"selected_model": "LR_VSA_BOT", "feature_columns": []},
            ), patch(
                "modules.live_pipeline.load_selected_model", return_value=object()
            ), patch(
                "modules.live_pipeline.predict_latest_setup_from_bars", return_value=(raw_signal, setup_row)
            ), patch(
                "modules.live_pipeline.explain_lr_setup", return_value=pd.DataFrame()
            ):
                result = run_live_cycle(
                    bars, project_signal, bridge_signal, status,
                    now_utc=datetime(2026, 8, 17, 10, 6, tzinfo=timezone.utc),
                    calendar_fetcher=calendar_fetcher,
                )

            self.assertEqual(result["state"], "DUPLICATE_SETUP_REFRESHED")
            self.assertEqual(result["news_status"], "HIGH_IMPACT_USD_BLOCK")
            refreshed = pd.read_csv(project_signal).iloc[-1]
            self.assertEqual(refreshed["advisor_output"], "WATCH_ONLY")
            self.assertTrue(bool(refreshed["news_blocked"]))
            self.assertTrue(bridge_signal.exists())

    def test_live_cycle_does_not_overwrite_signal_without_new_setup(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            bars = root / "bars.csv"
            bars.write_text(
                "time,open,high,low,close,tick_volume\n"
                "2026-08-17 10:00:00,1,2,0,1.5,100\n"
            )
            signal = root / "ml_signal.csv"
            bridge = root / "bridge.csv"
            status = root / "status.json"
            with patch("modules.live_pipeline.load_model_metadata", return_value={"selected_model": "LR_VSA_BOT"}), patch(
                "modules.live_pipeline.load_selected_model", return_value=object()
            ), patch(
                "modules.live_pipeline.predict_latest_setup_from_bars", side_effect=ValueError("NO_NEW_VSA_SETUP: test")
            ):
                result = run_live_cycle(
                    bars, signal, bridge, status,
                    now_utc=datetime(2026, 8, 17, 10, 6, tzinfo=timezone.utc),
                )
            self.assertEqual(result["state"], "NO_NEW_VSA_SETUP")
            self.assertTrue(result["vsa_scanner_executed"])
            self.assertFalse(result["ml_inference_executed"])
            self.assertFalse(result["retraining_executed"])
            self.assertIn("Offline labelled VSA examples", result["training_data_role"])
            self.assertIn("Completed MT5 M5 bars", result["live_data_role"])
            self.assertFalse(signal.exists())
            self.assertFalse(bridge.exists())
            self.assertTrue(status.exists())


if __name__ == "__main__":
    unittest.main()
