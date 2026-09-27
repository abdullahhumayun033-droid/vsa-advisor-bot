# Live MT5, VSA and trained-model traceability

This document is the examiner-facing map between the deployed components. It distinguishes rule detection, supervised learning, live inference and MT5 presentation.

## One canonical deployed flow

```text
MT5 VSA_M5_Bar_Exporter_EA.mq5
  CopyRates(XAUUSD, PERIOD_M5, shift=1)
             |
             v
MT5 Common Files/xauusd_m5_bars.csv
             |
             v
scripts/run_live_advisor.py -> modules/live_pipeline.py::run_live_cycle
             |
             +--> broker-clock freshness check
             |      stale / missing / clock-mismatched exports fail closed
             |
             +--> modules/vsa_strategy.py::scan_vsa_scenarios
             |      formal CAB + mirrored Scenario 1/2/3 rules
             |
             +--> modules/ml_pipeline.py::predict_vsa_setup_row
             |      load selected_model.joblib
             |      select the same 24 MODEL_FEATURES
             |      call predict_proba
             |
             +--> verified calendar coverage + high-impact USD safety gate
                    duplicate setup IDs are rechecked for safety every cycle
             |
             v
outputs/ml_signal.csv and MT5 Common Files/ml_signal.csv
             |
             +--> Streamlit Live MT5 End-to-End Proof page
             +--> VSA_ML_Signal_Bridge_EA.mq5 chart evidence/alerts/demo gate
```

## Training data versus live data

| Data | Purpose | Does it change the deployed model automatically? |
|---|---|---|
| `data/bars.csv` | Build labelled historical VSA examples; fit LR/RF; validate thresholds | Only when the controlled backend training command is run |
| MT5 `xauusd_m5_bars.csv` | Detect a new completed-bar VSA setup and construct its 24-feature inference row | No |
| `models/selected_model.joblib` | Stores the fitted scaler, LR coefficients, intercept and class mapping | Loaded read-only during inference |
| `ml_signal.csv` | Carries the selected model, VSA scenario, probability, threshold, risk, prices and news decision | No; it is one prediction contract |

Live data therefore affects the **current prediction**, not the learned coefficients. Controlled retraining is separate so that unseen evaluation, version checks and approval cannot be bypassed.

## Visible proof added for assessment

- `mt5/VSA_Live_Visual_Indicator.mq5` mirrors the completed-bar CAB and Scenario 1/2/3 state rules and draws the CAB zone, confirmation, entry, stop, 1R and 4R levels. Its panel explicitly says that it is a rule detector, not the ML model.
- The indicator writes `vsa_mql5_latest_audit.csv`. The website compares its signal time, CAB time, direction and scenario with Python and reports an explicit PASS or FAIL; a mismatch fails visibly rather than being hidden.
- `mt5/VSA_ML_Signal_Bridge_EA.mq5` displays the Python-selected scenario, trained model name, `predict_proba` result, threshold, decision, risk and news status. It uses a shared human-auditable signal ID.
- The Streamlit **Live Advisor** distinguishes LIVE, HISTORICAL DEMO and previous/expired states. Historical demo mode never writes to the MT5 bridge.
- The Live Advisor candlestick/tick-volume chart uses the same completed bars as the inference path and overlays the CAB zone, confirmation, entry, stop, 1R and 4R; it is presentation, not a second strategy implementation.
- The Streamlit **Live MT5 End-to-End Proof** page reads the same live-cycle status and canonical signal consumed by MT5. It exposes whether the VSA scanner and ML inference actually executed.
- `outputs/live_advisor_status.json` records the bars source, worker heartbeat, broker-clock freshness, expected/latest completed bar, saved-model artifact, scanner execution, inference execution, lack of automatic retraining, news status and local LR explanation.

## Important boundary

The MQL5 visual indicator mirrors the formal VSA domain rules for chart verification. The trained scikit-learn model remains in Python because `joblib` artifacts cannot be natively loaded by MQL5. MT5 receives the probability through the validated fixed CSV contract. This is a deliberate two-runtime deployment, not a claim that MT5 trains LR or RF.

## Demonstration checklist

1. Attach `VSA_M5_Bar_Exporter_EA` to one XAUUSD M5 chart.
2. Attach `VSA_ML_Signal_Bridge_EA` to a second XAUUSD M5 chart with demo execution disabled.
3. Attach `VSA_Live_Visual_Indicator` to either chart.
4. Run `./run_live_advisor.sh --broker-utc-offset <broker offset>`.
5. Open the website's **Live MT5 End-to-End Proof** page.
6. Show the live bars path and latest completed bar.
7. Show `selected_model.joblib` and `predict_proba()` evidence.
8. For a fresh setup, match the website and MT5 signal IDs, scenario, probability, threshold and decision.
9. Show the LIVE/STALE/DISCONNECTED status fields and explain that an old CSV cannot be presented as a current recommendation.
10. Explain that `NO_NEW_VSA_SETUP` is a valid state and that a duplicate setup is safety-refreshed without becoming a new signal ID or repeated alert.
