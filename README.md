# VSA Advisor Bot – Final VSA + ML Version


> **v9 calendar correctness fix (25 September 2026):** verifies the current-week and next-week Forex Factory exports independently, falls back from JSON to the official XML weekly export when needed, and never treats one successfully loaded week as proof that the other week contains no red-folder events. Empty-day claims are shown only for weeks whose export was actually verified.
> **Final build v9 (25 September 2026):** retains the frozen VSA+LR logic, TradingView-style novice market view, MT5 diagnostics and Trade Journal, and adds an embedded USD high-impact ("red-folder") calendar sourced from the official Forex Factory/Fair Economy weekly exports (JSON with XML fallback). The News Risk Filter now provides Today, This Week and Next Week views while the existing trading news-gate logic remains unchanged. The automated suite contains 37 passing tests.


This version fixes the project architecture according to the Financial Advisor Bot brief and supervisor feedback.

## Final logic

1. Two years of XAUUSD M5 candles are stored in `data/bars.csv`.
2. The VSA strategy scans the candles and creates candidate BUY/SELL setups.
3. Each setup is labelled using future TP/SL outcome: target before stop-loss = `VALID_TRADE`, otherwise `AVOID_OR_FAILED`.
4. Two separate advisor bots are trained in the backend:
   - `LR_VSA_BOT` = Logistic Regression trained on VSA setup outcomes.
   - `RF_VSA_BOT` = Random Forest trained on the same VSA setup outcomes.
5. Results are saved separately and compared in `results/model_comparison.csv`.
6. Logistic Regression won the stated validation-F1 criterion in this run and is saved as `models/selected_model.joblib`. Random Forest remains a separately trained comparison model.
7. The website is user-facing only; it does not expose training controls.
8. The latest selected-model signal is exported to `outputs/ml_signal.csv` for MT5 bridge integration.

## Set up the portable environment

The submitted project does **not** rely on a copied machine-specific virtual environment. Run:

```bash
cd ~/Documents/vsa_advisor_bot_FINAL_SUBMISSION_v9
./setup_environment.sh
```

This creates `.venv` locally and installs the pinned dependency versions, including the scikit-learn version required by the saved model.

## Run backend training for screenshots

```bash
./run_backend_training.sh
```

The dependency versions are pinned for reproducibility. The training metadata records the Python, scikit-learn, pandas, NumPy and joblib versions used to create the saved model; inference fails safely if a different scikit-learn version attempts to load it. Training prints progress in the terminal and saves `results/training_log.txt`.

The same command now also produces examiner-facing robustness and XAI evidence:

- `results/feature_set_walk_forward.csv` compares the 10-feature VSA core and full 24-feature context across four future folds.
- `results/feature_ablation_bootstrap.csv` reports 1,000-resample confidence intervals on the untouched test period.
- `results/additional_model_benchmarks.csv` compares Decision Tree and Histogram Gradient Boosting as contextual model families without changing the required LR/RF selection.
- `results/model_family_justification.csv` records why each family was retained, benchmarked or excluded.
- `results/xai_faithfulness_summary.csv` and `results/xai_local_probability_audit.csv` verify that all LR contributions plus the intercept reconstruct deployed probabilities.

Run the automated software and explanation-fidelity checks with:

```bash
./run_tests.sh
```

The finalisation pass adds regression checks for clock-based live-feed freshness, unrelated-calendar coverage, persistent journal behaviour, per-week Forex Factory verification and JSON-to-XML calendar fallback. The current suite contains **37 tests**.

## Run website

```bash
cd ~/Documents/vsa_advisor_bot_FINAL_SUBMISSION_v9
./run_website.sh
```

Open: `http://127.0.0.1:8501`

The default website navigation is intentionally novice-facing: Live Advisor, News Risk Filter, How the Bot Learns, Why This Advice, Risk & Safety, Learning Hub, Trade Journal, MT5 Connection, About the Bot and Feedback. Examiner/developer tables remain available through the sidebar's optional **Show technical audit pages** control, but are hidden from ordinary users by default.

The **News Risk Filter** embeds the official Forex Factory/Fair Economy weekly export directly inside the app; no external calendar page is required. It prefers the JSON export with a browser-compatible request, makes only the two weekly requests permitted by the source pattern (this week and next week), and caches the last successful response for display continuity. It filters to USD events marked High impact only (red-folder events), shows Today / This Week / Next Week calendar views, reports Actual / Forecast / Previous values when available, and explicitly distinguishes a verified no-event day from a failed/unverified feed. A cached or fallback calendar can never clear the trading news gate. The page is display/awareness functionality; it does not create a VSA setup or alter model probability.

The Live Advisor now includes a completed-candle XAUUSD M5 candlestick/tick-volume chart using the same pipeline candles, plus entry, stop, 1R/4R levels and the CAB region. It does **not** implement a second trading strategy. Historical demonstration mode is clearly separated from live mode and never overwrites the MT5 bridge signal.

The Trade Journal stores one local learning record per signal, including the source mode, advice snapshot, user decision, optional self-reported result and notes. Journal data is isolated from model training and never triggers an order.

## MT5 integration

The trained Python model writes `outputs/ml_signal.csv`. The included file `mt5/VSA_ML_Signal_Bridge_EA.mq5` shows how an MT5 Expert Advisor can read that signal from MT5 Common Files and send desktop/mobile alerts or demo trades.

For the final report/demo, describe MT5 as a bridge that consumes the trained model signal. MT5 does not train Logistic Regression or Random Forest.

## Live completed-M5 workflow

The live workflow has two deliberately separate MT5 EAs:

- `mt5/VSA_M5_Bar_Exporter_EA.mq5` is read-only. It exports 2,500 completed XAUUSD M5 candles to `xauusd_m5_bars.csv` in MT5 Common Files and never trades.
- `mt5/VSA_ML_Signal_Bridge_EA.mq5` consumes the fixed 29-column `ml_signal.csv` contract. Demo trading remains disabled by default.
- `mt5/VSA_Live_Visual_Indicator.mq5` visibly mirrors the Python CAB/Scenario 1/2/3 rules on completed M5 bars and draws the CAB zone, entry, stop and targets. It is rule evidence; the trained LR probability still comes from Python.

The website's **Live MT5 End-to-End Proof** page and the enhanced bridge show the same signal identity, scenario, trained-model probability, threshold and final decision. See `docs/LIVE_MT5_ML_TRACEABILITY.md` for the examiner-facing component map.

After compiling the EAs, open two XAUUSD M5 charts. Attach the exporter to one chart and the signal bridge to the other because MT5 permits only one EA per chart. Then start the Python worker:

```bash
cd ~/Documents/vsa_advisor_bot_FINAL_SUBMISSION_v9
./run_live_advisor.sh
```

The exporter now writes a small connection-status file, so the worker automatically detects the broker server UTC offset. You can still pass `--broker-utc-offset` manually as an override. The worker performs this sequence:

1. Read only completed M5 candles exported by MT5.
2. Apply the formal mirrored VSA Scenario 1/2/3 state machine.
3. Compare the latest exported completed bar with the current broker clock. A stale, missing or clock-mismatched feed fails closed and is labelled `STALE`, `DISCONNECTED` or `CLOCK_MISMATCH` rather than being presented as live.
4. If no new formal setup exists, write no new recommendation and report `NO_NEW_VSA_SETUP`.
5. If a recent setup exists, load the already-trained `LR_VSA_BOT` and estimate the probability of reaching 1R before the scenario stop.
6. Re-check the verified high-impact USD calendar on **every** cycle, including a duplicate setup. A calendar that is verified but does not cover the signal timestamp fails closed as `CALENDAR_OUT_OF_COVERAGE`.
7. Atomically refresh both the project signal and MT5 Common Files signal. The stable signal ID lets the MT5 bridge suppress repeat notifications while safety/news state can still change.
8. Save feed freshness, worker heartbeat, live state and faithful standardised LR feature contributions in `outputs/live_advisor_status.json`.

Use a single cycle for a diagnostic check:

```bash
./run_live_advisor.sh --once
```

The live worker is intentionally locked to `LR_VSA_BOT`. RF results are retained for fair examiner comparison, but RF cannot silently replace the transparent deployed advisor.

## MT5 Strategy Tester comparison

`mt5/VSA_LR_Backtest_Replay_EA.mq5` replays the frozen 65-setup untouched test period in MT5. Set `UseMLFilter=true` for VSA+LR and `false` for the matched VSA-only ablation. The source CSV is generated from the saved Python test predictions:

```bash
.venv/bin/python scripts/export_mt5_backtest_replay.py
```

The matched ExpertMACD, ExpertMAMA and ExpertMAPSAR runs are complete. Their genuine HTML reports are archived under `docs/evaluation/mt5_benchmarks/reports/`, the verified metrics are in `results/mt5_ea_comparison.csv`, and the interpretation protocol is in `docs/evaluation/mt5_benchmarks/MT5_EA_Benchmark_Protocol.md`.


## v4 UI status clarification
A healthy MT5 terminal + fresh M5 export + current Python worker is labelled **LIVE FEED** even when there is no newly confirmed VSA setup. Signal freshness is reported separately. Previous saved recommendations are collapsed under an audit-only expander so they cannot be mistaken for current advice.
