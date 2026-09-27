# MT5 Expert Advisor Benchmark Protocol

## Purpose

This evaluation compares the proposed VSA+LR advisor with its VSA-only ablation and three MetaQuotes indicator-based sample EAs. It is a contextual comparison on the same XAUUSD M5 market period, not a claim that unlike strategies take equivalent trades.

## Common tester window

- Symbol: broker XAUUSD symbol (for example `XAUUSD`)
- Timeframe: M5
- From: 2026-06-01
- To: 2026-08-17
- Modelling: Every tick based on real ticks, where available
- Deposit: USD 10,000
- Leverage: 1:100
- Execution delay: No delay for the primary reproducible run; add a realistic-delay sensitivity run if time permits
- Optimization: Disabled
- News filter: Disabled for every EA because equivalent verified historical news data is not included

Use the same broker, symbol, dates, deposit, leverage, modelling mode, commission and spread history for every row.

The frozen Python test partition contains formal VSA setup signals from 2026-06-16 onward. Starting the MT5 tester on 2026-06-01 supplies a common warm-up window while preserving the untouched VSA/LR signal boundary. It does not move training data into the LR evaluation.

## Systems and saved report names

1. `VSA_LR_Backtest_Replay_EA` with `UseMLFilter=true`, `UseStrategyRiskSizing=false`, `FixedLots=0.01` → `reports/01_VSA_LR/01_VSA_LR_fixed001.html`
2. `VSA_LR_Backtest_Replay_EA` with `UseMLFilter=false`, `UseStrategyRiskSizing=false`, `FixedLots=0.01` → `reports/02_VSA_only/02_VSA_only_fixed001.html`
3. `ExpertMACD` using its default parameters and 0.01 lot → `reports/03_ExpertMACD/03_ExpertMACD.html`
4. `ExpertMAMA` using its default parameters and 0.01 lot → `reports/04_ExpertMAMA/04_ExpertMAMA.html`
5. `ExpertMAPSAR` using its default parameters and 0.01 lot → `reports/05_ExpertMAPSAR/05_ExpertMAPSAR.html`

`ExpertMAPSARSizeOptimized` is excluded from the primary table because an optimized variant is not equivalent to untuned/default baselines. It may be reported separately only if its optimization period, objective and untouched forward period are documented.

## Installing the replay benchmark

1. Copy `mt5/VSA_LR_Backtest_Replay_EA.mq5` into the terminal's `MQL5/Experts/Advisors/` directory.
2. Copy `mt5/vsa_lr_test_signals.csv` into `MQL5/Files/`.
3. Open the EA in MetaEditor and compile. Require `0 errors, 0 warnings`.
4. Restart or refresh the MT5 Navigator.
5. Open Strategy Tester and select the settings above.

The EA declares `vsa_lr_test_signals.csv` using `#property tester_file`, allowing MT5 to copy the frozen replay file to the local tester agent. The CSV contains only the 65 untouched Python test setups. With the trained threshold of 0.35, 45 are LR candidates before realistic one-position-at-a-time execution constraints.

## Two VSA runs are required

The VSA-only and VSA+LR runs use the same setup times, stops, next-bar execution and trade management. `UseMLFilter` is the only model-related difference. This is the primary experiment for whether ML filtering adds value.

For an additional strategy-faithful sensitivity run, set `UseStrategyRiskSizing=true`. Scenario 1 then risks 0.5% and Scenarios 2/3 risk 1%. Do not mix this run with fixed-lot EA results in the same profit ranking.

## Metrics to transcribe from every HTML report

- Net profit and return percentage
- Maximum equity drawdown percentage
- Profit factor
- Expected payoff
- Sharpe ratio
- Recovery factor
- Total trades
- Win rate
- Maximum consecutive losses

Use `results/mt5_ea_comparison.csv`. Do not rank by net profit alone; discuss return together with drawdown, profit factor, Sharpe ratio, trade count and exposure differences.

## Required evidence

- MetaEditor compilation screenshot showing 0 errors and 0 warnings
- Strategy Tester Settings screenshot for the first run
- Overview/Backtest report screenshot for every EA
- Complete saved HTML report for every EA
- Experts/Journal screenshot showing the replay summary counts
- Completed `results/mt5_ea_comparison.csv`

## Completed results

| System | Net profit | Return | Max equity DD | Profit factor | Sharpe | Trades | Win rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| VSA+LR | $284.84 | 2.85% | 1.31% | 1.74 | 5.97 | 35 | 31.43% |
| VSA-only | $23.21 | 0.23% | 1.69% | 1.04 | 0.40 | 46 | 21.74% |
| ExpertMACD | -$180.58 | -1.81% | 1.81% | 0.30 | -5.00 | 1,118 | 12.34% |
| ExpertMAMA | -$101.80 | -1.02% | 5.93% | 0.39 | -0.39 | 22 | 90.91% |
| ExpertMAPSAR | $1,291.65 | 12.92% | 2.03% | 15.13 | 6.02 | 80 | 96.25% |

All five reports verify XAUUSD M5, 1 June–17 August 2026, USD 10,000, 1:100 leverage, 100% real-tick history and 0.01-lot orders. The VSA+LR versus VSA-only pair is the primary controlled ablation because `UseMLFilter` is the only model-related change. The MetaQuotes EAs are contextual baselines with different trading logic.

## Interpretation limits

- The replay EA evaluates frozen, genuinely unseen Python VSA/LR signals through MT5 execution. It does not retrain ML in MT5.
- The built-in EAs were designed around different indicators and may not be optimized for XAUUSD M5.
- The short untouched period and small VSA trade counts produce uncertain estimates.
- Results do not establish future profitability.
