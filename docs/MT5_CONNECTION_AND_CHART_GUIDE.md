# MT5 connection and novice chart guide

## What a frozen MT5 chart means

The Python advisor cannot create live market data. MetaTrader 5 must first have an active broker-server session. A chart can remain visible using cached candles even when the terminal is disconnected, so an old chart is not proof of a live feed.

The final v5 build checks three separate things:

1. **MT5 terminal connection** — exporter heartbeat plus `TERMINAL_CONNECTED`.
2. **Completed M5 feed freshness** — newest exported completed candle versus the current broker clock.
3. **Python worker heartbeat** — recent `run_live_advisor` cycle.

A failure at any layer prevents the website from labelling the recommendation LIVE.

## Reconnect MetaTrader 5

1. In MT5 choose **File → Login to Trade Account**.
2. Enter the account number/password inside MT5 and select the correct broker server. Use a demo account for project testing.
3. Wait for the bottom-right MT5 network indicator to show an active connection and non-zero traffic.
4. Open a fresh **XAUUSD M5** chart (broker suffixes such as `XAUUSD.a` are supported by the exporter).
5. Attach `VSA_M5_Bar_Exporter_EA` to one chart and `VSA_ML_Signal_Bridge_EA` to a second chart.
6. Keep `EnableDemoTrading=false` on the bridge during evidence collection.
7. Turn on **Algo Trading** so both EAs can run.
8. Start `./run_live_advisor.sh` from the project folder.

The exporter now refuses to rewrite cached candles when MT5 reports that it is disconnected. It still writes its connection heartbeat, allowing Streamlit to display `MT5 TERMINAL DISCONNECTED` rather than a vague stale-feed message.

## TradingView-style chart

The Live Advisor uses the same completed bars already supplied to the project pipeline. The visual layer does not recalculate VSA or change model inputs.

The chart includes:

- dark TradingView-style background and grid;
- green/red candlesticks;
- colour-matched tick-volume bars;
- right-side price scale;
- zoom/pan and hover OHLC details;
- CAB zone and scenario marker for a current/historical demonstration setup;
- Entry, Stop, 1R and 4R overlays when they fall inside the visible candle range;
- off-screen target text instead of stretching the chart until candles become unreadable;
- last completed close and timestamp;
- a novice legend below the chart.

When the live feed is stale/disconnected, an expired previous recommendation is deliberately not drawn over the current-candle panel. This prevents an old setup from looking actionable.
