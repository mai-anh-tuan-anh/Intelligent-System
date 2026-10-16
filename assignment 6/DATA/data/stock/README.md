# AAPL stock time series

## Source

Historical market data are retrieved using `yfinance` from Yahoo Finance.

`yfinance.download()` documentation:

https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html

## Experiment definition

- Ticker: `AAPL`
- Frequency: daily trading observations
- Period: `2010-01-01` through `2025-12-31`
- Download API end date: `2026-01-01` (exclusive)
- Target: next trading step closing price
- Lookback: 14 trading observations

The notebook caches the downloaded data locally as `AAPL_daily.csv` so the same historical dataset can be reused across Scratch, Keras, and PyTorch experiments.
