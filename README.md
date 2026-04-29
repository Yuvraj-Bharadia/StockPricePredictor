# Publishable Quant Research Starter

This workspace now contains a stronger version of the original hybrid transformer plus ensemble idea in [publishable_quant_platform.py](C:\Users\Yuvraj\Desktop\Quant Research Program\publishable_quant_platform.py).

Use [run_pipeline.py](C:\Users\Yuvraj\Desktop\Quant Research Program\run_pipeline.py) for a simple command-line run.

What is improved:

- Replaces the placeholder sentiment block with an interface that can ingest article-level data and aggregate it into daily sentiment factors.
- Expands the feature set into momentum, volatility, range-based estimators, liquidity, microstructure proxies, cross-asset context, calendar effects, and sentiment features.
- Uses a more realistic walk-forward process with periodic retraining, execution lag, turnover, transaction costs, and slippage.
- Adds a richer model stack: linear regularization, tree ensembles, histogram boosting, and an optional transformer sequence model.
- Adds reporting for Sharpe, Sortino, Calmar, hit rate, max drawdown, t-stat, p-value, and scenario-based stress tests.

Important research note:

- There is no "perfect algorithm ever." The best publishable setup is the one that is explicit about assumptions, data limits, validation design, and failure modes.
- True market microstructure modeling needs intraday trade/quote or depth data. Daily OHLCV data can only support proxies such as Amihud illiquidity, Roll spread proxy, range estimators, and signed dollar volume.

Suggested next upgrades before calling this paper-grade:

- Replace `yfinance` with a cleaner institutional data source for minute bars, trades, quotes, and corporate actions.
- Add purged and embargoed cross-validation for hyperparameter search.
- Add benchmark comparisons against buy-and-hold, simple momentum, and sector-neutral baselines.
- Add regime segmentation by volatility state, macro state, and earnings windows.
- Add point-in-time fundamentals, analyst revisions, options-implied features, and event labels.
- Add bootstrap confidence intervals and White's Reality Check or SPA-style multiple-testing controls.
- Add portfolio construction across a cross-section of names instead of a single ticker.

Current finance-news API sites worth evaluating for the sentiment layer:

- Alpha Vantage docs: https://www.alphavantage.co/documentation/
- Marketaux docs: https://www.marketaux.com/documentation
- Polygon news docs: https://polygon.io/docs/stocks/get_v2_reference_news
- EODHD sentiment docs: https://eodhd.com/financial-apis/stock-market-financial-news-api
- Benzinga docs: https://docs.benzinga.com/
- Financial Modeling Prep docs: https://site.financialmodelingprep.com/developer/docs

Run process:

1. Create or activate a Python 3.11 virtual environment.
2. Install dependencies from [requirements.txt](C:\Users\Yuvraj\Desktop\Quant Research Program\requirements.txt).
3. Copy [.env.example](C:\Users\Yuvraj\Desktop\Quant Research Program\.env.example) to `.env` if you want live Alpha Vantage news sentiment.
4. Add `ALPHAVANTAGE_API_KEY=...` to `.env`.
5. Run `python run_pipeline.py --symbol AAPL --sentiment-provider auto`.
5. If Yahoo data fails, retry or swap the market data loader to a more stable provider.

What is still a placeholder:

- If `ALPHAVANTAGE_API_KEY` is missing or Alpha Vantage returns no articles, the runner falls back to example sentiment data.
- You do not need a finance-news API key just to test the structure, but you do need one for real live sentiment.
- The default market data source is still Yahoo Finance through `yfinance`, which is convenient for research but not ideal for production or publication-quality reproducibility.

Recommended publishing workflow:

1. Lock the dataset date range and universe before tuning.
2. Save every experiment configuration and seed.
3. Compare against naive baselines and standard factor models.
4. Report gross and net results with explicit cost assumptions.
5. Show robustness by symbol, regime, and subperiod.
6. Replace daily proxies with true intraday microstructure data if your paper claims market microstructure alpha.
