from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from scipy import stats
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


PRICE_COLS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
BASE_FEATURES = [
    "log_ret_1",
    "log_ret_5",
    "log_ret_10",
    "vol_5",
    "vol_10",
    "vol_20",
    "down_vol_20",
    "up_vol_20",
    "skew_20",
    "kurt_20",
    "momentum_5",
    "momentum_10",
    "momentum_20",
    "price_to_ma_5",
    "price_to_ma_10",
    "price_to_ma_20",
    "price_to_ma_50",
    "ema_spread_12_26",
    "macd_signal_diff",
    "rsi_14",
    "stoch_k_14",
    "stoch_d_14",
    "atr_14",
    "parkinson_vol_10",
    "gk_vol_10",
    "rs_vol_10",
    "close_location_value",
    "intraday_range",
    "gap_return",
    "amihud_illiq_20",
    "roll_spread_proxy_20",
    "signed_dollar_volume",
    "dollar_volume_log",
    "volume_z_5",
    "volume_z_20",
    "obv_z_20",
    "beta_20",
    "idio_vol_20",
    "rel_strength_market_10",
    "rel_strength_sector_10",
    "corr_market_20",
    "market_ret_1",
    "market_vol_20",
    "vix_level",
    "vix_change_5",
    "bond_ret_5",
    "gold_ret_5",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "month_end",
    "quarter_end",
    "sentiment_mean",
    "sentiment_std",
    "sentiment_count",
    "sentiment_pos_share",
    "sentiment_neg_share",
    "sentiment_recency_decay",
    "sentiment_source_entropy",
    "sentiment_novelty",
]


@dataclass
class DataConfig:
    symbol: str = "AAPL"
    benchmark_symbol: str = "SPY"
    sector_symbol: str = "XLK"
    vix_symbol: str = "^VIX"
    bond_symbol: str = "TLT"
    gold_symbol: str = "GLD"
    lookback_period: str = "10y"
    interval: str = "1d"
    auto_adjust: bool = False


@dataclass
class ModelingConfig:
    target_horizon: int = 1
    min_train_size: int = 252 * 2
    retrain_frequency: int = 5
    sequence_window: int = 32
    transformer_epochs: int = 10
    transformer_batch_size: int = 32
    transformer_enabled: bool = True
    random_state: int = 42


@dataclass
class BacktestConfig:
    signal_clip: float = 2.0
    target_volatility: float = 0.15
    annualization: int = 252
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 2.0
    execution_lag: int = 1
    max_gross_leverage: float = 1.0


@dataclass
class ProviderSpec:
    name: str
    docs_url: str
    endpoint_hint: str
    notes: str


NEWS_PROVIDER_SPECS: Dict[str, ProviderSpec] = {
    "alphavantage": ProviderSpec(
        name="Alpha Vantage",
        docs_url="https://www.alphavantage.co/documentation/",
        endpoint_hint="function=NEWS_SENTIMENT",
        notes="Broad market and ticker-filtered news with built-in sentiment and topic filters.",
    ),
    "marketaux": ProviderSpec(
        name="Marketaux",
        docs_url="https://www.marketaux.com/documentation",
        endpoint_hint="GET https://api.marketaux.com/v1/news/all",
        notes="Finance-native article feed with entity mapping and per-entity sentiment.",
    ),
    "polygon": ProviderSpec(
        name="Polygon",
        docs_url="https://polygon.io/docs/stocks/get_v2_reference_news",
        endpoint_hint="GET /v2/reference/news",
        notes="Ticker-linked news feed with summaries and sentiment analysis in article insights.",
    ),
    "eodhd": ProviderSpec(
        name="EODHD",
        docs_url="https://eodhd.com/financial-apis/stock-market-financial-news-api",
        endpoint_hint="GET https://eodhd.com/api/sentiments",
        notes="Point-in-time sentiment scores from news and social media for instruments.",
    ),
    "benzinga": ProviderSpec(
        name="Benzinga",
        docs_url="https://docs.benzinga.com/",
        endpoint_hint="Stock News API / Benzinga Cloud",
        notes="Low-latency editorial feed often used by event-driven desks and platforms.",
    ),
    "fmp": ProviderSpec(
        name="Financial Modeling Prep",
        docs_url="https://site.financialmodelingprep.com/developer/docs",
        endpoint_hint="https://financialmodelingprep.com/stable/news/stock?symbols=AAPL",
        notes="Convenient stock news endpoint that pairs well with its fundamentals and estimates.",
    ),
}


def _download_single(symbol: str, config: DataConfig) -> pd.DataFrame:
    frame = yf.download(
        symbol,
        period=config.lookback_period,
        interval=config.interval,
        auto_adjust=config.auto_adjust,
        progress=False,
        threads=False,
    )
    if frame.empty:
        raise ValueError(
            f"No price data returned for {symbol}. "
            "The current default data source is yfinance/Yahoo, which can fail transiently "
            "or be rate-limited. Retry later, try a different symbol, or replace the loader "
            "with a more stable market data provider."
        )
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [c[0] for c in frame.columns]
    frame = frame.rename_axis("Date").reset_index()
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
    for col in PRICE_COLS:
        if col not in frame.columns and col == "Adj Close" and "Close" in frame.columns:
            frame["Adj Close"] = frame["Close"]
    return frame[["Date"] + [c for c in PRICE_COLS if c in frame.columns]].copy()


def load_market_data(config: DataConfig) -> Dict[str, pd.DataFrame]:
    symbols = {
        "asset": config.symbol,
        "benchmark": config.benchmark_symbol,
        "sector": config.sector_symbol,
        "vix": config.vix_symbol,
        "bond": config.bond_symbol,
        "gold": config.gold_symbol,
    }
    return {name: _download_single(symbol, config) for name, symbol in symbols.items()}


def _safe_log_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    return np.log(a.replace(0, np.nan) / b.replace(0, np.nan))


def _rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0, np.nan)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / window, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / window, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _stochastic_oscillator(close: pd.Series, low: pd.Series, high: pd.Series, window: int = 14) -> Tuple[pd.Series, pd.Series]:
    rolling_low = low.rolling(window).min()
    rolling_high = high.rolling(window).max()
    k = 100 * (close - rolling_low) / (rolling_high - rolling_low).replace(0, np.nan)
    d = k.rolling(3).mean()
    return k, d


def _average_true_range(df: pd.DataFrame, window: int = 14) -> pd.Series:
    prev_close = df["Adj Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


def _close_location_value(df: pd.DataFrame) -> pd.Series:
    denominator = (df["High"] - df["Low"]).replace(0, np.nan)
    return ((df["Adj Close"] - df["Low"]) - (df["High"] - df["Adj Close"])) / denominator


def _on_balance_volume(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["Adj Close"].diff()).fillna(0.0)
    return (direction * df["Volume"]).cumsum()


def _rolling_beta(asset_ret: pd.Series, benchmark_ret: pd.Series, window: int = 20) -> Tuple[pd.Series, pd.Series, pd.Series]:
    cov = asset_ret.rolling(window).cov(benchmark_ret)
    var = benchmark_ret.rolling(window).var()
    beta = cov / var.replace(0, np.nan)
    corr = asset_ret.rolling(window).corr(benchmark_ret)
    resid = asset_ret - beta * benchmark_ret
    idio_vol = resid.rolling(window).std()
    return beta, corr, idio_vol


def _range_estimators(df: pd.DataFrame, window: int = 10) -> Tuple[pd.Series, pd.Series, pd.Series]:
    log_hl = np.log(df["High"] / df["Low"])
    log_co = np.log(df["Adj Close"] / df["Open"])
    log_ho = np.log(df["High"] / df["Open"])
    log_lo = np.log(df["Low"] / df["Open"])
    log_hc = np.log(df["High"] / df["Adj Close"])
    log_lc = np.log(df["Low"] / df["Adj Close"])

    parkinson = (1.0 / (4.0 * np.log(2.0)) * log_hl.pow(2)).rolling(window).mean().pow(0.5)
    gk = (0.5 * log_hl.pow(2) - (2 * np.log(2) - 1) * log_co.pow(2)).rolling(window).mean().clip(lower=0).pow(0.5)
    rs = (log_ho * log_hc + log_lo * log_lc).rolling(window).mean().clip(lower=0).pow(0.5)
    return parkinson, gk, rs


def _amihud_illiq(log_ret: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    ratio = log_ret.abs() / dollar_volume.replace(0, np.nan)
    return ratio.rolling(window).mean()


def _roll_spread_proxy(price: pd.Series, window: int = 20) -> pd.Series:
    delta = price.diff()
    cov = delta.rolling(window).cov(delta.shift(1))
    return 2 * np.sqrt((-cov).clip(lower=0))


def _cyclical_encoding(index: pd.Series, period: int) -> Tuple[pd.Series, pd.Series]:
    angle = 2 * np.pi * index / period
    return np.sin(angle), np.cos(angle)


def build_sentiment_features(
    article_frame: Optional[pd.DataFrame],
    dates: pd.Series,
) -> pd.DataFrame:
    base = pd.DataFrame({"Date": pd.to_datetime(dates)})
    for col in [
        "sentiment_mean",
        "sentiment_std",
        "sentiment_count",
        "sentiment_pos_share",
        "sentiment_neg_share",
        "sentiment_recency_decay",
        "sentiment_source_entropy",
        "sentiment_novelty",
    ]:
        base[col] = 0.0

    if article_frame is None or article_frame.empty:
        return base

    df = article_frame.copy()
    df["published_at"] = pd.to_datetime(df["published_at"]).dt.tz_localize(None)
    if "sentiment" not in df.columns:
        raise ValueError("article_frame must contain a 'sentiment' column.")
    df["date"] = df["published_at"].dt.floor("D")
    df["is_positive"] = (df["sentiment"] > 0).astype(float)
    df["is_negative"] = (df["sentiment"] < 0).astype(float)

    if "source" not in df.columns:
        df["source"] = "unknown"
    if "novelty" not in df.columns:
        df["novelty"] = 1.0

    def source_entropy(sources: pd.Series) -> float:
        counts = sources.value_counts(normalize=True)
        return float(-(counts * np.log(counts + 1e-12)).sum())

    grouped = df.groupby("date").agg(
        sentiment_mean=("sentiment", "mean"),
        sentiment_std=("sentiment", "std"),
        sentiment_count=("sentiment", "size"),
        sentiment_pos_share=("is_positive", "mean"),
        sentiment_neg_share=("is_negative", "mean"),
        sentiment_novelty=("novelty", "mean"),
    )
    grouped["sentiment_source_entropy"] = df.groupby("date")["source"].apply(source_entropy)

    decay_values = []
    for date, group in df.groupby("date"):
        age_hours = (date + pd.Timedelta(days=1) - group["published_at"]).dt.total_seconds() / 3600.0
        weights = np.exp(-np.maximum(age_hours, 0.0) / 12.0)
        decay_values.append((date, float(np.average(group["sentiment"], weights=weights))))
    grouped["sentiment_recency_decay"] = pd.Series(dict(decay_values))

    grouped = grouped.reset_index().rename(columns={"date": "Date"})
    return base.merge(grouped, how="left", on="Date").fillna(0.0)


def build_feature_matrix(
    market_data: Dict[str, pd.DataFrame],
    sentiment_articles: Optional[pd.DataFrame] = None,
    target_horizon: int = 1,
) -> pd.DataFrame:
    asset = market_data["asset"].copy()
    asset["Date"] = pd.to_datetime(asset["Date"])
    asset = asset.sort_values("Date").reset_index(drop=True)

    if "Adj Close" not in asset.columns:
        asset["Adj Close"] = asset["Close"]

    for aux in ["benchmark", "sector", "vix", "bond", "gold"]:
        frame = market_data[aux][["Date", "Adj Close"]].copy()
        frame["Date"] = pd.to_datetime(frame["Date"])
        asset = asset.merge(frame, how="left", on="Date", suffixes=("", f"_{aux}"))

    close = asset["Adj Close"]
    high = asset["High"]
    low = asset["Low"]
    open_ = asset["Open"]
    volume = asset["Volume"].replace(0, np.nan)
    benchmark_close = asset["Adj Close_benchmark"]
    sector_close = asset["Adj Close_sector"]
    vix_close = asset["Adj Close_vix"]
    bond_close = asset["Adj Close_bond"]
    gold_close = asset["Adj Close_gold"]

    asset["log_ret_1"] = _safe_log_ratio(close, close.shift(1))
    asset["log_ret_5"] = _safe_log_ratio(close, close.shift(5))
    asset["log_ret_10"] = _safe_log_ratio(close, close.shift(10))

    asset["vol_5"] = asset["log_ret_1"].rolling(5).std()
    asset["vol_10"] = asset["log_ret_1"].rolling(10).std()
    asset["vol_20"] = asset["log_ret_1"].rolling(20).std()
    asset["down_vol_20"] = asset["log_ret_1"].clip(upper=0).rolling(20).std()
    asset["up_vol_20"] = asset["log_ret_1"].clip(lower=0).rolling(20).std()
    asset["skew_20"] = asset["log_ret_1"].rolling(20).skew()
    asset["kurt_20"] = asset["log_ret_1"].rolling(20).kurt()

    for window in [5, 10, 20]:
        asset[f"momentum_{window}"] = close / close.shift(window) - 1.0
        asset[f"price_to_ma_{window}"] = close / close.rolling(window).mean() - 1.0
    asset["price_to_ma_50"] = close / close.rolling(50).mean() - 1.0

    ema_12 = _ema(close, 12)
    ema_26 = _ema(close, 26)
    macd = ema_12 - ema_26
    macd_signal = _ema(macd, 9)
    asset["ema_spread_12_26"] = ema_12 / ema_26 - 1.0
    asset["macd_signal_diff"] = macd - macd_signal
    asset["rsi_14"] = _rsi(close, 14)
    stoch_k, stoch_d = _stochastic_oscillator(close, low, high, 14)
    asset["stoch_k_14"] = stoch_k
    asset["stoch_d_14"] = stoch_d

    asset["atr_14"] = _average_true_range(asset, 14)
    parkinson, gk, rs = _range_estimators(asset, 10)
    asset["parkinson_vol_10"] = parkinson
    asset["gk_vol_10"] = gk
    asset["rs_vol_10"] = rs

    asset["close_location_value"] = _close_location_value(asset)
    asset["intraday_range"] = (high - low) / close.replace(0, np.nan)
    asset["gap_return"] = open_ / close.shift(1) - 1.0

    asset["dollar_volume_log"] = np.log((close * volume).replace(0, np.nan))
    asset["amihud_illiq_20"] = _amihud_illiq(asset["log_ret_1"], close * volume, 20)
    asset["roll_spread_proxy_20"] = _roll_spread_proxy(close, 20)
    asset["signed_dollar_volume"] = np.sign(asset["log_ret_1"]).fillna(0.0) * close * volume
    asset["volume_z_5"] = _rolling_zscore(np.log(volume), 5)
    asset["volume_z_20"] = _rolling_zscore(np.log(volume), 20)
    asset["obv_z_20"] = _rolling_zscore(_on_balance_volume(asset), 20)

    market_ret = _safe_log_ratio(benchmark_close, benchmark_close.shift(1))
    sector_ret = _safe_log_ratio(sector_close, sector_close.shift(1))
    beta, corr, idio_vol = _rolling_beta(asset["log_ret_1"], market_ret, 20)
    asset["beta_20"] = beta
    asset["corr_market_20"] = corr
    asset["idio_vol_20"] = idio_vol
    asset["market_ret_1"] = market_ret
    asset["market_vol_20"] = market_ret.rolling(20).std()
    asset["rel_strength_market_10"] = asset["log_ret_10"] - _safe_log_ratio(benchmark_close, benchmark_close.shift(10))
    asset["rel_strength_sector_10"] = asset["log_ret_10"] - _safe_log_ratio(sector_close, sector_close.shift(10))

    asset["vix_level"] = vix_close
    asset["vix_change_5"] = _safe_log_ratio(vix_close, vix_close.shift(5))
    asset["bond_ret_5"] = _safe_log_ratio(bond_close, bond_close.shift(5))
    asset["gold_ret_5"] = _safe_log_ratio(gold_close, gold_close.shift(5))

    weekday = asset["Date"].dt.weekday
    month = asset["Date"].dt.month - 1
    asset["weekday_sin"], asset["weekday_cos"] = _cyclical_encoding(weekday, 7)
    asset["month_sin"], asset["month_cos"] = _cyclical_encoding(month, 12)
    asset["month_end"] = asset["Date"].dt.is_month_end.astype(float)
    asset["quarter_end"] = asset["Date"].dt.is_quarter_end.astype(float)

    sentiment_features = build_sentiment_features(sentiment_articles, asset["Date"])
    asset = asset.merge(sentiment_features, how="left", on="Date", suffixes=("", "_sent"))

    asset["target"] = _safe_log_ratio(close.shift(-target_horizon), close)
    asset = asset.replace([np.inf, -np.inf], np.nan).dropna(subset=["target"]).reset_index(drop=True)
    return asset


def build_tabular_models(random_state: int = 42) -> Dict[str, Pipeline]:
    return {
        "ridge": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", Ridge(alpha=3.0)),
            ]
        ),
        "elastic_net": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
                ("model", ElasticNet(alpha=0.001, l1_ratio=0.2, random_state=random_state, max_iter=2000)),
            ]
        ),
        "rf": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_estimators=400, max_depth=6, min_samples_leaf=5, random_state=random_state, n_jobs=-1)),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", ExtraTreesRegressor(n_estimators=500, max_depth=7, min_samples_leaf=4, random_state=random_state, n_jobs=-1)),
            ]
        ),
        "hgb": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(max_depth=4, learning_rate=0.03, max_iter=300, random_state=random_state)),
            ]
        ),
    }


def _load_tensorflow():
    try:
        import tensorflow as tf
        from tensorflow.keras import layers

        return tf, layers
    except Exception:  # pragma: no cover - optional dependency
        return None, None


def build_transformer(input_shape: Tuple[int, int]):
    tf, layers = _load_tensorflow()
    if tf is None or layers is None:
        return None

    inputs = tf.keras.Input(shape=input_shape)
    x = layers.LayerNormalization()(inputs)
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=16, dropout=0.1)(x, x)
    x = layers.Add()([x, attn])
    ff = layers.LayerNormalization()(x)
    ff = layers.Dense(64, activation="gelu")(ff)
    ff = layers.Dropout(0.1)(ff)
    ff = layers.Dense(32, activation="gelu")(ff)
    x = layers.Add()([x, ff])
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="gelu")(x)
    outputs = layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss="mse")
    return model


def create_sequences(frame: pd.DataFrame, feature_columns: Sequence[str], window: int) -> Tuple[np.ndarray, np.ndarray]:
    matrix = frame.loc[:, feature_columns].to_numpy()
    target = frame["target"].to_numpy()
    X, y = [], []
    for idx in range(window, len(frame)):
        X.append(matrix[idx - window : idx])
        y.append(target[idx])
    return np.asarray(X), np.asarray(y)


def _rank_inverse_error_weights(errors: Dict[str, float]) -> Dict[str, float]:
    clipped = {k: 1.0 / max(v, 1e-8) for k, v in errors.items()}
    total = sum(clipped.values())
    return {k: v / total for k, v in clipped.items()}


def _fit_and_predict_tabular(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: Sequence[str],
    random_state: int,
) -> Tuple[Dict[str, float], Dict[str, Pipeline]]:
    models = build_tabular_models(random_state=random_state)
    train_X = train.loc[:, feature_columns]
    train_y = train["target"]
    test_X = test.loc[:, feature_columns]

    preds = {}
    for name, model in models.items():
        model.fit(train_X, train_y)
        preds[name] = float(model.predict(test_X)[0])
    return preds, models


def walk_forward_backtest(
    feature_frame: pd.DataFrame,
    feature_columns: Optional[Sequence[str]] = None,
    modeling: Optional[ModelingConfig] = None,
    backtest: Optional[BacktestConfig] = None,
) -> pd.DataFrame:
    modeling = modeling or ModelingConfig()
    backtest = backtest or BacktestConfig()
    feature_columns = list(feature_columns or BASE_FEATURES)

    rows: List[Dict[str, float]] = []
    tabular_errors = {name: 1.0 for name in build_tabular_models(modeling.random_state)}
    tf, _ = _load_tensorflow() if modeling.transformer_enabled else (None, None)

    for idx in range(modeling.min_train_size, len(feature_frame)):
        if idx % modeling.retrain_frequency != 0 and rows:
            prev = rows[-1].copy()
            prev["Date"] = feature_frame.loc[idx, "Date"]
            prev["actual"] = feature_frame.loc[idx, "target"]
            rows.append(prev)
            continue

        train = feature_frame.iloc[:idx].copy()
        test = feature_frame.iloc[idx : idx + 1].copy()

        tab_preds, fitted_models = _fit_and_predict_tabular(train, test, feature_columns, modeling.random_state)

        validation_slice = train.iloc[-max(60, modeling.sequence_window + 20) :]
        model_weights = _rank_inverse_error_weights(tabular_errors)
        tabular_pred = sum(model_weights[name] * value for name, value in tab_preds.items())

        transformer_pred = tabular_pred
        if modeling.transformer_enabled and tf is not None:
            seq_train = train.dropna(subset=feature_columns)
            if len(seq_train) > modeling.sequence_window + 100:
                X_seq, y_seq = create_sequences(seq_train, feature_columns, modeling.sequence_window)
                if len(X_seq) > 50:
                    scaler = RobustScaler()
                    reshaped = X_seq.reshape(-1, X_seq.shape[-1])
                    scaled = scaler.fit_transform(reshaped).reshape(X_seq.shape)
                    t_model = build_transformer((scaled.shape[1], scaled.shape[2]))
                    if t_model is not None:
                        t_model.fit(
                            scaled,
                            y_seq,
                            epochs=modeling.transformer_epochs,
                            batch_size=modeling.transformer_batch_size,
                            verbose=0,
                        )
                        latest = seq_train.iloc[-modeling.sequence_window :, :].loc[:, feature_columns].to_numpy()
                        latest = scaler.transform(latest).reshape(1, modeling.sequence_window, len(feature_columns))
                        transformer_pred = float(t_model.predict(latest, verbose=0)[0][0])

        blend = 0.75 * tabular_pred + 0.25 * transformer_pred
        for name, model in fitted_models.items():
            val_pred = model.predict(validation_slice.loc[:, feature_columns])
            error = float(np.sqrt(np.mean((validation_slice["target"].to_numpy() - val_pred) ** 2)))
            tabular_errors[name] = error

        rows.append(
            {
                "Date": test["Date"].iloc[0],
                "actual": float(test["target"].iloc[0]),
                "pred_tabular": tabular_pred,
                "pred_transformer": transformer_pred,
                "prediction": blend,
            }
        )

    result = pd.DataFrame(rows).drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    if result.empty:
        raise ValueError("Not enough observations to run the walk-forward backtest.")

    signal_raw = result["prediction"] / result["prediction"].rolling(60, min_periods=20).std()
    result["signal_raw"] = signal_raw.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    result["signal"] = result["signal_raw"].clip(-backtest.signal_clip, backtest.signal_clip)
    result["position"] = (result["signal"] / backtest.signal_clip).clip(-1.0, 1.0) * backtest.max_gross_leverage
    result["position"] = result["position"].shift(backtest.execution_lag).fillna(0.0)
    result["turnover"] = result["position"].diff().abs().fillna(result["position"].abs())
    total_cost = (backtest.transaction_cost_bps + backtest.slippage_bps) / 10000.0
    result["gross_return"] = result["position"] * result["actual"]
    result["net_return"] = result["gross_return"] - result["turnover"] * total_cost
    result["equity"] = np.exp(result["net_return"].cumsum())
    return result


def performance_metrics(returns: pd.Series, annualization: int = 252) -> Dict[str, float]:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        raise ValueError("No returns supplied.")

    equity = np.exp(returns.cumsum())
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    downside = returns[returns < 0]

    sharpe = np.sqrt(annualization) * returns.mean() / returns.std(ddof=1)
    sortino = np.sqrt(annualization) * returns.mean() / downside.std(ddof=1) if len(downside) > 1 else np.nan
    ann_return = np.exp(returns.mean() * annualization) - 1.0
    ann_vol = returns.std(ddof=1) * np.sqrt(annualization)
    max_dd = float(drawdown.min())
    calmar = ann_return / abs(max_dd) if max_dd < 0 else np.nan
    hit_rate = float((returns > 0).mean())
    tail_ratio = returns.quantile(0.95) / abs(returns.quantile(0.05)) if returns.quantile(0.05) != 0 else np.nan
    t_stat, p_value = stats.ttest_1samp(returns, 0.0)

    return {
        "annual_return": float(ann_return),
        "annual_volatility": float(ann_vol),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": max_dd,
        "calmar": float(calmar),
        "hit_rate": hit_rate,
        "tail_ratio": float(tail_ratio),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
    }


def run_stress_tests(
    result_frame: pd.DataFrame,
    base_backtest: BacktestConfig,
) -> pd.DataFrame:
    scenarios = []
    net = result_frame["net_return"].copy()

    scenarios.append(("baseline", net))
    scenarios.append(("cost_x2", net - result_frame["turnover"] * (base_backtest.transaction_cost_bps / 10000.0)))
    scenarios.append(("cost_x4", net - result_frame["turnover"] * (3 * base_backtest.transaction_cost_bps / 10000.0)))
    scenarios.append(("prediction_haircut_25pct", result_frame["gross_return"] * 0.75 - (result_frame["gross_return"] - net)))
    scenarios.append(("prediction_haircut_50pct", result_frame["gross_return"] * 0.50 - (result_frame["gross_return"] - net)))
    scenarios.append(("volatility_shock_1p5x", net * 1.5))
    scenarios.append(("drawdown_shock_minus_5bps_daily", net - 0.0005))

    rows = []
    for name, series in scenarios:
        row = performance_metrics(series, annualization=base_backtest.annualization)
        row["scenario"] = name
        rows.append(row)
    return pd.DataFrame(rows)[
        [
            "scenario",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "sortino",
            "max_drawdown",
            "calmar",
            "hit_rate",
            "tail_ratio",
            "t_stat",
            "p_value",
        ]
    ]


def describe_news_providers() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "provider": spec.name,
                "docs_url": spec.docs_url,
                "endpoint_hint": spec.endpoint_hint,
                "notes": spec.notes,
            }
            for spec in NEWS_PROVIDER_SPECS.values()
        ]
    )


def build_alpha_vantage_news_request(symbol: str, api_key: str, limit: int = 200) -> str:
    return (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={symbol}&sort=LATEST&limit={limit}&apikey={api_key}"
    )


def build_marketaux_news_request(symbol: str, api_key: str, page: int = 1, limit: int = 50) -> str:
    return (
        "https://api.marketaux.com/v1/news/all"
        f"?symbols={symbol}&filter_entities=true&language=en&limit={limit}&page={page}&api_token={api_key}"
    )


def build_polygon_news_request(symbol: str, api_key: str, limit: int = 100) -> str:
    return f"https://api.polygon.io/v2/reference/news?ticker={symbol}&limit={limit}&apiKey={api_key}"


def build_eodhd_sentiment_request(symbol: str, api_key: str, start_date: str, end_date: str) -> str:
    return (
        "https://eodhd.com/api/sentiments"
        f"?s={symbol}&from={start_date}&to={end_date}&api_token={api_key}&fmt=json"
    )


def fetch_json(url: str, timeout: int = 30) -> dict:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def _period_to_time_from(period: str) -> Optional[str]:
    now = datetime.utcnow()
    mapping = {
        "1mo": timedelta(days=31),
        "3mo": timedelta(days=93),
        "6mo": timedelta(days=186),
        "1y": timedelta(days=366),
        "2y": timedelta(days=366 * 2),
        "5y": timedelta(days=366 * 5),
        "10y": timedelta(days=366 * 10),
        "max": timedelta(days=366 * 20),
    }
    delta = mapping.get(period.lower())
    if delta is None:
        return None
    return (now - delta).strftime("%Y%m%dT%H%M")


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_alpha_vantage_articles(payload: dict, symbol: str) -> pd.DataFrame:
    if "Error Message" in payload:
        raise ValueError(f"Alpha Vantage error: {payload['Error Message']}")
    if "Note" in payload:
        raise ValueError(f"Alpha Vantage note: {payload['Note']}")
    if "Information" in payload and "feed" not in payload:
        raise ValueError(f"Alpha Vantage information: {payload['Information']}")

    feed = payload.get("feed", [])
    rows: List[Dict[str, object]] = []
    normalized_symbol = symbol.upper()

    for item in feed:
        ticker_items = item.get("ticker_sentiment", []) or []
        ticker_match = None
        for ticker_item in ticker_items:
            if str(ticker_item.get("ticker", "")).upper() == normalized_symbol:
                ticker_match = ticker_item
                break

        score = _coerce_float(item.get("overall_sentiment_score"), 0.0)
        relevance = 1.0
        if ticker_match is not None:
            ticker_score = _coerce_float(ticker_match.get("ticker_sentiment_score"), score)
            ticker_relevance = _coerce_float(ticker_match.get("relevance_score"), 1.0)
            score = ticker_score
            relevance = ticker_relevance

        weighted_score = score * max(relevance, 0.0)
        rows.append(
            {
                "published_at": item.get("time_published"),
                "source": item.get("source", "alphavantage"),
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "url": item.get("url", ""),
                "sentiment": weighted_score,
                "raw_sentiment": score,
                "relevance": relevance,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["published_at", "source", "title", "summary", "url", "sentiment", "raw_sentiment", "relevance", "novelty"])

    frame = pd.DataFrame(rows)
    frame["published_at"] = pd.to_datetime(frame["published_at"], format="%Y%m%dT%H%M%S", errors="coerce")
    frame = frame.dropna(subset=["published_at"]).reset_index(drop=True)

    title_counts = frame["title"].fillna("").replace("", np.nan).value_counts()
    frame["novelty"] = frame["title"].map(lambda t: 1.0 / max(title_counts.get(t, 1), 1) if pd.notna(t) else 1.0)
    frame["novelty"] = frame["novelty"].fillna(1.0)
    return frame


def fetch_alpha_vantage_articles(
    symbol: str,
    api_key: str,
    limit: int = 1000,
    time_from: Optional[str] = None,
    sort: str = "LATEST",
) -> pd.DataFrame:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "sort": sort,
        "limit": str(limit),
        "apikey": api_key,
    }
    if time_from:
        params["time_from"] = time_from

    response = requests.get("https://www.alphavantage.co/query", params=params, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return _normalize_alpha_vantage_articles(payload, symbol)


def resolve_sentiment_articles(
    symbol: str,
    provider: str = "auto",
    api_key: Optional[str] = None,
    lookback_period: str = "10y",
    fallback_to_example: bool = True,
) -> Tuple[Optional[pd.DataFrame], str]:
    provider = provider.lower()

    if provider == "auto":
        if os.environ.get("ALPHAVANTAGE_API_KEY"):
            provider = "alphavantage"
        else:
            provider = "example"

    if provider == "example":
        return example_article_frame(), "example"

    if provider != "alphavantage":
        raise ValueError(f"Unsupported sentiment provider '{provider}'. Currently implemented: alphavantage, example, auto.")

    key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY")
    if not key:
        if fallback_to_example:
            return example_article_frame(), "example"
        raise ValueError("ALPHAVANTAGE_API_KEY is missing. Add it to .env or set it in your shell.")

    time_from = _period_to_time_from(lookback_period)
    try:
        articles = fetch_alpha_vantage_articles(
            symbol=symbol,
            api_key=key,
            limit=1000,
            time_from=time_from,
            sort="LATEST",
        )
    except Exception:
        if fallback_to_example:
            return example_article_frame(), "example"
        raise

    if articles.empty and fallback_to_example:
        return example_article_frame(), "example"
    return articles, "alphavantage"


def load_env_file(env_path: str = ".env") -> Dict[str, str]:
    loaded: Dict[str, str] = {}
    if not os.path.exists(env_path):
        return loaded

    with open(env_path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded[key] = value
    return loaded


def example_article_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"published_at": "2026-01-03 08:30:00", "sentiment": 0.45, "source": "examplewire", "novelty": 0.8},
            {"published_at": "2026-01-03 13:15:00", "sentiment": -0.10, "source": "examplewire", "novelty": 0.3},
            {"published_at": "2026-01-04 07:50:00", "sentiment": 0.62, "source": "streetnews", "novelty": 0.9},
        ]
    )


def run_research_pipeline(
    data_config: Optional[DataConfig] = None,
    modeling_config: Optional[ModelingConfig] = None,
    backtest_config: Optional[BacktestConfig] = None,
    sentiment_articles: Optional[pd.DataFrame] = None,
) -> Dict[str, pd.DataFrame]:
    data_config = data_config or DataConfig()
    modeling_config = modeling_config or ModelingConfig()
    backtest_config = backtest_config or BacktestConfig()

    market_data = load_market_data(data_config)
    feature_frame = build_feature_matrix(
        market_data=market_data,
        sentiment_articles=sentiment_articles,
        target_horizon=modeling_config.target_horizon,
    )
    result_frame = walk_forward_backtest(
        feature_frame=feature_frame,
        feature_columns=BASE_FEATURES,
        modeling=modeling_config,
        backtest=backtest_config,
    )
    metrics_frame = pd.DataFrame([performance_metrics(result_frame["net_return"], backtest_config.annualization)])
    stress_frame = run_stress_tests(result_frame, backtest_config)

    return {
        "features": feature_frame,
        "backtest": result_frame,
        "metrics": metrics_frame,
        "stress_tests": stress_frame,
        "providers": describe_news_providers(),
    }


if __name__ == "__main__":
    bundle = run_research_pipeline(sentiment_articles=example_article_frame())
    print("Performance metrics")
    print(bundle["metrics"].round(4).to_string(index=False))
    print("\nStress tests")
    print(bundle["stress_tests"].round(4).to_string(index=False))
