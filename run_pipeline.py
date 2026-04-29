import argparse
from publishable_quant_platform import (
    BacktestConfig,
    DataConfig,
    ModelingConfig,
    NEWS_PROVIDER_SPECS,
    load_env_file,
    resolve_sentiment_articles,
    run_research_pipeline,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the publishable quant research starter.")
    parser.add_argument("--symbol", default="AAPL", help="Primary asset ticker, default: AAPL")
    parser.add_argument("--benchmark", default="SPY", help="Benchmark ticker, default: SPY")
    parser.add_argument("--sector", default="XLK", help="Sector ETF ticker, default: XLK")
    parser.add_argument("--period", default="10y", help="Price lookback period for yfinance, default: 10y")
    parser.add_argument("--disable-transformer", action="store_true", help="Skip the transformer stage")
    parser.add_argument("--show-providers", action="store_true", help="Print finance-news provider docs and exit")
    parser.add_argument(
        "--sentiment-provider",
        default="auto",
        choices=["auto", "alphavantage", "example"],
        help="Sentiment source. 'auto' uses Alpha Vantage if ALPHAVANTAGE_API_KEY is set, otherwise demo data.",
    )
    return parser.parse_args()


def print_provider_setup() -> None:
    print("Finance news providers configured in this starter:")
    for key, spec in NEWS_PROVIDER_SPECS.items():
        env_key = f"{key.upper()}_API_KEY"
        print(f"- {spec.name}: {spec.docs_url}")
        print(f"  Endpoint hint: {spec.endpoint_hint}")
        print(f"  Suggested env var: {env_key}")
        print(f"  Notes: {spec.notes}")


def main() -> None:
    args = parse_args()
    load_env_file()

    if args.show_providers:
        print_provider_setup()
        return

    data_config = DataConfig(
        symbol=args.symbol,
        benchmark_symbol=args.benchmark,
        sector_symbol=args.sector,
        lookback_period=args.period,
    )
    modeling_config = ModelingConfig(transformer_enabled=not args.disable_transformer)
    backtest_config = BacktestConfig()
    sentiment_articles, sentiment_source = resolve_sentiment_articles(
        symbol=args.symbol,
        provider=args.sentiment_provider,
        lookback_period=args.period,
        fallback_to_example=True,
    )

    print("Running research pipeline...")
    print(f"Symbol: {data_config.symbol}")
    print(f"Benchmark: {data_config.benchmark_symbol}")
    print(f"Sector ETF: {data_config.sector_symbol}")
    print(f"Transformer enabled: {modeling_config.transformer_enabled}")
    print(f"Sentiment input: {sentiment_source}")

    try:
        bundle = run_research_pipeline(
            data_config=data_config,
            modeling_config=modeling_config,
            backtest_config=backtest_config,
            sentiment_articles=sentiment_articles,
        )
    except Exception as exc:
        print("")
        print("Run failed.")
        print(str(exc))
        print("")
        print("Most common fixes:")
        print("- Retry if Yahoo Finance temporarily failed.")
        print("- Change the ticker and try again.")
        print("- Replace the default price loader with a more stable provider for production use.")
        print("- Check your ALPHAVANTAGE_API_KEY if you asked for live sentiment.")
        return

    print("")
    print("Performance metrics")
    print(bundle["metrics"].round(4).to_string(index=False))
    print("")
    print("Stress tests")
    print(bundle["stress_tests"].round(4).to_string(index=False))
    print("")
    print("Publishing checklist:")
    print("- Save metrics, stress tests, and plots to files.")
    print("- Compare against buy-and-hold and simple factor baselines.")
    print("- Run robustness checks across symbols, regimes, and cost assumptions.")
    print("- Verify whether the run used example sentiment or live Alpha Vantage sentiment.")


if __name__ == "__main__":
    main()
