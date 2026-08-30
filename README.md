# funding-arb

Binance cash-and-carry funding-rate arbitrage. **M0**: public-data collector +
honest backtest with PnL decomposition (`net = funding + basis − fees`) — the
milestone that decides whether M1+ (forecast, paper executor, live) are worth building.

## Usage

```sh
uv sync
uv run farb collect                 # ~20 symbols, since 2024-12-01, into data/binance/
uv run farb backtest                # writes reports/backtest_YYYY-MM-DD.md
uv run pytest && uv run ruff check
```

Spec: `.omc/autopilot/spec.md`. Plan: `.omc/plans/autopilot-impl.md`.
