"""Markdown report rendering for the backtest engine.

Kept separate from engine.py so the numeric core (`simulate`) stays free of
any string formatting: this module only turns already-computed `SimResult`s
into text.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl

from farb.config import BacktestConfig

if TYPE_CHECKING:
    # Deferred (TYPE_CHECKING-only) to avoid a runtime circular import:
    # engine.py imports render_report (and YEAR_HOURS) from this module at
    # module scope.
    from farb.backtest.engine import SimResult

SimResults = dict[str, dict[str, "SimResult"]]

# Julian year in hours — the ONE definition shared by engine.py's signal
# annualization and this module's APR math, so the two can never drift
# apart (they used to use 365*24 and 365.25*24 respectively).
YEAR_HOURS = 365.25 * 24

CAVEATS = [
    "Single venue (Binance): no cross-exchange basis capture, no borrow costs "
    "modeled (spot leg assumed fully funded, no margin interest).",
    "Slippage is a configured bps assumption per scenario, not derived from "
    "historical order-book depth (not available via REST) — real fills may "
    "differ, especially in size.",
    "Entry/exit thresholds (`enter_apr`/`exit_apr`) are fixed across all "
    "symbols, not fitted per symbol; the sensitivity grid shows how much the "
    "result depends on them.",
    "Maker fee scenarios assume the resting order fills exactly at the 1h "
    "close — real maker fills carry queue risk (may not fill at all, or fill "
    "worse) not modeled here.",
    "Funding forecast (EMA of settled rates) does not use intra-period "
    "premium/TWAP data; that is collected but reserved for a later iteration.",
    "Funding PnL is accrued on the per-event mark price (from the funding "
    "parquet), matching how Binance actually settles funding, rather than on "
    "the perp price frozen at entry; when mark price is unavailable it falls "
    "back to the asof perp close at that event.",
    "Two APR figures are reported because they answer different questions: "
    "APR (alloc) is a capital-efficiency measure — return on the notional "
    "reserved for the whole period, idle time included. APR (deployed) is "
    "the edge measure — return earned while a position was actually open. "
    "Time-in-market is usually well under 100%, so deployed APR is the more "
    "honest read of the raw edge; alloc APR is the more honest read of what "
    "the capital actually earned sitting in this strategy.",
    "The collector excludes any still-forming (partial) final candle, so "
    "every 1h close used here for entry/exit/mark-to-market pricing reflects "
    "a fully closed bar.",
    "All symbols run the same signal logic and often share a funding-rate "
    "regime, so entries tend to cluster at the same funding event: the "
    "portfolio Max DD above mostly reflects entry costs paid simultaneously "
    "across symbols (near-zero entry diversification), not a synchronized "
    "market loss.",
]


def _years(start: datetime | None, end: datetime | None) -> float:
    if start is None or end is None:
        return 0.0
    return (end - start).total_seconds() / (YEAR_HOURS * 3600)


def _apr(net_pnl: float, notional: float, years: float) -> float:
    return net_pnl / notional / years if years > 0 else math.nan


def _fmt_pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.1%}"


def _portfolio_equity_max_drawdown(
    results: SimResults, symbols: list[str], scenario_name: str
) -> float:
    """Max drawdown of the SUMMED portfolio equity curve.

    This is NOT max(per-symbol drawdown) — that understates portfolio risk
    whenever symbols' worst days don't line up. Summing the curves first
    (aligned on time, forward-filled) and then measuring drawdown on the
    combined curve is the only way to see the portfolio's actual worst
    peak-to-trough loss.
    """
    curves: list[pl.DataFrame] = []
    for symbol in symbols:
        r = results[symbol].get(scenario_name)
        if r is None or r.equity_curve.is_empty():
            continue
        # Truncate to the hour: funding timestamps can differ by a few ms
        # across symbols, which would otherwise keep them from lining up in
        # the join below.
        curve = (
            r.equity_curve.with_columns(pl.col("time").dt.truncate("1h"))
            .group_by("time", maintain_order=True)
            .agg(pl.col("equity").last())
            .sort("time")
            .rename({"equity": symbol})
        )
        curves.append(curve)

    if not curves:
        return 0.0

    merged = curves[0]
    for curve in curves[1:]:
        merged = merged.join(curve, on="time", how="full", coalesce=True)
    merged = merged.sort("time")

    value_cols = [c for c in merged.columns if c != "time"]
    merged = merged.with_columns([pl.col(c).forward_fill().fill_null(0.0) for c in value_cols])
    portfolio_values = merged.select(pl.sum_horizontal(value_cols)).to_series().to_list()

    if not portfolio_values:
        return 0.0
    peak = portfolio_values[0]
    max_dd = 0.0
    for value in portfolio_values:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    return max_dd


def _render_scenario_table(
    results: SimResults, symbols: list[str], scenario_name: str, notional: float
) -> list[str]:
    """One scenario's per-symbol table plus a summed portfolio row.

    Two APR columns: APR (alloc) treats notional as reserved for the WHOLE
    period; APR (deployed) treats it as earning only while a position was
    actually open. See the Caveats section for why both are shown.
    """
    lines = [
        "| Symbol | Net PnL (USDT) | APR (alloc) | APR (deployed) | Funding | Basis | Fees | "
        "Trades | Avg hold (h) | Time in mkt | Max DD (USDT) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    totals = {
        "net": 0.0, "funding": 0.0, "basis": 0.0, "fees": 0.0, "trades": 0, "holding_hours": 0.0,
    }
    starts: list[datetime] = []
    ends: list[datetime] = []
    present = 0

    for symbol in symbols:
        r = results[symbol].get(scenario_name)
        if r is None:
            continue
        present += 1
        alloc_apr = _apr(r.net_pnl, notional, _years(r.period_start, r.period_end))
        deployed_apr = _apr(r.net_pnl, notional, r.holding_hours_sum / YEAR_HOURS)
        lines.append(
            f"| {symbol} | {r.net_pnl:,.2f} | {_fmt_pct(alloc_apr)} | {_fmt_pct(deployed_apr)} | "
            f"{r.funding_pnl:,.2f} | {r.basis_pnl:,.2f} | {r.fees_paid:,.2f} | {r.n_trades} | "
            f"{r.avg_holding_hours:.1f} | {r.time_in_market_frac:.1%} | {r.max_drawdown:,.2f} |"
        )
        totals["net"] += r.net_pnl
        totals["funding"] += r.funding_pnl
        totals["basis"] += r.basis_pnl
        totals["fees"] += r.fees_paid
        totals["trades"] += r.n_trades
        totals["holding_hours"] += r.holding_hours_sum
        if r.period_start is not None:
            starts.append(r.period_start)
        if r.period_end is not None:
            ends.append(r.period_end)

    port_period_hours = (max(ends) - min(starts)).total_seconds() / 3600 if starts and ends else 0.0
    port_years = port_period_hours / YEAR_HOURS
    # Denominator uses `present` (symbols actually contributing to THIS
    # scenario), not the full universe — a scenario missing a symbol's row
    # must not silently dilute its own portfolio APR.
    port_alloc_apr = _apr(totals["net"], notional * present, port_years) if present else math.nan
    port_deployed_apr = _apr(totals["net"], notional, totals["holding_hours"] / YEAR_HOURS)
    port_time_in_mkt = (
        totals["holding_hours"] / (present * port_period_hours)
        if present and port_period_hours > 0
        else 0.0
    )
    port_dd = _portfolio_equity_max_drawdown(results, symbols, scenario_name)
    lines.append(
        f"| **Portfolio** | {totals['net']:,.2f} | {_fmt_pct(port_alloc_apr)} | "
        f"{_fmt_pct(port_deployed_apr)} | {totals['funding']:,.2f} | {totals['basis']:,.2f} | "
        f"{totals['fees']:,.2f} | {totals['trades']} | – | {port_time_in_mkt:.1%} | "
        f"{port_dd:,.2f} |"
    )
    return lines


def _render_sensitivity_grid(
    sensitivity: list[tuple[float, float, float]], cfg: BacktestConfig
) -> list[str]:
    first_scenario_name = cfg.scenarios[0].name if cfg.scenarios else "n/a"
    header = " | ".join(f"{e:.1%}" for e in cfg.sensitivity_exit_apr)
    lines = [
        f"## Sensitivity grid — portfolio APR (alloc; scenario: {first_scenario_name})",
        "",
        f"| Enter APR \\ Exit APR | {header} |",
        "|---|" + "---:|" * len(cfg.sensitivity_exit_apr),
    ]
    by_enter: dict[float, dict[float, float]] = {}
    for enter_apr, exit_apr, port_apr in sensitivity:
        by_enter.setdefault(enter_apr, {})[exit_apr] = port_apr
    for enter_apr in cfg.sensitivity_enter_apr:
        row_values = by_enter.get(enter_apr, {})
        row_cells = (row_values.get(exit_apr) for exit_apr in cfg.sensitivity_exit_apr)
        row = " | ".join("–" if v is None else _fmt_pct(v) for v in row_cells)
        lines.append(f"| {enter_apr:.1%} | {row} |")
    lines.append("")
    return lines


def render_report(
    results: SimResults,
    cfg: BacktestConfig,
    sensitivity: list[tuple[float, float, float]],
) -> str:
    """Render the full markdown report: header, per-scenario per-symbol +
    portfolio tables, the threshold sensitivity grid, and honest caveats.
    """
    symbols = sorted(results.keys())
    lines: list[str] = [
        f"# funding-arb backtest report — {datetime.now(UTC).date().isoformat()}",
        "",
    ]

    all_results = [r for sym in results.values() for r in sym.values()]
    all_starts = [r.period_start for r in all_results if r.period_start is not None]
    all_ends = [r.period_end for r in all_results if r.period_end is not None]
    period_str = (
        f"{min(all_starts).date().isoformat()} to {max(all_ends).date().isoformat()}"
        if all_starts and all_ends
        else "n/a (no data)"
    )
    lines.append(f"- Period: {period_str}")
    lines.append(f"- Universe: {len(symbols)} symbols, {cfg.notional_usdt:,.0f} USDT notional each")
    lines.append(
        f"- Signal: EMA({cfg.ema_periods}) of settled funding rates, "
        f"enter APR > {cfg.enter_apr:.0%}, exit APR < {cfg.exit_apr:.0%} (hysteresis)"
    )
    lines.append(
        "- APR (alloc) = net / (notional x whole-period-years): return on capital "
        "reserved for the whole period. APR (deployed) = net / (notional x "
        "time-actually-in-position): return earned only while a position was open — "
        "the more honest edge measure when time-in-market is well under 100%."
    )
    lines.append("")

    for scenario in cfg.scenarios:
        lines.append(
            f"## Scenario: {scenario.name} (spot {scenario.spot_bps}bps, "
            f"perp {scenario.perp_bps}bps, slippage {scenario.slippage_bps}bps/leg)"
        )
        lines.append("")
        lines.extend(_render_scenario_table(results, symbols, scenario.name, cfg.notional_usdt))
        lines.append("")

    if sensitivity:
        lines.extend(_render_sensitivity_grid(sensitivity, cfg))

    lines.append("## Caveats")
    lines.append("")
    lines.extend(f"- {caveat}" for caveat in CAVEATS)
    lines.append("")

    return "\n".join(lines)
