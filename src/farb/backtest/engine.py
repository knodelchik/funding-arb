"""Cash-and-carry backtest engine: long spot + short USDⓈ-M perp, one venue.

The whole point of this module is to be honest about lookahead: every
decision at a funding event uses only data that had already settled strictly
before that event. `simulate()` is the pure, fully-testable core;
`run_backtest()` wires it to disk (parquet in, markdown report out).
"""

from __future__ import annotations

import bisect
import math
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from farb.backtest.report import YEAR_HOURS, render_report
from farb.config import BacktestConfig, FeeScenario
from farb.schemas import dataset_path, funding_info_path

DEFAULT_FUNDING_INTERVAL_HOURS = 8

# A truncated/short kline file must not be able to silently price an event
# with an ancient close and manufacture basis PnL out of stale data.
MAX_PRICE_STALENESS_HOURS = 24.0


@dataclass
class Trade:
    """One completed (or force-closed) round trip: long spot + short perp."""

    entry_time: datetime
    exit_time: datetime
    entry_spot: float
    entry_perp: float
    exit_spot: float
    exit_perp: float
    qty: float
    funding_pnl: float
    basis_pnl: float
    fees_paid: float
    net_pnl: float
    holding_hours: float
    force_closed: bool = False
    unpriced_exit: bool = False


@dataclass
class SimResult:
    """Output of one `simulate()` run: one symbol x one fee scenario x one
    pair of (enter_apr, exit_apr) thresholds.
    """

    trades: list[Trade]
    funding_pnl: float
    basis_pnl: float
    fees_paid: float
    net_pnl: float
    equity_curve: pl.DataFrame  # columns: time (Datetime, UTC), equity (Float64)
    max_drawdown: float
    n_trades: int
    avg_holding_hours: float
    holding_hours_sum: float
    time_in_market_frac: float
    period_start: datetime | None
    period_end: datetime | None
    unpriced_exit_count: int = 0


def _empty_equity_curve() -> pl.DataFrame:
    return pl.DataFrame(schema={"time": pl.Datetime("us", "UTC"), "equity": pl.Float64})


def _empty_result() -> SimResult:
    return SimResult(
        trades=[],
        funding_pnl=0.0,
        basis_pnl=0.0,
        fees_paid=0.0,
        net_pnl=0.0,
        equity_curve=_empty_equity_curve(),
        max_drawdown=0.0,
        n_trades=0,
        avg_holding_hours=0.0,
        holding_hours_sum=0.0,
        time_in_market_frac=0.0,
        period_start=None,
        period_end=None,
    )


def _asof_close(
    times: list[datetime],
    closes: list[float],
    t: datetime,
    max_staleness_hours: float = MAX_PRICE_STALENESS_HOURS,
) -> float | None:
    """Last close at-or-before `t` (backward asof join).

    Returns None if `t` is before the first available price, OR if the
    matched close is older than `max_staleness_hours` — a truncated/short
    kline file should be treated as missing data at that point, not priced
    with an ancient close that would manufacture bogus basis PnL.
    """
    idx = bisect.bisect_right(times, t) - 1
    if idx < 0:
        return None
    age_hours = (t - times[idx]).total_seconds() / 3600
    if age_hours > max_staleness_hours:
        return None
    return closes[idx]


def _close_times(open_times: list[datetime]) -> list[datetime]:
    """Shift kline open times to CLOSE times (open + bar step, inferred from
    the smallest gap between consecutive bars; 1h when there is only one bar
    or all bars share the same open time).

    A candle's close only exists once the candle has closed — asof-joining
    funding events against open times would price a 16:00 event with the
    16:00-17:00 close, an hour of future price. Joining against close times
    prices it with the candle that closed AT 16:00, i.e. the actual
    decision-moment price.
    """
    if len(open_times) >= 2:
        step = min(
            (b - a for a, b in zip(open_times, open_times[1:], strict=False) if b > a),
            default=timedelta(hours=1),
        )
    else:
        step = timedelta(hours=1)
    return [t + step for t in open_times]


def _forecast_series(rates: list[float], ema_periods: int) -> list[float]:
    """Causal EMA(ema_periods) forecast at each funding-event index.

    forecast[i] is the EMA of rates settled STRICTLY BEFORE event i, i.e. of
    rates[0:i] only; it is NaN until at least `ema_periods` prior rates exist
    (i < ema_periods). This is what makes the signal PIT-correct: it never
    uses the rate that settles at the event it is deciding.

    Standard span-based EMA, unadjusted recursive form seeded on the first
    rate: alpha = 2 / (ema_periods + 1), ema_k = alpha*rate_k + (1-alpha)*ema_{k-1}.
    """
    n = len(rates)
    forecast = [math.nan] * n
    if n == 0:
        return forecast
    alpha = 2.0 / (ema_periods + 1)
    ema = rates[0]
    seen = 1
    for i in range(1, n):
        if seen >= ema_periods:
            forecast[i] = ema
        ema = alpha * rates[i] + (1 - alpha) * ema
        seen += 1
    return forecast


def _oneway_fee(scenario: FeeScenario, notional: float) -> float:
    """Fee for one side of a roundtrip (both legs), one way (entry or exit).

    Equals half of `scenario.roundtrip_frac * notional` by construction —
    entry and exit cost the same, so 2 * this == the full roundtrip cost.
    """
    per_leg_bps = scenario.spot_bps + scenario.perp_bps + 2 * scenario.slippage_bps
    return notional * per_leg_bps / 10_000


def simulate(
    funding: pl.DataFrame,
    spot: pl.DataFrame,
    perp: pl.DataFrame,
    interval_hours: int,
    scenario: FeeScenario,
    notional: float,
    ema_periods: int,
    enter_apr: float,
    exit_apr: float,
) -> SimResult:
    """Run the cash-and-carry strategy over one symbol's settled funding
    events, using 1h spot/perp klines for entry/exit pricing.

    Assumes `funding`, `spot`, `perp` already hold a single symbol's rows
    (they are sorted defensively but not filtered by symbol).

    No lookahead: the decision made AT event T (enter/hold/exit) depends
    only on funding rates settled strictly before T (see `_forecast_series`)
    and on prices at-or-before T (backward asof against candle CLOSE times).
    Funding accrues on an open position at every settled event strictly
    AFTER its entry event; a position still open at the last event is
    force-closed there so the identity net == funding + basis - fees covers
    the whole period.
    """
    funding = funding.sort("funding_time")
    spot = spot.sort("open_time")
    perp = perp.sort("open_time")

    times: list[datetime] = funding["funding_time"].to_list()
    rates: list[float] = funding["funding_rate"].to_list()
    mark_prices: list[float | None] = funding["mark_price"].to_list()
    n = len(times)
    if n == 0:
        return _empty_result()

    forecasts = _forecast_series(rates, ema_periods)
    periods_per_year = YEAR_HOURS / interval_hours
    aprs = [f * periods_per_year if not math.isnan(f) else math.nan for f in forecasts]

    spot_times = _close_times(spot["open_time"].to_list())
    spot_closes: list[float] = spot["close"].to_list()
    perp_times = _close_times(perp["open_time"].to_list())
    perp_closes: list[float] = perp["close"].to_list()

    oneway_fee = _oneway_fee(scenario, notional)

    trades: list[Trade] = []
    realized_total = 0.0

    position_open = False
    entry_time_v: datetime | None = None
    entry_spot_px = 0.0
    entry_perp_px = 0.0
    qty_v = 0.0
    pos_funding = 0.0

    # Last point (within the current trade) where BOTH legs were actually
    # priced — the walk-back fallback for closing a position when the
    # decision event itself can't be priced (stale/short kline coverage).
    # Seeded at entry, since entry itself requires both legs to be priced.
    last_priced_time: datetime | None = None
    last_priced_spot: float | None = None
    last_priced_perp: float | None = None

    equity_times: list[datetime] = []
    equity_values: list[float] = []

    for i in range(n):
        t = times[i]
        apr = aprs[i]

        if position_open:
            # Entry (in the `else` branch below) only sets position_open —
            # it does not re-enter this branch on the same iteration. So
            # this branch only ever runs on an iteration where the position
            # was ALREADY open, i.e. strictly after its entry event: no
            # extra guard is needed for accrual to start after entry.
            #
            # Binance settles funding on the mark price at the event, not
            # the perp price frozen at entry — accrue on that when it's
            # available, falling back to the current asof perp close.
            mark_px = mark_prices[i]
            mark_missing = mark_px is None or math.isnan(mark_px)
            if not mark_missing:
                accrual_notional = qty_v * mark_px
            else:
                now_perp = _asof_close(perp_times, perp_closes, t)
                accrual_notional = qty_v * now_perp if now_perp is not None else 0.0
            pos_funding += rates[i] * accrual_notional

            is_last = i == n - 1
            # apr and forecast always share the sign of the (positive)
            # periods_per_year scaling, so `apr < exit_apr` already implies
            # `forecast < 0` whenever exit_apr >= 0 — no separate check needed.
            do_exit = not math.isnan(apr) and apr < exit_apr
            if do_exit or is_last:
                exit_time = t
                exit_spot_px = _asof_close(spot_times, spot_closes, t)
                exit_perp_px = _asof_close(perp_times, perp_closes, t)
                unpriced_exit = exit_spot_px is None or exit_perp_px is None
                if unpriced_exit:
                    # The decision event itself can't be priced (stale or
                    # short kline coverage) — walk back to the last point in
                    # this trade's life where both legs WERE priced, rather
                    # than silently dropping the position (net=0/trades=0
                    # while the equity curve still shows accrued PnL, and
                    # the identity check would be vacuously true over an
                    # empty trade list).
                    exit_time = last_priced_time
                    exit_spot_px = last_priced_spot
                    exit_perp_px = last_priced_perp
                if exit_time is not None and exit_spot_px is not None and exit_perp_px is not None:
                    basis_entry = entry_perp_px - entry_spot_px
                    basis_exit = exit_perp_px - exit_spot_px
                    trade_basis = qty_v * (basis_entry - basis_exit)
                    trade_fees = 2 * oneway_fee
                    trade_net = pos_funding + trade_basis - trade_fees
                    holding_hours = (exit_time - entry_time_v).total_seconds() / 3600
                    trades.append(
                        Trade(
                            entry_time=entry_time_v,
                            exit_time=exit_time,
                            entry_spot=entry_spot_px,
                            entry_perp=entry_perp_px,
                            exit_spot=exit_spot_px,
                            exit_perp=exit_perp_px,
                            qty=qty_v,
                            funding_pnl=pos_funding,
                            basis_pnl=trade_basis,
                            fees_paid=trade_fees,
                            net_pnl=trade_net,
                            holding_hours=holding_hours,
                            force_closed=(is_last and not do_exit) or unpriced_exit,
                            unpriced_exit=unpriced_exit,
                        )
                    )
                    realized_total += trade_net
                    position_open = False
                # else: no price data EVER existed for this trade (not even
                # at entry) — can't happen, entry itself requires prices.
        else:
            if not math.isnan(apr) and apr > enter_apr:
                e_spot = _asof_close(spot_times, spot_closes, t)
                e_perp = _asof_close(perp_times, perp_closes, t)
                if e_spot is not None and e_perp is not None and e_spot > 0:
                    qty_v = notional / e_spot
                    entry_spot_px = e_spot
                    entry_perp_px = e_perp
                    entry_time_v = t
                    pos_funding = 0.0
                    position_open = True
                    last_priced_time = t
                    last_priced_spot = e_spot
                    last_priced_perp = e_perp

        # Equity curve at event resolution: realized PnL from closed trades,
        # plus (if still open) funding collected so far and unrealized basis
        # mark-to-market on the open leg, less the entry fee already paid
        # (the exit fee only hits once the trade actually closes).
        if position_open:
            now_spot = _asof_close(spot_times, spot_closes, t)
            now_perp = _asof_close(perp_times, perp_closes, t)
            if now_spot is not None and now_perp is not None:
                last_priced_time = t
                last_priced_spot = now_spot
                last_priced_perp = now_perp
                unrealized_basis = qty_v * ((entry_perp_px - entry_spot_px) - (now_perp - now_spot))
            elif last_priced_spot is not None and last_priced_perp is not None:
                # Current price unavailable: mark at the last point we COULD
                # price, rather than assuming basis snapped back to flat.
                # This also keeps the curve consistent with the walk-back
                # closure above, which uses this same frozen mark.
                unrealized_basis = qty_v * (
                    (entry_perp_px - entry_spot_px) - (last_priced_perp - last_priced_spot)
                )
            else:
                unrealized_basis = 0.0
            equity = realized_total + pos_funding + unrealized_basis - oneway_fee
        else:
            equity = realized_total
        equity_times.append(t)
        equity_values.append(equity)

    funding_pnl = sum(tr.funding_pnl for tr in trades)
    basis_pnl = sum(tr.basis_pnl for tr in trades)
    fees_paid = sum(tr.fees_paid for tr in trades)
    net_pnl = sum(tr.net_pnl for tr in trades)
    identity_gap = abs(net_pnl - (funding_pnl + basis_pnl - fees_paid))
    if identity_gap >= 1e-9:
        # A bare `assert` is stripped under `python -O`; this invariant is
        # load-bearing (it's the whole point of the engine), so raise
        # unconditionally instead.
        raise RuntimeError(
            f"PnL identity violated: net must equal funding + basis - fees (gap={identity_gap})"
        )

    peak = equity_values[0]
    max_drawdown = 0.0
    for e in equity_values:
        peak = max(peak, e)
        max_drawdown = max(max_drawdown, peak - e)

    period_start = times[0]
    period_end = times[-1]
    total_hours = (period_end - period_start).total_seconds() / 3600
    holding_sum = sum(tr.holding_hours for tr in trades)
    time_in_market_frac = holding_sum / total_hours if total_hours > 0 else 0.0
    avg_holding_hours = holding_sum / len(trades) if trades else 0.0

    equity_curve = pl.DataFrame(
        {"time": equity_times, "equity": equity_values},
        schema={"time": pl.Datetime("us", "UTC"), "equity": pl.Float64},
    )

    return SimResult(
        trades=trades,
        funding_pnl=funding_pnl,
        basis_pnl=basis_pnl,
        fees_paid=fees_paid,
        net_pnl=net_pnl,
        equity_curve=equity_curve,
        max_drawdown=max_drawdown,
        n_trades=len(trades),
        avg_holding_hours=avg_holding_hours,
        holding_hours_sum=holding_sum,
        time_in_market_frac=time_in_market_frac,
        period_start=period_start,
        period_end=period_end,
        unpriced_exit_count=sum(1 for tr in trades if tr.unpriced_exit),
    )


def _load_interval_map(data_root: Path) -> dict[str, int]:
    """symbol -> funding_interval_hours from the funding_info snapshot.

    Only used as a cross-check now (see `_derive_interval_hours`): this
    file reflects only the symbol's CURRENT interval, which is unreliable
    applied to ~2 years of history.
    """
    path = funding_info_path(data_root)
    if not path.exists():
        return {}
    info = pl.read_parquet(path)
    symbols = info["symbol"].to_list()
    hours = info["funding_interval_hours"].to_list()
    return dict(zip(symbols, hours, strict=True))


def _derive_interval_hours(funding_times: list[datetime]) -> int:
    """Median gap between consecutive settled funding events, rounded to
    the hour — the authoritative source for `interval_hours`.

    Preferred over the funding_info snapshot, which reflects only the
    symbol's CURRENT interval and would misprice the whole history if that
    interval had ever changed. This is a median, though, not a true
    reconstruction of interval history: if a symbol's interval changed
    mid-history, this returns whichever interval covers MORE THAN HALF the
    history (the majority interval) and silently mis-annualizes events
    settled under the other one. That's a real limitation, just a smaller
    one than trusting a current-state snapshot for the whole span.
    """
    unique_times = sorted(set(funding_times))
    if len(unique_times) < 2:
        return DEFAULT_FUNDING_INTERVAL_HOURS
    gaps = [b - a for a, b in zip(unique_times[:-1], unique_times[1:], strict=True)]
    gaps.sort()
    mid = len(gaps) // 2
    median_gap = gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2
    return max(1, round(median_gap.total_seconds() / 3600))


def _portfolio_alloc_apr(
    net: float, notional: float, n_symbols: int, starts: list[datetime], ends: list[datetime]
) -> float:
    """Capital-efficiency APR: net / (notional * n_symbols) / whole-period-years."""
    if not starts or not ends or n_symbols == 0:
        return math.nan
    years = (max(ends) - min(starts)).total_seconds() / (YEAR_HOURS * 3600)
    return net / (notional * n_symbols) / years if years > 0 else math.nan


def _portfolio_deployed_apr(net: float, notional: float, holding_hours_sum: float) -> float:
    """Edge-measure APR: net / (notional * years actually deployed), where
    deployed hours are SUMMED across symbols (each symbol's notional is its
    own pool, so summing hours already accounts for having several pools).
    """
    deployed_years = holding_hours_sum / YEAR_HOURS
    return net / (notional * deployed_years) if deployed_years > 0 else math.nan


def run_backtest(cfg: BacktestConfig, reports_dir: Path) -> None:
    """Run the backtest for every symbol x fee scenario in `cfg`, plus a
    portfolio-level sensitivity grid for the first scenario, write the
    markdown report, and print a console summary.
    """
    interval_snapshot = _load_interval_map(cfg.data_root)

    # symbol -> (funding, spot, perp, interval_hours), cached so the
    # sensitivity grid doesn't re-read parquet for every threshold pair.
    frames: dict[str, tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, int]] = {}
    warned: set[str] = set()

    for symbol in cfg.symbols:
        funding_path = dataset_path(cfg.data_root, "funding", symbol)
        spot_path = dataset_path(cfg.data_root, "klines_spot_1h", symbol)
        perp_path = dataset_path(cfg.data_root, "klines_perp_1h", symbol)
        if not (funding_path.exists() and spot_path.exists() and perp_path.exists()):
            if symbol not in warned:
                print(f"warning: skipping {symbol} — missing data", file=sys.stderr)
                warned.add(symbol)
            continue
        funding_df = pl.read_parquet(funding_path).sort("funding_time")
        interval_hours = _derive_interval_hours(funding_df["funding_time"].to_list())
        snapshot_hours = interval_snapshot.get(symbol)
        if snapshot_hours is not None and snapshot_hours != interval_hours:
            print(
                f"warning: {symbol} funding_info says {snapshot_hours}h but "
                f"{interval_hours}h is derived from settled history; using derived",
                file=sys.stderr,
            )
        frames[symbol] = (
            funding_df,
            pl.read_parquet(spot_path),
            pl.read_parquet(perp_path),
            interval_hours,
        )

    if not frames:
        print("warning: no symbols had data; nothing to backtest", file=sys.stderr)

    results: dict[str, dict[str, SimResult]] = {}
    for symbol, (funding, spot, perp, interval_hours) in frames.items():
        results[symbol] = {}
        for scenario in cfg.scenarios:
            r = simulate(
                funding=funding,
                spot=spot,
                perp=perp,
                interval_hours=interval_hours,
                scenario=scenario,
                notional=cfg.notional_usdt,
                ema_periods=cfg.ema_periods,
                enter_apr=cfg.enter_apr,
                exit_apr=cfg.exit_apr,
            )
            if r.unpriced_exit_count > 0:
                print(
                    f"warning: {symbol}/{scenario.name}: {r.unpriced_exit_count} "
                    "position(s) force-closed at a stale/missing price (kline coverage gap)",
                    file=sys.stderr,
                )
            results[symbol][scenario.name] = r

    sensitivity: list[tuple[float, float, float]] = []
    if frames and cfg.scenarios:
        first_scenario = cfg.scenarios[0]
        n_symbols = len(frames)
        for enter_apr in cfg.sensitivity_enter_apr:
            for exit_apr in cfg.sensitivity_exit_apr:
                port_net = 0.0
                starts: list[datetime] = []
                ends: list[datetime] = []
                for funding, spot, perp, interval_hours in frames.values():
                    r = simulate(
                        funding=funding,
                        spot=spot,
                        perp=perp,
                        interval_hours=interval_hours,
                        scenario=first_scenario,
                        notional=cfg.notional_usdt,
                        ema_periods=cfg.ema_periods,
                        enter_apr=enter_apr,
                        exit_apr=exit_apr,
                    )
                    port_net += r.net_pnl
                    if r.period_start is not None:
                        starts.append(r.period_start)
                    if r.period_end is not None:
                        ends.append(r.period_end)
                port_apr = _portfolio_alloc_apr(
                    port_net, cfg.notional_usdt, n_symbols, starts, ends
                )
                sensitivity.append((enter_apr, exit_apr, port_apr))

    report_md = render_report(results, cfg, sensitivity)

    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / f"backtest_{datetime.now(UTC).date().isoformat()}.md"
    out_path.write_text(report_md)

    print(f"=== funding-arb backtest ({len(frames)} symbols) ===")
    for scenario in cfg.scenarios:
        scen_results = [results[sym][scenario.name] for sym in results]
        port_net = sum(r.net_pnl for r in scen_results)
        port_trades = sum(r.n_trades for r in scen_results)
        starts = [r.period_start for r in scen_results if r.period_start is not None]
        ends = [r.period_end for r in scen_results if r.period_end is not None]
        alloc_apr = _portfolio_alloc_apr(port_net, cfg.notional_usdt, len(results), starts, ends)
        holding_hours_sum = sum(r.holding_hours_sum for r in scen_results)
        deployed_apr = _portfolio_deployed_apr(port_net, cfg.notional_usdt, holding_hours_sum)
        print(
            f"-- scenario {scenario.name}: portfolio net {port_net:,.2f} USDT, "
            f"APR alloc {alloc_apr:.1%} / deployed {deployed_apr:.1%}, {port_trades} trades"
        )
        for symbol in sorted(results):
            r = results[symbol][scenario.name]
            print(
                f"   {symbol}: net {r.net_pnl:,.2f} USDT, {r.n_trades} trades, "
                f"maxDD {r.max_drawdown:,.2f}"
            )
    print(f"report written to {out_path}")
