"""Backtest engine tests. No files, no network — `simulate()` is a pure
function over polars frames built with the conftest helpers.

Each test is designed to DISCRIMINATE: it is built so that the specific
plausible bug (flipped funding sign, dropped/doubled fees, lookahead,
wrong hysteresis, frozen accrual notional, hardcoded annualization) makes
the assertion fail, not just "look wrong".
"""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from conftest import T0, make_funding, make_klines

from farb.backtest.engine import SimResult, _derive_interval_hours, simulate
from farb.backtest.report import YEAR_HOURS, _portfolio_equity_max_drawdown, render_report
from farb.config import BacktestConfig, FeeScenario

ZERO_FEES = FeeScenario(name="zero", spot_bps=0.0, perp_bps=0.0, slippage_bps=0.0)
TAKER = FeeScenario(name="taker", spot_bps=10.0, perp_bps=5.0, slippage_bps=1.0)
INTERVAL_HOURS = 8  # matches make_funding's default


def _flat_klines(symbol: str, price: float, hours: int) -> tuple:
    spot = make_klines(symbol, [price] * hours)
    perp = make_klines(symbol, [price] * hours)
    return spot, perp


def test_constant_positive_funding_gives_expected_positive_net() -> None:
    """Constant positive funding, flat prices (basis always 0, mark price
    equal to the flat perp close): net must be exactly the funding
    collected while in position, minus the one entry+exit fee, computed by
    hand — and must be positive.
    """
    rate = 0.0001  # 0.01% per 8h
    n_events = 30
    notional = 10_000.0
    funding = make_funding("BTCUSDT", [rate] * n_events, interval_hours=INTERVAL_HOURS)
    spot, perp = _flat_klines("BTCUSDT", 100.0, hours=(n_events - 1) * INTERVAL_HOURS + 8)

    scenario = FeeScenario(name="low", spot_bps=1.0, perp_bps=1.0, slippage_bps=0.0)
    result = simulate(
        funding=funding,
        spot=spot,
        perp=perp,
        interval_hours=INTERVAL_HOURS,
        scenario=scenario,
        notional=notional,
        ema_periods=2,
        enter_apr=0.05,
        exit_apr=0.01,
    )

    # forecast (EMA of a constant series) equals `rate` from index 2 onward,
    # apr = rate * (365.25*24/8) ~= 10.96% > enter_apr(5%), so entry at index
    # 2; apr never drops below exit_apr(1%), so the position is
    # force-closed at the last event (index 29). In-position events
    # (strictly after entry): 27. mark_price defaults to 100.0 (== the flat
    # perp close), so accrual notional == qty*100 == notional exactly.
    n_in_position_events = (n_events - 1) - 2
    expected_funding = n_in_position_events * rate * notional
    expected_fees = 2 * (notional * (scenario.spot_bps + scenario.perp_bps) / 10_000)
    expected_net = expected_funding - expected_fees

    assert result.basis_pnl == pytest.approx(0.0, abs=1e-9)
    assert result.funding_pnl == pytest.approx(expected_funding)
    assert result.fees_paid == pytest.approx(expected_fees)
    assert result.net_pnl == pytest.approx(expected_net)
    assert result.net_pnl > 0


def test_negative_funding_never_enters_even_with_a_permissive_threshold() -> None:
    """Mirror of the positive-funding case: constant NEGATIVE funding means
    the EMA forecast is negative, so apr is negative and must never clear
    enter_apr — no position is ever opened. This exercises the ENTRY
    threshold comparison's sign handling (a flipped comparison, or a sign
    bug in parsing/EMA, would wrongly admit entry here); it does NOT
    exercise the funding accrual's sign convention, since no trade ever
    opens here — that path is covered by the constant-positive-funding test.

    The second threshold (enter_apr=0.0) is deliberately permissive — lower
    than the realistic default of 12% — so a sign bug that makes the
    (actually negative) apr look positive would clear it and wrongly enter.
    """
    rate = -0.0001
    n_events = 20
    funding = make_funding("BTCUSDT", [rate] * n_events, interval_hours=INTERVAL_HOURS)
    spot, perp = _flat_klines("BTCUSDT", 100.0, hours=(n_events - 1) * INTERVAL_HOURS + 8)

    for enter_apr in (0.05, 0.0):
        result = simulate(
            funding=funding,
            spot=spot,
            perp=perp,
            interval_hours=INTERVAL_HOURS,
            scenario=TAKER,
            notional=10_000.0,
            ema_periods=3,
            enter_apr=enter_apr,
            exit_apr=0.01,
        )
        assert result.n_trades == 0
        assert result.net_pnl == 0.0
        assert result.funding_pnl == 0.0


def test_fee_scenarios_differ_by_exactly_the_roundtrip_fee() -> None:
    """Same PnL-generating scenario run once with zero fees and once with
    the taker scenario: the net-PnL difference must equal exactly
    `notional * scenario.roundtrip_frac` (one entry + one exit, both legs).
    Fails if fees are dropped, or charged once instead of per leg/side.
    """
    rate = 0.0001
    n_events = 30
    notional = 10_000.0
    funding = make_funding("BTCUSDT", [rate] * n_events, interval_hours=INTERVAL_HOURS)
    spot, perp = _flat_klines("BTCUSDT", 100.0, hours=(n_events - 1) * INTERVAL_HOURS + 8)

    common = {
        "funding": funding,
        "spot": spot,
        "perp": perp,
        "interval_hours": INTERVAL_HOURS,
        "notional": notional,
        "ema_periods": 2,
        "enter_apr": 0.05,
        "exit_apr": 0.01,
    }
    zero_result = simulate(scenario=ZERO_FEES, **common)
    taker_result = simulate(scenario=TAKER, **common)

    roundtrip_fee = notional * TAKER.roundtrip_frac
    assert zero_result.fees_paid == 0.0
    assert taker_result.fees_paid == pytest.approx(roundtrip_fee)
    assert (zero_result.net_pnl - taker_result.net_pnl) == pytest.approx(roundtrip_fee)
    # funding/basis are identical between the two runs — only fees differ.
    assert zero_result.funding_pnl == pytest.approx(taker_result.funding_pnl)
    assert zero_result.basis_pnl == pytest.approx(taker_result.basis_pnl)


def test_identity_net_equals_funding_plus_basis_minus_fees() -> None:
    """A churny run (multiple entries/exits from varying, non-monotone rates
    and moving prices): the PnL identity must hold to 1e-9 regardless of how
    many trades occurred.
    """
    rates = [
        0.0002, -0.0001, 0.00015, 0.00005, -0.0003, 0.0001,
        0.00025, -0.00005, 0.0004, -0.0002, 0.00018, 0.00007,
    ]
    n_events = len(rates)
    hours = (n_events - 1) * INTERVAL_HOURS + 8
    funding = make_funding("ETHUSDT", rates, interval_hours=INTERVAL_HOURS)
    spot_closes = [100.0 + 0.3 * i for i in range(hours)]
    perp_closes = [100.0 + 0.3 * i + (1.5 if (i // 7) % 2 == 0 else -0.5) for i in range(hours)]
    spot = make_klines("ETHUSDT", spot_closes)
    perp = make_klines("ETHUSDT", perp_closes)

    result = simulate(
        funding=funding,
        spot=spot,
        perp=perp,
        interval_hours=INTERVAL_HOURS,
        scenario=TAKER,
        notional=10_000.0,
        ema_periods=3,
        enter_apr=0.05,
        exit_apr=0.02,
    )

    assert result.net_pnl == pytest.approx(
        result.funding_pnl + result.basis_pnl - result.fees_paid, abs=1e-9
    )


def test_no_lookahead_early_periods_cant_signal_future_rates_dont_move_past_decisions() -> None:
    """The first `ema_periods` events can never carry a signal (the EMA
    needs that many prior rates); changing a rate far in the future must not
    change a decision already made at an earlier event; and changing the
    rate that settles AT the decision event itself must not move that same
    decision either (the forecast used to decide at event i is built only
    from rates[0:i], excluding rates[i] — that's the whole no-lookahead
    property, and this directly kills an "EMA includes the current rate"
    bug).
    """
    ema_periods = 3
    base_rates = [1.0, 1.0, 1.0] + [0.5] * 9  # 12 events; huge, then moderate positive
    hours = (len(base_rates) - 1) * INTERVAL_HOURS + 8

    def _run(rates: list[float]) -> SimResult:
        funding = make_funding("SOLUSDT", rates, interval_hours=INTERVAL_HOURS)
        spot, perp = _flat_klines("SOLUSDT", 100.0, hours=hours)
        return simulate(
            funding=funding,
            spot=spot,
            perp=perp,
            interval_hours=INTERVAL_HOURS,
            scenario=ZERO_FEES,
            notional=10_000.0,
            ema_periods=ema_periods,
            enter_apr=0.05,
            # exit_apr set far below anything reachable here so the only
            # exit is the forced close at the last event — keeps this test
            # about entry timing only, not exit timing.
            exit_apr=-10.0,
        )

    result_a = _run(base_rates)
    future_changed = list(base_rates)
    future_changed[10] = -5.0  # a rate 7 events after entry — must not matter
    result_b = _run(future_changed)
    same_event_changed = list(base_rates)
    same_event_changed[ema_periods] = -100.0  # the rate settling AT the entry event itself
    result_c = _run(same_event_changed)

    expected_entry_time = T0 + timedelta(hours=INTERVAL_HOURS * ema_periods)

    assert result_a.n_trades == 1
    assert result_b.n_trades == 1
    assert result_c.n_trades >= 1
    assert result_a.trades[0].entry_time == expected_entry_time
    assert result_b.trades[0].entry_time == expected_entry_time
    assert result_c.trades[0].entry_time == expected_entry_time
    # no entry could possibly have happened before the signal exists
    for early_index in range(ema_periods):
        assert result_a.trades[0].entry_time != T0 + timedelta(hours=INTERVAL_HOURS * early_index)


def test_basis_pnl_matches_hand_computed_value_with_zero_funding_and_fees() -> None:
    """Enter with a +1%-of-price basis, exit with zero basis, zero funding
    while in position, zero fees: net must equal qty * basis_entry exactly.
    """
    # rates[0:3] prime the EMA to enter at index 3; rates[3:] are 0 so the
    # single in-position (and force-closing) event contributes no funding.
    rates = [0.001, 0.001, 0.001, 0.0, 0.0]
    funding = make_funding("BTCUSDT", rates, interval_hours=INTERVAL_HOURS)

    hours = 40
    spot_closes = [100.0] * hours
    perp_closes = [100.0] * hours
    # Klines are asof-joined by CLOSE time (open + 1h), so the bar whose
    # close prices the event at hour H is the one opened at H-1.
    spot_closes[31] = 105.0  # closes at 32h (exit event): basis -> 0
    perp_closes[23] = 101.0  # closes at 24h (entry event): basis -> +1 (1% of 100)
    perp_closes[31] = 105.0  # closes at 32h (exit event): basis -> 0
    spot = make_klines("BTCUSDT", spot_closes)
    perp = make_klines("BTCUSDT", perp_closes)

    result = simulate(
        funding=funding,
        spot=spot,
        perp=perp,
        interval_hours=INTERVAL_HOURS,
        scenario=ZERO_FEES,
        notional=10_000.0,
        ema_periods=3,
        enter_apr=0.05,
        exit_apr=-1.0,  # never triggers naturally; forced close at last event
    )

    assert result.n_trades == 1
    assert result.funding_pnl == pytest.approx(0.0)
    assert result.fees_paid == 0.0
    trade = result.trades[0]
    basis_entry_diff = trade.entry_perp - trade.entry_spot
    assert basis_entry_diff == pytest.approx(1.0)
    assert result.net_pnl == pytest.approx(trade.qty * basis_entry_diff)


def test_hysteresis_holds_through_a_mid_band_dip_without_churning() -> None:
    """apr dips between exit_apr and enter_apr after entry: the position
    must be held, not churned (re-entered/exited). Uses ema_periods=1 so
    forecast[i] == rates[i-1] exactly, making the apr path fully explicit.
    """
    enter_apr = 0.12
    exit_apr = 0.02
    periods_per_year = 365.25 * 24 / INTERVAL_HOURS
    # forecast[1]=rates[0] -> apr ~16.4% (enter); forecast[2]=rates[1] -> ~5.5%
    # (mid-band dip: below enter_apr, above exit_apr -> must hold);
    # forecast[3]=rates[2] -> ~21.9% (back up, still just holding);
    # forecast[4]=rates[3] -> ~11% (mid-band again, last event forces close).
    rates = [0.00015, 0.00005, 0.0002, 0.0001, 0.0001]
    assert rates[0] * periods_per_year > enter_apr
    assert exit_apr < rates[1] * periods_per_year < enter_apr

    funding = make_funding("BTCUSDT", rates, interval_hours=INTERVAL_HOURS)
    hours = (len(rates) - 1) * INTERVAL_HOURS + 8
    spot, perp = _flat_klines("BTCUSDT", 100.0, hours=hours)

    result = simulate(
        funding=funding,
        spot=spot,
        perp=perp,
        interval_hours=INTERVAL_HOURS,
        scenario=TAKER,
        notional=10_000.0,
        ema_periods=1,
        enter_apr=enter_apr,
        exit_apr=exit_apr,
    )

    assert result.n_trades == 1
    trade = result.trades[0]
    assert trade.entry_time == T0 + timedelta(hours=INTERVAL_HOURS * 1)
    assert trade.exit_time == T0 + timedelta(hours=INTERVAL_HOURS * (len(rates) - 1))
    assert trade.force_closed


def test_funding_accrual_uses_per_event_mark_price_not_frozen_entry_notional() -> None:
    """Binance settles funding on the mark price AT EACH EVENT, not on the
    perp price frozen at entry. Set mark_price well away from the (flat)
    perp/spot close so a frozen-entry-notional bug gives a different, wrong
    answer than accrual computed from the event's mark price.
    """
    rate = 0.0001
    n_events = 10
    notional = 10_000.0
    mark_price = 110.0  # deliberately far from the flat 100.0 perp/spot close
    funding = make_funding(
        "BTCUSDT", [rate] * n_events, interval_hours=INTERVAL_HOURS, mark_price=mark_price
    )
    spot, perp = _flat_klines("BTCUSDT", 100.0, hours=(n_events - 1) * INTERVAL_HOURS + 8)

    result = simulate(
        funding=funding,
        spot=spot,
        perp=perp,
        interval_hours=INTERVAL_HOURS,
        scenario=ZERO_FEES,
        notional=notional,
        ema_periods=2,
        enter_apr=0.05,
        exit_apr=0.01,
    )

    # qty = notional / entry_spot(100) = 100. entry_perp is also 100 (flat),
    # so a frozen-entry-notional bug would accrue on qty*entry_perp=10,000;
    # the correct accrual is on qty*mark_price=11,000.
    qty = notional / 100.0
    n_in_position_events = (n_events - 1) - 2
    expected_funding = n_in_position_events * rate * qty * mark_price
    wrong_funding_if_frozen_entry_notional = n_in_position_events * rate * qty * 100.0

    assert result.funding_pnl == pytest.approx(expected_funding)
    assert result.funding_pnl != pytest.approx(wrong_funding_if_frozen_entry_notional)


def test_annualization_uses_the_passed_interval_hours_not_a_hardcoded_8h() -> None:
    """Same constant rate, but funding settles every 4h instead of 8h: the
    annualized apr must roughly double (YEAR_HOURS/4 vs YEAR_HOURS/8),
    crossing enter_apr in one case but not the other. A hardcoded 8h
    annualization constant would make both cases identical (neither enters).
    """
    rate = 0.0001
    n_events = 10
    enter_apr = 0.15  # between the 8h apr (~11%) and the 4h apr (~22%)

    def _run(interval_hours: int) -> SimResult:
        funding = make_funding("BTCUSDT", [rate] * n_events, interval_hours=interval_hours)
        spot, perp = _flat_klines(
            "BTCUSDT", 100.0, hours=(n_events - 1) * interval_hours + interval_hours
        )
        return simulate(
            funding=funding,
            spot=spot,
            perp=perp,
            interval_hours=interval_hours,
            scenario=ZERO_FEES,
            notional=10_000.0,
            ema_periods=2,
            enter_apr=enter_apr,
            exit_apr=0.01,
        )

    result_8h = _run(8)
    result_4h = _run(4)

    assert result_8h.n_trades == 0
    assert result_4h.n_trades >= 1


def test_unpriced_exit_walks_back_to_last_priced_event_instead_of_dropping_the_trade() -> None:
    """Klines end well before the funding history does (a common real-world
    shape: a truncated/short kline file). The position must still be
    force-closed — at the last event where both legs COULD be priced — not
    silently dropped. This is also exactly the case that would slip through
    if the staleness guard in `_asof_close` were ever disabled: without the
    guard, exit would (wrongly) succeed directly at the final event using an
    ancient close, giving unpriced_exit=False and a different exit_time.
    """
    rate = 0.0001
    n_events = 10  # events at 0,8,16,...,72h
    notional = 10_000.0
    funding = make_funding("BTCUSDT", [rate] * n_events, interval_hours=INTERVAL_HOURS)
    # Klines only cover hours 0..20 (21 bars) -> close times 1..21h. Entry at
    # 16h prices fine; by the last funding event (72h) the last available
    # close (21h) is 51h stale, well past the 24h staleness bound.
    spot, perp = _flat_klines("BTCUSDT", 100.0, hours=21)

    result = simulate(
        funding=funding,
        spot=spot,
        perp=perp,
        interval_hours=INTERVAL_HOURS,
        scenario=ZERO_FEES,
        notional=notional,
        ema_periods=2,
        enter_apr=0.05,
        exit_apr=0.01,
    )

    # Entry at index 2 (16h). Last point both legs were priced within the
    # 24h staleness bound: event at 40h (age 40-21=19h <= 24h); the next
    # event (48h, age 27h) is stale. So the walk-back exit lands at 40h.
    assert result.n_trades == 1
    assert result.trades[0].unpriced_exit is True
    assert result.trades[0].exit_time == T0 + timedelta(hours=40)
    assert result.unpriced_exit_count == 1

    # Funding accrues on mark_price (unaffected by kline staleness) for
    # every in-position event (indices 3..9, i.e. 7 events) regardless of
    # where the walk-back landed — the trade must not vanish nor lose PnL.
    qty = notional / 100.0
    expected_funding = 7 * rate * qty * 100.0
    assert result.funding_pnl == pytest.approx(expected_funding)
    assert result.net_pnl == pytest.approx(expected_funding)  # zero basis, zero fees

    # The equity curve's own last value must agree with the totals it fed —
    # this is the "vacuous identity" failure mode: before the fix, totals
    # were 0/0 while the curve still showed accrued PnL.
    last_equity = result.equity_curve["equity"].to_list()[-1]
    assert last_equity == pytest.approx(result.net_pnl)


def test_portfolio_dd_is_drawdown_of_the_summed_curve_not_max_of_per_symbol_dd() -> None:
    """Direct regression test for the portfolio-DD fix: symbol A drops 50
    and holds; symbol B (independently) drops to 100 later. Naive
    max(per-symbol DD) = 100, but the two losses overlap in the combined
    portfolio equity — its true max drawdown is 150. A mutant that reverts
    `_portfolio_equity_max_drawdown` to `max(r.max_drawdown for r in ...)`
    must fail this.
    """

    def _curve(equity_values: list[float]) -> pl.DataFrame:
        times = [T0 + timedelta(hours=i) for i in range(len(equity_values))]
        return pl.DataFrame(
            {"time": times, "equity": equity_values},
            schema={"time": pl.Datetime("us", "UTC"), "equity": pl.Float64},
        )

    def _result(equity_values: list[float], dd: float) -> SimResult:
        return SimResult(
            trades=[], funding_pnl=0.0, basis_pnl=0.0, fees_paid=0.0,
            net_pnl=equity_values[-1], equity_curve=_curve(equity_values),
            max_drawdown=dd, n_trades=0, avg_holding_hours=0.0, holding_hours_sum=0.0,
            time_in_market_frac=0.0, period_start=T0, period_end=T0 + timedelta(hours=3),
        )

    a = _result([0.0, -50.0, -50.0, -50.0], dd=50.0)
    b = _result([0.0, 0.0, -50.0, -100.0], dd=100.0)
    results = {"A": {"s": a}, "B": {"s": b}}

    naive_max = max(a.max_drawdown, b.max_drawdown)
    combined_dd = _portfolio_equity_max_drawdown(results, ["A", "B"], "s")

    assert naive_max == 100.0
    assert combined_dd == pytest.approx(150.0)


def test_portfolio_dd_join_forward_fills_across_staggered_symbol_timestamps() -> None:
    """Symbols A and B report equity at DIFFERENT, staggered timestamps (as
    real symbols with different funding schedules would). Without
    forward-filling each symbol's value across the other's timestamps
    before summing, the join would (wrongly) treat "no reading yet" as 0
    instead of "unchanged since its last reading", understating the
    portfolio drawdown. Correct: combined DD = 100. Without forward_fill:
    combined DD = 50.
    """

    def _curve(times: list[datetime], equity_values: list[float]) -> pl.DataFrame:
        return pl.DataFrame(
            {"time": times, "equity": equity_values},
            schema={"time": pl.Datetime("us", "UTC"), "equity": pl.Float64},
        )

    def _result(times: list[datetime], equity_values: list[float]) -> SimResult:
        return SimResult(
            trades=[], funding_pnl=0.0, basis_pnl=0.0, fees_paid=0.0,
            net_pnl=equity_values[-1], equity_curve=_curve(times, equity_values),
            max_drawdown=0.0, n_trades=0, avg_holding_hours=0.0, holding_hours_sum=0.0,
            time_in_market_frac=0.0, period_start=times[0], period_end=times[-1],
        )

    # A drops to -50 at hour 2 and holds; B drops to -50 at hour 3 and holds.
    a = _result([T0 + timedelta(hours=h) for h in (0, 2, 4)], [0.0, -50.0, -50.0])
    b = _result([T0 + timedelta(hours=h) for h in (1, 3)], [0.0, -50.0])
    results = {"A": {"s": a}, "B": {"s": b}}

    combined_dd = _portfolio_equity_max_drawdown(results, ["A", "B"], "s")
    assert combined_dd == pytest.approx(100.0)


def test_derive_interval_hours_returns_actual_gap_not_hardcoded_8h() -> None:
    """Feed a 4h-spaced funding timeline directly to `_derive_interval_hours`
    and require 4, not 8 — kills a hardcoded-8h mutant directly at the
    source, independent of any downstream annualization behavior.
    """
    times = [T0 + timedelta(hours=4 * i) for i in range(12)]
    assert _derive_interval_hours(times) == 4


def test_render_report_deployed_apr_uses_holding_hours_not_period_hours() -> None:
    """Two symbols with DIFFERENT known holding_hours_sum (so their deployed
    APRs, and the portfolio's, are all numerically distinct): render the
    report and check each symbol's OWN row carries its own hand-computed
    deployed-APR percentage.

    Pinning to the per-symbol ROW (not just "somewhere in the markdown")
    matters: with only one symbol, the portfolio row's separately-computed
    deployed APR happens to equal the per-symbol one, so a mutant that
    breaks only the per-symbol computation (e.g. collapses it to the
    alloc-APR formula) would slip through undetected if the portfolio row's
    correct figure were still floating around in the document.
    """
    notional = 10_000.0
    period_start = T0
    period_end = T0 + timedelta(days=100)
    empty_curve = pl.DataFrame(schema={"time": pl.Datetime("us", "UTC"), "equity": pl.Float64})

    def _make(net_pnl: float, holding_hours: float) -> SimResult:
        return SimResult(
            trades=[], funding_pnl=net_pnl, basis_pnl=0.0, fees_paid=0.0, net_pnl=net_pnl,
            equity_curve=empty_curve, max_drawdown=0.0, n_trades=1, avg_holding_hours=holding_hours,
            holding_hours_sum=holding_hours, time_in_market_frac=holding_hours / (100 * 24),
            period_start=period_start, period_end=period_end,
        )

    btc = _make(net_pnl=50.0, holding_hours=24.0 * 10)   # 10 days deployed
    eth = _make(net_pnl=80.0, holding_hours=24.0 * 20)   # 20 days deployed
    results = {"BTCUSDT": {"taker": btc}, "ETHUSDT": {"taker": eth}}
    cfg = BacktestConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        notional_usdt=notional,
        scenarios=[FeeScenario(name="taker", spot_bps=0.0, perp_bps=0.0, slippage_bps=0.0)],
    )

    report_md = render_report(results, cfg, sensitivity=[])
    lines = report_md.splitlines()

    def _row(symbol: str) -> str:
        matches = [line for line in lines if line.startswith(f"| {symbol} |")]
        assert len(matches) == 1, f"expected exactly one {symbol} row, got {matches}"
        return matches[0]

    def _deployed_apr_str(r: SimResult) -> str:
        return f"{r.net_pnl / notional / (r.holding_hours_sum / YEAR_HOURS):.1%}"

    btc_str, eth_str = _deployed_apr_str(btc), _deployed_apr_str(eth)
    assert btc_str != eth_str  # sanity: the two must actually be distinguishable
    assert btc_str in _row("BTCUSDT")
    assert eth_str in _row("ETHUSDT")
