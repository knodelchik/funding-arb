"""EXPERIMENT P — premium signal (clamp-aware predictor D) in PnL terms.

Arms:
  (a) D-as-forecast: monkeypatch engine._forecast_series so forecast[i] = D(T_i),
      run stock simulate at baseline thresholds (12%/2% APR). Control arm.
  (b) HYBRID: entry requires ema6 apr > 12% AND D > 0; exit when ema6 apr < 2%
      OR D < -8%-APR-equivalent (rate threshold = -0.08 * interval_h / YEAR_HOURS).
  (c) D-exit-only: baseline entry (ema6 apr > 12%), exit ONLY when D < 0.

D(T) = premium_mean + clamp(0.0001 - premium_mean, +-0.0005), premium_mean =
mean of premium_1h bar closes whose CLOSE time lies in (T-8h, T-1h] with T
truncated to the hour — every bar closed by T-1h => strictly PIT.

Sanity gates:
  1. stock engine reproduces baseline taker net 1891.57 (abort loudly otherwise);
  2. the copied gated loop with gates disabled reproduces stock simulate() per
     symbol x scenario to 1e-9 (abort otherwise).
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, "/Users/bogdan/funding-arb/src")

import farb.backtest.engine as eng  # noqa: E402
from farb.backtest.engine import (  # noqa: E402
    SimResult,
    Trade,
    _asof_close,
    _close_times,
    _derive_interval_hours,
    _empty_result,
    _forecast_series,
    _oneway_fee,
    simulate,
)
from farb.backtest.report import YEAR_HOURS  # noqa: E402
from farb.config import DEFAULT_SCENARIOS  # noqa: E402

DATA = Path("/Users/bogdan/funding-arb/data/binance")
SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "TRXUSDT", "NEARUSDT", "ATOMUSDT", "FILUSDT", "ETCUSDT", "APTUSDT",
    "SUIUSDT", "ARBUSDT",
]
NOTIONAL = 10_000.0
EMA_PERIODS = 6
ENTER_APR = 0.12
EXIT_APR = 0.02
BASELINE_TAKER_NET = 1891.57
BASELINE_MAKER_NET = 2446.57

HOUR = timedelta(hours=1)


# ---------------------------------------------------------------- predictor D
def compute_d(funding_times: list[datetime], premium: pl.DataFrame) -> list[float]:
    """D per funding event; NaN when no premium bar is available in window."""
    premium = premium.sort("open_time")
    p_open: list[datetime] = premium["open_time"].to_list()
    p_close_val: list[float] = premium["close"].to_list()
    # hourly premium bars: bar closes at open + 1h
    by_open = dict(zip(p_open, p_close_val, strict=True))
    d_list: list[float] = []
    for t in funding_times:
        t_h = t.replace(minute=0, second=0, microsecond=0)
        # bars whose close in (t_h-8h, t_h-1h]  <=>  open in [t_h-8h, t_h-2h]
        vals = []
        for k in range(8, 1, -1):  # open at t_h - 8h ... t_h - 2h  (7 bars)
            v = by_open.get(t_h - k * HOUR)
            if v is not None and not math.isnan(v):
                vals.append(v)
        if not vals:
            d_list.append(math.nan)
            continue
        pm = sum(vals) / len(vals)
        c = max(-0.0005, min(0.0005, 0.0001 - pm))
        d_list.append(pm + c)
    return d_list


# ------------------------------------------------- faithful copy of simulate()
# Identical to engine.simulate except:
#   * entry/exit decisions delegated to entry_rule(i)/exit_rule(i) callables,
#   * per-event bookkeeping returned: open_entering[i] (position open at top of
#     event i) and accrual[i] (funding USDT accrued at event i; 0 when flat).
def simulate_gated(
    funding: pl.DataFrame,
    spot: pl.DataFrame,
    perp: pl.DataFrame,
    interval_hours: int,
    scenario,
    notional: float,
    entry_rule,
    exit_rule,
) -> tuple[SimResult, list[bool], list[float]]:
    funding = funding.sort("funding_time")
    spot = spot.sort("open_time")
    perp = perp.sort("open_time")

    times: list[datetime] = funding["funding_time"].to_list()
    rates: list[float] = funding["funding_rate"].to_list()
    mark_prices: list[float | None] = funding["mark_price"].to_list()
    n = len(times)
    if n == 0:
        return _empty_result(), [], []

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

    last_priced_time: datetime | None = None
    last_priced_spot: float | None = None
    last_priced_perp: float | None = None

    equity_times: list[datetime] = []
    equity_values: list[float] = []

    open_entering = [False] * n
    accrual = [0.0] * n

    for i in range(n):
        t = times[i]
        open_entering[i] = position_open

        if position_open:
            mark_px = mark_prices[i]
            mark_missing = mark_px is None or math.isnan(mark_px)
            if not mark_missing:
                accrual_notional = qty_v * mark_px
            else:
                now_perp = _asof_close(perp_times, perp_closes, t)
                accrual_notional = qty_v * now_perp if now_perp is not None else 0.0
            pos_funding += rates[i] * accrual_notional
            accrual[i] = rates[i] * accrual_notional

            is_last = i == n - 1
            do_exit = exit_rule(i)
            if do_exit or is_last:
                exit_time = t
                exit_spot_px = _asof_close(spot_times, spot_closes, t)
                exit_perp_px = _asof_close(perp_times, perp_closes, t)
                unpriced_exit = exit_spot_px is None or exit_perp_px is None
                if unpriced_exit:
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
        else:
            if entry_rule(i):
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

        if position_open:
            now_spot = _asof_close(spot_times, spot_closes, t)
            now_perp = _asof_close(perp_times, perp_closes, t)
            if now_spot is not None and now_perp is not None:
                last_priced_time = t
                last_priced_spot = now_spot
                last_priced_perp = now_perp
                unrealized_basis = qty_v * ((entry_perp_px - entry_spot_px) - (now_perp - now_spot))
            elif last_priced_spot is not None and last_priced_perp is not None:
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
        raise RuntimeError(f"PnL identity violated (gap={identity_gap})")

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

    res = SimResult(
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
    return res, open_entering, accrual


# --------------------------------------------------------------------- runner
def main() -> None:
    scenarios = {s.name: s for s in DEFAULT_SCENARIOS}

    frames = {}
    for sym in SYMBOLS:
        funding = pl.read_parquet(DATA / "funding" / f"{sym}.parquet").sort("funding_time")
        spot = pl.read_parquet(DATA / "klines_spot_1h" / f"{sym}.parquet")
        perp = pl.read_parquet(DATA / "klines_perp_1h" / f"{sym}.parquet")
        premium = pl.read_parquet(DATA / "premium_1h" / f"{sym}.parquet")
        ih = _derive_interval_hours(funding["funding_time"].to_list())
        assert ih == 8, f"{sym}: derived interval {ih}h != 8h"
        frames[sym] = (funding, spot, perp, premium, ih)

    # ---- sanity gate 1: reproduce baseline with stock engine
    base_stock = {}
    for scen_name, scen in scenarios.items():
        tot = 0.0
        per_sym = {}
        for sym, (funding, spot, perp, _prem, ih) in frames.items():
            r = simulate(funding, spot, perp, ih, scen, NOTIONAL, EMA_PERIODS, ENTER_APR, EXIT_APR)
            per_sym[sym] = r
            tot += r.net_pnl
        base_stock[scen_name] = (tot, per_sym)
    taker_net = base_stock["taker"][0]
    maker_net = base_stock["maker_bnb"][0]
    if abs(taker_net - BASELINE_TAKER_NET) > 0.01:
        print(f"ABORT: baseline taker net {taker_net:.2f} != {BASELINE_TAKER_NET}", file=sys.stderr)
        sys.exit(1)
    print(f"sanity gate 1 OK: taker {taker_net:.2f}, maker_bnb {maker_net:.2f}")

    # ---- precompute per-symbol ema aprs, D, D thresholds
    sig = {}
    d_nan_events = 0
    total_events = 0
    for sym, (funding, spot, perp, premium, ih) in frames.items():
        times = funding["funding_time"].to_list()
        rates = funding["funding_rate"].to_list()
        ppy = YEAR_HOURS / ih
        f = _forecast_series(rates, EMA_PERIODS)
        aprs = [x * ppy if not math.isnan(x) else math.nan for x in f]
        d = compute_d(times, premium)
        d_nan_events += sum(1 for x in d if math.isnan(x))
        total_events += len(d)
        sig[sym] = (aprs, d, ppy)
    print(f"D coverage: {total_events - d_nan_events}/{total_events} events "
          f"({d_nan_events} NaN)")

    # ---- sanity gate 2: gated copy with gates disabled == stock simulate
    base_gated = {}   # scen -> {sym: (res, open_entering, accrual)}
    for scen_name, scen in scenarios.items():
        base_gated[scen_name] = {}
        for sym, (funding, spot, perp, _prem, ih) in frames.items():
            aprs, _d, _ppy = sig[sym]
            entry = lambda i, a=aprs: (not math.isnan(a[i])) and a[i] > ENTER_APR
            exitr = lambda i, a=aprs: (not math.isnan(a[i])) and a[i] < EXIT_APR
            res, oe, acc = simulate_gated(funding, spot, perp, ih, scen, NOTIONAL, entry, exitr)
            r0 = base_stock[scen_name][1][sym]
            for attr in ("net_pnl", "funding_pnl", "basis_pnl", "fees_paid",
                         "n_trades", "holding_hours_sum"):
                v0, v1 = getattr(r0, attr), getattr(res, attr)
                if isinstance(v0, float):
                    ok = abs(v0 - v1) < 1e-9
                else:
                    ok = v0 == v1
                if not ok:
                    print(f"ABORT: gated copy mismatch {sym}/{scen_name}.{attr}: "
                          f"{v0} vs {v1}", file=sys.stderr)
                    sys.exit(1)
            base_gated[scen_name][sym] = (res, oe, acc)
    print("sanity gate 2 OK: gated copy == stock engine (all 20 symbols x 2 scenarios)")

    # ---- arm (a): D-as-forecast via sanctioned monkeypatch
    arm_a_stock = {}
    orig = eng._forecast_series
    try:
        for scen_name, scen in scenarios.items():
            arm_a_stock[scen_name] = {}
            for sym, (funding, spot, perp, _prem, ih) in frames.items():
                _aprs, d, _ppy = sig[sym]

                def patched(rates, ema_periods, _d=d):
                    assert len(rates) == len(_d), "patched forecast length mismatch"
                    return _d

                eng._forecast_series = patched
                r = eng.simulate(funding, spot, perp, ih, scen, NOTIONAL,
                                 EMA_PERIODS, ENTER_APR, EXIT_APR)
                arm_a_stock[scen_name][sym] = r
    finally:
        eng._forecast_series = orig

    # same arm through the gated copy (cross-check + per-event bookkeeping)
    def run_arm(entry_fn_factory, exit_fn_factory):
        out = {}
        for scen_name, scen in scenarios.items():
            out[scen_name] = {}
            for sym, (funding, spot, perp, _prem, ih) in frames.items():
                aprs, d, ppy = sig[sym]
                entry = entry_fn_factory(aprs, d, ppy)
                exitr = exit_fn_factory(aprs, d, ppy)
                out[scen_name][sym] = simulate_gated(
                    funding, spot, perp, ih, scen, NOTIONAL, entry, exitr)
        return out

    # (a) forecast = D at baseline thresholds: enter d*ppy > 12%, exit d*ppy < 2%
    arm_a = run_arm(
        lambda aprs, d, ppy: (lambda i: (not math.isnan(d[i])) and d[i] * ppy > ENTER_APR),
        lambda aprs, d, ppy: (lambda i: (not math.isnan(d[i])) and d[i] * ppy < EXIT_APR),
    )
    for scen_name in scenarios:
        for sym in frames:
            v0 = arm_a_stock[scen_name][sym].net_pnl
            v1 = arm_a[scen_name][sym][0].net_pnl
            if abs(v0 - v1) > 1e-9:
                print(f"ABORT: arm(a) monkeypatch vs copy mismatch {sym}/{scen_name}: "
                      f"{v0} vs {v1}", file=sys.stderr)
                sys.exit(1)
    print("arm (a) cross-check OK: monkeypatched stock == gated copy")

    # (b) HYBRID: entry ema apr > 12% AND D > 0; exit ema apr < 2% OR D < -8%-APR-equiv
    arm_b = run_arm(
        lambda aprs, d, ppy: (lambda i: (not math.isnan(aprs[i])) and aprs[i] > ENTER_APR
                              and (not math.isnan(d[i])) and d[i] > 0.0),
        lambda aprs, d, ppy: (lambda i:
                              ((not math.isnan(aprs[i])) and aprs[i] < EXIT_APR)
                              or ((not math.isnan(d[i])) and d[i] < -0.08 / ppy)),
    )

    # (c) D-exit-only: baseline entry, exit ONLY on D < 0
    arm_c = run_arm(
        lambda aprs, d, ppy: (lambda i: (not math.isnan(aprs[i])) and aprs[i] > ENTER_APR),
        lambda aprs, d, ppy: (lambda i: (not math.isnan(d[i])) and d[i] < 0.0),
    )

    # ---- reporting
    def agg(arm, scen_name):
        rs = [arm[scen_name][sym][0] for sym in frames]
        net = sum(r.net_pnl for r in rs)
        trades = sum(r.n_trades for r in rs)
        fees = sum(r.fees_paid for r in rs)
        fund = sum(r.funding_pnl for r in rs)
        basis = sum(r.basis_pnl for r in rs)
        hh = sum(r.holding_hours_sum for r in rs)
        dep_apr = net / (NOTIONAL * hh / YEAR_HOURS) if hh > 0 else math.nan
        tim = hh / sum(
            (r.period_end - r.period_start).total_seconds() / 3600 for r in rs)
        return dict(net=net, trades=trades, fees=fees, funding=fund, basis=basis,
                    holding_hours=hh, deployed_apr=dep_apr, time_in_mkt=tim)

    def funding_decomp(arm, scen_name):
        """vs baseline: avoided negative events, missed positive, extra events."""
        avoided_n = 0
        avoided_usd = 0.0     # negative funding baseline paid while arm was flat
        missed_n = 0
        missed_usd = 0.0      # positive funding baseline collected while arm flat
        extra_pos_n = 0
        extra_pos_usd = 0.0   # positive funding arm collected while baseline flat
        extra_neg_n = 0
        extra_neg_usd = 0.0   # negative funding arm paid while baseline flat
        for sym in frames:
            _rb, oe_b, acc_b = base_gated[scen_name][sym]
            _ra, oe_a, acc_a = arm[scen_name][sym]
            for i in range(len(oe_b)):
                if oe_b[i] and not oe_a[i]:
                    if acc_b[i] < 0:
                        avoided_n += 1
                        avoided_usd += -acc_b[i]
                    elif acc_b[i] > 0:
                        missed_n += 1
                        missed_usd += acc_b[i]
                elif oe_a[i] and not oe_b[i]:
                    if acc_a[i] > 0:
                        extra_pos_n += 1
                        extra_pos_usd += acc_a[i]
                    elif acc_a[i] < 0:
                        extra_neg_n += 1
                        extra_neg_usd += -acc_a[i]
        return dict(avoided_n=avoided_n, avoided_usd=avoided_usd,
                    missed_n=missed_n, missed_usd=missed_usd,
                    extra_pos_n=extra_pos_n, extra_pos_usd=extra_pos_usd,
                    extra_neg_n=extra_neg_n, extra_neg_usd=extra_neg_usd)

    arms = {
        "baseline": base_gated,
        "a_D_as_forecast": {s: {sym: v for sym, v in arm_a[s].items()} for s in arm_a},
        "b_hybrid": arm_b,
        "c_D_exit_only": arm_c,
    }

    print("\n=== EXPERIMENT P results (20 symbols, 10k USDT each) ===")
    header = (f"{'arm':>16} {'scen':>9} {'net':>10} {'dNet':>9} {'trades':>6} "
              f"{'fees':>9} {'funding':>9} {'basis':>8} {'depAPR':>7} {'TiM':>6}")
    print(header)
    results = {}
    for arm_name, arm in arms.items():
        for scen_name in scenarios:
            a = agg(arm, scen_name)
            b = agg(base_gated, scen_name)
            d_net = a["net"] - b["net"]
            results[(arm_name, scen_name)] = a
            print(f"{arm_name:>16} {scen_name:>9} {a['net']:>10.2f} {d_net:>9.2f} "
                  f"{a['trades']:>6d} {a['fees']:>9.2f} {a['funding']:>9.2f} "
                  f"{a['basis']:>8.2f} {a['deployed_apr']:>7.1%} {a['time_in_mkt']:>6.1%}")

    print("\n--- funding-event decomposition vs baseline (scenario-independent; taker shown) ---")
    for arm_name in ("a_D_as_forecast", "b_hybrid", "c_D_exit_only"):
        d = funding_decomp(arms[arm_name], "taker")
        results[(arm_name, "decomp")] = d
        print(f"{arm_name:>16}: avoided NEG {d['avoided_n']:>4d} ev = +{d['avoided_usd']:.2f} USDT | "
              f"missed POS {d['missed_n']:>4d} ev = -{d['missed_usd']:.2f} | "
              f"extra POS {d['extra_pos_n']:>4d} ev = +{d['extra_pos_usd']:.2f} | "
              f"extra NEG {d['extra_neg_n']:>4d} ev = -{d['extra_neg_usd']:.2f}")

    # negative-funding actually PAID by each arm (incl. baseline)
    print("\n--- negative funding paid while in position ---")
    for arm_name, arm in arms.items():
        neg_n = 0
        neg_usd = 0.0
        for sym in frames:
            _r, _oe, acc = arm["taker"][sym]
            for x in acc:
                if x < 0:
                    neg_n += 1
                    neg_usd += -x
        print(f"{arm_name:>16}: {neg_n:>4d} negative events paid, -{neg_usd:.2f} USDT")

    # per-symbol nets for the winner arms (taker)
    print("\n--- per-symbol net (taker): baseline / hybrid / d_exit_only ---")
    for sym in sorted(frames):
        b = base_gated["taker"][sym][0].net_pnl
        h = arm_b["taker"][sym][0].net_pnl
        c = arm_c["taker"][sym][0].net_pnl
        print(f"  {sym:>10}: {b:>9.2f} {h:>9.2f} {c:>9.2f}")


if __name__ == "__main__":
    main()
