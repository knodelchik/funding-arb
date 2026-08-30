"""E2 — allocator value: rotate ONE pool of 200k USDT across the 20 symbols.

Portfolio event-loop over the union of all symbols' 8h funding events
(timestamps truncated to the hour to align the grid). At each event:
  1) held positions accrue funding on per-event mark notional and exit when
     PIT forecast APR < exit_apr (force-close at the symbol's final event);
  2) free slots (of K total) are filled by symbols with forecast APR >
     enter_apr, ranked by forecast APR desc, entered at asof close prices.

PIT: forecast[i] is the exact `_forecast_series` recursion imported from
farb.backtest.engine — EMA(6) over rates settled STRICTLY before event i.
Prices are backward-asof against kline CLOSE times (engine helpers reused).
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, "/Users/bogdan/funding-arb/src")

import polars as pl

from farb.backtest.engine import _asof_close, _close_times, _forecast_series, _oneway_fee
from farb.backtest.report import YEAR_HOURS
from farb.config import DEFAULT_SCENARIOS

DATA = Path("/Users/bogdan/funding-arb/data/binance")
TOTAL_CAPITAL = 200_000.0
INTERVAL_HOURS = 8
EMA_PERIODS = 6
ENTER_APR = 0.12
EXIT_APR = 0.02
KS = [2, 4, 6, 10]
BASELINE = {"taker": 1891.57, "maker_bnb": 2446.57}

SYMBOLS = sorted(p.stem for p in (DATA / "funding").glob("*.parquet"))


def load_symbol(sym: str) -> dict:
    funding = pl.read_parquet(DATA / "funding" / f"{sym}.parquet").sort("funding_time")
    spot = pl.read_parquet(DATA / "klines_spot_1h" / f"{sym}.parquet").sort("open_time")
    perp = pl.read_parquet(DATA / "klines_perp_1h" / f"{sym}.parquet").sort("open_time")

    raw_times: list[datetime] = funding["funding_time"].to_list()
    # Truncate to the hour so all symbols share the exact 8h grid.
    times = [t.replace(minute=0, second=0, microsecond=0) for t in raw_times]
    rates: list[float] = funding["funding_rate"].to_list()
    marks: list[float | None] = funding["mark_price"].to_list()

    forecasts = _forecast_series(rates, EMA_PERIODS)
    ppy = YEAR_HOURS / INTERVAL_HOURS
    aprs = [f * ppy if not math.isnan(f) else math.nan for f in forecasts]

    return {
        "times": times,
        "index": {t: i for i, t in enumerate(times)},
        "rates": rates,
        "marks": marks,
        "aprs": aprs,
        "n": len(times),
        "spot_times": _close_times(spot["open_time"].to_list()),
        "spot_closes": spot["close"].to_list(),
        "perp_times": _close_times(perp["open_time"].to_list()),
        "perp_closes": perp["close"].to_list(),
    }


def run_portfolio(data: dict[str, dict], k: int, scenario) -> dict:
    pos_notional = TOTAL_CAPITAL / k
    oneway = _oneway_fee(scenario, pos_notional)

    union_times = sorted({t for d in data.values() for t in d["times"]})
    t0, t1 = union_times[0], union_times[-1]
    total_hours = (t1 - t0).total_seconds() / 3600

    positions: dict[str, dict] = {}  # sym -> open position state
    trades: list[dict] = []
    funding_pnl = basis_pnl = fees_paid = 0.0
    position_hours = 0.0

    for t in union_times:
        # ---- 1) accrue funding on held positions, decide exits ----
        for sym in list(positions):
            d = data[sym]
            i = d["index"].get(t)
            if i is None:
                continue  # symbol has no event at this grid point (data gap)
            p = positions[sym]

            mark = d["marks"][i]
            if mark is None or (isinstance(mark, float) and math.isnan(mark)):
                mark = _asof_close(d["perp_times"], d["perp_closes"], t)
            if mark is not None:
                p["funding"] += d["rates"][i] * p["qty"] * mark

            apr = d["aprs"][i]
            is_last = i == d["n"] - 1
            do_exit = (not math.isnan(apr) and apr < EXIT_APR) or is_last
            if not do_exit:
                # keep marking the last priceable point for walk-back exits
                s = _asof_close(d["spot_times"], d["spot_closes"], t)
                q = _asof_close(d["perp_times"], d["perp_closes"], t)
                if s is not None and q is not None:
                    p["last_t"], p["last_s"], p["last_p"] = t, s, q
                continue

            exit_t = t
            exit_s = _asof_close(d["spot_times"], d["spot_closes"], t)
            exit_p = _asof_close(d["perp_times"], d["perp_closes"], t)
            if exit_s is None or exit_p is None:
                exit_t, exit_s, exit_p = p["last_t"], p["last_s"], p["last_p"]

            tb = p["qty"] * ((p["entry_p"] - p["entry_s"]) - (exit_p - exit_s))
            tf = 2 * oneway
            trades.append(
                {
                    "sym": sym,
                    "entry": p["t"],
                    "exit": exit_t,
                    "funding": p["funding"],
                    "basis": tb,
                    "fees": tf,
                    "net": p["funding"] + tb - tf,
                    "hours": (exit_t - p["t"]).total_seconds() / 3600,
                }
            )
            funding_pnl += p["funding"]
            basis_pnl += tb
            fees_paid += tf
            position_hours += (exit_t - p["t"]).total_seconds() / 3600
            del positions[sym]

        # ---- 2) fill free slots with the highest-forecast candidates ----
        free = k - len(positions)
        if free > 0:
            cands = []
            for sym, d in data.items():
                if sym in positions:
                    continue
                i = d["index"].get(t)
                if i is None or i == d["n"] - 1:  # no event / final event: no entry
                    continue
                apr = d["aprs"][i]
                if not math.isnan(apr) and apr > ENTER_APR:
                    cands.append((-apr, sym, i))
            cands.sort()
            for _, sym, i in cands[:free]:
                d = data[sym]
                s = _asof_close(d["spot_times"], d["spot_closes"], t)
                q = _asof_close(d["perp_times"], d["perp_closes"], t)
                if s is None or q is None or s <= 0:
                    continue
                positions[sym] = {
                    "t": t,
                    "entry_s": s,
                    "entry_p": q,
                    "qty": pos_notional / s,
                    "funding": 0.0,
                    "last_t": t,
                    "last_s": s,
                    "last_p": q,
                }

    assert not positions, "positions must all be force-closed at final events"
    net = funding_pnl + basis_pnl - fees_paid
    gap = abs(net - sum(tr["net"] for tr in trades))
    assert gap < 1e-6, f"PnL identity violated (gap={gap})"

    years = total_hours / YEAR_HOURS
    return {
        "k": k,
        "scenario": scenario.name,
        "pos_notional": pos_notional,
        "net": net,
        "funding": funding_pnl,
        "basis": basis_pnl,
        "fees": fees_paid,
        "trades": len(trades),
        "apr_total": net / TOTAL_CAPITAL / years,
        "avg_concurrent": position_hours / total_hours,
        "utilization": position_hours / (k * total_hours),
        "position_hours": position_hours,
        "total_hours": total_hours,
        "period": (t0, t1),
    }


def main() -> None:
    data = {sym: load_symbol(sym) for sym in SYMBOLS}

    # sanity: grid alignment after hour-truncation
    union = sorted({t for d in data.values() for t in d["times"]})
    gaps = {(b - a) for a, b in zip(union, union[1:])}
    print(f"{len(SYMBOLS)} symbols, union events={len(union)}, "
          f"span {union[0]} .. {union[-1]}, unique gaps={sorted(gaps)}")
    for sym, d in data.items():
        missing = len(union) - d["n"]
        if missing:
            print(f"  note: {sym} missing {missing} grid events")

    print()
    hdr = (f"{'K':>3} {'scenario':<10} {'pos_notional':>12} {'net':>10} {'APR@200k':>9} "
           f"{'funding':>10} {'basis':>10} {'fees':>9} {'trades':>6} "
           f"{'avg_conc':>8} {'util':>6} {'vs_baseline':>12}")
    print(hdr)
    for scenario in DEFAULT_SCENARIOS:
        for k in KS:
            r = run_portfolio(data, k, scenario)
            base = BASELINE[scenario.name]
            print(
                f"{r['k']:>3} {r['scenario']:<10} {r['pos_notional']:>12,.0f} "
                f"{r['net']:>10,.2f} {r['apr_total']:>8.2%} "
                f"{r['funding']:>10,.2f} {r['basis']:>10,.2f} {r['fees']:>9,.2f} "
                f"{r['trades']:>6} {r['avg_concurrent']:>8.2f} {r['utilization']:>6.1%} "
                f"{r['net'] - base:>+11,.2f} ({r['net'] / base:>.2f}x)"
            )
        print()


if __name__ == "__main__":
    main()
