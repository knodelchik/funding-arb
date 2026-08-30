#!/usr/bin/env python
"""EXPERIMENT C — HL-short + Binance-SPOT carry (hybrid cash-and-carry).

Long Binance SPOT (real coins as the hedge), short the Hyperliquid perp,
collect HL's hourly funding.  20 core symbols, 10k notional per symbol.

Signal (8h grid, 00/08/16 UTC): PIT EMA(6) of the trailing 8h-summed HL
hourly funding, using only 8h buckets that ended STRICTLY BEFORE the
decision time T (engine convention: forecast[i] = EMA of rates[0:i]).
Annualize with PPY = 8766/8.  Enter apr > 12%, exit apr < 0%.

Accrual: an entry at event T starts collecting with the bucket ending at
the NEXT event; the exit event's bucket IS collected (accrue-then-decide,
mirrors farb engine / Experiment X).  Buckets are complete sums of the 8
hourly HL funding prints, so this equals hourly accrual.

Prices at decision time T: close of the 1h candle that CLOSES at T
(open_time = T - 1h), for both Binance spot and HL perp klines.

Fees per roundtrip on the 10k notional:
  spot leg  2 fills x spot_bps   (7.5 bps BNB variant / 0 bps promo variant)
  HL leg    2 fills x 1.5 bps maker (verified tier-0 maker 0.015%, docs 2026-08-30)

Basis leg per trade: qty = 10k / spot_entry;
  basis_pnl = qty * ((hl_entry - spot_entry) - (hl_exit - spot_exit)).
HL 1h klines only reach back to ~2026-02-02 (API caps at most recent ~5000
candles), so basis is computable only for trades whose entry AND exit fall
inside coverage; reported separately with coverage stats.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

sys.path.insert(0, "/Users/bogdan/funding-arb/src")
from farb.backtest.engine import _forecast_series  # noqa: E402
from farb.backtest.report import YEAR_HOURS  # noqa: E402

DATA = Path("/Users/bogdan/funding-arb/data")
HL = DATA / "venues" / "hyperliquid"

SYMBOLS = [
    s + "USDT"
    for s in (
        "BTC ETH BNB SOL XRP DOGE ADA AVAX LINK DOT "
        "LTC BCH TRX NEAR ATOM FIL ETC APT SUI ARB"
    ).split()
]

NOTIONAL = 10_000.0
INTERVAL_H = 8
PPY = YEAR_HOURS / INTERVAL_H  # 1095.75
EMA_PERIODS = 6
ENTER_APR = 0.12
EXIT_APR = 0.0

HL_MAKER_BPS = 1.5  # verified 2026-08-30: hyperliquid docs, tier 0 maker 0.015%
SPOT_BPS_VARIANTS = {"spot7.5bnb": 7.5, "spot0promo": 0.0}

BASELINE_MAKER_BNB = 2446.57  # Binance-only cash-and-carry, same rule family
PERP_PERP_NET = 6727.0  # naive HL-short/Binance-perp-long funding-only, 12% enter

EIGHT_H = timedelta(hours=8)
ONE_H = timedelta(hours=1)


def load_hl_8h_events(sym: str) -> pl.DataFrame:
    """8h-summed HL hourly funding, bucket = ceil to the 00/08/16 UTC grid
    (grid-exact maps to itself); complete 8-hour buckets only."""
    hl = (
        pl.read_parquet(HL / "funding" / f"{sym}.parquet")
        .with_columns(pl.col("funding_time").dt.truncate("1h"))
        .unique(subset="funding_time")
        .sort("funding_time")
    )
    hl = hl.with_columns(
        (
            (pl.col("funding_time") - pl.duration(milliseconds=1)).dt.truncate("8h")
            + pl.duration(hours=8)
        ).alias("bucket")
    )
    return (
        hl.group_by("bucket")
        .agg(pl.col("funding_rate").sum().alias("rate_8h"), pl.len().alias("n_h"))
        .filter(pl.col("n_h") == 8)
        .sort("bucket")
        .select(pl.col("bucket").alias("event_time"), "rate_8h")
    )


def close_price_map(path: Path) -> dict[datetime, float]:
    """Map decision-time T -> close of the candle closing at T (open_time = T-1h)."""
    df = pl.read_parquet(path).sort("open_time")
    return {
        (t + ONE_H): c
        for t, c in zip(df["open_time"].to_list(), df["close"].to_list())
    }


def simulate_symbol(sym: str) -> dict:
    ev = load_hl_8h_events(sym)
    times: list[datetime] = ev["event_time"].to_list()
    rates: list[float] = ev["rate_8h"].to_list()
    n = len(times)
    forecasts = _forecast_series(rates, EMA_PERIODS)
    aprs = [f * PPY if not math.isnan(f) else math.nan for f in forecasts]

    spot_px = close_price_map(DATA / "binance" / "klines_spot_1h" / f"{sym}.parquet")
    hl_px = close_price_map(HL / "klines_1h" / f"{sym}.parquet")

    pos_open = False
    entry_time: datetime | None = None
    spot_entry = hl_entry = qty = 0.0
    pos_funding = 0.0

    funding_gross = 0.0
    hold_hours = 0.0
    trades: list[dict] = []

    for i in range(n):
        t = times[i]
        apr = aprs[i]
        if pos_open:
            pos_funding += rates[i] * NOTIONAL  # short HL: +rate accrues to us
            is_last = i == n - 1
            if (not math.isnan(apr) and apr < EXIT_APR) or is_last:
                spot_exit = spot_px.get(t)
                hl_exit = hl_px.get(t)
                basis = None
                if hl_entry is not None and hl_exit is not None and spot_exit is not None:
                    basis = qty * ((hl_entry - spot_entry) - (hl_exit - spot_exit))
                spot_leg = qty * (spot_exit - spot_entry) if spot_exit is not None else None
                funding_gross += pos_funding
                hold_hours += (t - entry_time).total_seconds() / 3600
                trades.append(
                    {
                        "entry": entry_time,
                        "exit": t,
                        "hours": (t - entry_time).total_seconds() / 3600,
                        "funding": pos_funding,
                        "basis": basis,
                        "spot_leg": spot_leg,
                        "forced": is_last and not (not math.isnan(apr) and apr < EXIT_APR),
                    }
                )
                pos_open = False
        else:
            if not math.isnan(apr) and apr > ENTER_APR:
                se = spot_px.get(t)
                if se is None:
                    continue  # cannot price the hedge -> skip entry
                pos_open = True
                entry_time = t
                spot_entry = se
                hl_entry = hl_px.get(t)
                qty = NOTIONAL / spot_entry
                pos_funding = 0.0

    span_h = (times[-1] - times[0]).total_seconds() / 3600 if n >= 2 else 0.0
    return {
        "sym": sym,
        "n_events": n,
        "span_hours": span_h,
        "funding_gross": funding_gross,
        "hold_hours": hold_hours,
        "trades": trades,
    }


def main() -> None:
    results = [simulate_symbol(s) for s in SYMBOLS]

    hl_fee_rt = 2 * HL_MAKER_BPS / 1e4 * NOTIONAL  # 3.0 USDT per roundtrip

    print("=" * 100)
    print(
        f"EXPERIMENT C — long Binance spot / short HL perp | EMA({EMA_PERIODS}) of "
        f"8h-summed HL funding | enter >{ENTER_APR:.0%} APR, exit <{EXIT_APR:.0%} | "
        f"10k/symbol, 20 symbols"
    )
    print("=" * 100)

    total_trades = sum(len(r["trades"]) for r in results)
    total_funding = sum(r["funding_gross"] for r in results)
    total_hold = sum(r["hold_hours"] for r in results)
    span_years = max(r["span_hours"] for r in results) / YEAR_HOURS

    cov_trades = [t for r in results for t in r["trades"] if t["basis"] is not None]
    all_trades = [t for r in results for t in r["trades"]]
    basis_total = sum(t["basis"] for t in cov_trades)
    cov_hold = sum(t["hours"] for t in cov_trades)
    cov_funding = sum(t["funding"] for t in cov_trades)
    wins_funding = sum(1 for t in all_trades if t["funding"] > 0)

    print(f"\nspan: {span_years:.2f} years | events per symbol ~{results[0]['n_events']}")
    print(
        f"trades: {total_trades} (funding-positive {wins_funding}) | "
        f"portfolio hold {total_hold:,.0f} pos-hours "
        f"({total_hold / (20 * span_years * YEAR_HOURS):.1%} time-in-market)"
    )
    print(f"funding gross (full span): {total_funding:,.2f} USDT")

    for name, spot_bps in SPOT_BPS_VARIANTS.items():
        spot_fee_rt = 2 * spot_bps / 1e4 * NOTIONAL
        fee_rt = spot_fee_rt + hl_fee_rt
        fees = total_trades * fee_rt
        net = total_funding - fees
        deployed_years = total_hold / YEAR_HOURS
        apr_notional = net / (NOTIONAL * deployed_years)
        apr_capital = net / (1.5 * NOTIONAL * deployed_years)
        alloc_apr_notional = net / (20 * NOTIONAL * span_years)
        alloc_apr_capital = net / (20 * 1.5 * NOTIONAL * span_years)
        print(f"\n--- fee variant {name}: spot {spot_bps}bps + HL maker {HL_MAKER_BPS}bps "
              f"= {fee_rt:.1f} USDT/roundtrip ---")
        print(
            f"  FUNDING-ONLY NET {net:,.2f} USDT  (gross {total_funding:,.2f} "
            f"- fees {fees:,.2f} [spot {total_trades * spot_fee_rt:,.0f} + "
            f"HL {total_trades * hl_fee_rt:,.0f}])"
        )
        print(
            f"  deployed APR {apr_notional:6.2%} on 10k notional | "
            f"{apr_capital:6.2%} on 15k capital (10k spot + 5k HL margin @50%)"
        )
        print(
            f"  allocated APR {alloc_apr_notional:6.2%} on 200k notional | "
            f"{alloc_apr_capital:6.2%} on 300k capital"
        )
        print(
            f"  vs Binance-only maker_bnb {BASELINE_MAKER_BNB:,.2f} "
            f"({net / BASELINE_MAKER_BNB:.2f}x) | vs perp-perp naive {PERP_PERP_NET:,.0f} "
            f"({net / PERP_PERP_NET:.2f}x)"
        )

    # ---------------- basis leg (partial coverage) ----------------
    print("\n" + "=" * 100)
    print("BASIS LEG (HL perp klines reach back only to ~2026-02-02 — API caps at ~5000 candles)")
    print("=" * 100)
    print(
        f"covered trades: {len(cov_trades)}/{total_trades} "
        f"({cov_hold:,.0f} of {total_hold:,.0f} pos-hours, {cov_hold / total_hold:.1%})"
    )
    if cov_trades:
        b = sorted(t["basis"] for t in cov_trades)
        m = len(b)
        print(
            f"basis PnL on covered trades: total {basis_total:+,.2f} USDT | "
            f"mean {basis_total / m:+.2f} | median {b[m // 2]:+.2f} | "
            f"min {b[0]:+.2f} | max {b[-1]:+.2f}"
        )
        print(
            f"funding on those same trades: {cov_funding:,.2f} USDT -> basis adds "
            f"{basis_total / cov_funding:+.1%} of funding on the covered subset"
        )
        neg = sum(1 for x in b if x < 0)
        print(f"basis-negative trades: {neg}/{m}")
        for name, spot_bps in SPOT_BPS_VARIANTS.items():
            fee_rt = (2 * spot_bps / 1e4 + 2 * HL_MAKER_BPS / 1e4) * NOTIONAL
            cov_net = cov_funding + basis_total - len(cov_trades) * fee_rt
            cy = cov_hold / YEAR_HOURS
            print(
                f"  covered-subset ALL-IN net ({name}): {cov_net:,.2f} USDT | "
                f"deployed APR {cov_net / (NOTIONAL * cy):6.2%} on notional / "
                f"{cov_net / (1.5 * NOTIONAL * cy):6.2%} on 15k-capital"
            )

    # ---------------- per-symbol table ----------------
    print("\n" + "=" * 100)
    print("PER-SYMBOL (funding-only net at spot7.5bnb fees; basis over covered trades)")
    print("=" * 100)
    fee_rt_main = (2 * 7.5 / 1e4 + 2 * HL_MAKER_BPS / 1e4) * NOTIONAL
    print(
        f"{'symbol':10s} {'trades':>6s} {'fund_gross':>10s} {'fees':>7s} {'net':>9s} "
        f"{'TiM':>6s} {'depAPR':>7s} {'basisCov':>8s} {'basisPnL':>9s}"
    )
    for r in sorted(results, key=lambda x: -(x["funding_gross"] - len(x["trades"]) * fee_rt_main)):
        ntr = len(r["trades"])
        fees = ntr * fee_rt_main
        net = r["funding_gross"] - fees
        tim = r["hold_hours"] / r["span_hours"] if r["span_hours"] else math.nan
        dep = net / (NOTIONAL * r["hold_hours"] / YEAR_HOURS) if r["hold_hours"] else math.nan
        ct = [t for t in r["trades"] if t["basis"] is not None]
        bsum = sum(t["basis"] for t in ct)
        print(
            f"{r['sym']:10s} {ntr:6d} {r['funding_gross']:10.2f} {fees:7.1f} {net:9.2f} "
            f"{tim:6.1%} {dep:7.1%} {len(ct):3d}/{ntr:<3d} {bsum:+9.2f}"
        )

    # largest basis hits
    if cov_trades:
        print("\nworst 8 covered trades by basis PnL:")
        for t in sorted(cov_trades, key=lambda x: x["basis"])[:8]:
            sym = next(r["sym"] for r in results if t in r["trades"])
            print(
                f"  {sym:10s} {t['entry']} -> {t['exit']} ({t['hours']:5.0f}h) "
                f"funding {t['funding']:+8.2f} basis {t['basis']:+8.2f}"
                f"{'  [forced-close]' if t['forced'] else ''}"
            )

    forced = sum(1 for t in all_trades if t["forced"])
    print(f"\nforce-closed at end of data: {forced} positions (their PnL included above)")


if __name__ == "__main__":
    main()
