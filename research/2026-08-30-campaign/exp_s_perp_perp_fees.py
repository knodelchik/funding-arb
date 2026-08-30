#!/usr/bin/env python
"""EXPERIMENT S: perp-vs-perp HL/Binance funding-spread backtest with real fees
and hourly accrual.

Position: dir=+1 -> short HL perp + long Binance perp, 10k USDT notional per leg
(flat notional, funding-only PnL, no basis/price PnL, no depth modeling).

Signal (PIT, reference-engine conventions from src/farb/backtest/engine.py):
  - 8h grid = Binance settlement times (hour-truncated).
  - spread(T) = [sum of 8 HL hourly funding rates in (T-8h, T]] - bn_rate(T).
  - forecast at event i = EMA(6, alpha=2/7, unadjusted, seeded on first obs)
    of spread[0:i]  (STRICTLY before i; NaN until 6 prior observations).
  - ann = forecast * (8766/8). Enter dir=+1 when ann > thr; exit when
    ann < exit_thr. In 'both' mode also enter dir=-1 when ann < -thr,
    exit when ann > -exit_thr.
  - Entry fill at T. Accrual starts strictly after entry event, includes exit
    event. Force-close at last event.

Accrual while held (dir=+1):
  - HL leg: hourly; short receives +h(t)*N for every HL hourly settlement
    t in (T_entry, T_exit].
  - BN leg: at 8h settlements T in (T_entry, T_exit]; long pays positive
    rate: -bn(T)*N.
  dir=-1 flips both signs.

Fees: per-fill rate on 10k notional; roundtrip = entry+exit on both legs
  = 2*(hl_fill + bn_fill)*N; +1bp slippage per taker fill.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import polars as pl

DATA = "/Users/bogdan/funding-arb/data"
SYMS = ["BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "AVAX", "LINK", "DOT",
        "LTC", "BCH", "TRX", "NEAR", "ATOM", "FIL", "ETC", "APT", "SUI", "ARB"]
N = 10_000.0  # notional per leg, USDT
YEAR_HOURS = 365.25 * 24  # 8766
PPY = YEAR_HOURS / 8.0    # 8h periods per year = 1095.75
START = datetime(2024, 12, 1, tzinfo=timezone.utc)
END = datetime(2026, 8, 29, 16, 0, tzinfo=timezone.utc)  # common last settlement

# Fee schedule (verified 2026-08-30):
# HL official docs (hyperliquid.gitbook.io/hyperliquid-docs/trading/fees):
#   base tier perps maker 0.015%, taker 0.045%.
# Binance USDT-M VIP0 (fee page login-gated; multiple current third-party
#   sources consistent): maker 0.02%, taker 0.05%; BNB pay -10%.
HL_MAKER, HL_TAKER = 0.00015, 0.00045
BN_MAKER, BN_TAKER = 0.00020, 0.00050
SLIP = 0.0001  # 1bp per taker fill

FEE_COMBOS = {
    "MM": (HL_MAKER, BN_MAKER),
    "MT": (HL_MAKER, BN_TAKER + SLIP),
    "TM": (HL_TAKER + SLIP, BN_MAKER),
    "TT": (HL_TAKER + SLIP, BN_TAKER + SLIP),
}


def roundtrip_fee(hl_fill: float, bn_fill: float) -> float:
    return 2.0 * (hl_fill + bn_fill) * N


def load_symbol(sym: str):
    bn = (
        pl.read_parquet(f"{DATA}/binance/funding/{sym}USDT.parquet")
        .with_columns(pl.col("funding_time").dt.truncate("1h"))
        .filter((pl.col("funding_time") >= START) & (pl.col("funding_time") <= END))
        .sort("funding_time")
    )
    hl = (
        pl.read_parquet(f"{DATA}/venues/hyperliquid/funding/{sym}USDT.parquet")
        .filter((pl.col("funding_time") >= START) & (pl.col("funding_time") <= END))
        .sort("funding_time")
    )
    bn_times = bn["funding_time"].to_list()
    bn_rates = bn["funding_rate"].to_list()
    hl_times = hl["funding_time"].to_list()
    hl_rates = hl["funding_rate"].to_list()
    # hourly index for prefix sums
    h0 = hl_times[0]
    idx = [int((t - h0).total_seconds() // 3600) for t in hl_times]
    assert all(b - a == 1 for a, b in zip(idx, idx[1:])), f"{sym}: HL hourly gaps"
    csum = [0.0]  # csum[k] = sum of first k
    for r in hl_rates:
        csum.append(csum[-1] + r)

    def hl_sum(t_from: datetime, t_to: datetime) -> float:
        """Sum of HL hourly rates with settlement time in (t_from, t_to]."""
        a = int((t_from - h0).total_seconds() // 3600)  # exclude t_from
        b = int((t_to - h0).total_seconds() // 3600)    # include t_to
        a = max(a, -1)
        b = min(b, len(hl_rates) - 1)
        if b < 0 or b <= a:
            return 0.0
        return float(csum[b + 1] - csum[a + 1])

    hl8 = [hl_sum(t - timedelta(hours=8), t) for t in bn_times]
    return bn_times, bn_rates, hl8, hl_sum


def forecast_series(vals: list, span: int = 6) -> list:
    """forecast[i] = EMA of vals[0:i] (strictly before i); NaN until span prior."""
    alpha = 2.0 / (span + 1)
    out = [math.nan] * len(vals)
    ema = vals[0] if vals else 0.0
    for i in range(len(vals)):
        if i >= span:
            out[i] = ema
        ema = alpha * vals[i] + (1 - alpha) * ema
    return out


def run_symbol(sym_data, thr_apr: float, exit_apr: float, mode: str,
               accrual: str = "hourly"):
    """Returns list of trades: (dir, t_entry, t_exit, hl_pnl, bn_pnl, hours)."""
    bn_times, bn_rates, hl8, hl_sum = sym_data
    spread = [h - b for h, b in zip(hl8, bn_rates)]
    ann = [f * PPY for f in forecast_series(spread)]
    n = len(bn_times)
    trades = []
    open_dir = 0
    t_entry = None
    bn_acc = 0.0
    hl8_acc = 0.0
    for i in range(n):
        a = ann[i]
        if open_dir != 0:
            # accrue this event (strictly after entry by construction)
            bn_acc += -open_dir * bn_rates[i] * N
            hl8_acc += open_dir * hl8[i] * N
            is_last = i == n - 1
            do_exit = (not math.isnan(a)) and (
                a < exit_apr if open_dir == 1 else a > -exit_apr
            )
            if do_exit or is_last:
                t_exit = bn_times[i]
                if accrual == "hourly":
                    hl_pnl = open_dir * hl_sum(t_entry, t_exit) * N
                else:  # 8h-summed at events
                    hl_pnl = hl8_acc
                hours = (t_exit - t_entry).total_seconds() / 3600
                trades.append((open_dir, t_entry, t_exit, hl_pnl, bn_acc, hours))
                open_dir = 0
        else:
            if not math.isnan(a) and i < n - 1:
                if a > thr_apr:
                    open_dir = 1
                elif mode == "both" and a < -thr_apr:
                    open_dir = -1
                if open_dir != 0:
                    t_entry = bn_times[i]
                    bn_acc = 0.0
                    hl8_acc = 0.0
    return trades


def portfolio(all_data, thr, exit_apr, mode, rt_fee, accrual="hourly"):
    tot_gross = tot_hl = tot_bn = 0.0
    n_trades = 0
    held_hours = 0.0
    per_sym = {}
    n_neg = 0
    gross_neg = 0.0
    for sym in SYMS:
        trades = run_symbol(all_data[sym], thr, exit_apr, mode, accrual)
        g = sum(t[3] + t[4] for t in trades)
        f = rt_fee * len(trades)
        per_sym[sym] = (g - f, len(trades))
        tot_gross += g
        tot_hl += sum(t[3] for t in trades)
        tot_bn += sum(t[4] for t in trades)
        n_trades += len(trades)
        held_hours += sum(t[5] for t in trades)
        neg = [t for t in trades if t[0] == -1]
        n_neg += len(neg)
        gross_neg += sum(t[3] + t[4] for t in neg)
    net = tot_gross - rt_fee * n_trades
    cap_hours = held_hours * 2 * N  # both legs deployed
    dep_apr = net / cap_hours * YEAR_HOURS if cap_hours > 0 else float("nan")
    period_hours = (END - START).total_seconds() / 3600
    util = held_hours / (period_hours * len(SYMS))
    return dict(net=net, gross=tot_gross, hl=tot_hl, bn=tot_bn, trades=n_trades,
                held_hours=held_hours, dep_apr=dep_apr, util=util,
                per_sym=per_sym, n_neg=n_neg, gross_neg=gross_neg)


def main():
    print("Loading data...")
    all_data = {s: load_symbol(s) for s in SYMS}

    # ---- Sanity gate: flat 8bps roundtrip, 8h-summed accrual, enter 12% ----
    print("\n=== SANITY GATE (target +6727, flat 8bps roundtrip, 8h accrual) ===")
    flat_rt = 0.0008 * N
    for mode in ("pos", "both"):
        for ex in (0.0, 0.02):
            r = portfolio(all_data, 0.12, ex, mode, flat_rt, accrual="8h")
            print(f"mode={mode:4s} exit={ex:4.0%}  net={r['net']:9.2f}  "
                  f"trades={r['trades']:4d}  (dev vs 6727: {r['net']/6727-1:+.1%})")

    # ---- Accrual check: hourly vs 8h-summed must coincide on this grid ----
    r_h = portfolio(all_data, 0.12, 0.0, "pos", flat_rt, accrual="hourly")
    r_8 = portfolio(all_data, 0.12, 0.0, "pos", flat_rt, accrual="8h")
    print(f"\nhourly vs 8h-summed accrual (12%/exit0/pos, flat fees): "
          f"{r_h['net']:.2f} vs {r_8['net']:.2f}")

    # ---- Exit rule comparison at 12% MM ----
    print("\n=== EXIT RULE (thr=12%, MM fees, pos-only, hourly accrual) ===")
    for ex in (0.0, 0.02):
        r = portfolio(all_data, 0.12, ex, "pos", roundtrip_fee(*FEE_COMBOS["MM"]))
        print(f"exit={ex:4.0%}  net={r['net']:9.2f}  trades={r['trades']:4d}  "
              f"depAPR={r['dep_apr']:6.2%}")

    # ---- Main fee grid ----
    print("\n=== FEE GRID (exit<0, pos-only, hourly accrual) ===")
    print(f"{'thr':>5s} {'combo':>5s} {'rt_fee$':>8s} {'net':>10s} {'gross':>10s} "
          f"{'trades':>6s} {'depAPR':>8s} {'util':>6s}")
    grid = {}
    for thr in (0.08, 0.12, 0.20):
        for name, (hf, bf) in FEE_COMBOS.items():
            rt = roundtrip_fee(hf, bf)
            r = portfolio(all_data, thr, 0.0, "pos", rt)
            grid[(thr, name)] = r
            print(f"{thr:5.0%} {name:>5s} {rt:8.2f} {r['net']:10.2f} "
                  f"{r['gross']:10.2f} {r['trades']:6d} {r['dep_apr']:8.2%} "
                  f"{r['util']:6.1%}")

    # ---- Directionality: both vs pos-only at each thr (MM) ----
    print("\n=== DIRECTIONALITY (MM fees, exit<0) ===")
    for thr in (0.08, 0.12, 0.20):
        rt = roundtrip_fee(*FEE_COMBOS["MM"])
        rp = portfolio(all_data, thr, 0.0, "pos", rt)
        rb = portfolio(all_data, thr, 0.0, "both", rt)
        rev_net = rb["gross_neg"] - rt * rb["n_neg"]
        print(f"thr={thr:4.0%}  pos-only={rp['net']:9.2f} ({rp['trades']} tr)  "
              f"both={rb['net']:9.2f} ({rb['trades']} tr)  "
              f"reverse-dir contribution={rev_net:+8.2f} ({rb['n_neg']} tr)")

    # ---- Per-symbol at headline 12% MM ----
    print("\n=== PER-SYMBOL (thr=12%, MM, exit<0, pos-only) ===")
    r = grid[(0.12, "MM")]
    for sym, (net, k) in sorted(r["per_sym"].items(), key=lambda x: -x[1][0]):
        print(f"{sym:6s} net={net:9.2f}  trades={k:3d}")
    print(f"leg split: HL={r['hl']:.2f}  BN={r['bn']:.2f}")
    pos_syms = sum(1 for v in r["per_sym"].values() if v[0] > 0)
    print(f"symbols positive: {pos_syms}/20")


if __name__ == "__main__":
    main()
