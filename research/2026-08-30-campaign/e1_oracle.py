"""E1 — ORACLE FORECAST CEILING.

Monkeypatch farb.backtest.engine._forecast_series with a perfect one-step-ahead
oracle (forecast[i] = rates[i+1], a DELIBERATE, labeled PIT violation; last
event = NaN), run all 20 symbols x both fee scenarios at the baseline
thresholds (enter 12% APR, exit 2% APR), and compare against the unpatched
EMA(6) baseline reproduced in the same process. Also run the oracle (and the
EMA baseline, for reference) at enter 8% to see whether a perfect forecast
makes cheap entries profitable.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl

import farb.backtest.engine as engine
from farb.backtest.report import YEAR_HOURS
from farb.config import DEFAULT_SCENARIOS

DATA_ROOT = Path("/Users/bogdan/funding-arb/data/binance")
NOTIONAL = 10_000.0
EMA_PERIODS = 6
EXIT_APR = 0.02

SYMBOLS = sorted(p.stem for p in (DATA_ROOT / "funding").glob("*.parquet"))
assert len(SYMBOLS) == 20, f"expected 20 symbols, got {len(SYMBOLS)}"

SCENARIOS = {s.name: s for s in DEFAULT_SCENARIOS}  # taker, maker_bnb


def oracle_forecast_series(rates: list[float], ema_periods: int) -> list[float]:
    """PERFECT one-step-ahead forecast: forecast[i] = rates[i+1], i.e. the
    rate that will settle at the NEXT event. DELIBERATE PIT VIOLATION
    (lookahead by exactly one funding interval). Last event has no next rate
    -> NaN. `ema_periods` is ignored."""
    n = len(rates)
    out = [math.nan] * n
    for i in range(n - 1):
        out[i] = rates[i + 1]
    return out


def load_frames():
    frames = {}
    for sym in SYMBOLS:
        funding = pl.read_parquet(DATA_ROOT / "funding" / f"{sym}.parquet").sort("funding_time")
        spot = pl.read_parquet(DATA_ROOT / "klines_spot_1h" / f"{sym}.parquet")
        perp = pl.read_parquet(DATA_ROOT / "klines_perp_1h" / f"{sym}.parquet")
        interval = engine._derive_interval_hours(funding["funding_time"].to_list())
        frames[sym] = (funding, spot, perp, interval)
    return frames


def run_portfolio(frames, scenario, enter_apr, exit_apr):
    per_symbol = {}
    tot_net = tot_funding = tot_basis = tot_fees = 0.0
    tot_trades = 0
    tot_holding = 0.0
    tot_period_hours = 0.0
    for sym, (funding, spot, perp, interval) in frames.items():
        r = engine.simulate(
            funding=funding, spot=spot, perp=perp, interval_hours=interval,
            scenario=scenario, notional=NOTIONAL, ema_periods=EMA_PERIODS,
            enter_apr=enter_apr, exit_apr=exit_apr,
        )
        period_hours = (
            (r.period_end - r.period_start).total_seconds() / 3600
            if r.period_start and r.period_end else 0.0
        )
        per_symbol[sym] = {
            "net": r.net_pnl, "funding": r.funding_pnl, "basis": r.basis_pnl,
            "fees": r.fees_paid, "trades": r.n_trades,
            "tim": r.time_in_market_frac, "maxdd": r.max_drawdown,
            "avg_hold_h": r.avg_holding_hours,
        }
        tot_net += r.net_pnl
        tot_funding += r.funding_pnl
        tot_basis += r.basis_pnl
        tot_fees += r.fees_paid
        tot_trades += r.n_trades
        tot_holding += r.holding_hours_sum
        tot_period_hours += period_hours
    years = tot_period_hours / len(frames) / YEAR_HOURS if frames else float("nan")
    alloc_apr = tot_net / (NOTIONAL * len(frames)) / years
    deployed_years = tot_holding / YEAR_HOURS
    deployed_apr = tot_net / (NOTIONAL * deployed_years) if deployed_years > 0 else math.nan
    return {
        "net": tot_net, "funding": tot_funding, "basis": tot_basis, "fees": tot_fees,
        "trades": tot_trades, "tim": tot_holding / tot_period_hours,
        "alloc_apr": alloc_apr, "deployed_apr": deployed_apr,
        "per_symbol": per_symbol,
    }


def main():
    frames = load_frames()
    intervals = {sym: f[3] for sym, f in frames.items()}
    assert set(intervals.values()) == {8}, f"non-8h intervals: {intervals}"

    results = {}

    # --- baseline (unpatched EMA(6)) ---
    for name, scen in SCENARIOS.items():
        results[f"ema_{name}_enter12"] = run_portfolio(frames, scen, 0.12, EXIT_APR)
        results[f"ema_{name}_enter08"] = run_portfolio(frames, scen, 0.08, EXIT_APR)

    base_taker = results["ema_taker_enter12"]["net"]
    gate_ok = abs(base_taker - 1891.57) < 0.5
    print(f"SANITY GATE: reproduced EMA(6) taker net = {base_taker:.2f} "
          f"(expected ~1891.57) -> {'OK' if gate_ok else 'FAILED'}")
    if not gate_ok:
        print("!!! BASELINE NOT REPRODUCED — oracle numbers below are not comparable !!!")

    # --- oracle (patched) ---
    original = engine._forecast_series
    try:
        engine._forecast_series = oracle_forecast_series
        for name, scen in SCENARIOS.items():
            results[f"oracle_{name}_enter12"] = run_portfolio(frames, scen, 0.12, EXIT_APR)
            results[f"oracle_{name}_enter08"] = run_portfolio(frames, scen, 0.08, EXIT_APR)
    finally:
        engine._forecast_series = original
    assert engine._forecast_series is original, "patch not restored"
    print("patch restored: _forecast_series is the original again\n")

    # --- report ---
    def fmt(key):
        r = results[key]
        return (f"net {r['net']:>10.2f}  funding {r['funding']:>10.2f}  "
                f"basis {r['basis']:>9.2f}  fees {r['fees']:>8.2f}  "
                f"trades {r['trades']:>3d}  TiM {r['tim']:6.2%}  "
                f"APRalloc {r['alloc_apr']:6.2%}  APRdeployed {r['deployed_apr']:7.2%}")

    for name in SCENARIOS:
        for ent in ("enter12", "enter08"):
            e, o = results[f"ema_{name}_{ent}"], results[f"oracle_{name}_{ent}"]
            d = o["net"] - e["net"]
            pct = d / abs(e["net"]) * 100 if e["net"] != 0 else math.nan
            print(f"== {name} / {ent} ==")
            print(f"  EMA(6) : {fmt(f'ema_{name}_{ent}')}")
            print(f"  ORACLE : {fmt(f'oracle_{name}_{ent}')}")
            print(f"  DELTA  : {d:+.2f} USDT ({pct:+.1f}% vs EMA)")
            print()

    # per-symbol table, taker enter12
    print("=== per-symbol taker enter12: EMA vs ORACLE (net USDT / trades) ===")
    print(f"{'symbol':10s} {'ema_net':>9s} {'orc_net':>9s} {'delta':>9s} "
          f"{'ema_tr':>6s} {'orc_tr':>6s} {'ema_TiM':>8s} {'orc_TiM':>8s}")
    e12 = results["ema_taker_enter12"]["per_symbol"]
    o12 = results["oracle_taker_enter12"]["per_symbol"]
    for sym in SYMBOLS:
        a, b = e12[sym], o12[sym]
        print(f"{sym:10s} {a['net']:9.2f} {b['net']:9.2f} {b['net']-a['net']:+9.2f} "
              f"{a['trades']:6d} {b['trades']:6d} {a['tim']:8.2%} {b['tim']:8.2%}")

    print("\n=== per-symbol taker enter08: EMA vs ORACLE (net USDT / trades) ===")
    e08 = results["ema_taker_enter08"]["per_symbol"]
    o08 = results["oracle_taker_enter08"]["per_symbol"]
    for sym in SYMBOLS:
        a, b = e08[sym], o08[sym]
        print(f"{sym:10s} {a['net']:9.2f} {b['net']:9.2f} {b['net']-a['net']:+9.2f} "
              f"{a['trades']:6d} {b['trades']:6d}")

    # losers under each
    for key in ("ema_taker_enter12", "oracle_taker_enter12", "oracle_taker_enter08"):
        losers = {s: v["net"] for s, v in results[key]["per_symbol"].items() if v["net"] < 0}
        print(f"\nlosers under {key}: "
              + (", ".join(f"{s} {n:.2f}" for s, n in sorted(losers.items())) or "none"))

    out = {k: {kk: vv for kk, vv in v.items() if kk != "per_symbol"} for k, v in results.items()}
    out["per_symbol_taker_enter12"] = {"ema": e12, "oracle": o12}
    out["per_symbol_taker_enter08"] = {"ema": e08, "oracle": o08}
    out["per_symbol_maker_enter12"] = {
        "ema": results["ema_maker_bnb_enter12"]["per_symbol"],
        "oracle": results["oracle_maker_bnb_enter12"]["per_symbol"],
    }
    json_path = Path(__file__).parent / "e1_oracle_results.json"
    json_path.write_text(json.dumps(out, indent=2))
    print(f"\nfull results -> {json_path}")


if __name__ == "__main__":
    main()
