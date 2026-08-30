"""Experiment R — entry/exit rule redesign for funding-arb (20 core symbols).

(a) Cumulative gate: enter when EMA(6)_rate * H > roundtrip_frac * m, translated
    to the EQUIVALENT enter_apr and run through the stock engine (zero new signal
    code, fully PIT):
        EMA_rate * H > rt * m
    <=> EMA_rate > rt * m / H
    <=> apr = EMA_rate * ppy > rt * m / H * ppy,   ppy = YEAR_HOURS / 8
(b) Exit laziness sweep: exit_apr in {-5%,0%,2%,5%} at baseline 12% entry.
(c) Min-hold {0,24h,72h}: faithful copy of simulate()'s loop with an extra
    holding-time gate on the signal exit; verified to reproduce the stock
    engine exactly at min_hold=0 before use.

Sanity gate: baseline (EMA6, 12%/2%) must reproduce taker net 1891.57 or ABORT.
"""

from __future__ import annotations

import math
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

from farb.backtest.engine import (
    _asof_close,
    _close_times,
    _derive_interval_hours,
    _forecast_series,
    _oneway_fee,
    simulate,
)
from farb.backtest.report import YEAR_HOURS
from farb.config import DEFAULT_SCENARIOS, FeeScenario

DATA = Path("/Users/bogdan/funding-arb/data/binance")
CORE = ("BTC ETH BNB SOL XRP DOGE ADA AVAX LINK DOT LTC BCH TRX NEAR "
        "ATOM FIL ETC APT SUI ARB").split()
SYMBOLS = [s + "USDT" for s in CORE]
NOTIONAL = 10_000.0
EMA = 6
BASE_ENTER, BASE_EXIT = 0.12, 0.02
BASELINE_NET = {"taker": 1891.57, "maker_bnb": 2446.57}
SCEN = {s.name: s for s in DEFAULT_SCENARIOS}


def load_frames():
    frames = {}
    for sym in SYMBOLS:
        fpath = DATA / "funding" / f"{sym}.parquet"
        spath = DATA / "klines_spot_1h" / f"{sym}.parquet"
        ppath = DATA / "klines_perp_1h" / f"{sym}.parquet"
        for p in (fpath, spath, ppath):
            if not p.exists():
                raise SystemExit(f"ABORT: missing data file {p}")
        funding = pl.read_parquet(fpath).sort("funding_time")
        ih = _derive_interval_hours(funding["funding_time"].to_list())
        if ih != 8:
            raise SystemExit(f"ABORT: {sym} derived interval {ih}h != 8h")
        frames[sym] = (funding, pl.read_parquet(spath), pl.read_parquet(ppath), ih)
    return frames


def agg_cell(frames, scenario: FeeScenario, enter_apr: float, exit_apr: float):
    """Portfolio aggregates for one (scenario, enter, exit) cell via stock simulate."""
    a = dict(net=0.0, funding=0.0, basis=0.0, fees=0.0, trades=0, hold_h=0.0)
    starts, ends = [], []
    for sym, (f, s, p, ih) in frames.items():
        r = simulate(f, s, p, ih, scenario, NOTIONAL, EMA, enter_apr, exit_apr)
        a["net"] += r.net_pnl
        a["funding"] += r.funding_pnl
        a["basis"] += r.basis_pnl
        a["fees"] += r.fees_paid
        a["trades"] += r.n_trades
        a["hold_h"] += r.holding_hours_sum
        if r.period_start:
            starts.append(r.period_start)
        if r.period_end:
            ends.append(r.period_end)
    a["dep_apr"] = (a["net"] / (NOTIONAL * a["hold_h"] / YEAR_HOURS)
                    if a["hold_h"] > 0 else math.nan)
    hours = (max(ends) - min(starts)).total_seconds() / 3600 if starts else 0.0
    a["tim"] = a["hold_h"] / (len(frames) * hours) if hours else 0.0
    return a


def simulate_minhold(funding, spot, perp, interval_hours, scenario, notional,
                     ema_periods, enter_apr, exit_apr, min_hold_hours):
    """Faithful copy of farb.backtest.engine.simulate()'s event loop with ONE
    change: the signal exit is ignored until the position has been held for
    min_hold_hours (force-close at the last event is NOT gated). Equity-curve
    storage is dropped, but the last_priced_* walk-back updates that live in
    that block are kept (they feed unpriced-exit closure). Returns aggregates.
    """
    funding = funding.sort("funding_time")
    spot = spot.sort("open_time")
    perp = perp.sort("open_time")

    times = funding["funding_time"].to_list()
    rates = funding["funding_rate"].to_list()
    mark_prices = funding["mark_price"].to_list()
    n = len(times)
    zero = dict(net=0.0, funding=0.0, basis=0.0, fees=0.0, trades=0, hold_h=0.0)
    if n == 0:
        return zero

    forecasts = _forecast_series(rates, ema_periods)
    periods_per_year = YEAR_HOURS / interval_hours
    aprs = [f * periods_per_year if not math.isnan(f) else math.nan for f in forecasts]

    spot_times = _close_times(spot["open_time"].to_list())
    spot_closes = spot["close"].to_list()
    perp_times = _close_times(perp["open_time"].to_list())
    perp_closes = perp["close"].to_list()

    oneway_fee = _oneway_fee(scenario, notional)

    out = dict(zero)
    position_open = False
    entry_time_v = None
    entry_spot_px = entry_perp_px = 0.0
    qty_v = 0.0
    pos_funding = 0.0
    last_priced_time = None
    last_priced_spot = None
    last_priced_perp = None

    for i in range(n):
        t = times[i]
        apr = aprs[i]

        if position_open:
            mark_px = mark_prices[i]
            mark_missing = mark_px is None or math.isnan(mark_px)
            if not mark_missing:
                accrual_notional = qty_v * mark_px
            else:
                now_perp = _asof_close(perp_times, perp_closes, t)
                accrual_notional = qty_v * now_perp if now_perp is not None else 0.0
            pos_funding += rates[i] * accrual_notional

            is_last = i == n - 1
            held_h = (t - entry_time_v).total_seconds() / 3600
            do_exit = (not math.isnan(apr) and apr < exit_apr
                       and held_h >= min_hold_hours)
            if do_exit or is_last:
                exit_time = t
                exit_spot_px = _asof_close(spot_times, spot_closes, t)
                exit_perp_px = _asof_close(perp_times, perp_closes, t)
                unpriced_exit = exit_spot_px is None or exit_perp_px is None
                if unpriced_exit:
                    exit_time = last_priced_time
                    exit_spot_px = last_priced_spot
                    exit_perp_px = last_priced_perp
                if (exit_time is not None and exit_spot_px is not None
                        and exit_perp_px is not None):
                    basis_entry = entry_perp_px - entry_spot_px
                    basis_exit = exit_perp_px - exit_spot_px
                    trade_basis = qty_v * (basis_entry - basis_exit)
                    trade_fees = 2 * oneway_fee
                    trade_net = pos_funding + trade_basis - trade_fees
                    out["net"] += trade_net
                    out["funding"] += pos_funding
                    out["basis"] += trade_basis
                    out["fees"] += trade_fees
                    out["trades"] += 1
                    out["hold_h"] += (exit_time - entry_time_v).total_seconds() / 3600
                    position_open = False
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

        # Mirrors the equity-curve block's price tracking (curve itself dropped)
        if position_open:
            now_spot = _asof_close(spot_times, spot_closes, t)
            now_perp = _asof_close(perp_times, perp_closes, t)
            if now_spot is not None and now_perp is not None:
                last_priced_time = t
                last_priced_spot = now_spot
                last_priced_perp = now_perp

    return out


def agg_cell_minhold(frames, scenario, enter_apr, exit_apr, min_hold_hours):
    a = dict(net=0.0, funding=0.0, basis=0.0, fees=0.0, trades=0, hold_h=0.0)
    for sym, (f, s, p, ih) in frames.items():
        r = simulate_minhold(f, s, p, ih, scenario, NOTIONAL, EMA,
                             enter_apr, exit_apr, min_hold_hours)
        for k in ("net", "funding", "basis", "fees", "hold_h"):
            a[k] += r[k]
        a["trades"] += r["trades"]
    a["dep_apr"] = (a["net"] / (NOTIONAL * a["hold_h"] / YEAR_HOURS)
                    if a["hold_h"] > 0 else math.nan)
    return a


def fmt(a):
    dep = "n/a " if math.isnan(a["dep_apr"]) else f"{a['dep_apr']:6.1%}"
    return (f"net {a['net']:9.2f}  trades {a['trades']:3d}  fees {a['fees']:8.2f}  "
            f"funding {a['funding']:8.2f}  basis {a['basis']:8.2f}  depAPR {dep}")


def main():
    frames = load_frames()
    print(f"loaded {len(frames)} symbols")

    # ---------- sanity gate ----------
    base = {}
    for name, sc in SCEN.items():
        base[name] = agg_cell(frames, sc, BASE_ENTER, BASE_EXIT)
        print(f"BASELINE {name}: {fmt(base[name])}")
    if abs(base["taker"]["net"] - 1891.57) > 0.01:
        raise SystemExit(f"ABORT ABORT ABORT: baseline taker net "
                         f"{base['taker']['net']:.2f} != 1891.57 — engine/data drift, "
                         f"all downstream numbers would be meaningless.")
    if abs(base["maker_bnb"]["net"] - 2446.57) > 0.01:
        raise SystemExit(f"ABORT: baseline maker_bnb net "
                         f"{base['maker_bnb']['net']:.2f} != 2446.57")
    print("sanity gate PASSED\n")

    # ---------- (a) cumulative gate ----------
    print("=" * 100)
    print("(a) CUMULATIVE GATE: enter when EMA_rate*H > rt_frac*m  "
          "<=> enter_apr = rt_frac*m/H * (YEAR_HOURS/8)")
    a_results = {}  # (H, m, exit) -> {scen: agg}
    Hs = [3, 9, 21, 42]
    ms = [1.5, 2.0, 3.0]
    exits = [0.0, 0.02]
    ppy = YEAR_HOURS / 8
    for name, sc in SCEN.items():
        rt = sc.roundtrip_frac
        print(f"\n--- scenario {name} (roundtrip {rt*1e4:.0f} bps) --- "
              f"[baseline net {base[name]['net']:.2f}]")
        for H in Hs:
            for m in ms:
                enter = rt * m / H * ppy
                for ex in exits:
                    if enter <= ex:
                        print(f"H={H:2d} m={m:3.1f} exit={ex:5.1%}  SKIP enter<=exit")
                        continue
                    a = agg_cell(frames, sc, enter, ex)
                    a_results.setdefault((H, m, ex), {})[name] = (a, enter)
                    print(f"H={H:2d} m={m:3.1f} exit={ex:5.1%} "
                          f"(enter_apr={enter:7.1%})  {fmt(a)}")

    # ---------- (b) exit laziness ----------
    print("\n" + "=" * 100)
    print("(b) EXIT LAZINESS at baseline 12% entry")
    b_results = {}
    for name, sc in SCEN.items():
        print(f"\n--- scenario {name} --- [baseline net {base[name]['net']:.2f}]")
        for ex in [-0.05, 0.0, 0.02, 0.05]:
            a = agg_cell(frames, sc, BASE_ENTER, ex)
            b_results.setdefault(ex, {})[name] = a
            tag = " <= BASELINE" if ex == BASE_EXIT else ""
            print(f"exit={ex:6.1%}  {fmt(a)}{tag}")

    # ---------- (c) min-hold ----------
    print("\n" + "=" * 100)
    print("(c) MIN-HOLD (faithful loop copy) — verification at min_hold=0 first")
    worst = 0.0
    for name, sc in SCEN.items():
        for sym, (f, s, p, ih) in frames.items():
            r0 = simulate(f, s, p, ih, sc, NOTIONAL, EMA, BASE_ENTER, BASE_EXIT)
            r1 = simulate_minhold(f, s, p, ih, sc, NOTIONAL, EMA,
                                  BASE_ENTER, BASE_EXIT, 0.0)
            for k, v in (("net", r0.net_pnl), ("funding", r0.funding_pnl),
                         ("basis", r0.basis_pnl), ("fees", r0.fees_paid),
                         ("hold_h", r0.holding_hours_sum)):
                worst = max(worst, abs(r1[k] - v))
            if r1["trades"] != r0.n_trades:
                raise SystemExit(f"ABORT: min-hold copy diverges on {sym}/{name}: "
                                 f"trades {r1['trades']} != {r0.n_trades}")
    if worst > 1e-9:
        raise SystemExit(f"ABORT: min-hold copy diverges from stock engine, "
                         f"worst abs diff {worst}")
    print(f"min_hold=0 reproduces stock engine exactly "
          f"(worst abs diff {worst:.2e}) — VERIFIED\n")

    c_results = {}
    for name, sc in SCEN.items():
        print(f"--- scenario {name} (entry 12%, exit 2%) --- "
              f"[baseline net {base[name]['net']:.2f}]")
        for mh in [0, 24, 72]:
            a = agg_cell_minhold(frames, sc, BASE_ENTER, BASE_EXIT, mh)
            c_results.setdefault(mh, {})[name] = a
            print(f"min_hold={mh:3d}h  {fmt(a)}")
        print()

    # ---------- summary: top-5 per scenario, both-scenario winners ----------
    print("=" * 100)
    print("SUMMARY")
    all_cells = []  # (label, {scen: net_dict})
    for (H, m, ex), per in a_results.items():
        all_cells.append((f"(a) H={H} m={m} exit={ex:.0%}",
                          {n: per[n][0] for n in per}))
    for ex, per in b_results.items():
        all_cells.append((f"(b) enter=12% exit={ex:.0%}", per))
    for mh, per in c_results.items():
        all_cells.append((f"(c) 12%/2% min_hold={mh}h", per))

    for name in SCEN:
        ranked = sorted(all_cells, key=lambda c: -c[1][name]["net"])[:5]
        print(f"\nTOP-5 by net — {name} (baseline {base[name]['net']:.2f}):")
        for label, per in ranked:
            a = per[name]
            gain = a["net"] / base[name]["net"] - 1
            print(f"  {label:34s} net {a['net']:9.2f} ({gain:+6.1%} vs base)  "
                  f"trades {a['trades']:3d}  depAPR "
                  f"{a['dep_apr']:.1%}" if not math.isnan(a["dep_apr"])
                  else f"  {label:34s} net {a['net']:9.2f}")

    print("\nConfigs beating baseline on BOTH scenarios:")
    both = []
    for label, per in all_cells:
        if all(per[n]["net"] > base[n]["net"] for n in SCEN):
            margins = {n: per[n]["net"] / base[n]["net"] - 1 for n in SCEN}
            both.append((label, per, margins))
    both.sort(key=lambda x: -min(x[2].values()))
    if not both:
        print("  NONE")
    for label, per, margins in both:
        flag = "  [<10% margin — NOISE-LEVEL]" if min(margins.values()) < 0.10 else ""
        print(f"  {label:34s} taker {per['taker']['net']:9.2f} "
              f"({margins['taker']:+6.1%})  maker {per['maker_bnb']['net']:9.2f} "
              f"({margins['maker_bnb']:+6.1%}){flag}")

    # ---------- bonus: min-hold on top of the best both-scenario (a) rule ----------
    a_both = [(lbl, per, mg) for lbl, per, mg in both if lbl.startswith("(a)")]
    if a_both:
        label, per, _ = a_both[0]
        # recover (H, m, ex)
        parts = label.replace("(a) ", "").split()
        H = int(parts[0].split("=")[1])
        m = float(parts[1].split("=")[1])
        ex = float(parts[2].split("=")[1].rstrip("%")) / 100
        print(f"\nBONUS: min-hold on top of best (a) rule {label}:")
        for name, sc in SCEN.items():
            enter = sc.roundtrip_frac * m / H * ppy
            for mh in [0, 24, 72]:
                a = agg_cell_minhold(frames, sc, enter, ex, mh)
                print(f"  {name:9s} min_hold={mh:3d}h  {fmt(a)}")


if __name__ == "__main__":
    main()
