#!/usr/bin/env python
"""Experiment WA — rotating allocator on the wide universe (354 symbols).

200k USDT total capital, K rotating slots. At each union funding event:
- held positions accrue funding on per-event mark notional (rate*qty*mark,
  fallback asof perp close), exit when that symbol's PIT EMA(6) APR < 0%,
  force-close at the symbol's last settled event;
- free slots filled from symbols with an event at that time whose forecast
  APR > enter threshold, ranked by forecast APR desc, subject to a liquidity
  cap: notional = min(200k/K, 0.5% * trailing-30d PIT median hourly quote
  volume, min(spot, perp) leg); skip when the liquidity cap < 2k.

Semantics deliberately mirror farb.backtest.engine.simulate: PIT EMA(6) over
settled rates (forecast at event i uses rates[0:i] only), prices asof candle
CLOSE with 24h staleness guard, fees on entry notional both legs entry+exit,
walk-back exit pricing when the decision event can't be priced. A symbol is
actionable only at its own funding events (engine parity); a symbol that
exits at time T cannot re-enter at the same T.

SANITY GATE: reproduces the core-20 fixed-bank baseline (taker 2385.17 /
maker_bnb 2880.17 at 12%/0%) with the stock engine before anything else,
hard-aborting on mismatch.
"""
from __future__ import annotations

import bisect
import json
import math
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

ROOT = Path("/Users/bogdan/funding-arb")
DATA = ROOT / "data" / "binance"
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-bogdan/22c3fe7f-2325-4918-9a61-1adbc0bd23d3/scratchpad"
)

sys.path.insert(0, str(ROOT / "src"))
from farb.backtest.engine import (  # noqa: E402
    _asof_close,
    _close_times,
    _derive_interval_hours,
    _forecast_series,
    simulate,
)
from farb.backtest.report import YEAR_HOURS  # noqa: E402
from farb.config import DEFAULT_SCENARIOS, FeeScenario  # noqa: E402

CORE20 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "BCHUSDT", "TRXUSDT", "NEARUSDT", "ATOMUSDT",
    "FILUSDT", "ETCUSDT", "APTUSDT", "SUIUSDT", "ARBUSDT",
]

TOTAL_CAPITAL = 200_000.0
EMA_PERIODS = 6
ENTER_APR = 0.12
EXIT_APR = 0.0
MIN_CAP = 2_000.0
LIQ_FRAC = 0.005  # 0.5% of median hourly quote volume
LIQ_WINDOW = "30d"
LIQ_MIN_SAMPLES = 24  # need >= 24 hourly bars before liquidity is trusted
STALENESS_H = 24.0

# The verified core-20 baseline (2385.17/2880.17) covers 2024-12-01..2026-08-29;
# the raw files have since been extended ~10h into 2026-08-30. Clamp funding
# events to the baseline period so the gate and the experiment share one span.
CUTOFF = datetime(2026, 8, 30, 0, 0, tzinfo=UTC)

SCEN = {s.name: s for s in DEFAULT_SCENARIOS}


def oneway_fee(scen: FeeScenario, notional: float) -> float:
    per_leg_bps = scen.spot_bps + scen.perp_bps + 2 * scen.slippage_bps
    return notional * per_leg_bps / 10_000


# ---------------------------------------------------------------- sanity gate
def sanity_gate() -> None:
    expected = {"taker": 2385.17, "maker_bnb": 2880.17}
    for scen in DEFAULT_SCENARIOS:
        tot, ntr = 0.0, 0
        for sym in CORE20:
            funding = (
                pl.read_parquet(DATA / "funding" / f"{sym}.parquet")
                .filter(pl.col("funding_time") < CUTOFF)
                .sort("funding_time")
            )
            spot = pl.read_parquet(DATA / "klines_spot_1h" / f"{sym}.parquet")
            perp = pl.read_parquet(DATA / "klines_perp_1h" / f"{sym}.parquet")
            ih = _derive_interval_hours(funding["funding_time"].to_list())
            r = simulate(funding, spot, perp, ih, scen, 10_000, EMA_PERIODS, 0.12, 0.0)
            tot += r.net_pnl
            ntr += r.n_trades
        print(f"[gate] {scen.name}: net {tot:.2f} USDT, {ntr} trades "
              f"(expected {expected[scen.name]:.2f})")
        if abs(tot - expected[scen.name]) > 0.01:
            print("SANITY GATE FAILED — aborting.", file=sys.stderr)
            sys.exit(2)
    print("[gate] PASSED: core-20 baseline reproduced with the stock engine\n")


# ---------------------------------------------------------------- data loading
def load_symbol(sym: str) -> dict | None:
    fp = DATA / "funding" / f"{sym}.parquet"
    sp = DATA / "klines_spot_1h" / f"{sym}.parquet"
    pp = DATA / "klines_perp_1h" / f"{sym}.parquet"
    if not (fp.exists() and sp.exists() and pp.exists()):
        return None
    funding = pl.read_parquet(fp).filter(pl.col("funding_time") < CUTOFF).sort("funding_time")
    times = funding["funding_time"].to_list()
    if len(times) < 2:
        return None
    ih = _derive_interval_hours(times)
    rates = funding["funding_rate"].to_list()
    marks = funding["mark_price"].to_list()
    fc = _forecast_series(rates, EMA_PERIODS)
    ppy = YEAR_HOURS / ih
    aprs = [f * ppy if not math.isnan(f) else math.nan for f in fc]
    ev_times = [t.replace(minute=0, second=0, microsecond=0) for t in times]

    spot = pl.read_parquet(sp).sort("open_time")
    perp = pl.read_parquet(pp).sort("open_time")
    if spot.height == 0 or perp.height == 0:
        return None
    spot_times = _close_times(spot["open_time"].to_list())
    spot_closes = spot["close"].to_list()
    perp_times = _close_times(perp["open_time"].to_list())
    perp_closes = perp["close"].to_list()

    liq = {}
    for name, df in (("spot", spot), ("perp", perp)):
        qv = df.with_columns((pl.col("volume") * pl.col("close")).alias("qv"))
        try:
            med = qv.select(
                pl.col("qv").rolling_median_by(
                    "open_time", window_size=LIQ_WINDOW, min_samples=LIQ_MIN_SAMPLES
                )
            ).to_series().to_list()
        except Exception:
            med = qv.select(
                pl.col("qv").rolling_median(window_size=720, min_samples=LIQ_MIN_SAMPLES)
            ).to_series().to_list()
        ct = [t + timedelta(hours=1) for t in df["open_time"].to_list()]
        liq[name] = (ct, med)

    return {
        "sym": sym,
        "interval_hours": ih,
        "ev_times": ev_times,
        "rates": rates,
        "marks": marks,
        "aprs": aprs,
        "n": len(ev_times),
        "spot_times": spot_times,
        "spot_closes": spot_closes,
        "perp_times": perp_times,
        "perp_closes": perp_closes,
        "liq_spot": liq["spot"],
        "liq_perp": liq["perp"],
    }


def liq_lookup(d: dict, t) -> float | None:
    """min(spot, perp) trailing-30d median hourly quote volume asof t (PIT)."""
    vals = []
    for ct, med in (d["liq_spot"], d["liq_perp"]):
        idx = bisect.bisect_right(ct, t) - 1
        if idx < 0:
            return None
        if (t - ct[idx]).total_seconds() / 3600 > STALENESS_H:
            return None
        m = med[idx]
        if m is None or (isinstance(m, float) and math.isnan(m)):
            return None
        vals.append(m)
    return min(vals)


# ------------------------------------------------------------------ allocator
def run_alloc(symbols_data: dict, by_time: dict, times_sorted: list,
              K: int, enter_apr: float, use_cap: bool) -> tuple[list, dict]:
    per_slot = TOTAL_CAPITAL / K
    open_pos: dict[str, dict] = {}
    trades: list[dict] = []
    int_pos = 0.0
    int_notional = 0.0
    blocked_full = 0
    skipped_cap = 0
    skipped_noprice = 0
    capped_entries = 0  # entries whose notional was reduced by the liquidity cap
    prev_t = times_sorted[0]

    for t in times_sorted:
        dt_h = (t - prev_t).total_seconds() / 3600
        if dt_h > 0:
            int_pos += len(open_pos) * dt_h
            int_notional += sum(p["notional"] for p in open_pos.values()) * dt_h
        prev_t = t

        events = by_time[t]
        closed_this_t: set[str] = set()

        # 1) accrual + exits for held symbols with an event at t
        for sym, i in events:
            pos = open_pos.get(sym)
            if pos is None:
                continue
            d = symbols_data[sym]
            rate = d["rates"][i]
            mark = d["marks"][i]
            if mark is None or (isinstance(mark, float) and math.isnan(mark)):
                mp = _asof_close(d["perp_times"], d["perp_closes"], t)
                accr = pos["qty"] * mp if mp is not None else 0.0
            else:
                accr = pos["qty"] * mark
            pos["funding"] += rate * accr

            apr = d["aprs"][i]
            is_last = i == d["n"] - 1
            do_exit = (not math.isnan(apr)) and apr < EXIT_APR
            es = _asof_close(d["spot_times"], d["spot_closes"], t)
            ep = _asof_close(d["perp_times"], d["perp_closes"], t)
            if do_exit or is_last:
                unpriced = es is None or ep is None
                if unpriced:
                    xt, xs, xp = pos["last_priced"]
                else:
                    xt, xs, xp = t, es, ep
                basis = pos["qty"] * ((pos["entry_perp"] - pos["entry_spot"]) - (xp - xs))
                trades.append({
                    "symbol": sym,
                    "entry_time": pos["entry_time"],
                    "exit_time": xt,
                    "notional": pos["notional"],
                    "funding": pos["funding"],
                    "basis": basis,
                    "force_closed": is_last and not do_exit,
                    "unpriced": unpriced,
                })
                del open_pos[sym]
                closed_this_t.add(sym)
            elif es is not None and ep is not None:
                pos["last_priced"] = (t, es, ep)

        # 2) entries into free slots, ranked by forecast APR
        free = K - len(open_pos)
        cands = []
        for sym, i in events:
            if sym in open_pos or sym in closed_this_t:
                continue
            d = symbols_data[sym]
            apr = d["aprs"][i]
            # no entry at a symbol's last event (nothing can ever accrue/close)
            if (not math.isnan(apr)) and apr > enter_apr and i < d["n"] - 1:
                cands.append((apr, sym, i))
        cands.sort(key=lambda x: (-x[0], x[1]))
        for apr, sym, i in cands:
            if free <= 0:
                blocked_full += 1
                continue
            d = symbols_data[sym]
            if use_cap:
                liq = liq_lookup(d, t)
                cap = LIQ_FRAC * liq if liq is not None else 0.0
                if cap < MIN_CAP:
                    skipped_cap += 1
                    continue
                notional = min(per_slot, cap)
            else:
                notional = per_slot
            es = _asof_close(d["spot_times"], d["spot_closes"], t)
            ep = _asof_close(d["perp_times"], d["perp_closes"], t)
            if es is None or ep is None or es <= 0:
                skipped_noprice += 1
                continue
            if use_cap and notional < per_slot:
                capped_entries += 1
            open_pos[sym] = {
                "qty": notional / es,
                "notional": notional,
                "entry_spot": es,
                "entry_perp": ep,
                "entry_time": t,
                "funding": 0.0,
                "last_priced": (t, es, ep),
            }
            free -= 1

    assert not open_pos, f"positions left open after final events: {list(open_pos)}"

    total_h = (times_sorted[-1] - times_sorted[0]).total_seconds() / 3600
    stats = {
        "avg_concurrent": int_pos / total_h,
        "utilization_pct": 100.0 * int_notional / (TOTAL_CAPITAL * total_h),
        "blocked_full": blocked_full,
        "skipped_cap": skipped_cap,
        "skipped_noprice": skipped_noprice,
        "capped_entries": capped_entries,
        "total_hours": total_h,
        "years": total_h / YEAR_HOURS,
    }
    return trades, stats


def cell_metrics(trades: list, stats: dict, scen: FeeScenario) -> dict:
    years = stats["years"]
    rows = []
    for tr in trades:
        fee = 2 * oneway_fee(scen, tr["notional"])
        rows.append({**tr, "fees": fee, "net": tr["funding"] + tr["basis"] - fee})
    net = sum(r["net"] for r in rows)
    by_year: dict[int, dict] = {}
    for r in rows:
        y = r["entry_time"].year
        b = by_year.setdefault(y, {"net": 0.0, "funding": 0.0, "fees": 0.0, "trades": 0})
        b["net"] += r["net"]
        b["funding"] += r["funding"]
        b["fees"] += r["fees"]
        b["trades"] += 1
    return {
        "net": net,
        "apr_on_200k": net / TOTAL_CAPITAL / years,
        "gross_funding": sum(r["funding"] for r in rows),
        "basis": sum(r["basis"] for r in rows),
        "fees": sum(r["fees"] for r in rows),
        "trades": len(rows),
        "unpriced_exits": sum(1 for r in rows if r["unpriced"]),
        "by_year": by_year,
        "rows": rows,
    }


def print_cell(label: str, m: dict, stats: dict) -> None:
    print(f"--- {label}")
    print(f"    net {m['net']:>12,.2f} USDT | APR on 200k {m['apr_on_200k']:>7.2%} | "
          f"gross funding {m['gross_funding']:>12,.2f} | basis {m['basis']:>10,.2f} | "
          f"fees {m['fees']:>10,.2f}")
    print(f"    trades {m['trades']} (unpriced exits {m['unpriced_exits']}) | "
          f"avg concurrent {stats['avg_concurrent']:.2f} | "
          f"utilization {stats['utilization_pct']:.1f}% | "
          f"blocked(full) {stats['blocked_full']} | skipped(cap<2k) {stats['skipped_cap']} | "
          f"cap-reduced entries {stats['capped_entries']}")
    for y in sorted(m["by_year"]):
        b = m["by_year"][y]
        print(f"      entries {y}: net {b['net']:>11,.2f} | funding {b['funding']:>11,.2f} | "
              f"fees {b['fees']:>9,.2f} | trades {b['trades']}")


# ----------------------------------------------------------------------- main
def main() -> None:
    t0 = time.time()
    sanity_gate()

    all_syms = sorted(p.stem for p in (DATA / "funding").glob("*.parquet"))
    symbols_data = {}
    skipped_files = []
    for sym in all_syms:
        d = load_symbol(sym)
        if d is None:
            skipped_files.append(sym)
        else:
            symbols_data[sym] = d
    print(f"[load] {len(symbols_data)} symbols usable, "
          f"{len(skipped_files)} skipped (missing/empty files): {skipped_files}")
    iv: dict[int, int] = {}
    for d in symbols_data.values():
        iv[d["interval_hours"]] = iv.get(d["interval_hours"], 0) + 1
    print(f"[load] funding interval distribution: {dict(sorted(iv.items()))} "
          f"({time.time()-t0:.1f}s)\n")

    by_time: dict = {}
    for sym, d in symbols_data.items():
        for i, t in enumerate(d["ev_times"]):
            by_time.setdefault(t, []).append((sym, i))
    times_sorted = sorted(by_time)
    print(f"[load] union timeline: {len(times_sorted)} distinct event times, "
          f"{times_sorted[0]} .. {times_sorted[-1]}")
    n_ev = sum(len(v) for v in by_time.values())
    print(f"[load] {n_ev} symbol-events total\n")

    results = {}
    print("=" * 100)
    print("MAIN GRID: enter 12% / exit 0%, liquidity cap ON")
    print("=" * 100)
    for K in (5, 10, 20):
        trades, stats = run_alloc(symbols_data, by_time, times_sorted, K, ENTER_APR, True)
        for scen_name in ("taker", "maker_bnb"):
            m = cell_metrics(trades, stats, SCEN[scen_name])
            results[(K, scen_name)] = (m, stats)
            print_cell(f"K={K} | {scen_name}", m, stats)
        print()

    best_key = max(results, key=lambda k: results[k][0]["net"])
    best_K, best_scen = best_key
    print(f"BEST CELL: K={best_K} {best_scen} net {results[best_key][0]['net']:,.2f}\n")

    print("=" * 100)
    print(f"ROBUSTNESS on best K={best_K}")
    print("=" * 100)
    robust = {}
    for enter in (0.15, 0.10):
        trades, stats = run_alloc(symbols_data, by_time, times_sorted, best_K, enter, True)
        for scen_name in ("taker", "maker_bnb"):
            m = cell_metrics(trades, stats, SCEN[scen_name])
            robust[(enter, scen_name, "cap")] = (m, stats)
            print_cell(f"K={best_K} | enter {enter:.0%} | {scen_name} | cap ON", m, stats)
        print()
    trades, stats = run_alloc(symbols_data, by_time, times_sorted, best_K, ENTER_APR, False)
    for scen_name in ("taker", "maker_bnb"):
        m = cell_metrics(trades, stats, SCEN[scen_name])
        robust[(ENTER_APR, scen_name, "nocap")] = (m, stats)
        print_cell(f"K={best_K} | enter 12% | {scen_name} | cap OFF (liquidity fantasy)", m, stats)
    print()

    # K-stability at other enter thresholds for the best scenario
    print("=" * 100)
    print(f"K-STABILITY: net by K at enter 10/12/15% ({best_scen})")
    print("=" * 100)
    kstab = {}
    for K in (5, 10, 20):
        for enter in (0.10, 0.12, 0.15):
            if enter == ENTER_APR:
                m, stats = results[(K, best_scen)]
                kstab[(K, enter)] = m["net"]
            elif K == best_K:
                kstab[(K, enter)] = robust[(enter, best_scen, "cap")][0]["net"]
            else:
                tr, st = run_alloc(symbols_data, by_time, times_sorted, K, enter, True)
                kstab[(K, enter)] = cell_metrics(tr, st, SCEN[best_scen])["net"]
    print(f"{'':>6} " + " ".join(f"{f'enter {e:.0%}':>14}" for e in (0.10, 0.12, 0.15)))
    for K in (5, 10, 20):
        print(f"K={K:<4} " + " ".join(f"{kstab[(K, e)]:>14,.2f}" for e in (0.10, 0.12, 0.15)))

    # persist
    out = {
        "main_grid": {
            f"K{K}_{s}": {k: v for k, v in results[(K, s)][0].items() if k != "rows"}
            | {"stats": results[(K, s)][1]}
            for K in (5, 10, 20) for s in ("taker", "maker_bnb")
        },
        "best_cell": f"K{best_K}_{best_scen}",
        "kstab": {f"K{K}_enter{e:.2f}": kstab[(K, e)] for K in (5, 10, 20)
                  for e in (0.10, 0.12, 0.15)},
    }

    def _default(o):
        if hasattr(o, "isoformat"):
            return o.isoformat()
        raise TypeError(type(o))

    for key, blob in out["main_grid"].items():
        blob["by_year"] = {str(y): v for y, v in blob["by_year"].items()}
    (SCRATCH / "wa_results.json").write_text(json.dumps(out, indent=2, default=_default))

    best_rows = results[best_key][0]["rows"]
    pl.DataFrame([
        {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in r.items()}
        for r in best_rows
    ]).write_csv(SCRATCH / "wa_best_cell_trades.csv")
    print(f"\n[done] {time.time()-t0:.1f}s; results -> wa_results.json, "
          f"best-cell trades -> wa_best_cell_trades.csv")


if __name__ == "__main__":
    main()
