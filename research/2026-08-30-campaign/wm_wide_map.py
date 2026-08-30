"""EXPERIMENT WM — wide opportunity map over all symbols with data.

Stock engine per symbol, 12%/0%, taker + maker_bnb, 10k notional,
per-symbol derived funding interval. Adds liquidity overlay, concurrency
profile on union 8h grid, and new-listing attribution.

SANITY GATE: reproduces core-20 taker 2385.17 / maker 2880.17 first;
hard-aborts otherwise.
"""

from __future__ import annotations

import bisect
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl

sys.path.insert(0, "/Users/bogdan/funding-arb/src")

from farb.backtest.engine import _derive_interval_hours, _forecast_series, simulate
from farb.backtest.report import YEAR_HOURS
from farb.config import DEFAULT_SCENARIOS

DATA = Path("/Users/bogdan/funding-arb/data/binance")
SCRATCH = Path("/private/tmp/claude-501/-Users-bogdan/22c3fe7f-2325-4918-9a61-1adbc0bd23d3/scratchpad")
NOTIONAL = 10_000.0
EMA = 6
ENTER, EXIT = 0.12, 0.0
CORE20 = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "DOGEUSDT", "ADAUSDT", "AVAXUSDT", "LINKUSDT", "DOTUSDT",
    "LTCUSDT", "BCHUSDT", "TRXUSDT", "NEARUSDT", "ATOMUSDT",
    "FILUSDT", "ETCUSDT", "APTUSDT", "SUIUSDT", "ARBUSDT",
]
TAKER, MAKER = DEFAULT_SCENARIOS  # taker, maker_bnb
assert TAKER.name == "taker" and MAKER.name == "maker_bnb"

# Deployability: 10k position must be <= 0.5% of median hourly quote volume
# => median hourly quote volume >= 2,000,000 USDT/h.
DEPLOYABLE_MIN_QVOL = NOTIONAL / 0.005  # 2M


def load_symbol(sym: str):
    fp = DATA / "funding" / f"{sym}.parquet"
    sp = DATA / "klines_spot_1h" / f"{sym}.parquet"
    pp = DATA / "klines_perp_1h" / f"{sym}.parquet"
    if not (fp.exists() and sp.exists() and pp.exists()):
        return None
    funding = pl.read_parquet(fp).sort("funding_time")
    spot = pl.read_parquet(sp).sort("open_time")
    perp = pl.read_parquet(pp).sort("open_time")
    if funding.height == 0 or spot.height == 0 or perp.height == 0:
        return None
    interval = _derive_interval_hours(funding["funding_time"].to_list())
    return funding, spot, perp, interval


def run_one(sym: str, funding, spot, perp, interval):
    out = {}
    for scen in (TAKER, MAKER):
        r = simulate(
            funding=funding, spot=spot, perp=perp, interval_hours=interval,
            scenario=scen, notional=NOTIONAL, ema_periods=EMA,
            enter_apr=ENTER, exit_apr=EXIT,
        )
        out[scen.name] = r
    return out


# ---------------------------------------------------------------- sanity gate
print("=== SANITY GATE: core-20 @ 12%/0% ===", flush=True)
gate_net = {"taker": 0.0, "maker_bnb": 0.0}
gate_trades = {"taker": 0, "maker_bnb": 0}
for sym in CORE20:
    loaded = load_symbol(sym)
    assert loaded is not None, f"core-20 symbol {sym} missing data"
    res = run_one(sym, *loaded)
    for name, r in res.items():
        gate_net[name] += r.net_pnl
        gate_trades[name] += r.n_trades
print(f"core-20 taker net    = {gate_net['taker']:.2f} ({gate_trades['taker']} trades)  [baseline 2385.17, 33]")
print(f"core-20 maker_bnb net = {gate_net['maker_bnb']:.2f} ({gate_trades['maker_bnb']} trades)  [baseline 2880.17, 33]")
# The dataset was re-collected 2026-08-30 11:40-12:05 (after the 2026-08-29
# baseline report). Verified separately: at 12%/2% today's snapshot gives
# 1894.84/2449.84 with 37/37 trades and 19 of 20 symbols matching the
# 2026-08-29 report TO THE CENT; the entire delta is BCHUSDT's still-open
# tail trade force-closing at the new last event (+3.27). So the gate here
# demands exact trade counts and net within that documented tail drift.
if (
    gate_trades["taker"] != 33
    or gate_trades["maker_bnb"] != 33
    or abs(gate_net["taker"] - 2385.17) > 4.0
    or abs(gate_net["maker_bnb"] - 2880.17) > 4.0
    or abs((gate_net["maker_bnb"] - gate_net["taker"]) - (2880.17 - 2385.17)) > 0.01
):
    print("SANITY GATE FAILED — aborting", file=sys.stderr)
    sys.exit(1)
print("SANITY GATE PASSED (exact trade counts; net within documented "
      "BCHUSDT tail-trade drift from the 2026-08-30 data refresh)\n", flush=True)

# ---------------------------------------------------------------- wide run
symbols = sorted(p.stem for p in (DATA / "funding").glob("*.parquet"))
print(f"wide universe: {len(symbols)} funding files")

rows = []
skipped = []
# For concurrency + listing analysis keep per-symbol event series.
apr_series: dict[str, tuple[list[datetime], list[float]]] = {}
trades_by_sym: dict[str, dict[str, list]] = {}
first_event: dict[str, datetime] = {}

for i, sym in enumerate(symbols):
    loaded = load_symbol(sym)
    if loaded is None:
        skipped.append(sym)
        continue
    funding, spot, perp, interval = loaded
    res = run_one(sym, funding, spot, perp, interval)
    rt, rm = res["taker"], res["maker_bnb"]

    # liquidity: median hourly quote volume over the symbol's spot history
    med_qvol = spot.select((pl.col("close") * pl.col("volume")).median()).item()
    med_qvol = float(med_qvol) if med_qvol is not None else 0.0

    times = funding["funding_time"].to_list()
    rates = funding["funding_rate"].to_list()
    fcst = _forecast_series(rates, EMA)
    ppy = YEAR_HOURS / interval
    aprs = [f * ppy if not math.isnan(f) else math.nan for f in fcst]
    apr_series[sym] = (times, aprs)
    first_event[sym] = times[0]
    trades_by_sym[sym] = {"taker": rt.trades, "maker_bnb": rm.trades}

    deployed_years_t = rt.holding_hours_sum / YEAR_HOURS
    rows.append({
        "symbol": sym,
        "interval_h": interval,
        "data_start": times[0].isoformat(),
        "data_end": times[-1].isoformat(),
        "n_events": len(times),
        "net_taker": rt.net_pnl,
        "net_maker": rm.net_pnl,
        "trades_taker": rt.n_trades,
        "trades_maker": rm.n_trades,
        "funding_pnl_taker": rt.funding_pnl,
        "basis_pnl_taker": rt.basis_pnl,
        "fees_taker": rt.fees_paid,
        "deployed_apr_taker": (rt.net_pnl / (NOTIONAL * deployed_years_t))
        if deployed_years_t > 0 else float("nan"),
        "time_in_mkt_taker": rt.time_in_market_frac,
        "holding_hours_taker": rt.holding_hours_sum,
        "med_hourly_qvol_usdt": med_qvol,
        "unpriced_exits_taker": rt.unpriced_exit_count,
        "unpriced_exits_maker": rm.unpriced_exit_count,
    })
    if (i + 1) % 50 == 0:
        print(f"  ... {i + 1}/{len(symbols)} done", flush=True)

df = pl.DataFrame(rows).sort("net_taker", descending=True)
csv_path = SCRATCH / "wm_per_symbol.csv"
df.write_csv(csv_path)
print(f"\nper-symbol table -> {csv_path}  ({df.height} symbols, {len(skipped)} skipped: {skipped})")

# liquidity buckets
def bucket(q):
    if q >= 5e6:
        return ">=5M/h"
    if q >= 1e6:
        return "1-5M/h"
    if q >= 2e5:
        return "0.2-1M/h"
    return "<0.2M/h"

df = df.with_columns(
    pl.col("med_hourly_qvol_usdt").map_elements(bucket, return_dtype=pl.Utf8).alias("liq_bucket"),
    (pl.col("med_hourly_qvol_usdt") >= DEPLOYABLE_MIN_QVOL).alias("deployable_10k"),
)
df.write_csv(csv_path)  # rewrite with bucket columns

tot_t = df["net_taker"].sum()
tot_m = df["net_maker"].sum()
tot_trades_t = df["trades_taker"].sum()
tot_trades_m = df["trades_maker"].sum()
hold_sum = df["holding_hours_taker"].sum()
port_deployed_apr = tot_t / (NOTIONAL * hold_sum / YEAR_HOURS)
print(f"\n=== TOTALS (wide, {df.height} symbols) ===")
print(f"taker net {tot_t:,.2f} USDT over {tot_trades_t} trades; maker_bnb net {tot_m:,.2f} over {tot_trades_m}")
print(f"portfolio deployed APR (taker) {port_deployed_apr:.1%}; sum holding-hours {hold_sum:,.0f}")
pos_t = df.filter(pl.col("net_taker") > 0)["net_taker"].sum()
neg_t = df.filter(pl.col("net_taker") < 0)["net_taker"].sum()
print(f"gross positive {pos_t:,.2f} from {df.filter(pl.col('net_taker') > 0).height} syms; "
      f"gross negative {neg_t:,.2f} from {df.filter(pl.col('net_taker') < 0).height} syms; "
      f"zero-trade syms {df.filter(pl.col('trades_taker') == 0).height}")

print("\n=== TOP 25 by taker net ===")
top = df.head(25).select(["symbol", "interval_h", "net_taker", "net_maker", "trades_taker",
                          "deployed_apr_taker", "time_in_mkt_taker", "med_hourly_qvol_usdt",
                          "liq_bucket", "data_start"])
with pl.Config(tbl_rows=30, tbl_cols=12, fmt_str_lengths=30, tbl_width_chars=200):
    print(top)

print("\n=== BOTTOM 10 by taker net ===")
bot = df.tail(10).select(["symbol", "interval_h", "net_taker", "net_maker", "trades_taker",
                          "deployed_apr_taker", "med_hourly_qvol_usdt", "liq_bucket"])
with pl.Config(tbl_rows=15, tbl_cols=12, fmt_str_lengths=30, tbl_width_chars=200):
    print(bot)

print("\n=== LIQUIDITY BUCKETS (taker net) ===")
lb = (df.group_by("liq_bucket").agg(
        pl.col("net_taker").sum().alias("net_taker"),
        pl.col("net_maker").sum().alias("net_maker"),
        pl.len().alias("n_symbols"),
        pl.col("trades_taker").sum().alias("trades"),
     ).sort("net_taker", descending=True))
with pl.Config(tbl_rows=10, tbl_width_chars=200):
    print(lb)
dep = df.filter(pl.col("deployable_10k"))
ndep = df.filter(~pl.col("deployable_10k"))
print(f"deployable (med qvol >= {DEPLOYABLE_MIN_QVOL/1e6:.0f}M/h): {dep.height} syms, "
      f"taker net {dep['net_taker'].sum():,.2f} ({dep['net_taker'].sum()/tot_t:.1%} of total), "
      f"maker {dep['net_maker'].sum():,.2f}")
print(f"NOT deployable: {ndep.height} syms, taker net {ndep['net_taker'].sum():,.2f}")
# share of gross positive profit that is deployable
dep_pos = dep.filter(pl.col("net_taker") > 0)["net_taker"].sum()
print(f"of gross-positive taker profit ({pos_t:,.2f}), deployable share = {dep_pos/pos_t:.1%}")

# ---------------------------------------------------------------- concurrency
print("\n=== CONCURRENCY on union 8h grid (forecast APR > 12%) ===")
grid_start = min(fe for fe in first_event.values())
grid_end = max(t[-1] for t, _ in apr_series.values())
# align to 8h grid (00/08/16 UTC)
g = grid_start.replace(minute=0, second=0, microsecond=0)
g = g.replace(hour=(g.hour // 8) * 8)
grid = []
while g <= grid_end:
    grid.append(g)
    g += timedelta(hours=8)

deployable_set = set(dep["symbol"].to_list())
STALE = timedelta(hours=8, minutes=5)  # forecast must come from an event within one grid step

counts_all, counts_liq = [], []
for T in grid:
    c_all = c_liq = 0
    for sym, (times, aprs) in apr_series.items():
        idx = bisect.bisect_right(times, T) - 1
        if idx < 0:
            continue
        if T - times[idx] > STALE:
            continue
        a = aprs[idx]
        if not math.isnan(a) and a > ENTER:
            c_all += 1
            if sym in deployable_set:
                c_liq += 1
    counts_all.append(c_all)
    counts_liq.append(c_liq)

conc = pl.DataFrame({"time": grid, "n_all": counts_all, "n_liq": counts_liq})
conc.write_csv(SCRATCH / "wm_concurrency_8h.csv")

def stats(s: pl.Series, label: str):
    print(f"  {label}: median {s.median():.0f}, p75 {s.quantile(0.75):.0f}, "
          f"p90 {s.quantile(0.9):.0f}, max {s.max()}, mean {s.mean():.2f}")

stats(conc["n_all"], "unfiltered  ")
stats(conc["n_liq"], "liq-filtered")
for yr in (2025, 2026):
    sub = conc.filter(pl.col("time").dt.year() == yr)
    if sub.height:
        print(f"  -- {yr} ({sub.height} grid pts):")
        stats(sub["n_all"], "unfiltered  ")
        stats(sub["n_liq"], "liq-filtered")
# quarterly trend
qtr = (conc.with_columns(
        (pl.col("time").dt.year().cast(pl.Utf8) + "Q" + pl.col("time").dt.quarter().cast(pl.Utf8)).alias("q"))
       .group_by("q").agg(pl.col("n_all").median().alias("med_all"),
                          pl.col("n_liq").median().alias("med_liq"),
                          pl.col("n_all").mean().alias("mean_all"),
                          pl.col("n_liq").mean().alias("mean_liq"))
       .sort("q"))
with pl.Config(tbl_rows=12):
    print(qtr)

# ---------------------------------------------------------------- new-listing effect
print("\n=== NEW-LISTING EFFECT (trades entered within 30d of symbol data start) ===")
WINDOW_START = datetime(2024, 12, 1, tzinfo=timezone.utc)
listing_net = {"taker": 0.0, "maker_bnb": 0.0}
listing_net_late = {"taker": 0.0, "maker_bnb": 0.0}
late_syms = set()
for sym, scen_trades in trades_by_sym.items():
    fe = first_event[sym]
    cutoff = fe + timedelta(days=30)
    is_late = fe > WINDOW_START + timedelta(days=7)  # genuinely listed after window start
    if is_late:
        late_syms.add(sym)
    for scen, trs in scen_trades.items():
        s = sum(tr.net_pnl for tr in trs if tr.entry_time <= cutoff)
        listing_net[scen] += s
        if is_late:
            listing_net_late[scen] += s

late_total_t = df.filter(pl.col("symbol").is_in(late_syms))["net_taker"].sum()
print(f"symbols with data starting >7d after 2024-12-01 (proxy for true new listings): {len(late_syms)}")
print(f"first-30d trade net, ALL symbols: taker {listing_net['taker']:,.2f} "
      f"({listing_net['taker']/tot_t:.1%} of wide total), maker {listing_net['maker_bnb']:,.2f}")
print(f"first-30d trade net, LATE-LISTED only: taker {listing_net_late['taker']:,.2f} "
      f"({listing_net_late['taker']/tot_t:.1%} of wide total; "
      f"{(listing_net_late['taker']/late_total_t if late_total_t else float('nan')):.1%} of late-listed syms' own total {late_total_t:,.2f}), "
      f"maker {listing_net_late['maker_bnb']:,.2f}")
dep_late = dep.filter(pl.col("symbol").is_in(late_syms))
print(f"late-listed AND deployable at 10k: {dep_late.height} syms, taker net {dep_late['net_taker'].sum():,.2f}")
print("\nDONE")
