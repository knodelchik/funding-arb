"""Experiment W step 3: funding-premium map of the HL universe (60 coins).

Per coin: mean hourly funding annualized, share hours positive, lag-1 autocorr
of 8h sums, naive PIT short-only carry (EMA(6)-gated 8h blocks) minus fees.
Fees: HL maker 1.5 bps x2 legs + Binance perp maker 2 bps x2 legs = 7 bps
roundtrip per entry/exit episode (3 bps HL-only for unhedgeable coins).
PIT: gate for block T uses EMA of block sums through T-1 only.
"""
import json
from datetime import datetime, timezone

import polars as pl

SCRATCH = "/private/tmp/claude-501/-Users-bogdan/22c3fe7f-2325-4918-9a61-1adbc0bd23d3/scratchpad"
FUND_DIR = "/Users/bogdan/funding-arb/data/venues/hyperliquid/funding"
YEAR_HOURS = 365.25 * 24  # 8766
START = datetime(2024, 12, 1, tzinfo=timezone.utc)
END = datetime(2026, 8, 29, tzinfo=timezone.utc)

CORE = ["BTC","ETH","BNB","SOL","XRP","DOGE","ADA","AVAX","LINK","DOT",
        "LTC","BCH","TRX","NEAR","ATOM","FIL","ETC","APT","SUI","ARB"]
K_MAP = {"kPEPE": "1000PEPEUSDT", "kBONK": "1000BONKUSDT", "kSHIB": "1000SHIBUSDT",
         "kFLOKI": "1000FLOKIUSDT", "kNEIRO": "1000NEIROUSDT", "kLUNC": "1000LUNCUSDT"}

FEE_RT_HEDGED = 0.0007    # 2*1.5bps HL maker + 2*2bps Binance perp maker
FEE_RT_HL_ONLY = 0.0003   # 2*1.5bps HL maker (no hedge — reference only)

with open(f"{SCRATCH}/hl_universe.json") as f:
    universe = {r["coin"]: r for r in json.load(f)}
with open(f"{SCRATCH}/binance_perps.json") as f:
    bperps = set(json.load(f))
with open(f"{SCRATCH}/top40_selection.json") as f:
    top40 = [r["coin"] for r in json.load(f)]

def hedge_symbol(coin):
    if coin in K_MAP:
        return K_MAP[coin] if K_MAP[coin] in bperps else None
    s = coin + "USDT"
    return s if s in bperps else None

def analyze(coin):
    path = f"{FUND_DIR}/{coin}USDT.parquet"
    df = pl.read_parquet(path).filter(
        (pl.col("funding_time") >= START) & (pl.col("funding_time") < END)
    ).sort("funding_time")
    if len(df) == 0:
        return None
    n = len(df)
    years = n / YEAR_HOURS
    mean_apr = df["funding_rate"].mean() * YEAR_HOURS
    share_pos = (df["funding_rate"] > 0).mean()

    # 8h calendar blocks (00/08/16 UTC), drop blocks with <6 hourly points
    blocks = (
        df.group_by(pl.col("funding_time").dt.truncate("8h").alias("block"))
        .agg(pl.col("funding_rate").sum().alias("f8"), pl.len().alias("nh"))
        .sort("block")
        .filter(pl.col("nh") >= 6)
    )
    f8 = blocks["f8"]
    nb = len(f8)
    if nb < 30:
        return None
    ac1 = pl.DataFrame({"a": f8, "b": f8.shift(1)}).drop_nulls().select(
        pl.corr("a", "b")
    ).item()

    # PIT EMA(6) gate: decision for block T uses EMA through T-1
    ema_prev = f8.ewm_mean(span=6).shift(1)
    d = pl.DataFrame({"f8": f8, "ema_prev": ema_prev}).drop_nulls()
    gate = d["ema_prev"] > 0
    gross = d.filter(gate)["f8"].sum()
    episodes = int((gate.cast(pl.Int8).diff().fill_null(gate.cast(pl.Int8).first()) == 1).sum())
    hedge = hedge_symbol(coin)
    fee_rt = FEE_RT_HEDGED if hedge else FEE_RT_HL_ONLY
    net = gross - episodes * fee_rt
    net_apr = net / years
    gross_apr = gross / years

    # variant: hysteresis enter EMA>12% APR, exit EMA<0 (established best rule)
    thr_hi = 0.12 / YEAR_HOURS * 8
    in_pos = False
    gross_h, ep_h = 0.0, 0
    for fv, ev in zip(d["f8"], d["ema_prev"]):
        if not in_pos and ev > thr_hi:
            in_pos = True
            ep_h += 1
        elif in_pos and ev < 0:
            in_pos = False
        if in_pos:
            gross_h += fv
    net_h_apr = (gross_h - ep_h * fee_rt) / years

    u = universe.get(coin, {})
    return {
        "coin": coin,
        "core": coin in [c for c in CORE],
        "hedge": hedge or "none",
        "oi_usd": u.get("oi_usd", 0.0),
        "max_lev": u.get("max_leverage"),
        "hours": n,
        "years": round(years, 3),
        "mean_apr_pct": round(mean_apr * 100, 2),
        "share_pos_pct": round(share_pos * 100, 1),
        "ac1_8h": round(ac1, 3) if ac1 is not None else None,
        "episodes": episodes,
        "gross_apr_pct": round(gross_apr * 100, 2),
        "net_apr_pct": round(net_apr * 100, 2),
        "net_hyst_apr_pct": round(net_h_apr * 100, 2),
    }

rows = []
for coin in CORE + top40:
    try:
        r = analyze(coin)
    except FileNotFoundError:
        print(f"missing file: {coin}")
        continue
    if r is None:
        print(f"insufficient data: {coin}")
        continue
    rows.append(r)

res = pl.DataFrame(rows).sort("net_apr_pct", descending=True)
res.write_csv(f"{SCRATCH}/w_experiment_results.csv")

pl.Config.set_tbl_rows(70)
pl.Config.set_tbl_cols(20)
pl.Config.set_tbl_width_chars(220)
print(res)

# --- top-15 tradable (hedgeable, OI >= $5M) ---
tradable = res.filter((pl.col("hedge") != "none") & (pl.col("oi_usd") >= 5e6))
top15 = tradable.head(15)
print("\n=== TOP-15 TRADABLE (hedgeable on Binance, OI >= $5M) by net carry APR ===")
print(top15.select(["coin","core","hedge","oi_usd","years","mean_apr_pct","share_pos_pct",
                    "ac1_8h","episodes","gross_apr_pct","net_apr_pct","net_hyst_apr_pct"]))

# --- capacity at 100k: greedy allocation, per-coin cap = min(25k, 1% OI) ---
DEPLOY = 100_000.0
alloc, left = [], DEPLOY
for r in top15.iter_rows(named=True):
    if left <= 0 or r["net_apr_pct"] <= 0:
        break
    cap = min(25_000.0, 0.01 * r["oi_usd"])
    take = min(cap, left)
    alloc.append((r["coin"], take, r["net_apr_pct"], 0.01 * r["oi_usd"]))
    left -= take
w_apr = sum(a * n for _, a, n, _ in alloc) / max(DEPLOY - left, 1)
print("\n=== 100k allocation (cap/coin = min(25k, 1% OI)) ===")
for c, a, napr, cap1 in alloc:
    print(f"  {c:>10}  alloc=${a:>9,.0f}  net_apr={napr:6.1f}%  1%OI=${cap1:>12,.0f}")
print(f"  deployed ${DEPLOY-left:,.0f} of ${DEPLOY:,.0f}; weighted net APR = {w_apr:.1f}%")

# aggregates: core vs non-core
for label, sub in [("core20", res.filter(pl.col("core"))), ("new40", res.filter(~pl.col("core")))]:
    print(f"\n{label}: n={len(sub)}, median mean_apr={sub['mean_apr_pct'].median():.1f}%, "
          f"median net_apr={sub['net_apr_pct'].median():.1f}%, "
          f"median ac1={sub['ac1_8h'].median():.3f}, "
          f"total 1%OI of hedgeable>=5M = ${sub.filter((pl.col('hedge')!='none') & (pl.col('oi_usd')>=5e6))['oi_usd'].sum()*0.01:,.0f}")
