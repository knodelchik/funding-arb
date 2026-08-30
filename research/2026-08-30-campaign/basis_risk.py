"""Experiment B: cross-venue basis risk, short-HL / long-Binance.

Per symbol:
  r(t) = hl_close(t) / binance_perp_close(t)   (1h closes, inner join on hour)
  - median r, p1/p99 of r, max |r-1| (+timestamp)
  - worst 24h swing of r (max r - min r within any rolling 24h window)
  - top-3 daily excursions of |r-1| with dates
  - funding-vs-basis on those dates: daily carry of the pair
      carry_day = sum(HL hourly funding rates that day) - sum(Binance 8h funding that day)
    (short HL receives +rate; long Binance pays rate)
  - margin stress: per-symbol HL maintenance margin MM = 1/(2*maxLeverage)
      short-leg liquidation price ratio at leverage L: x_liq = (1 + 1/L)/(1 + MM)
      divergence-only loss on 10k leg at p99/max |r-1|
      raw HL price drawups (max run-up) over 24h/72h/7d/30d rolling windows and
      max run-up from any trough -> subsequent peak (worst-case hold with no top-up)
"""
import json
from pathlib import Path

import polars as pl

ROOT = Path("/Users/bogdan/funding-arb")
HL_K = ROOT / "data/venues/hyperliquid/klines_1h"
BN_K = ROOT / "data/binance/klines_perp_1h"
HL_F = ROOT / "data/venues/hyperliquid/funding"
BN_F = ROOT / "data/binance/funding"
SCRATCH = Path("/private/tmp/claude-501/-Users-bogdan/22c3fe7f-2325-4918-9a61-1adbc0bd23d3/scratchpad")

CORE = "BTC ETH BNB SOL XRP DOGE ADA AVAX LINK DOT LTC BCH TRX NEAR ATOM FIL ETC APT SUI ARB".split()

# per-symbol HL max leverage (fetched from {"type":"meta"} on 2026-08-30)
MAX_LEV = {
    "BTC": 40, "ETH": 25, "SOL": 20, "XRP": 20,
    "BNB": 10, "DOGE": 10, "ADA": 10, "AVAX": 10, "LINK": 10, "DOT": 10,
    "LTC": 10, "BCH": 10, "TRX": 10, "NEAR": 10, "APT": 10, "SUI": 10, "ARB": 10,
    "ATOM": 5, "FIL": 5, "ETC": 5,
}

NOTIONAL = 10_000.0


def load_pair(coin: str) -> pl.DataFrame:
    sym = coin + "USDT"
    hl = pl.read_parquet(HL_K / f"{sym}.parquet").select(
        pl.col("open_time").cast(pl.Datetime("us", "UTC")),
        pl.col("close").alias("hl_close"),
    )
    bn = pl.read_parquet(BN_K / f"{sym}.parquet").select(
        pl.col("open_time").cast(pl.Datetime("us", "UTC")),
        pl.col("close").alias("bn_close"),
    )
    df = hl.join(bn, on="open_time", how="inner").sort("open_time")
    df = df.with_columns((pl.col("hl_close") / pl.col("bn_close")).alias("r"))
    df = df.with_columns((pl.col("r") - 1.0).abs().alias("absdev"))
    return df


def rolling_swing(df: pl.DataFrame, hours: int) -> float:
    """max over rolling windows of (max r - min r) within window."""
    s = df.select(
        (
            pl.col("r").rolling_max(window_size=hours)
            - pl.col("r").rolling_min(window_size=hours)
        ).max()
    ).item()
    return s if s is not None else float("nan")


def rolling_drawup(closes: pl.Series, hours: int) -> float:
    """Time-ordered worst rally bounded by `hours`: max over t of
    close[t] / min(close[t-hours+1 .. t]) - 1 (trough occurs before peak)."""
    df = pl.DataFrame({"c": closes})
    v = df.select(
        (pl.col("c") / pl.col("c").rolling_min(window_size=hours) - 1.0).max()
    ).item()
    return v if v is not None else float("nan")


def liq_hit_frac(closes: pl.Series, hold_hours: int, thresh: float) -> float:
    """Fraction of entry hours whose forward path within `hold_hours` rallies
    >= thresh above entry close (i.e. a short at that entry gets liquidated
    with no margin top-up)."""
    c = closes.to_list()
    rev = pl.Series(list(reversed(c)))
    fwd_max_rev = pl.DataFrame({"c": rev}).select(
        pl.col("c").rolling_max(window_size=hold_hours, min_samples=1)
    )["c"]
    fwd_max = list(reversed(fwd_max_rev.to_list()))
    n = len(c)
    hits = sum(1 for i in range(n) if fwd_max[i] / c[i] - 1.0 >= thresh)
    return hits / n


def max_runup(closes: pl.Series) -> float:
    """max over t of close[t] / min(close[0..t]) - 1 : worst trough->peak rally."""
    df = pl.DataFrame({"c": closes})
    v = df.select((pl.col("c") / pl.col("c").cum_min() - 1.0).max()).item()
    return v


def daily_carry(coin: str) -> pl.DataFrame:
    sym = coin + "USDT"
    hf = pl.read_parquet(HL_F / f"{sym}.parquet")  # hourly rates
    bf = pl.read_parquet(BN_F / f"{sym}.parquet")  # 8h rates
    hl_day = (
        hf.with_columns(pl.col("funding_time").dt.date().alias("d"))
        .group_by("d")
        .agg(pl.col("funding_rate").sum().alias("hl_day"))
    )
    bn_day = (
        bf.with_columns(pl.col("funding_time").dt.date().alias("d"))
        .group_by("d")
        .agg(pl.col("funding_rate").sum().alias("bn_day"))
    )
    return hl_day.join(bn_day, on="d", how="full", coalesce=True).fill_null(0.0).sort("d")


def main() -> None:
    rows = []
    excursions = []
    for coin in CORE:
        sym = coin + "USDT"
        p = HL_K / f"{sym}.parquet"
        if not p.exists():
            print(f"{coin}: HL klines missing, skipped")
            continue
        df = load_pair(coin)
        n = len(df)
        if n < 500:
            print(f"{coin}: only {n} joined hours, skipped")
            continue
        med = df["r"].median()
        p1, p99 = df["r"].quantile(0.01), df["r"].quantile(0.99)
        mx_i = df["absdev"].arg_max()
        mx = df["absdev"][mx_i]
        mx_t = df["open_time"][mx_i]
        p99_dev = df["absdev"].quantile(0.99)
        swing24 = rolling_swing(df, 24)

        daily = (
            df.with_columns(pl.col("open_time").dt.date().alias("d"))
            .group_by("d")
            .agg(pl.col("absdev").max().alias("day_max"))
            .sort("day_max", descending=True)
        )
        top3 = daily.head(3)

        carry = daily_carry(coin)
        top3 = top3.join(carry, on="d", how="left").with_columns(
            (pl.col("hl_day") - pl.col("bn_day")).alias("carry_day")
        )

        mm = 1.0 / (2 * MAX_LEV[coin])
        liq = {L: (1 + 1 / L) / (1 + mm) - 1 for L in (1, 2, 3)}

        # Drawups over the FULL 21-month window from Binance perp closes
        # (proxy for HL price — they track within bps, as shown by this analysis;
        # HL only serves ~5000 recent candles so its own history is ~7 months).
        full = pl.read_parquet(BN_K / f"{sym}.parquet").sort("open_time")
        closes = full["close"]
        du24 = rolling_drawup(closes, 24)
        du72 = rolling_drawup(closes, 72)
        du7d = rolling_drawup(closes, 24 * 7)
        du30d = rolling_drawup(closes, 24 * 30)
        runup = max_runup(closes)
        # liquidation-hit frequency over entries, 7d and 30d holds, no top-up
        hitfreq = {
            (L, H): liq_hit_frac(closes, H * 24, liq[L])
            for L in (1, 2, 3)
            for H in (7, 30)
        }

        rows.append(
            {
                "coin": coin,
                "hours": n,
                "start": str(df["open_time"].min())[:10],
                "end": str(df["open_time"].max())[:10],
                "median_r": med,
                "p1_r": p1,
                "p99_r": p99,
                "p99_absdev_bps": p99_dev * 1e4,
                "max_absdev_bps": mx * 1e4,
                "max_absdev_time": str(mx_t),
                "swing24_bps": swing24 * 1e4,
                "p99_loss_usd": NOTIONAL * p99_dev,
                "max_loss_usd": NOTIONAL * mx,
                "mm_pct": mm * 100,
                "liq_move_1x_pct": liq[1] * 100,
                "liq_move_2x_pct": liq[2] * 100,
                "liq_move_3x_pct": liq[3] * 100,
                "drawup24h_pct": du24 * 100,
                "drawup72h_pct": du72 * 100,
                "drawup7d_pct": du7d * 100,
                "drawup30d_pct": du30d * 100,
                "max_runup_pct": runup * 100,
                "hit3x_7d_pct": hitfreq[(3, 7)] * 100,
                "hit2x_7d_pct": hitfreq[(2, 7)] * 100,
                "hit1x_7d_pct": hitfreq[(1, 7)] * 100,
                "hit3x_30d_pct": hitfreq[(3, 30)] * 100,
                "hit2x_30d_pct": hitfreq[(2, 30)] * 100,
                "hit1x_30d_pct": hitfreq[(1, 30)] * 100,
            }
        )
        for row in top3.iter_rows(named=True):
            excursions.append(
                {
                    "coin": coin,
                    "date": str(row["d"]),
                    "day_max_absdev_bps": row["day_max"] * 1e4,
                    "hl_funding_day_bps": (row["hl_day"] or 0) * 1e4,
                    "bn_funding_day_bps": (row["bn_day"] or 0) * 1e4,
                    "pair_carry_day_bps": (row["carry_day"] or 0) * 1e4,
                }
            )

    res = pl.DataFrame(rows)
    exc = pl.DataFrame(excursions)
    res.write_parquet(SCRATCH / "basis_risk_table.parquet")
    exc.write_parquet(SCRATCH / "basis_excursions.parquet")

    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_fmt_str_lengths(24)
    print("=== BASIS RATIO STATS (r = HL close / Binance perp close) ===")
    print(
        res.select(
            "coin", "hours", "start", "end",
            (pl.col("median_r") - 1).mul(1e4).round(2).alias("med_dev_bps"),
            (pl.col("p1_r") - 1).mul(1e4).round(2).alias("p1_bps"),
            (pl.col("p99_r") - 1).mul(1e4).round(2).alias("p99_bps"),
            pl.col("p99_absdev_bps").round(2),
            pl.col("max_absdev_bps").round(1),
            "max_absdev_time",
            pl.col("swing24_bps").round(1),
        )
    )
    print("=== MARGIN STRESS (10k/leg, short HL) ===")
    print(
        res.select(
            "coin",
            pl.col("p99_loss_usd").round(2),
            pl.col("max_loss_usd").round(2),
            pl.col("mm_pct").round(2),
            pl.col("liq_move_3x_pct").round(1),
            pl.col("liq_move_2x_pct").round(1),
            pl.col("liq_move_1x_pct").round(1),
            pl.col("drawup24h_pct").round(1),
            pl.col("drawup72h_pct").round(1),
            pl.col("drawup7d_pct").round(1),
            pl.col("drawup30d_pct").round(1),
            pl.col("max_runup_pct").round(1),
            pl.col("hit3x_7d_pct").round(2),
            pl.col("hit2x_7d_pct").round(2),
            pl.col("hit1x_7d_pct").round(2),
            pl.col("hit3x_30d_pct").round(2),
            pl.col("hit2x_30d_pct").round(2),
            pl.col("hit1x_30d_pct").round(2),
        )
    )
    print("=== TOP-3 DAILY EXCURSIONS PER SYMBOL + FUNDING CARRY THAT DAY ===")
    print(exc)

    print("=== EXCURSION DATE CLUSTERING (across symbols) ===")
    print(exc.group_by("date").len().sort("len", descending=True).head(12))


if __name__ == "__main__":
    main()
