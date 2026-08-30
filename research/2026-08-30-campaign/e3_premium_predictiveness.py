"""E3 — Premium-TWAP predictiveness of the next settled funding rate.

Compares, per funding event T (all 20 symbols, first 7 events per symbol skipped):
  A  = EMA(6) of funding rates settled STRICTLY before T (exact engine recursion,
       copied from farb.backtest.engine._forecast_series semantics).
  B  = mean of premium_1h CLOSES for bars fully closed by T-1h within (T-8h, T-1h]
       (7 bars, open times T-8h .. T-2h)  -> decision made ONE HOUR BEFORE settlement.
  C  = B + 0.0001/3  (simple const interest add-on, clamp-free, per task spec).
  D  = clamp-aware bonus: B + clamp(0.0001 - B, -0.0005, +0.0005)  (Binance-style
       F = P + clamp(I - P, ±0.05%) with I = 0.01% per 8h; NOT requested, labeled extra).
  B4 = same premium mean but only bars closed by T-4h within (T-8h, T-4h]
       (4 bars, open times T-8h .. T-5h) -> decision ~4h before settlement.
Target = funding_rate settled at T.

All predictors evaluated on the SAME aligned rows (A finite, all 7 premium bars
present). Metrics: MAE (bps), Spearman rank corr, sign accuracy, and
precision/recall of pred > tau vs target > tau, tau = 0.12 * 8 / YEAR_HOURS.

Pure-python numerics (no numpy in project env).
"""

from __future__ import annotations

import math
from datetime import timedelta
from pathlib import Path
from statistics import mean, median

import polars as pl

DATA = Path("/Users/bogdan/funding-arb/data/binance")
YEAR_HOURS = 365.25 * 24  # farb.backtest.report.YEAR_HOURS
TAU = 0.12 * 8 / YEAR_HOURS  # per-8h funding-rate threshold equivalent to 12% APR
EMA_PERIODS = 6
SKIP_FIRST = 7
INTEREST_CONST = 0.0001 / 3  # per task spec for predictor C
CLAMP = 0.0005  # Binance clamp half-width (0.05%) for bonus predictor D
INTEREST_8H = 0.0001  # canonical Binance interest component per 8h (bonus D)

SYMBOLS = sorted(p.stem for p in (DATA / "funding").glob("*.parquet"))
PRED_KEYS = ["A", "B", "C", "D", "B4"]


def forecast_ema(rates: list[float], ema_periods: int) -> list[float]:
    """Exact copy of engine._forecast_series recursion (causal EMA)."""
    n = len(rates)
    forecast = [math.nan] * n
    if n == 0:
        return forecast
    alpha = 2.0 / (ema_periods + 1)
    ema = rates[0]
    seen = 1
    for i in range(1, n):
        if seen >= ema_periods:
            forecast[i] = ema
        ema = alpha * rates[i] + (1 - alpha) * ema
        seen += 1
    return forecast


def rankdata(a: list[float]) -> list[float]:
    """Average ranks (ties share the mean rank), like scipy.stats.rankdata."""
    n = len(a)
    order = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    mx, my = mean(x), mean(y)
    sxy = sxx = syy = 0.0
    for a, b in zip(x, y):
        da, db = a - mx, b - my
        sxy += da * db
        sxx += da * da
        syy += db * db
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else math.nan


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return math.nan
    return pearson(rankdata(x), rankdata(y))


def sign(v: float) -> int:
    return (v > 0) - (v < 0)


def sign_acc(pred: list[float], tgt: list[float]) -> float:
    return mean(1.0 if sign(p) == sign(t) else 0.0 for p, t in zip(pred, tgt))


def prec_recall(pred: list[float], tgt: list[float]) -> tuple[float, float, int, int]:
    tp = n_pp = n_ap = 0
    for p, t in zip(pred, tgt):
        pp, ap = p > TAU, t > TAU
        n_pp += pp
        n_ap += ap
        tp += pp and ap
    prec = tp / n_pp if n_pp else math.nan
    rec = tp / n_ap if n_ap else math.nan
    return prec, rec, n_pp, n_ap


def mae_bps(pred: list[float], tgt: list[float]) -> float:
    return mean(abs(p - t) for p, t in zip(pred, tgt)) * 1e4


def nanmedian(vals: list[float]) -> float:
    v = [x for x in vals if not math.isnan(x)]
    return median(v) if v else math.nan


rows: list[dict] = []  # per-event aligned records
dropped_missing_premium = 0
gap_anomalies = 0
total_events_considered = 0

for sym in SYMBOLS:
    f = pl.read_parquet(DATA / "funding" / f"{sym}.parquet").sort("funding_time")
    p = pl.read_parquet(DATA / "premium_1h" / f"{sym}.parquet").sort("open_time")

    times = f["funding_time"].to_list()
    rates = f["funding_rate"].to_list()
    ema = forecast_ema(rates, EMA_PERIODS)

    # hour-truncated funding event times (ms jitter -> align to hour grid)
    th = [t.replace(minute=0, second=0, microsecond=0) for t in times]
    # sanity: interval anomalies (informational only)
    gap_anomalies += sum(
        1 for a, b in zip(th[:-1], th[1:]) if (b - a) != timedelta(hours=8)
    )

    prem = dict(zip(p["open_time"].to_list(), p["close"].to_list()))

    for i in range(SKIP_FIRST, len(times)):
        total_events_considered += 1
        if math.isnan(ema[i]):
            continue
        T = th[i]
        # B: 7 bars fully closed by T-1h within (T-8h, T-1h] -> opens T-8h..T-2h
        opens7 = [T - timedelta(hours=h) for h in range(8, 1, -1)]
        vals7 = [prem.get(o) for o in opens7]
        if any(v is None for v in vals7):
            dropped_missing_premium += 1
            continue
        b = mean(vals7)
        # B4: bars closed by T-4h within (T-8h, T-4h] -> opens T-8h..T-5h (subset)
        b4 = mean(vals7[:4])
        rows.append(
            {
                "symbol": sym,
                "A": ema[i],
                "B": b,
                "C": b + INTEREST_CONST,
                "D": b + min(max(INTEREST_8H - b, -CLAMP), CLAMP),
                "B4": b4,
                "y": rates[i],
            }
        )

y = [r["y"] for r in rows]
preds = {k: [r[k] for r in rows] for k in PRED_KEYS}
n_actual_pos = sum(1 for t in y if t > TAU)

print(f"symbols: {len(SYMBOLS)}, aligned events: {len(rows)} "
      f"(considered {total_events_considered}, dropped for missing premium bars: "
      f"{dropped_missing_premium}, non-8h gaps seen: {gap_anomalies})")
print(f"tau (12% APR per-8h): {TAU:.10f} = {TAU*1e4:.4f} bps; "
      f"actual positives pooled: {n_actual_pos} ({n_actual_pos/len(rows):.2%})")
print()

# ---------- pooled metrics ----------
print("== POOLED ==")
hdr = (f"{'pred':>4} {'MAE_bps':>8} {'spearman':>9} {'sign_acc':>9} "
       f"{'precision':>9} {'recall':>7} {'n_pred_pos':>10}")
print(hdr)
pooled = {}
for k in PRED_KEYS:
    pr = preds[k]
    prec, rec, n_pp, n_ap = prec_recall(pr, y)
    pooled[k] = dict(mae=mae_bps(pr, y), rho=spearman(pr, y), sacc=sign_acc(pr, y),
                     prec=prec, rec=rec, n_pp=n_pp)
    m = pooled[k]
    print(f"{k:>4} {m['mae']:8.3f} {m['rho']:9.4f} {m['sacc']:9.4f} "
          f"{m['prec']:9.4f} {m['rec']:7.4f} {n_pp:10d}")

# ---------- per-symbol medians ----------
print("\n== PER-SYMBOL MEDIANS (across symbols) ==")
print(hdr.replace("n_pred_pos", "n_sym_def"))
per_sym = {k: {"mae": [], "rho": [], "sacc": [], "prec": [], "rec": []} for k in PRED_KEYS}
for sym in SYMBOLS:
    idx = [i for i, r in enumerate(rows) if r["symbol"] == sym]
    ys = [y[i] for i in idx]
    for k in PRED_KEYS:
        pr = [preds[k][i] for i in idx]
        prec, rec, _, _ = prec_recall(pr, ys)
        per_sym[k]["mae"].append(mae_bps(pr, ys))
        per_sym[k]["rho"].append(spearman(pr, ys))
        per_sym[k]["sacc"].append(sign_acc(pr, ys))
        per_sym[k]["prec"].append(prec)
        per_sym[k]["rec"].append(rec)

med = {}
for k in PRED_KEYS:
    s = per_sym[k]
    med[k] = {m: nanmedian(s[m]) for m in s}
    ndef = sum(1 for v in s["prec"] if not math.isnan(v))
    print(f"{k:>4} {med[k]['mae']:8.3f} {med[k]['rho']:9.4f} {med[k]['sacc']:9.4f} "
          f"{med[k]['prec']:9.4f} {med[k]['rec']:7.4f} {ndef:10d}")

# ---------- decay: B(T-1h) vs B4(T-4h) ----------
print("\n== DECAY WITH DECISION LEAD TIME ==")
print(f"B  (T-1h, 7/8 bars): pooled spearman {pooled['B']['rho']:.4f}, MAE {pooled['B']['mae']:.3f} bps")
print(f"B4 (T-4h, 4/8 bars): pooled spearman {pooled['B4']['rho']:.4f}, MAE {pooled['B4']['mae']:.3f} bps")
print(f"per-symbol median spearman: B {med['B']['rho']:.4f} -> B4 {med['B4']['rho']:.4f}")

# ---------- extra context ----------
frac_default = mean(1.0 if t == 0.0001 else 0.0 for t in y)
print(f"\ntarget == 0.0001 (interest-rate default, clamp active) fraction: {frac_default:.2%}")

b_beats_a_rho = sum(1 for br, ar in zip(per_sym["B"]["rho"], per_sym["A"]["rho"]) if br > ar)
b_beats_a_mae = sum(1 for bm, am in zip(per_sym["B"]["mae"], per_sym["A"]["mae"]) if bm < am)
d_beats_a_mae = sum(1 for dm, am in zip(per_sym["D"]["mae"], per_sym["A"]["mae"]) if dm < am)
print(f"symbols where B beats A: spearman {b_beats_a_rho}/{len(SYMBOLS)}, MAE {b_beats_a_mae}/{len(SYMBOLS)}")
print(f"symbols where D beats A on MAE: {d_beats_a_mae}/{len(SYMBOLS)}")
