"""EXPERIMENT F — fee-reduction variants on the baseline rule (EMA6).

Runs the stock engine (no monkeypatching — the signal is untouched; only
FeeScenario varies) over the 20 core symbols, for two threshold pairs:
  A) baseline enter 12% / exit 2% APR
  B) cheap-entry enter 8% / exit 0% APR

Fee scenario construction notes (bps per leg, one way):
- taker        : spot 10 + perp 5 + slip 1/leg  (repo default, no BNB)
- maker_bnb    : spot 7.5 + perp 2 + slip 0     (repo default)
- maker_pXX    : queue-risk haircut. Maker fills w.p. p, else falls back to
                 taker+slip. Blended per leg:
                   spot = p*7.5 + (1-p)*(10+1)
                   perp = p*2.0 + (1-p)*(5+1)
                 slippage_bps=0 (slip already baked into the blend).
- promo_spot0  : 0-fee spot promo pair + perp maker 2, no slip.
- perpperp_proxy: BOTH legs perp maker 2bps (spot_bps=2 as proxy) — PROXY
                 for cross-venue perp-vs-perp, not achievable on one venue.
- taker_bnb    : BNB 25% off spot taker (10->7.5), 10% off perp taker
                 (5->4.5), slip 1/leg.
- vip1_taker   : ASSUMED table per task spec: spot VIP1 taker 9bps x 0.75
                 BNB = 6.75; perp VIP1 taker 4.5; slip 1/leg.
- vip1_maker   : spot VIP1 maker 8.1bps x 0.75 BNB = 6.075; perp VIP1
                 maker 1.8; slip 0.
- onesided_maker: practical easy mode — spot taker 10 + 1 slip (=11 on the
                 spot leg), perp maker 2, no perp slip.

Because the signal does not depend on fees, the trade set at a given
threshold pair is IDENTICAL across scenarios; net differences are purely
fee arithmetic (funding and basis invariant). We still run the engine per
scenario as ground truth and verify that invariance.

SANITY GATE: taker @ 12/2 must reproduce portfolio net 1891.57 USDT.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, "/Users/bogdan/funding-arb/src")

from farb.backtest.engine import _derive_interval_hours, simulate  # noqa: E402
from farb.config import FeeScenario  # noqa: E402

DATA = Path("/Users/bogdan/funding-arb/data/binance")
NOTIONAL = 10_000.0
EMA = 6

SYMBOLS = [
    s + "USDT"
    for s in (
        "BTC ETH BNB SOL XRP DOGE ADA AVAX LINK DOT "
        "LTC BCH TRX NEAR ATOM FIL ETC APT SUI ARB"
    ).split()
]

# ---- scenario table -------------------------------------------------------
MAKER_SPOT, MAKER_PERP = 7.5, 2.0          # maker_bnb legs
TAKER_SPOT_ALLIN, TAKER_PERP_ALLIN = 11.0, 6.0  # taker fee + 1bp slip, per leg


def blended(p: float) -> FeeScenario:
    return FeeScenario(
        name=f"maker_p{int(p*100)}",
        spot_bps=p * MAKER_SPOT + (1 - p) * TAKER_SPOT_ALLIN,
        perp_bps=p * MAKER_PERP + (1 - p) * TAKER_PERP_ALLIN,
        slippage_bps=0.0,
    )


SCENARIOS: list[FeeScenario] = [
    FeeScenario(name="taker", spot_bps=10.0, perp_bps=5.0, slippage_bps=1.0),
    FeeScenario(name="maker_bnb", spot_bps=7.5, perp_bps=2.0, slippage_bps=0.0),
    blended(0.9),
    blended(0.7),
    blended(0.5),
    FeeScenario(name="promo_spot0", spot_bps=0.0, perp_bps=2.0, slippage_bps=0.0),
    FeeScenario(name="perpperp_proxy", spot_bps=2.0, perp_bps=2.0, slippage_bps=0.0),
    FeeScenario(name="taker_bnb", spot_bps=7.5, perp_bps=4.5, slippage_bps=1.0),
    FeeScenario(name="vip1_taker", spot_bps=6.75, perp_bps=4.5, slippage_bps=1.0),
    FeeScenario(name="vip1_maker", spot_bps=6.075, perp_bps=1.8, slippage_bps=0.0),
    FeeScenario(name="onesided_maker", spot_bps=11.0, perp_bps=2.0, slippage_bps=0.0),
]

THRESHOLDS = [("base_12_2", 0.12, 0.02), ("cheap_8_0", 0.08, 0.0)]


def main() -> None:
    frames = {}
    for sym in SYMBOLS:
        f = pl.read_parquet(DATA / "funding" / f"{sym}.parquet").sort("funding_time")
        s = pl.read_parquet(DATA / "klines_spot_1h" / f"{sym}.parquet")
        p = pl.read_parquet(DATA / "klines_perp_1h" / f"{sym}.parquet")
        iv = _derive_interval_hours(f["funding_time"].to_list())
        assert iv == 8, f"{sym}: expected 8h interval, derived {iv}"
        frames[sym] = (f, s, p, iv)
    assert len(frames) == 20

    # rows[(tag, scen_name)] = dict of portfolio sums
    rows: dict[tuple[str, str], dict[str, float]] = {}
    for tag, enter, exitv in THRESHOLDS:
        for scen in SCENARIOS:
            agg = dict(net=0.0, funding=0.0, basis=0.0, fees=0.0, trades=0, hold=0.0)
            for sym, (f, s, p, iv) in frames.items():
                r = simulate(
                    funding=f, spot=s, perp=p, interval_hours=iv, scenario=scen,
                    notional=NOTIONAL, ema_periods=EMA,
                    enter_apr=enter, exit_apr=exitv,
                )
                agg["net"] += r.net_pnl
                agg["funding"] += r.funding_pnl
                agg["basis"] += r.basis_pnl
                agg["fees"] += r.fees_paid
                agg["trades"] += r.n_trades
                agg["hold"] += r.holding_hours_sum
            rows[(tag, scen.name)] = agg

    # ---- SANITY GATE ----
    base = rows[("base_12_2", "taker")]
    if abs(base["net"] - 1891.57) > 0.01:
        print(f"ABORT: sanity gate FAILED — taker@12/2 net {base['net']:.2f} != 1891.57")
        sys.exit(1)
    print(f"SANITY GATE OK: taker@12/2 net {base['net']:.2f} (=1891.57)")

    # verify funding/basis invariance across scenarios per threshold pair
    for tag, _, _ in THRESHOLDS:
        fs = {round(rows[(tag, sc.name)]["funding"], 6) for sc in SCENARIOS}
        bs = {round(rows[(tag, sc.name)]["basis"], 6) for sc in SCENARIOS}
        ts = {rows[(tag, sc.name)]["trades"] for sc in SCENARIOS}
        assert len(fs) == 1 and len(bs) == 1 and len(ts) == 1, (tag, fs, bs, ts)
        print(f"invariance OK [{tag}]: funding={fs.pop():.2f} basis={bs.pop():.2f} "
              f"trades={ts.pop()}")

    print()
    hdr = (f"{'scenario':<16}{'rt_bps':>7}"
           f"{'net@12/2':>11}{'fees@12/2':>11}{'d_vs_taker':>11}"
           f"{'net@8/0':>12}{'fees@8/0':>11}{'trades8/0':>10}")
    print(hdr)
    print("-" * len(hdr))
    t12 = rows[("base_12_2", "taker")]["net"]
    for sc in SCENARIOS:
        a = rows[("base_12_2", sc.name)]
        b = rows[("cheap_8_0", sc.name)]
        rt_bps = sc.roundtrip_frac * 10_000
        print(f"{sc.name:<16}{rt_bps:>7.2f}"
              f"{a['net']:>11.2f}{a['fees']:>11.2f}{a['net']-t12:>11.2f}"
              f"{b['net']:>12.2f}{b['fees']:>11.2f}{b['trades']:>10}")

    print()
    print(f"trades@12/2 = {rows[('base_12_2','taker')]['trades']}, "
          f"trades@8/0 = {rows[('cheap_8_0','taker')]['trades']}")

    # fine p-sweep for the break-even question
    print()
    print("queue-risk p-sweep (blended maker, thresholds 12/2):")
    fund = rows[("base_12_2", "taker")]["funding"]
    basi = rows[("base_12_2", "taker")]["basis"]
    ntr = rows[("base_12_2", "taker")]["trades"]
    for pct in range(0, 101, 10):
        p = pct / 100
        sc = blended(p)
        fees = ntr * sc.roundtrip_frac * NOTIONAL
        net = fund + basi - fees
        print(f"  p={p:.1f}: rt={sc.roundtrip_frac*10_000:5.2f}bps  "
              f"net={net:8.2f}  vs taker {net - t12:+7.2f}")


if __name__ == "__main__":
    main()
