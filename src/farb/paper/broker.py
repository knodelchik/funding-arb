"""Virtual fills for the paper executor: short HL perp + long Binance hedge.

Maker-first fill model: each leg independently draws maker vs taker with
probability `cfg.maker_fill_prob` from a RNG seeded deterministically from
(symbol, hour) — never the `random` module's global state — so replaying
the same journal (or re-running a test) reproduces the exact same fills.

Sign convention (see farb.paper.models' module docstring): the HL leg is
SHORT (receives positive funding), the hedge leg is LONG (a perp hedge PAYS
positive funding; a spot hedge pays nothing).
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from farb.paper.models import Fill, HedgeMode, PaperConfig, Position, SymbolSpec

if TYPE_CHECKING:
    # Only for type hints — farb.hl may not exist yet while this lane is
    # under construction; annotations are strings (see __future__ import
    # above) so this import is never attempted at runtime.
    from datetime import datetime

    from farb.hl.client import BookTop


def _seeded_rng(symbol: str, ts: datetime) -> random.Random:
    """A fresh RNG seeded from (symbol, UTC hour) — NOT `random`'s global
    state. Determinism is the whole point: replaying a journal (or a test)
    must reproduce bit-exact fills, and the global `random` module's state
    depends on call order across the whole process, which a paper-trading
    daemon cannot promise to hold constant across restarts.
    """
    from farb.paper.journal import hour_bucket

    seed = f"{symbol}:{hour_bucket(ts).isoformat()}"
    return random.Random(seed)  # noqa: S311 - deterministic paper-fill sim, not security


def hl_funding_usd(rate: float, qty: float, mark_price: float) -> float:
    """Amount the SHORT HL leg receives this settlement (can be negative if
    `rate` is negative, in which case the short PAYS).
    """
    return rate * qty * mark_price


def hedge_funding_usd(rate: float, qty: float, mark_price: float) -> float:
    """Amount the LONG hedge-perp leg pays this settlement (can be negative
    if `rate` is negative, in which case the long RECEIVES). Zero for a
    spot hedge — spot never funds, so callers must not call this for
    HedgeMode.SPOT positions.
    """
    return rate * qty * mark_price


def open_pair(
    symbol_spec: SymbolSpec,
    notional: float,
    hl_book: BookTop,
    hedge_price: float,
    ts: datetime,
    cfg: PaperConfig,
) -> list[Fill]:
    """Open a pair: SHORT `notional` worth of HL perp, LONG the same qty on
    the Binance hedge. `hedge_price` must be the ASK (buying).

    Both legs' qty is fixed by the HL leg's own fill price, so the two legs
    end up with EQUAL qty but not necessarily equal notional value (that
    gap — basis risk — is real and shows up in `basis` at close).
    """
    rng = _seeded_rng(symbol_spec.symbol, ts)

    # HL leg: opening SHORT. Maker = resting sell at the ask; taker = cross
    # the spread by hitting the bid, worse (lower) by slippage.
    hl_is_maker = rng.random() < cfg.maker_fill_prob
    if hl_is_maker:
        hl_price = hl_book.best_ask
        hl_fee_bps = cfg.hl_maker_bps
    else:
        hl_price = hl_book.best_bid * (1 - cfg.taker_slippage_bps / 10_000)
        hl_fee_bps = cfg.hl_taker_bps
    qty = notional / hl_price
    hl_fee_usd = notional * hl_fee_bps / 10_000  # qty * hl_price == notional by construction
    hl_fill = Fill(
        venue="hyperliquid", leg="hl_short", side="open",
        price=hl_price, qty=qty, fee_usd=hl_fee_usd, is_maker=hl_is_maker, ts=ts,
    )

    # Hedge leg: opening LONG at the given (ask) price. Spot has no
    # maker/taker split (one flat fee); perp draws its own maker/taker coin
    # flip and slips the given price on a taker fill.
    if symbol_spec.hedge_mode == HedgeMode.SPOT:
        hedge_is_maker = False
        hedge_fill_price = hedge_price
        hedge_fee_bps = cfg.spot_fee_bps
    else:
        hedge_is_maker = rng.random() < cfg.maker_fill_prob
        if hedge_is_maker:
            hedge_fill_price = hedge_price
            hedge_fee_bps = cfg.perp_maker_bps
        else:
            hedge_fill_price = hedge_price * (1 + cfg.taker_slippage_bps / 10_000)
            hedge_fee_bps = cfg.perp_taker_bps
    hedge_fee_usd = qty * hedge_fill_price * hedge_fee_bps / 10_000
    hedge_fill = Fill(
        venue="binance", leg="hedge_long", side="open",
        price=hedge_fill_price, qty=qty, fee_usd=hedge_fee_usd, is_maker=hedge_is_maker, ts=ts,
    )

    return [hl_fill, hedge_fill]


def close_pair(
    position: Position,
    hl_book: BookTop,
    hedge_price: float,
    ts: datetime,
    cfg: PaperConfig,
) -> list[Fill]:
    """Close a pair: BUY BACK the HL short, SELL the hedge long, both at
    `position.qty`. `hedge_price` must be the BID (selling).

    Unlike `open_pair`, no SymbolSpec is passed in — hedge_mode comes from
    `position.hedge_mode` (set at entry, carried on the Position itself),
    so closing works even if the symbol has since been removed from
    `cfg.universe` — a config change must never be able to strand an open
    position.
    """
    hedge_mode = position.hedge_mode
    rng = _seeded_rng(position.symbol, ts)
    qty = position.qty

    # HL leg: closing SHORT = buying back. Maker = resting buy at the bid;
    # taker = cross by hitting the ask, worse (higher) by slippage.
    hl_is_maker = rng.random() < cfg.maker_fill_prob
    if hl_is_maker:
        hl_price = hl_book.best_bid
        hl_fee_bps = cfg.hl_maker_bps
    else:
        hl_price = hl_book.best_ask * (1 + cfg.taker_slippage_bps / 10_000)
        hl_fee_bps = cfg.hl_taker_bps
    hl_fee_usd = qty * hl_price * hl_fee_bps / 10_000
    hl_fill = Fill(
        venue="hyperliquid", leg="hl_short", side="close",
        price=hl_price, qty=qty, fee_usd=hl_fee_usd, is_maker=hl_is_maker, ts=ts,
    )

    # Hedge leg: closing LONG = selling at the given (bid) price.
    if hedge_mode == HedgeMode.SPOT:
        hedge_is_maker = False
        hedge_fill_price = hedge_price
        hedge_fee_bps = cfg.spot_fee_bps
    else:
        hedge_is_maker = rng.random() < cfg.maker_fill_prob
        if hedge_is_maker:
            hedge_fill_price = hedge_price
            hedge_fee_bps = cfg.perp_maker_bps
        else:
            hedge_fill_price = hedge_price * (1 - cfg.taker_slippage_bps / 10_000)
            hedge_fee_bps = cfg.perp_taker_bps
    hedge_fee_usd = qty * hedge_fill_price * hedge_fee_bps / 10_000
    hedge_fill = Fill(
        venue="binance", leg="hedge_long", side="close",
        price=hedge_fill_price, qty=qty, fee_usd=hedge_fee_usd, is_maker=hedge_is_maker, ts=ts,
    )

    return [hl_fill, hedge_fill]


def liquidate_pair(
    position: Position,
    liq_price: float,
    hedge_price: float,
    ts: datetime,
    cfg: PaperConfig,
) -> list[Fill]:
    """Force-close both legs on a simulated liquidation: the HL short at
    the liquidation trigger price itself (not the live book — a
    liquidation fills at its trigger, not at a chosen resting/crossing
    price), the hedge leg at the given live price. Both legs pay TAKER
    fees — a liquidation is by definition a forced, aggressive fill, never
    a resting maker order. No RNG: unlike `open_pair`/`close_pair`, there
    is no maker/taker draw to make.
    """
    qty = position.qty
    hl_fee_usd = qty * liq_price * cfg.hl_taker_bps / 10_000
    hl_fill = Fill(
        venue="hyperliquid", leg="hl_short", side="close",
        price=liq_price, qty=qty, fee_usd=hl_fee_usd, is_maker=False, ts=ts,
    )

    hedge_fee_bps = (
        cfg.spot_fee_bps if position.hedge_mode == HedgeMode.SPOT else cfg.perp_taker_bps
    )
    hedge_fee_usd = qty * hedge_price * hedge_fee_bps / 10_000
    hedge_fill = Fill(
        venue="binance", leg="hedge_long", side="close",
        price=hedge_price, qty=qty, fee_usd=hedge_fee_usd, is_maker=False, ts=ts,
    )

    return [hl_fill, hedge_fill]
