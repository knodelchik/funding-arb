"""M2 paper executor: `farb paper {tick,status,report}`.

`tick` is one hourly decision cycle: replay the journal, pull live HL asset
contexts, compute each tracked symbol's signal, accrue funding on open
positions, check liquidation/margin, exit/enter as the signal dictates, and
log everything. `status` and `report` never touch the network — both replay
the journal and render from what's already recorded (report's basis MtM on
open positions uses the LAST marks seen in the journal, not a fresh live
price).

Every live client (HLClient, BinanceClient) is dependency-injected with a
lazy-import default: `run_paper(action, cfg)` is what the CLI calls (no
clients), so real ones get constructed lazily inside `_tick`; tests always
pass fakes directly, so `farb.hl` never needs to exist for this module to
import or for its tests to run.

A position's `hl_coin`/`hedge_mode` live on the Position itself (not looked
up from `cfg.universe`), so a position stays fully priceable, accruable,
and closeable even after its symbol is removed from the config — a config
edit must never be able to strand an open position with no way to unwind.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import polars as pl

from farb.paper import broker, journal
from farb.paper.journal import PaperState
from farb.paper.models import HedgeMode, PaperConfig, Position, SymbolSpec
from farb.paper.report import render_report, render_status

if TYPE_CHECKING:
    from farb.hl.client import AssetCtx, HLClient

HOURS_PER_YEAR = 8766.0  # 365.25 * 24
# HL funds hourly; sum 8h to compare like-for-like with Binance's native 8h rate.
ROLLING_WINDOW_HOURS = 8
# The EMA is over per-8h-window values (not per-hour), so annualizing means
# "how many 8h windows per year", not "how many hours per year" — dividing
# by the window size here is what a prior bug got wrong (it multiplied a
# per-8h rate by HOURS_PER_YEAR directly, an 8x overstatement: e.g. HL's
# default hourly rate 1.25e-5 sums to a per-8h rate of 1e-4, which is the
# correct ~11% APR at x1095.75 but a wrong ~88% at x8766).
PERIODS_PER_YEAR = HOURS_PER_YEAR / ROLLING_WINDOW_HOURS
# Extra lookback margin (beyond the strict minimum) requested from the live
# HL history endpoint, so a missed tick or two doesn't starve the EMA warmup.
WARMUP_MARGIN_HOURS = 24

# HL's low-leverage tiers maintain roughly ~1.25% margin. A real per-symbol
# maintenance-margin tier table belongs in models.py (PaperConfig is
# another lane's contract) — this is a conservative module-level constant
# until that's wired through.
MAINT_MARGIN_FRACTION = 0.0125

# Position sizing liquidity cap: never risk more than 0.5% of the trailing
# 30d median quote volume on one position (so paper size stays realistic
# even when max_notional/capital would allow more).
LIQUIDITY_CAP_FRACTION = 0.005
LIQUIDITY_LOOKBACK_HOURS = 24 * 30
# Participation base: quote volume is summed over rolling 4h windows (not
# taken per-hour) before the median is computed. A position here is
# typically HELD for days, so its execution risk amortizes across many
# hours of trading, not one — capping against a single hour's volume (which
# can be spiky/thin) was too conservative in practice: week-1 paper entries
# got capped to sub-$2k notionals, where taker/maker fees on entry+exit ate
# the whole funding edge (e.g. one closed UNI trade: fees $5.74 vs funding
# $1.92 over 2.2 days held). A 4h base better reflects the volume a
# position can realistically trade into/out of over a meaningful slice of
# its holding period, while staying far shorter than that holding period
# (days) so it's still a genuinely conservative cap, not a loophole.
LIQUIDITY_WINDOW_HOURS = 4
# Hard floor: below this, fees on entry+exit reliably exceed any realistic
# funding edge over a typical holding period — not worth opening at all.
MIN_ENTRY_NOTIONAL = 2_000.0


@dataclass
class _Signal:
    ema_apr: float | None
    next_hour_funding: float | None
    hourly: list[tuple[datetime, float]]
    ctx: Any | None  # AssetCtx, kept loose to avoid a runtime farb.hl import


def run_paper(
    action: str,
    cfg: PaperConfig,
    *,
    hl_client: HLClient | None = None,
    binance_client: Any | None = None,
    now: datetime | None = None,
) -> None:
    """Entry point wired by the CLI (`farb paper {tick,status,report}`).

    `hl_client`/`binance_client`/`now` are injection points for tests, not
    part of the CLI's call shape — the CLI always calls with just
    (action, cfg), letting real clients and wall-clock `now` apply.
    """
    now = now or datetime.now(UTC)
    if action == "tick":
        _tick(cfg, hl_client, binance_client, now)
    elif action == "status":
        print(render_status(cfg, journal.replay(journal.journal_path(cfg))))
    elif action == "report":
        print(render_report(cfg, journal.replay(journal.journal_path(cfg))))
    else:
        raise ValueError(f"unknown paper action: {action!r}")


def _rolling_8h_sums(hourly: list[tuple[datetime, float]]) -> list[float]:
    """Trailing 8h-summed series: value[i] = sum of the 8 most recent
    hourly rates ending at `hourly[i]` (needs >=8 points to start).
    """
    return [
        sum(r for _, r in hourly[i - ROLLING_WINDOW_HOURS + 1 : i + 1])
        for i in range(ROLLING_WINDOW_HOURS - 1, len(hourly))
    ]


def signal_apr(
    hourly: list[tuple[datetime, float]], as_of: datetime, ema_periods: int
) -> float | None:
    """PIT EMA(ema_periods) of the trailing-8h-summed HL hourly funding,
    annualized x PERIODS_PER_YEAR (= HOURS_PER_YEAR / 8, since each summed
    value is a per-8h-window rate, not a per-hour one). None if there isn't
    enough settled history yet to warm up the EMA.

    PIT by construction: any (time, rate) pair with time >= `as_of` is
    dropped before anything else touches it, so a caller accidentally
    handing in a "future" rate can never move the result.
    """
    pit_hourly = sorted((t, r) for t, r in hourly if t < as_of)
    summed = _rolling_8h_sums(pit_hourly)
    if len(summed) < ema_periods:
        return None
    alpha = 2.0 / (ema_periods + 1)
    ema = summed[0]
    for value in summed[1:]:
        ema = alpha * value + (1 - alpha) * ema
    return ema * PERIODS_PER_YEAR


def _hl_history_parquet_path(cfg: PaperConfig, symbol: str) -> Any:
    return cfg.data_root / "venues" / "hyperliquid" / "funding" / f"{symbol}.parquet"


def _hourly_funding_series(
    symbol: str,
    hl_coin: str,
    hl_client: Any,
    state: PaperState,
    cfg: PaperConfig,
    hour: datetime,
    now: datetime,
) -> list[tuple[datetime, float]]:
    """Merge journal + parquet + a live HL fetch into one PIT hourly rate
    series. Never raises: a missing parquet file, or a live-fetch failure,
    degrades to whatever sources remain (journal alone in the worst case)
    rather than crashing the tick — a live-fetch failure is logged as a
    warning so it's visible without being fatal.

    Takes `symbol`/`hl_coin` directly (not a SymbolSpec) so it works for a
    position whose symbol has been removed from `cfg.universe` too.
    """
    merged: dict[datetime, float] = {}

    for iso, rate in state.hl_funding_history.get(symbol, {}).items():
        merged[datetime.fromisoformat(iso)] = rate

    parquet_path = _hl_history_parquet_path(cfg, symbol)
    if parquet_path.exists():
        try:
            df = pl.read_parquet(parquet_path)
            times = df["funding_time"].to_list()
            rates = df["funding_rate"].to_list()
            for t, r in zip(times, rates, strict=True):
                merged.setdefault(t, r)
        except Exception as exc:  # noqa: BLE001 - a bad parquet must degrade, not crash
            journal.append_event(
                cfg, journal.WARNING,
                {"symbol": symbol, "message": f"HL history parquet unreadable: {exc}"},
                now,
            )

    lookback_hours = cfg.ema_periods + ROLLING_WINDOW_HOURS + WARMUP_MARGIN_HOURS
    start_ms = round((hour - timedelta(hours=lookback_hours)).timestamp() * 1000)
    try:
        events = hl_client.funding_history(hl_coin, start_ms)
        for event in events:
            merged[event.time] = event.rate  # live is authoritative on conflict
    except Exception as exc:  # noqa: BLE001 - farb.hl is another lane; never let it crash a tick
        journal.append_event(
            cfg, journal.WARNING,
            {"symbol": symbol, "message": f"HL funding_history fetch failed: {exc}"},
            now,
        )

    return sorted(merged.items())


def _binance_bookticker(binance_client: Any, symbol: str, hedge_mode: HedgeMode) -> dict[str, Any]:
    from farb.data.client import FAPI_BASE, SPOT_BASE

    if hedge_mode == HedgeMode.SPOT:
        base, path = SPOT_BASE, "/api/v3/ticker/bookTicker"
    else:
        base, path = FAPI_BASE, "/fapi/v1/ticker/bookTicker"
    return binance_client.get(base, path, {"symbol": symbol})


def _binance_book_price(
    binance_client: Any, symbol: str, hedge_mode: HedgeMode, side: str
) -> float:
    """Best ask (buying, i.e. entering the long hedge) or bid (selling,
    i.e. exiting it) from Binance's public bookTicker — spot if the hedge
    is spot, else USDT-perp. Takes symbol/hedge_mode directly (not a
    SymbolSpec) so this works from a Position after universe removal too.
    """
    row = _binance_bookticker(binance_client, symbol, hedge_mode)
    return float(row["askPrice"] if side == "ask" else row["bidPrice"])


def _refresh_hedge_mark(
    cfg: PaperConfig, state: PaperState, position: Position, binance_client: Any, now: datetime
) -> None:
    """Fetch the CURRENT hedge-side mark (mid of bid/ask) and store it in
    last_marks, every tick an open position exists — not just when a
    hedge-perp funding event happens to settle.

    Without this, `last_marks[symbol]["hedge_price"]` only ever gets set at
    entry (and, for a perp hedge, refreshed incidentally by
    `_accrue_hedge_perp` roughly every 8h) — for a spot hedge it NEVER
    updates again before close. The unrealized basis MtM then compares a
    fresh, ticking HL mark against a stale, entry-frozen hedge price: one
    leg's price move shows up as "basis" in the display even though the
    position is genuinely hedged. On a fetch failure the previous mark is
    carried forward (not overwritten with a stale/wrong value) and a
    warning is logged.
    """
    try:
        row = _binance_bookticker(binance_client, position.symbol, position.hedge_mode)
        mid = (float(row["askPrice"]) + float(row["bidPrice"])) / 2
    except Exception as exc:  # noqa: BLE001 - a hedge-mark fetch failure must not crash the tick
        journal.append_event(
            cfg, journal.WARNING,
            {
                "symbol": position.symbol,
                "message": f"hedge mark fetch failed, carrying previous mark: {exc}",
            },
            now,
        )
        return
    # Journaled (not just set in-memory): PaperState is rebuilt fresh from
    # the journal on every tick via replay(), so a mark update that isn't
    # recorded here is invisible on the very next tick.
    journal.append_event(
        cfg, journal.HEDGE_MARK_REFRESHED, {"symbol": position.symbol, "hedge_price": mid}, now
    )
    state.last_marks.setdefault(position.symbol, {})["hedge_price"] = mid


def _liquidity_cap(cfg: PaperConfig, spec: SymbolSpec, cfg_now: datetime) -> float | None:
    """0.5% of the trailing-30d median 4-HOUR quote volume (close*volume,
    summed over rolling/bucketed LIQUIDITY_WINDOW_HOURS-hour windows —
    see that constant's comment for why 4h rather than 1h), from the
    collected Binance klines — spot klines for a spot hedge, perp klines
    for a perp hedge (the venue the hedge leg actually trades on). None (no
    cap from this source) if that parquet is missing/unreadable or has no
    rows in the lookback window; the caller is responsible for warning in
    that case.
    """
    dataset = "klines_spot_1h" if spec.hedge_mode == HedgeMode.SPOT else "klines_perp_1h"
    path = cfg.data_root / "binance" / dataset / f"{spec.symbol}.parquet"
    if not path.exists():
        return None
    try:
        df = pl.read_parquet(path)
        cutoff = cfg_now - timedelta(hours=LIQUIDITY_LOOKBACK_HOURS)
        recent = df.filter(pl.col("open_time") >= cutoff).sort("open_time")
        bucketed = (
            recent.with_row_index("_idx")
            .with_columns((pl.col("_idx") // LIQUIDITY_WINDOW_HOURS).alias("_bucket"))
            .with_columns((pl.col("close") * pl.col("volume")).alias("_qv"))
            .group_by("_bucket")
            .agg(pl.col("_qv").sum())
        )
        window_volumes = bucketed["_qv"].to_list()
    except Exception:  # noqa: BLE001 - a bad klines parquet must degrade, not crash
        return None
    if not window_volumes:
        return None
    return LIQUIDITY_CAP_FRACTION * statistics.median(window_volumes)


def _entry_notional(cfg: PaperConfig, state: PaperState, spec: SymbolSpec, now: datetime) -> float:
    caps = [spec.max_notional, cfg.capital_usd / cfg.max_positions]
    liquidity_cap = _liquidity_cap(cfg, spec, now)
    if liquidity_cap is not None:
        caps.append(liquidity_cap)
    else:
        journal.append_event(
            cfg, journal.WARNING,
            {
                "symbol": spec.symbol,
                "message": "liquidity cap unavailable (missing/empty klines parquet); "
                "using static caps only",
            },
            now,
        )
    return min(caps)


def _open_position(
    cfg: PaperConfig,
    state: PaperState,
    spec: SymbolSpec,
    ctx: Any,
    hl_client: Any,
    binance_client: Any,
    ema_apr: float,
    notional: float,
    now: datetime,
) -> None:
    """`notional` is precomputed by the caller (via `_entry_notional`) at
    candidate-filtering time, not here — the MIN_ENTRY_NOTIONAL floor has
    to reject a candidate BEFORE it's ranked/slotted, not after opening a
    tiny position.
    """
    hl_book = hl_client.book_top(spec.hl_coin)
    hedge_price = _binance_book_price(binance_client, spec.symbol, spec.hedge_mode, side="ask")
    fills = broker.open_pair(spec, notional, hl_book, hedge_price, now, cfg)
    hl_fill, hedge_fill = fills

    position = Position(
        symbol=spec.symbol,
        hl_coin=spec.hl_coin,
        hedge_mode=spec.hedge_mode,
        qty=hl_fill.qty,
        notional_entry=notional,
        entry_ts=now,
        entry_fills=fills,
        funding_received_usd=0.0,
        funding_paid_usd=0.0,
        signal_apr_at_entry=ema_apr,
    )
    state.open_positions[spec.symbol] = position
    state.last_marks[spec.symbol] = {"hl_mark_px": hl_fill.price, "hedge_price": hedge_fill.price}
    journal.append_event(cfg, journal.POSITION_OPENED, position.model_dump(mode="json"), now)


def _finalize_close(
    cfg: PaperConfig,
    state: PaperState,
    position: Position,
    fills: list[Any],
    now: datetime,
    *,
    liquidated: bool,
) -> None:
    """Shared decomposition + journal-write for both a normal close and a
    liquidation — they differ only in how `fills` were obtained and the
    `liquidated` flag, not in how the resulting PnL is computed or logged.
    """
    hl_exit, hedge_exit = fills
    entry_hl = next(f for f in position.entry_fills if f.leg == "hl_short")
    entry_hedge = next(f for f in position.entry_fills if f.leg == "hedge_long")
    basis = position.qty * (
        (hedge_exit.price - entry_hedge.price) - (hl_exit.price - entry_hl.price)
    )
    entry_fees = sum(f.fee_usd for f in position.entry_fills)
    exit_fees = sum(f.fee_usd for f in fills)
    fees = entry_fees + exit_fees
    funding_hl = position.funding_received_usd
    funding_hedge_paid = position.funding_paid_usd
    net = funding_hl - funding_hedge_paid + basis - fees

    state.open_positions.pop(position.symbol, None)
    state.last_marks[position.symbol] = {
        "hl_mark_px": hl_exit.price,
        "hedge_price": hedge_exit.price,
    }
    journal.append_event(
        cfg,
        journal.POSITION_CLOSED,
        {
            "symbol": position.symbol,
            "exit_ts": now.isoformat(),
            "funding_hl": funding_hl,
            "funding_hedge_paid": funding_hedge_paid,
            "basis": basis,
            "fees": fees,
            "net": net,
            "exit_fills": [f.model_dump(mode="json") for f in fills],
            "liquidated": liquidated,
        },
        now,
    )
    # Keep the in-memory state consistent with what replay would reconstruct
    # from the event just appended, for phases later in this same tick.
    closed = journal.ClosedPosition(
        symbol=position.symbol,
        entry_ts=position.entry_ts,
        exit_ts=now,
        notional_entry=position.notional_entry,
        funding_hl=funding_hl,
        funding_hedge_paid=funding_hedge_paid,
        basis=basis,
        fees=fees,
        net=net,
        liquidated=liquidated,
    )
    state.closed.append(closed)
    for k, v in (("funding_hl", funding_hl), ("funding_hedge_paid", funding_hedge_paid),
                 ("basis", basis), ("fees", fees), ("net", net)):
        state.realized_totals[k] += v


def _close_position(
    cfg: PaperConfig,
    state: PaperState,
    position: Position,
    hl_client: Any,
    binance_client: Any,
    now: datetime,
) -> None:
    hl_book = hl_client.book_top(position.hl_coin)
    hedge_price = _binance_book_price(
        binance_client, position.symbol, position.hedge_mode, side="bid"
    )
    fills = broker.close_pair(position, hl_book, hedge_price, now, cfg)
    _finalize_close(cfg, state, position, fills, now, liquidated=False)


def _liq_price(position: Position, cfg: PaperConfig) -> float:
    """Simulated HL liquidation trigger price for the SHORT leg: the mark
    price at which unrealized loss has consumed margin down to the
    maintenance threshold. Initial margin = notional/leverage; the price
    move that exhausts (leverage_headroom - maintenance) of it is the
    trigger — see MAINT_MARGIN_FRACTION's docstring above for the
    maintenance-margin assumption.

    Approximation direction: this computes maintenance margin as a
    fraction of the FIXED entry notional (qty * entry_price), where real
    HL maintenance margin is a fraction of the position's current MARK
    value (qty * mark_price), which is larger than entry notional at the
    trigger price (mark > entry for a short's adverse move). That makes
    this sim's trigger price ~0.5% further from entry than real HL's would
    be — i.e. this simulation liquidates slightly LATER than the real
    venue would. Net effect: conservative for realized PnL on a
    liquidation (a bit more adverse price move is absorbed before the
    forced close), optimistic for survival (a position here can ride out a
    move that would have triggered real HL's liquidation first).
    """
    entry_hl_price = next(f.price for f in position.entry_fills if f.leg == "hl_short")
    return entry_hl_price * (1 + 1 / cfg.hl_leverage - MAINT_MARGIN_FRACTION)


def _check_liquidation(
    cfg: PaperConfig,
    state: PaperState,
    position: Position,
    mark_px: float,
    binance_client: Any,
    now: datetime,
) -> bool:
    """Simulated forced liquidation of the HL short leg when `mark_px`
    reaches (or passes) the trigger price. Returns True if it fired — the
    caller must treat the position as already closed this tick (never
    silently ignored, per the spec's "hard simulated liquidation logging").
    """
    liq_price = _liq_price(position, cfg)
    if mark_px < liq_price:
        return False
    hedge_price = _binance_book_price(
        binance_client, position.symbol, position.hedge_mode, side="bid"
    )
    fills = broker.liquidate_pair(position, liq_price, hedge_price, now, cfg)
    journal.append_event(
        cfg, journal.LIQUIDATION,
        {"symbol": position.symbol, "mark_px": mark_px, "liq_price": liq_price},
        now,
    )
    _finalize_close(cfg, state, position, fills, now, liquidated=True)
    return True


def _accrue_hl(
    cfg: PaperConfig,
    state: PaperState,
    position: Position,
    hourly: list[tuple[datetime, float]],
    mark_px: float,
    hour: datetime,
    now: datetime,
) -> None:
    """Accrue every settled HL hour not yet applied, from (strictly after)
    entry up to (but excluding) the current, still-forming hour — catching
    up on more than one hour if a tick was missed, guarded per-hour against
    double-accrual.

    The guard is against `position.entry_ts` itself, not its hour bucket:
    an event settling AT the top of the entry hour (e.g. entry at 14:35,
    event at 14:00) settled BEFORE the position existed and must not be
    accrued to it.

    Whenever an hour IS accrued, its own FUNDING_ACCRUED event already
    carries `mark_px` and journal.py's replay applies it to last_marks —
    that's sufficient. But if there's nothing new to accrue this call (e.g.
    entry landed exactly on an hour boundary, leaving no accruable hour
    before the very next tick), the current mark would otherwise only be
    set in-memory on `state` here and never persisted: PaperState is
    rebuilt fresh from the journal on every tick, so an unjournaled mutation
    made now is invisible on the next replay. HL_MARK_REFRESHED covers
    exactly that gap.
    """
    symbol = position.symbol
    accrued_any = False
    for t, rate in hourly:
        if t <= position.entry_ts or t >= hour:
            continue
        key = (symbol, journal.hour_bucket(t).isoformat())
        if key in state.hl_accrual_keys:
            continue
        amount = broker.hl_funding_usd(rate, position.qty, mark_px)
        state.hl_accrual_keys.add(key)
        state.hl_funding_history.setdefault(symbol, {})[key[1]] = rate
        position.funding_received_usd += amount
        accrued_any = True
        journal.append_event(
            cfg, journal.FUNDING_ACCRUED,
            {
                "symbol": symbol, "leg": journal.LEG_HL, "key": key[1],
                "rate": rate, "mark_px": mark_px, "qty": position.qty, "amount_usd": amount,
            },
            now,
        )
    if not accrued_any:
        journal.append_event(
            cfg, journal.HL_MARK_REFRESHED, {"symbol": symbol, "hl_mark_px": mark_px}, now
        )
    state.last_marks.setdefault(symbol, {})["hl_mark_px"] = mark_px


def _accrue_hedge_perp(
    cfg: PaperConfig, state: PaperState, position: Position, binance_client: Any, now: datetime,
) -> None:
    """Fetch any Binance perp funding events settled since the last
    accrual for this hedge leg (its own ~8h schedule, independent of HL's
    hourly one) and apply them: LONG pays positive rate.

    The fetch cursor is bounded below by `position.entry_ts` as well as by
    the last accrual key: accrual keys are a symbol-wide set, so if this
    symbol was previously traded, closed, and re-entered, the OLD
    position's last accrual key would otherwise resume the cursor from
    before this position existed — charging it for funding that settled
    while flat. Each row is also individually guarded against this.
    """
    from farb.data.client import FAPI_BASE

    symbol = position.symbol
    existing_keys = [k for (sym, k) in state.hedge_accrual_keys if sym == symbol]
    entry_ms = round(position.entry_ts.timestamp() * 1000)
    if existing_keys:
        last_key_ms = round(datetime.fromisoformat(max(existing_keys)).timestamp() * 1000) + 1
        start_ms = max(entry_ms, last_key_ms)
    else:
        start_ms = entry_ms
    try:
        rows = binance_client.get_list(
            FAPI_BASE, "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": start_ms, "limit": 100},
        )
    except Exception as exc:  # noqa: BLE001 - a hedge-funding fetch failure must not crash the tick
        journal.append_event(
            cfg, journal.WARNING,
            {"symbol": symbol, "message": f"Binance hedge funding fetch failed: {exc}"}, now,
        )
        return

    for row in rows:
        funding_time = datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=UTC)
        if funding_time >= now or funding_time < position.entry_ts:
            continue  # PIT (not yet settled), or settled before this position existed
        key = (symbol, funding_time.isoformat())
        if key in state.hedge_accrual_keys:
            continue
        mark_price_raw = row.get("markPrice")
        if mark_price_raw:
            mark_px = float(mark_price_raw)
        else:
            fallback = state.last_marks.get(symbol, {}).get("hedge_price")
            if fallback is None:
                journal.append_event(
                    cfg, journal.WARNING,
                    {
                        "symbol": symbol,
                        "message": f"hedge funding at {key[1]} missing markPrice, "
                        "no fallback mark available, skipped",
                    },
                    now,
                )
                state.hedge_accrual_keys.add(key)  # never retry this row again
                continue
            mark_px = fallback
            journal.append_event(
                cfg, journal.WARNING,
                {
                    "symbol": symbol,
                    "message": f"hedge funding at {key[1]} missing markPrice, "
                    f"used last known hedge mark {mark_px}",
                },
                now,
            )
        rate = float(row["fundingRate"])
        amount = broker.hedge_funding_usd(rate, position.qty, mark_px)
        state.hedge_accrual_keys.add(key)
        position.funding_paid_usd += amount
        journal.append_event(
            cfg, journal.FUNDING_ACCRUED,
            {
                "symbol": symbol, "leg": journal.LEG_HEDGE, "key": key[1],
                "rate": rate, "mark_px": mark_px, "qty": position.qty, "amount_usd": amount,
            },
            now,
        )


def _check_margin(
    cfg: PaperConfig, state: PaperState, position: Position, mark_px: float, now: datetime
) -> None:
    entry_hl_price = next(f.price for f in position.entry_fills if f.leg == "hl_short")
    hl_margin = position.notional_entry / cfg.hl_leverage
    if hl_margin <= 0:
        return
    unrealized_pnl = position.qty * (entry_hl_price - mark_px)  # short: profits when price falls
    margin_ratio = (hl_margin + unrealized_pnl) / hl_margin
    if margin_ratio < cfg.margin_warn_buffer:
        journal.append_event(
            cfg, journal.WARNING,
            {
                "symbol": position.symbol,
                "message": "margin buffer breached",
                "margin_ratio": margin_ratio,
                "mark_px": mark_px,
            },
            now,
        )


def _tick(cfg: PaperConfig, hl_client: Any, binance_client: Any, now: datetime) -> None:
    state = journal.replay(journal.journal_path(cfg))
    hour = journal.hour_bucket(now)

    if state.last_tick_hour is not None and state.last_tick_hour == hour:
        journal.append_event(cfg, journal.TICK_SKIPPED_DUPLICATE, {"hour": hour.isoformat()}, now)
        return
    journal.append_event(cfg, journal.TICK_STARTED, {"hour": hour.isoformat()}, now)

    if hl_client is None:
        from farb.hl.client import HLClient

        hl_client = HLClient()
    if binance_client is None:
        from farb.data.client import BinanceClient

        binance_client = BinanceClient()

    asset_ctxs: dict[str, AssetCtx] = hl_client.asset_ctxs()
    universe_by_symbol = {s.symbol: s for s in cfg.universe}

    # Signals are computed for the UNION of universe symbols and symbols
    # with an open position: a position whose symbol left the universe
    # must still be priceable (and closeable) from the Position's own
    # hl_coin, not from a config lookup that may no longer find it.
    hl_coin_by_symbol: dict[str, str] = {s.symbol: s.hl_coin for s in cfg.universe}
    for symbol, position in state.open_positions.items():
        hl_coin_by_symbol.setdefault(symbol, position.hl_coin)

    signals: dict[str, _Signal] = {}
    for symbol, hl_coin in hl_coin_by_symbol.items():
        hourly = _hourly_funding_series(symbol, hl_coin, hl_client, state, cfg, hour, now)
        ctx = asset_ctxs.get(hl_coin)
        signals[symbol] = _Signal(
            ema_apr=signal_apr(hourly, hour, cfg.ema_periods),
            next_hour_funding=ctx.funding_hourly if ctx is not None else None,
            hourly=hourly,
            ctx=ctx,
        )

    decisions: dict[str, str] = {}

    # Accrue + liquidation-check + exit existing positions first (mirrors
    # the backtest: accrue, then decide whether this event is also the exit).
    for symbol in list(state.open_positions):
        position = state.open_positions[symbol]
        spec = universe_by_symbol.get(symbol)
        sig = signals.get(symbol)
        can_price = sig is not None and sig.ctx is not None

        if can_price:
            _accrue_hl(cfg, state, position, sig.hourly, sig.ctx.mark_px, hour, now)
        if position.hedge_mode == HedgeMode.PERP:
            _accrue_hedge_perp(cfg, state, position, binance_client, now)
        _refresh_hedge_mark(cfg, state, position, binance_client, now)

        liquidated = can_price and _check_liquidation(
            cfg, state, position, sig.ctx.mark_px, binance_client, now
        )
        if liquidated:
            decisions[symbol] = "liquidated"
            continue

        left_universe = spec is None
        signal_exit = sig is not None and sig.ema_apr is not None and sig.ema_apr < cfg.exit_apr
        should_exit = left_universe or signal_exit
        if should_exit and can_price:
            _close_position(cfg, state, position, hl_client, binance_client, now)
            decisions[symbol] = "exit"
        elif should_exit:
            journal.append_event(
                cfg, journal.WARNING,
                {
                    "symbol": symbol,
                    "message": "exit signaled but no live HL context to price the close",
                },
                now,
            )
            decisions[symbol] = "exit_blocked"
        else:
            decisions[symbol] = "hold"

    for symbol, position in state.open_positions.items():
        sig = signals.get(symbol)
        if sig is not None and sig.ctx is not None:
            _check_margin(cfg, state, position, sig.ctx.mark_px, now)

    free_slots = cfg.max_positions - len(state.open_positions)
    candidates: list[tuple[float, SymbolSpec, Any, float]] = []
    for spec in cfg.universe:
        if spec.symbol in state.open_positions:
            continue
        sig = signals[spec.symbol]
        gate_ok = sig.next_hour_funding is not None and sig.next_hour_funding > 0
        if sig.ema_apr is None or sig.ctx is None:
            decisions[spec.symbol] = "skip_no_signal"
        elif sig.ema_apr > cfg.enter_apr and gate_ok:
            # Compute notional now (not inside _open_position): the
            # MIN_ENTRY_NOTIONAL floor has to reject a thin-liquidity
            # candidate BEFORE ranking/slotting it, not after opening a
            # position too small for fees to ever be worth it.
            notional = _entry_notional(cfg, state, spec, now)
            if notional < MIN_ENTRY_NOTIONAL:
                decisions[spec.symbol] = "skip_below_min_notional"
            else:
                candidates.append((sig.ema_apr, spec, sig.ctx, notional))
        else:
            decisions[spec.symbol] = "skip_below_threshold"
    candidates.sort(key=lambda c: c[0], reverse=True)
    for ema_apr, spec, ctx, notional in candidates[:free_slots] if free_slots > 0 else []:
        _open_position(cfg, state, spec, ctx, hl_client, binance_client, ema_apr, notional, now)
        decisions[spec.symbol] = "enter"
    for _, spec, _, _ in candidates[free_slots:] if free_slots > 0 else candidates:
        decisions.setdefault(spec.symbol, "skip_no_slot")

    for spec in cfg.universe:
        sig = signals[spec.symbol]
        journal.append_event(
            cfg, journal.SIGNAL_SNAPSHOT,
            {
                "symbol": spec.symbol,
                "ema_apr": sig.ema_apr,
                "next_hour_funding": sig.next_hour_funding,
                "decision": decisions.get(spec.symbol, "hold"),
            },
            now,
        )

    journal.write_parquet_mirror(cfg)
