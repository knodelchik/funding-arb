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

import json
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

# Binance's own funding intervals vary by symbol/regime (1h/2h/4h/8h --
# verified live: HYPEUSDT and PUMPUSDT both settle every 4h, not 8h) and
# are never wider than 8h. When no parquet warmup exists at all for the
# spread signal, the live top-up's cold-start lookback conservatively
# ASSUMES the widest (8h) interval throughout purely to size how far back
# to ask for `ema_periods` settlements' worth of calendar time -- this
# assumption is used ONLY for that lookback size, never for the actual
# per-event interval/annualization math in `_spread_signal`, which always
# derives each event's real interval from its own gap to the previous one.
# BN_WARMUP_MARGIN_HOURS is a flat extra-hours buffer on top of that,
# mirroring WARMUP_MARGIN_HOURS's role for the HL leg's own series.
BN_WARMUP_MARGIN_HOURS = 24

# A newest-PIT-Binance-event age (relative to the tick hour) beyond this
# many multiples of that event's OWN settlement interval means the data is
# stale, not merely "the live top-up happened to fail this tick" -- see
# `_binance_series_is_stale`.
BN_STALENESS_INTERVAL_MULTIPLE = 2

# Journaled right alongside POSITION_OPENED (see `_open_position`'s
# docstring for why it can't be merged into that event's own payload) --
# an event type journal.py doesn't know about, so `journal.replay` just
# skips it as inert audit trail, same as SIGNAL_SNAPSHOT.
POSITION_OPENED_SIGNAL = "position_opened_signal"


@dataclass
class _Signal:
    """`decision_apr` is whichever of `hl_apr`/`spread_apr` is actually
    compared against cfg.enter_apr this tick for a NEW entry (see `gate`);
    it can be None (a perp hedge's spread hasn't got `ema_periods` settled
    pairs yet) even when `hl_apr` isn't, in which case entries stay
    blocked but an OPEN position's own exit decision separately falls back
    to `hl_apr` (see `_tick`'s exit loop) so it never goes unsignalable.

    `hl_apr` is always the EMA APR of the HL leg's own absolute funding
    (spot's decision value; a perp-hedge symbol's fallback when Binance
    data is unavailable/stale, and its exit-only fallback when Binance
    data is on hand but not yet warmed up). `bn_apr`/`spread_apr` are
    populated only for a perp-hedge symbol with usable (fresh, non-stale)
    Binance funding data on hand — the EMA APR of Binance's own
    settlement rate, and of the HL-vs-Binance spread, respectively, both
    interval-aware and computed on the Binance settlement grid (see
    `_spread_signal`; they run on a different grid/horizon than `hl_apr`,
    so no `hl_apr - bn_apr == spread_apr` identity is expected).
    """

    decision_apr: float | None
    hl_apr: float | None
    bn_apr: float | None
    spread_apr: float | None
    gate: str  # "spread" | "absolute" — what decision_apr actually is, for the journal
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


def _binance_funding_parquet_path(cfg: PaperConfig, symbol: str) -> Any:
    return cfg.data_root / "binance" / "funding" / f"{symbol}.parquet"


def _binance_funding_series(
    symbol: str, binance_client: Any, cfg: PaperConfig, now: datetime, ema_periods: int
) -> list[tuple[datetime, float]]:
    """Merge a (possibly stale) parquet warmup with a live top-up fetch into
    a PIT Binance funding-rate series for the perp-hedge spread signal --
    settlement INTERVAL may vary by symbol (1h/2h/4h/8h); this function
    only collects the raw (time, rate) series, it never assumes a fixed
    spacing (see `_spread_signal`, which derives each event's own interval
    from its gap to the previous one).

    Never raises: a missing/unreadable parquet or a live-fetch failure
    degrades to whatever source remains (logged as a warning), not fatal —
    an empty result (no parquet, no live data either) is one of the two
    "Binance data unavailable" cases the caller falls back to the
    absolute-HL gate on (the other being `_binance_series_is_stale`).
    """
    from farb.data.client import FAPI_BASE

    merged: dict[datetime, float] = {}
    parquet_path = _binance_funding_parquet_path(cfg, symbol)
    if parquet_path.exists():
        try:
            df = pl.read_parquet(parquet_path)
            times = df["funding_time"].to_list()
            rates = df["funding_rate"].to_list()
            for t, r in zip(times, rates, strict=True):
                merged[t] = r
        except Exception as exc:  # noqa: BLE001 - a bad parquet must degrade, not crash
            journal.append_event(
                cfg, journal.WARNING,
                {"symbol": symbol, "message": f"Binance funding parquet unreadable: {exc}"},
                now,
            )

    # Cursor forward from the last known event when we have one (mirrors
    # _accrue_hedge_perp's cursoring); otherwise pull enough whole 8h
    # periods of history to warm the EMA from a cold start.
    if merged:
        start_ms = round((max(merged) + timedelta(milliseconds=1)).timestamp() * 1000)
    else:
        # Cold start: no cursor to resume from, so size the lookback off
        # the worst-case (widest, 8h) interval assumption -- see
        # BN_WARMUP_MARGIN_HOURS's docstring above. A symbol whose real
        # interval is narrower (e.g. 4h) just gets more history than the
        # strict minimum, never less.
        lookback_hours = ema_periods * ROLLING_WINDOW_HOURS + BN_WARMUP_MARGIN_HOURS
        start_ms = round((now - timedelta(hours=lookback_hours)).timestamp() * 1000)
    try:
        rows = binance_client.get_list(
            FAPI_BASE, "/fapi/v1/fundingRate",
            {"symbol": symbol, "startTime": start_ms, "limit": 1000},
        )
        for row in rows:
            t = datetime.fromtimestamp(int(row["fundingTime"]) / 1000, tz=UTC)
            merged[t] = float(row["fundingRate"])  # live is authoritative on conflict
    except Exception as exc:  # noqa: BLE001 - farb.data is another lane; never let it crash a tick
        journal.append_event(
            cfg, journal.WARNING,
            {"symbol": symbol, "message": f"Binance fundingRate fetch failed: {exc}"},
            now,
        )

    return sorted(merged.items())


def _spread_signal(
    hl_hourly: list[tuple[datetime, float]],
    binance_series: list[tuple[datetime, float]],
    as_of: datetime,
    ema_periods: int,
) -> tuple[float | None, float | None]:
    """(bn_apr, spread_apr): PIT EMA(ema_periods) of Binance's own
    ANNUALIZED settlement rate, and of the ANNUALIZED (HL funding over that
    same window MINUS that rate) — computed on the BINANCE SETTLEMENT GRID,
    INTERVAL-AWARE: Binance's own settlement interval is not fixed at 8h
    (verified live: HYPEUSDT/PUMPUSDT both settle every 4h) and can even
    switch mid-series (a funding-interval regime change). Treating it as a
    fixed 8h window regardless of the REAL gap mis-pairs a narrower-interval
    Binance rate against a too-wide HL sum and roughly HALVES the effective
    spread APR at a 4h interval — silently letting the bot enter/hold
    exactly what this gate exists to block.

    For each settled Binance event, its own interval is the gap to the
    PREVIOUS settled event (falling back to ROLLING_WINDOW_HOURS=8 hours
    for the very first event, where there is no prior settlement to measure
    from). It pairs with the HL hourly rates covering exactly
    (previous_settlement, this_settlement] — not a fixed trailing-8h
    window — summed and compared against the Binance rate over that SAME
    window, then annualized by (HOURS_PER_YEAR / that event's own interval)
    — a 4h event is annualized twice as aggressively as an 8h one, exactly
    as it should be. Each per-event value is annualized BEFORE the EMA
    runs over the sequence (not after, unlike `signal_apr`), since
    different events in the same series can carry different annualization
    factors.

    None, None if there aren't yet `ema_periods` settlements with full HL
    coverage of their own window. PIT by construction: a Binance event
    at/after `as_of` is dropped before anything else touches it.

    `hl_apr` (the hourly-grid EMA from `signal_apr`, used elsewhere for
    audit and as the absolute-HL fallback) and this function's `bn_apr` run
    on entirely different grids and horizons (hourly-EMA vs
    settlement-EMA, fixed vs per-event annualization) — the journaled
    `hl_apr - bn_apr == spread_apr` identity a prior version of this
    docstring claimed is NOT expected to hold, even approximately.
    """
    pit_bn = sorted((t, r) for t, r in binance_series if t < as_of)
    if not pit_bn:
        return None, None
    hl_sorted = sorted(hl_hourly)

    bn_annualized: list[float] = []
    spread_annualized: list[float] = []
    prev_settlement: datetime | None = None
    for settlement_t, bn_rate in pit_bn:
        if prev_settlement is None:
            interval_hours = float(ROLLING_WINDOW_HOURS)  # no prior settlement to measure from
            window_start = settlement_t - timedelta(hours=interval_hours)
        else:
            interval_hours = (settlement_t - prev_settlement).total_seconds() / 3600
            window_start = prev_settlement
        prev_settlement = settlement_t
        if interval_hours <= 0:
            continue  # malformed/duplicate settlement timestamps -- never divide by zero
        window = [r for t, r in hl_sorted if window_start < t <= settlement_t]
        if len(window) < round(interval_hours):
            continue  # not enough HL hourly coverage over this event's own window yet
        hl_sum = sum(window)
        periods_per_year = HOURS_PER_YEAR / interval_hours
        bn_annualized.append(bn_rate * periods_per_year)
        spread_annualized.append((hl_sum - bn_rate) * periods_per_year)

    if len(spread_annualized) < ema_periods:
        return None, None

    def _ema(values: list[float]) -> float:
        alpha = 2.0 / (ema_periods + 1)
        ema = values[0]
        for value in values[1:]:
            ema = alpha * value + (1 - alpha) * ema
        return ema

    return _ema(bn_annualized), _ema(spread_annualized)


def _binance_series_is_stale(bn_series: list[tuple[datetime, float]], as_of: datetime) -> bool:
    """True when the newest PIT-eligible Binance event is old enough
    (more than BN_STALENESS_INTERVAL_MULTIPLE x its own settlement
    interval before `as_of`) that building a spread signal from it would
    be silently trusting stale data just because the live top-up happened
    to fail this tick — e.g. monthy-old parquet rows left over after a
    live outage. The caller treats this exactly like "no Binance data at
    all": fall back to the absolute-HL gate, never build a spread EMA from
    it.
    """
    pit = sorted((t, r) for t, r in bn_series if t < as_of)
    if not pit:
        return True  # caught separately by the caller's own empty-series check too
    newest_t, _ = pit[-1]
    if len(pit) >= 2:
        interval_hours = (pit[-1][0] - pit[-2][0]).total_seconds() / 3600
    else:
        interval_hours = float(ROLLING_WINDOW_HOURS)  # no prior settlement to measure from
    if interval_hours <= 0:
        interval_hours = float(ROLLING_WINDOW_HOURS)
    age_hours = (as_of - newest_t).total_seconds() / 3600
    return age_hours > BN_STALENESS_INTERVAL_MULTIPLE * interval_hours


def _last_binance_gate_by_symbol(cfg: PaperConfig) -> dict[str, str]:
    """The most recent SIGNAL_SNAPSHOT `gate` value per symbol, read
    directly from the journal file (PaperState/`journal.replay` doesn't
    reconstruct SIGNAL_SNAPSHOT history — it's audit-trail-only, see
    journal.py's own docstring) — used ONLY to tell a NEW Binance-data
    outage (last tick's gate for this symbol was "spread") from an ONGOING
    one (already "absolute"), so the fallback warning fires once per
    outage rather than every tick it persists.
    """
    path = journal.journal_path(cfg)
    if not path.exists():
        return {}
    last: dict[str, str] = {}
    with path.open() as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            event = json.loads(line)
            if event["type"] == journal.SIGNAL_SNAPSHOT:
                last[event["payload"]["symbol"]] = event["payload"]["gate"]
    return last


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
    decision_apr: float,
    gate: str,
    notional: float,
    now: datetime,
) -> None:
    """`notional` is precomputed by the caller (via `_entry_notional`) at
    candidate-filtering time, not here — the MIN_ENTRY_NOTIONAL floor has
    to reject a candidate BEFORE it's ranked/slotted, not after opening a
    tiny position. `decision_apr`/`gate` are whichever signal (absolute-HL
    or spread) actually triggered the entry, per `_Signal.gate` — the
    Position model already carries `signal_apr_at_entry` (decision_apr),
    but not which GATE it came from. `gate` is NOT merged into the
    POSITION_OPENED payload itself: journal.py's `_apply_position_opened`
    parses that payload straight through `Position.model_validate` (the
    Position model has `extra="forbid"`, off-limits here), so any extra
    key there would break every future replay. It's journaled as its own
    POSITION_OPENED_SIGNAL event instead, right alongside, at the same
    timestamp — an audit-trail-only event type `journal.replay` safely
    ignores (same as SIGNAL_SNAPSHOT), so this needs no journal.py change.
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
        signal_apr_at_entry=decision_apr,
    )
    state.open_positions[spec.symbol] = position
    state.last_marks[spec.symbol] = {"hl_mark_px": hl_fill.price, "hedge_price": hedge_fill.price}
    journal.append_event(cfg, journal.POSITION_OPENED, position.model_dump(mode="json"), now)
    journal.append_event(
        cfg, POSITION_OPENED_SIGNAL,
        {"symbol": spec.symbol, "entry_gate": gate, "entry_decision_apr": decision_apr},
        now,
    )


def _finalize_close(
    cfg: PaperConfig,
    state: PaperState,
    position: Position,
    fills: list[Any],
    now: datetime,
    *,
    liquidated: bool,
    exit_gate: str | None = None,
    exit_decision_apr: float | None = None,
) -> None:
    """Shared decomposition + journal-write for both a normal close and a
    liquidation — they differ only in how `fills` were obtained and the
    `liquidated` flag, not in how the resulting PnL is computed or logged.

    `exit_gate`/`exit_decision_apr` (None for a liquidation, which isn't
    signal-driven) record which gate and APR actually triggered a
    signal-driven close, for auditability -- extends the JOURNAL payload
    only, journal.py's `ClosedPosition`/schema is untouched.
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
            "exit_gate": exit_gate,
            "exit_decision_apr": exit_decision_apr,
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
    *,
    exit_gate: str | None = None,
    exit_decision_apr: float | None = None,
) -> None:
    hl_book = hl_client.book_top(position.hl_coin)
    hedge_price = _binance_book_price(
        binance_client, position.symbol, position.hedge_mode, side="bid"
    )
    fills = broker.close_pair(position, hl_book, hedge_price, now, cfg)
    _finalize_close(
        cfg, state, position, fills, now, liquidated=False,
        exit_gate=exit_gate, exit_decision_apr=exit_decision_apr,
    )


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
    # Same union-with-open-positions logic for hedge_mode, but an OPEN
    # position's own hedge_mode (frozen at entry) ALWAYS wins over the
    # config's, even when the symbol is STILL in cfg.universe -- unlike
    # hl_coin above, hedge_mode can be edited in place for an existing
    # universe entry (a config flip PERP->SPOT), and re-gating an already
    # -open perp position onto the absolute-HL signal just because the
    # config now says SPOT would silently stop accounting for the Binance
    # funding it's actually still paying away.
    hedge_mode_by_symbol: dict[str, HedgeMode] = {s.symbol: s.hedge_mode for s in cfg.universe}
    for symbol, position in state.open_positions.items():
        hedge_mode_by_symbol[symbol] = position.hedge_mode

    # Only consulted for PERP symbols' fallback-warning dedup below; a full
    # extra journal scan, but a cheap one relative to `journal.replay`'s
    # own full pass moments ago, and SIGNAL_SNAPSHOT history isn't
    # reconstructed by replay() at all (see its own docstring).
    last_gate_by_symbol = _last_binance_gate_by_symbol(cfg)

    signals: dict[str, _Signal] = {}
    for symbol, hl_coin in hl_coin_by_symbol.items():
        hourly = _hourly_funding_series(symbol, hl_coin, hl_client, state, cfg, hour, now)
        ctx = asset_ctxs.get(hl_coin)
        hl_apr = signal_apr(hourly, hour, cfg.ema_periods)

        # SPOT hedges keep the absolute-HL gate untouched. PERP hedges gate
        # on the HL-vs-Binance SPREAD instead (a perp hedge pays away its
        # own Binance funding, so gating on HL alone overstates the edge
        # whenever Binance funding is also rich) — unless Binance data for
        # this symbol is unavailable (missing) or too old to trust (stale:
        # e.g. a live outage left only month-old parquet rows), in which
        # case this degrades to the same absolute-HL gate SPOT uses, never
        # a crash or a silent zero. The fallback warning fires once per
        # OUTAGE (a transition into "absolute"), not every tick it persists.
        bn_apr: float | None = None
        spread_apr: float | None = None
        if hedge_mode_by_symbol[symbol] == HedgeMode.PERP:
            bn_series = _binance_funding_series(symbol, binance_client, cfg, now, cfg.ema_periods)
            stale = bool(bn_series) and _binance_series_is_stale(bn_series, hour)
            if bn_series and not stale:
                bn_apr, spread_apr = _spread_signal(hourly, bn_series, hour, cfg.ema_periods)
                decision_apr, gate = spread_apr, "spread"
            else:
                if last_gate_by_symbol.get(symbol) != "absolute":
                    reason = "too stale to trust" if stale else "unavailable"
                    journal.append_event(
                        cfg, journal.WARNING,
                        {
                            "symbol": symbol,
                            "message": f"Binance funding data {reason} for the spread signal; "
                            "falling back to the absolute-HL gate",
                        },
                        now,
                    )
                decision_apr, gate = hl_apr, "absolute"
        else:
            decision_apr, gate = hl_apr, "absolute"

        signals[symbol] = _Signal(
            decision_apr=decision_apr,
            hl_apr=hl_apr,
            bn_apr=bn_apr,
            spread_apr=spread_apr,
            gate=gate,
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

        # An OPEN position must never become unsignalable for EXIT purposes:
        # if the primary decision_apr is unavailable this tick (a perp
        # hedge's spread hasn't got `ema_periods` settled pairs yet, or the
        # tick fell back to "absolute" already) but the absolute-HL EMA
        # itself IS available, exits fall back to that -- entries are a
        # SEPARATE, unmutated read of `sig.decision_apr` below and stay
        # blocked in this same condition, exactly as before.
        exit_apr, exit_gate = (sig.decision_apr, sig.gate) if sig is not None else (None, None)
        if exit_apr is None and sig is not None and sig.hl_apr is not None:
            exit_apr, exit_gate = sig.hl_apr, "absolute"

        left_universe = spec is None
        signal_exit = exit_apr is not None and exit_apr < cfg.exit_apr
        should_exit = left_universe or signal_exit
        if should_exit and can_price:
            _close_position(
                cfg, state, position, hl_client, binance_client, now,
                exit_gate=exit_gate, exit_decision_apr=exit_apr,
            )
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
    candidates: list[tuple[float, SymbolSpec, Any, float, str]] = []
    for spec in cfg.universe:
        if spec.symbol in state.open_positions:
            continue
        sig = signals[spec.symbol]
        gate_ok = sig.next_hour_funding is not None and sig.next_hour_funding > 0
        if sig.decision_apr is None or sig.ctx is None:
            decisions[spec.symbol] = "skip_no_signal"
        elif sig.decision_apr > cfg.enter_apr and gate_ok:
            # Compute notional now (not inside _open_position): the
            # MIN_ENTRY_NOTIONAL floor has to reject a thin-liquidity
            # candidate BEFORE ranking/slotting it, not after opening a
            # position too small for fees to ever be worth it.
            notional = _entry_notional(cfg, state, spec, now)
            if notional < MIN_ENTRY_NOTIONAL:
                decisions[spec.symbol] = "skip_below_min_notional"
            else:
                candidates.append((sig.decision_apr, spec, sig.ctx, notional, sig.gate))
        else:
            decisions[spec.symbol] = "skip_below_threshold"
    candidates.sort(key=lambda c: c[0], reverse=True)
    for decision_apr, spec, ctx, notional, gate in (
        candidates[:free_slots] if free_slots > 0 else []
    ):
        _open_position(
            cfg, state, spec, ctx, hl_client, binance_client, decision_apr, gate, notional, now
        )
        decisions[spec.symbol] = "enter"
    for _, spec, _, _, _ in candidates[free_slots:] if free_slots > 0 else candidates:
        decisions.setdefault(spec.symbol, "skip_no_slot")

    for spec in cfg.universe:
        sig = signals[spec.symbol]
        journal.append_event(
            cfg, journal.SIGNAL_SNAPSHOT,
            {
                "symbol": spec.symbol,
                "ema_apr": sig.decision_apr,
                "hl_apr": sig.hl_apr,
                "bn_apr": sig.bn_apr,
                "spread_apr": sig.spread_apr,
                "gate": sig.gate,
                "next_hour_funding": sig.next_hour_funding,
                "decision": decisions.get(spec.symbol, "hold"),
            },
            now,
        )

    journal.write_parquet_mirror(cfg)
