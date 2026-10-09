"""M2.5 dynamic universe scanner: finds, ranks, and hysteresis-gates new
paper-tradeable symbols from HL + Binance market data, on top of the
config's own PINNED `cfg.universe` entries (which this module never
touches — a pin is the operator's own call, not the scanner's business).
Whether removing a pin force-exits its position is decided in engine.py
(by comparing this tick's pin set against the PREVIOUS tick's, both
journaled directly there) — this module only ever reasons about the
SCANNER's own membership, never about pins.

Pipeline (`scan_candidates`): HL's metaAndAssetCtxs (all coins: funding,
OI, markPx) intersected with Binance's exchangeInfo (spot USDT TRADING
preferred, else USDT-perp TRADING) by EXACT name match `hl_coin + "USDT"`.
This exact-match-by-construction is deliberate, not an oversight: it
excludes any HL/Binance pair whose symbols don't literally agree (e.g. HL's
"kPEPE" never matches Binance's real "1000PEPEUSDT" — a unit-mismatch that
would silently mis-size a position if ever paired) by never considering it
a candidate in the first place, rather than by a separate denylist that
would need to be kept up to date by hand. Pins are excluded from candidacy
BEFORE ranking/capping (not after) — a pin competing for a scan slot could
otherwise starve the genuinely-addable candidates ranked just below it,
even though the scanner would never actually add a pinned symbol anyway.

`apply_hysteresis`'s add/remove thresholds operate on two DIFFERENT
notions on purpose: an EXISTING dynamic member's continued membership
depends only on whether it still PASSES THE FILTERS (`ScanResult.
qualifying`, unranked and uncapped) — rank/cap play no role in whether a
member gets to KEEP its seat. A NEW candidate's priority for a FREE slot
depends on rank (`qualifying` is sorted carry-desc, so ties for scarce
slots go to the best-ranked candidates deterministically) and the cap
applies only there, at add time.

Persistence: every piece of scanner state (dynamic membership, the
in-progress add/remove hysteresis counters, and the separate hedge
-delisting check's own counters) is derived from journal event types this
module owns (UNIVERSE_SCANNED, UNIVERSE_MEMBER_ADDED,
UNIVERSE_MEMBER_REMOVED, HEDGE_DELIST_CHECKED) via `replay_membership` —
mirroring `journal.replay()`'s own "rebuild from events, trust nothing
in-memory across a restart" discipline, kept here (not in journal.py,
off-limits for this lane) since these event types are local to the
scanner, not part of journal.py's own schema. `engine.py` owns the actual
journal I/O; every function in this module is a pure, directly-testable
decision given data already fetched by the caller.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from farb.paper.models import HedgeMode, SymbolSpec, UniverseScanConfig

# Mirrors farb.schemas._SYMBOL_RE exactly (same security boundary: symbol
# names from an exchange API become file names via dataset_path later, so
# an unvalidated one could carry "../" or similar). Kept as a local literal
# rather than importing that private name — this module validates the
# scanner's OWN constructed candidate symbol, a narrower and earlier check
# than dataset_path's, not a replacement for it.
_SYMBOL_RE = re.compile(r"[A-Z0-9]{2,20}")

# A per-1h fraction annualizes by the number of hours in a year — this is
# the scanner's OWN simple "mean hourly rate x hours/year" carry heuristic
# for membership screening, deliberately NOT the paper engine's EMA-based
# trading signal (farb.paper.engine.signal_apr) -- the two measure
# different things (a 30d screening average vs. a live decision EMA) and
# must not be confused for each other.
HOURS_PER_YEAR = 8766.0  # 365.25 * 24

# How close the first HL funding event actually returned must be to the
# probe's requested start for a coin to count as having that much history
# -- a coin whose real history starts well AFTER the requested start is
# younger than min_history_days, even though SOME events came back.
AGE_PROBE_TOLERANCE_DAYS = 2

# How far past the age probe's own start the bounded fetch extends --
# only the EARLIEST event near the start matters for the age check, so
# this just needs to be wide enough to reliably catch at least one event
# if the coin's real history reaches back that far (HL settles hourly,
# so even a single day comfortably contains one), not wide enough to
# reach `now`.
AGE_PROBE_FETCH_WINDOW_DAYS = 3

TRAILING_CARRY_LOOKBACK_DAYS = 30

# More than this fraction of per-coin funding_history probes failing in
# one scan means the scan itself is unreliable (not just a few unlucky
# coins) -- the whole scan is treated as FAILED (see ScanProbeDegradedError).
PROBE_FAILURE_FRACTION_LIMIT = 0.5

UNIVERSE_SCANNED = "universe_scanned"
UNIVERSE_MEMBER_ADDED = "universe_member_added"
UNIVERSE_MEMBER_REMOVED = "universe_member_removed"
HEDGE_DELIST_CHECKED = "hedge_delist_checked"

SPOT_EXCHANGE_INFO_PATH = "/api/v3/exchangeInfo"
PERP_EXCHANGE_INFO_PATH = "/fapi/v1/exchangeInfo"

# The hedge-delisting check runs on this cadence REGARDLESS of whether
# universe_scan itself is enabled (it's just 2 exchangeInfo calls) -- an
# open position must never be exposed to a delisted hedge indefinitely
# just because the scanner feature is off.
HEDGE_DELIST_CHECK_INTERVAL_HOURS = 24
# A hedge must look delisted on this many CONSECUTIVE checks before the
# position is actually force-closed -- the first sighting is logged as a
# warning only (a transient/flaky exchangeInfo response must not close a
# healthy position).
HEDGE_DELIST_CONFIRM_CHECKS = 2
# The sanity breaker judges the exchangeInfo RESPONSE ITSELF, not how many
# open positions happen to look delisted (with few/one open positions, a
# single GENUINE delisting can legitimately make 100% of them look
# delisted -- that must still force-close, not get waved off as "probably
# a bad response"). A response is degenerate if either venue's symbol
# count collapses relative to its own last known-good count: below
# HEDGE_DELIST_ABSOLUTE_FLOOR outright (a real exchange always lists far
# more than this many USDT pairs), or below HEDGE_DELIST_PREVIOUS_FRACTION
# of the last trustworthy count, whichever floor is higher. On a trip,
# nothing closes and the per-symbol streaks are left exactly as they
# were; the known-good counts baseline ALSO carries forward unchanged
# (never regresses to a bad response's collapsed count), so a later
# equally-bad response can't look "normal" by comparison.
HEDGE_DELIST_ABSOLUTE_FLOOR = 200
HEDGE_DELIST_PREVIOUS_FRACTION = 0.5


class ScanProbeDegradedError(RuntimeError):
    """More than PROBE_FAILURE_FRACTION_LIMIT of this scan's per-coin
    funding_history probes failed -- the whole scan is unreliable, not
    just a few unlucky coins. Raised so the caller's existing "total scan
    failure" handling (warn, don't journal UNIVERSE_SCANNED, don't touch
    streaks) applies uniformly, the same as an exchangeInfo outage.
    """


@dataclass
class BinanceSymbols:
    """The set of symbols currently TRADING, quoted in USDT, on each
    Binance venue — spot and USDT-margined perpetual futures.
    """

    spot: set[str] = field(default_factory=set)
    perp: set[str] = field(default_factory=set)


def _tradeable_symbols_from_exchange_info(
    response: Any, *, require_contract_type: str | None
) -> set[str]:
    """Symbols with status=="TRADING" and quoteAsset=="USDT" from one
    Binance exchangeInfo response — `require_contract_type` additionally
    filters to e.g. "PERPETUAL" on the futures endpoint (which also lists
    delivery/quarterly contracts we never want to hedge with). A single
    malformed entry is skipped, not fatal to the rest of the response.
    """
    symbols = response["symbols"]
    result: set[str] = set()
    for entry in symbols:
        try:
            if entry.get("status") != "TRADING" or entry.get("quoteAsset") != "USDT":
                continue
            if require_contract_type is not None and entry.get("contractType") != (
                require_contract_type
            ):
                continue
            result.add(entry["symbol"])
        except (TypeError, AttributeError):
            continue  # one malformed entry must not drop the rest of the response
    return result


def fetch_binance_tradeable_symbols(binance_client: Any) -> BinanceSymbols:
    """Fetch BOTH Binance exchangeInfo endpoints. Unlike the per-entry
    tolerance above, a TOTAL fetch failure (network error, malformed top
    -level response) is NOT swallowed here — it propagates, so the caller
    (engine.py) can treat "couldn't reach Binance at all" as a whole-scan
    (or whole-delist-check) failure: warn and keep the previous state,
    never silently treat "fetch failed" as "nothing is tradeable anymore"
    (which would look identical to a mass hard-delisting otherwise).
    """
    from farb.data.client import FAPI_BASE, SPOT_BASE

    spot_response = binance_client.get(SPOT_BASE, SPOT_EXCHANGE_INFO_PATH, {})
    perp_response = binance_client.get(FAPI_BASE, PERP_EXCHANGE_INFO_PATH, {})
    return BinanceSymbols(
        spot=_tradeable_symbols_from_exchange_info(spot_response, require_contract_type=None),
        perp=_tradeable_symbols_from_exchange_info(
            perp_response, require_contract_type="PERPETUAL"
        ),
    )


@dataclass
class ScanResult:
    """One scan's output. `qualifying` is EVERY symbol that passes every
    filter (pins already excluded), sorted by trailing carry APR
    descending — deliberately NOT capped at `max_dynamic_members` here:
    capping belongs to `apply_hysteresis`'s add-time decision (existing
    members occupy slots too, not just this scan's own candidate count),
    and an EXISTING member's continued qualification must never depend on
    whether it happens to rank above or below an arbitrary cap.
    """

    qualifying: list[SymbolSpec] = field(default_factory=list)
    dropped_invalid_names: list[str] = field(default_factory=list)
    probe_attempts: int = 0
    probe_failures: int = 0


def scan_candidates(
    asset_ctxs: dict[str, Any],
    binance_symbols: BinanceSymbols,
    hl_client: Any,
    scan_cfg: UniverseScanConfig,
    pinned_symbols: set[str],
    now: datetime,
) -> ScanResult:
    """Filter + rank candidate symbols from already-fetched market data.

    `asset_ctxs` (HL's metaAndAssetCtxs, keyed by HL coin name) and
    `binance_symbols` are fetched ONCE by the caller and handed in here
    (not re-fetched) — both are needed elsewhere in the same tick anyway
    (asset_ctxs for the trading signal itself, binance_symbols for the
    separate hedge-delisting check), so this function never makes its own
    exchangeInfo/asset_ctxs call. `hl_client` is still needed directly for
    the per-coin `funding_history` age/carry probes below, which have no
    cheaper batched equivalent.

    `pinned_symbols` (config `universe` entries) are excluded FIRST, before
    any filter runs — a pin is never a scanner candidate, so there's no
    point spending a network probe on one either.

    Filter order is cheapest-first (pin exclusion, symbol-name validity,
    Binance presence, open interest) before the two per-coin network
    probes: AGE (a SINGLE BOUNDED fetch — `funding_history`'s `end_ms`
    stops it a few days past the requested start instead of paginating
    all the way to the present; only the earliest returned event's time
    matters) and CARRY (a SEPARATE, always-30-day window — never
    entangled with however large `min_history_days` is configured, so it
    stays bounded regardless). Splitting these means a coin that fails
    the cheap age check never pays for the carry probe at all. If more
    than PROBE_FAILURE_FRACTION_LIMIT of all attempted probes fail,
    `ScanProbeDegradedError` is raised — the caller treats the WHOLE scan
    as failed, not just those individual coins.
    """
    qualifying: list[tuple[float, SymbolSpec]] = []
    dropped_invalid_names: list[str] = []
    probe_attempts = 0
    probe_failures = 0

    for hl_coin, ctx in asset_ctxs.items():
        symbol = f"{hl_coin}USDT"
        if symbol in pinned_symbols:
            continue  # a pin is the operator's call, never a scanner candidate

        if not _SYMBOL_RE.fullmatch(symbol):
            dropped_invalid_names.append(symbol)
            continue

        if symbol in binance_symbols.spot:
            hedge_mode = HedgeMode.SPOT
        elif symbol in binance_symbols.perp:
            hedge_mode = HedgeMode.PERP
        else:
            continue  # no Binance hedge at all -- not a candidate

        oi_usd = ctx.open_interest * ctx.mark_px
        if oi_usd < scan_cfg.min_open_interest_usd:
            continue

        probe_attempts += 1
        try:
            # AGE: a single BOUNDED fetch -- (min_history_days + tolerance)
            # back, ending AGE_PROBE_FETCH_WINDOW_DAYS later. Probing from
            # exactly min_history_days back (as a prior version of this
            # function did) and then allowing the usual +/-tolerance slack
            # means a coin with as little as (min_history_days - tolerance)
            # of REAL history could still pass; probing from
            # (min_history_days + tolerance) back and applying the SAME
            # tolerance against THAT point closes that undercount -- the
            # coin must show at least min_history_days of real history,
            # not min_history_days-minus-slack. Only the EARLIEST event
            # near the start matters, so `end_ms` stops the fetch a few
            # days later instead of paginating all the way to `now`.
            age_probe_start = now - timedelta(
                days=scan_cfg.min_history_days + AGE_PROBE_TOLERANCE_DAYS
            )
            age_probe_end = age_probe_start + timedelta(days=AGE_PROBE_FETCH_WINDOW_DAYS)
            age_events = hl_client.funding_history(
                hl_coin,
                round(age_probe_start.timestamp() * 1000),
                round(age_probe_end.timestamp() * 1000),
            )
        except Exception:  # noqa: BLE001 - one coin's fetch failure must not drop the whole scan
            probe_failures += 1
            continue
        pit_age_events = sorted(e.time for e in age_events if e.time < now)
        if not pit_age_events:
            continue
        first_event_time = pit_age_events[0]
        if first_event_time - age_probe_start > timedelta(days=AGE_PROBE_TOLERANCE_DAYS):
            continue  # real history doesn't reach back far enough -- too young

        try:
            # CARRY: always exactly the trailing 30d, independent of
            # min_history_days -- bounded regardless of how that's
            # configured, and a SEPARATE fetch from the age probe above
            # (narrower windows cost less than one big combined one when
            # min_history_days differs from 30).
            carry_probe_start = now - timedelta(days=TRAILING_CARRY_LOOKBACK_DAYS)
            carry_events = hl_client.funding_history(
                hl_coin, round(carry_probe_start.timestamp() * 1000)
            )
        except Exception:  # noqa: BLE001 - one coin's fetch failure must not drop the whole scan
            probe_failures += 1
            continue
        trailing_rates = [e.rate for e in carry_events if e.time < now]
        if not trailing_rates:
            continue
        carry_apr = statistics.mean(trailing_rates) * HOURS_PER_YEAR
        if carry_apr < scan_cfg.min_trailing_carry_apr:
            continue

        spec = SymbolSpec(
            symbol=symbol, hl_coin=hl_coin, hedge_mode=hedge_mode,
            max_notional=scan_cfg.default_max_notional,
        )
        qualifying.append((carry_apr, spec))

    if probe_attempts > 0 and probe_failures / probe_attempts > PROBE_FAILURE_FRACTION_LIMIT:
        raise ScanProbeDegradedError(
            f"{probe_failures}/{probe_attempts} per-coin funding_history probes failed"
        )

    qualifying.sort(key=lambda c: c[0], reverse=True)
    return ScanResult(
        qualifying=[spec for _, spec in qualifying],
        dropped_invalid_names=dropped_invalid_names,
        probe_attempts=probe_attempts,
        probe_failures=probe_failures,
    )


@dataclass
class UniverseMembership:
    """Reconstructed from the journal's own scanner event types by
    `replay_membership` — the scanner's entire working memory, exactly
    like `journal.PaperState` is reconstructed for the paper engine's own
    events. Nothing here is ever trusted across a process restart except
    what came out of that replay.
    """

    dynamic_members: dict[str, SymbolSpec] = field(default_factory=dict)
    # Consecutive-scan counters as of the LATEST scan -- each
    # UNIVERSE_SCANNED event carries a full snapshot (not a delta), so
    # replay just takes the most recent one verbatim rather than folding
    # increments across many scan events.
    qualify_streak: dict[str, int] = field(default_factory=dict)
    disqualify_streak: dict[str, int] = field(default_factory=dict)
    last_scan_hour: datetime | None = None
    scan_count: int = 0
    # The separate hedge-delisting check's own counters/clock -- these
    # advance on their OWN cadence (HEDGE_DELIST_CHECK_INTERVAL_HOURS),
    # independent of whether a scan happened to run too.
    delist_streak: dict[str, int] = field(default_factory=dict)
    last_delist_check_hour: datetime | None = None
    # The last KNOWN-GOOD (non-degenerate) exchangeInfo symbol counts --
    # the sanity breaker's own baseline. Only ever updated from a check
    # that itself passed the breaker; a degenerate response never moves
    # this (see `check_hedge_delisting`).
    last_delist_spot_count: int = 0
    last_delist_perp_count: int = 0


def replay_membership(events: list[dict[str, Any]]) -> UniverseMembership:
    """Reconstruct dynamic-universe membership and the hedge-delisting
    check's own state from a flat list of raw `{"type": ..., "payload":
    ...}` journal events, in order. `events` is the WHOLE journal (same
    shape `journal.replay` consumes from the raw file) — event types this
    module doesn't own are silently skipped, mirroring `journal.replay`'s
    own tolerance of SIGNAL_SNAPSHOT-style audit-only events it doesn't
    reconstruct state from either.
    """
    state = UniverseMembership()
    for event in events:
        event_type = event.get("type")
        payload = event.get("payload", {})
        if event_type == UNIVERSE_SCANNED:
            state.last_scan_hour = datetime.fromisoformat(payload["hour"])
            state.scan_count += 1
            state.qualify_streak = dict(payload.get("qualify_streak", {}))
            state.disqualify_streak = dict(payload.get("disqualify_streak", {}))
        elif event_type == UNIVERSE_MEMBER_ADDED:
            spec = SymbolSpec.model_validate(payload["spec"])
            state.dynamic_members[spec.symbol] = spec
        elif event_type == UNIVERSE_MEMBER_REMOVED:
            state.dynamic_members.pop(payload["symbol"], None)
        elif event_type == HEDGE_DELIST_CHECKED:
            state.last_delist_check_hour = datetime.fromisoformat(payload["hour"])
            state.delist_streak = dict(payload.get("delist_streak", {}))
            state.last_delist_spot_count = payload.get(
                "spot_count", state.last_delist_spot_count
            )
            state.last_delist_perp_count = payload.get(
                "perp_count", state.last_delist_perp_count
            )
    return state


@dataclass
class HysteresisUpdate:
    """One scan's worth of hysteresis bookkeeping — the NEXT full streak
    snapshot (to journal verbatim in the UNIVERSE_SCANNED event) plus the
    adds/removes that cross their thresholds THIS scan.
    """

    qualify_streak: dict[str, int] = field(default_factory=dict)
    disqualify_streak: dict[str, int] = field(default_factory=dict)
    to_add: list[SymbolSpec] = field(default_factory=list)
    to_remove: list[tuple[str, str]] = field(default_factory=list)  # (symbol, reason)


def apply_hysteresis(
    membership: UniverseMembership, scan_result: ScanResult, scan_cfg: UniverseScanConfig,
) -> HysteresisUpdate:
    """Decide this scan's adds/removes from the CURRENT reconstructed
    membership and this scan's fresh (pin-free, ranked, UNCAPPED)
    qualifying list. Pure and replay-safe: callers journal `to_add`/
    `to_remove` as their own events and the returned `qualify_streak`/
    `disqualify_streak` verbatim as the next UNIVERSE_SCANNED snapshot —
    this function never touches a journal.

    An EXISTING member's survival depends only on whether its symbol is
    still IN `scan_result.qualifying` at all (rank/cap play no role — a
    member that passes every filter but ranks, say, 8th when the cap is
    5 keeps its seat and accrues NO disqualify streak). A NEW candidate's
    priority for a free slot DOES depend on rank: `scan_result.qualifying`
    is already sorted carry-desc, and this function walks it in that
    order, so when several candidates cross `add_after_scans` in the same
    scan but only some slots are free, the best-ranked ones win
    deterministically, not whichever happened to come first in set
    iteration order.
    """
    qualifying_symbols = {c.symbol for c in scan_result.qualifying}

    new_qualify: dict[str, int] = {}
    to_add: list[SymbolSpec] = []
    for spec in scan_result.qualifying:  # already pin-free and rank-ordered
        symbol = spec.symbol
        if symbol in membership.dynamic_members:
            continue  # already a member -- nothing left to qualify toward
        streak = membership.qualify_streak.get(symbol, 0) + 1
        new_qualify[symbol] = streak
        if (
            streak >= scan_cfg.add_after_scans
            and len(membership.dynamic_members) + len(to_add) < scan_cfg.max_dynamic_members
        ):
            to_add.append(spec)

    new_disqualify: dict[str, int] = {}
    to_remove: list[tuple[str, str]] = []
    for symbol in membership.dynamic_members:
        if symbol in qualifying_symbols:
            continue  # still PASSES THE FILTERS -- rank/cap is irrelevant here
        streak = membership.disqualify_streak.get(symbol, 0) + 1
        new_disqualify[symbol] = streak
        if streak >= scan_cfg.remove_after_scans:
            to_remove.append((symbol, "degraded"))

    return HysteresisUpdate(
        qualify_streak=new_qualify, disqualify_streak=new_disqualify,
        to_add=to_add, to_remove=to_remove,
    )


@dataclass
class DelistCheckResult:
    """One hedge-delisting check's outcome. `newly_confirmed` symbols
    must be force-closed NOW (this is their HEDGE_DELIST_CONFIRM_CHECKS-th
    consecutive sighting); `first_sighting` symbols are logged as a
    warning only, not yet closed. `delist_streak` is the NEXT full
    snapshot to journal verbatim (same discipline as the add/remove
    streaks). `spot_count`/`perp_count` are the counts to journal/carry
    forward as the sanity breaker's OWN next baseline -- the current
    response's counts when trustworthy, or the membership's PREVIOUS
    (still trustworthy) counts unchanged when this check tripped the
    breaker. `sanity_breaker_tripped` means the check itself was
    distrusted and skipped entirely this time -- `delist_streak` is then
    the UNCHANGED previous snapshot (never silently reset by a bad
    response), and `newly_confirmed`/`first_sighting` are both empty.
    """

    newly_confirmed: list[str] = field(default_factory=list)
    first_sighting: list[str] = field(default_factory=list)
    delist_streak: dict[str, int] = field(default_factory=dict)
    sanity_breaker_tripped: bool = False
    spot_count: int = 0
    perp_count: int = 0


def check_hedge_delisting(
    positions: dict[str, HedgeMode],
    binance_symbols: BinanceSymbols,
    membership: UniverseMembership,
) -> DelistCheckResult:
    """Which OPEN positions' hedges are no longer TRADING on Binance AT
    ALL, venue-specific: a SPOT-hedged position is checked against the
    SPOT tradeable set, a PERP-hedged one against the PERP set (never the
    other venue's set, which would read as "delisted" for a perfectly
    fine hedge simply traded on the other venue).

    The sanity breaker judges the RESPONSE ITSELF (its spot/perp symbol
    counts against their own last known-good baseline), NOT what fraction
    of open positions look delisted -- with few (or one) open positions, a
    single GENUINE delisting legitimately makes 100% of them look
    delisted, and that must still be allowed to confirm and force-close,
    not get waved off by a position-count heuristic. A response is
    degenerate if EITHER venue's count falls below
    max(HEDGE_DELIST_ABSOLUTE_FLOOR, HEDGE_DELIST_PREVIOUS_FRACTION x that
    venue's last known-good count) -- on a trip, nothing closes, the
    per-symbol streaks are left exactly as they were, and the baseline
    counts themselves are carried forward UNCHANGED (never regress to the
    bad response's collapsed count).
    """
    if not positions:
        # Nothing open to protect -- a pure no-op, no breaker judgment at
        # all (and no point updating the baseline counts for a check that
        # has no positions to apply them to).
        return DelistCheckResult(
            delist_streak=dict(membership.delist_streak),
            spot_count=membership.last_delist_spot_count,
            perp_count=membership.last_delist_perp_count,
        )

    spot_count = len(binance_symbols.spot)
    perp_count = len(binance_symbols.perp)
    spot_floor = max(
        HEDGE_DELIST_ABSOLUTE_FLOOR,
        HEDGE_DELIST_PREVIOUS_FRACTION * membership.last_delist_spot_count,
    )
    perp_floor = max(
        HEDGE_DELIST_ABSOLUTE_FLOOR,
        HEDGE_DELIST_PREVIOUS_FRACTION * membership.last_delist_perp_count,
    )
    degenerate = spot_count < spot_floor or perp_count < perp_floor
    next_spot_count = membership.last_delist_spot_count if degenerate else spot_count
    next_perp_count = membership.last_delist_perp_count if degenerate else perp_count

    if degenerate:
        return DelistCheckResult(
            delist_streak=dict(membership.delist_streak),
            sanity_breaker_tripped=True,
            spot_count=next_spot_count, perp_count=next_perp_count,
        )

    currently_delisted = set()
    for symbol, hedge_mode in positions.items():
        tradeable = binance_symbols.spot if hedge_mode == HedgeMode.SPOT else binance_symbols.perp
        if symbol not in tradeable:
            currently_delisted.add(symbol)

    new_streak: dict[str, int] = {}
    newly_confirmed: list[str] = []
    first_sighting: list[str] = []
    for symbol in currently_delisted:
        streak = membership.delist_streak.get(symbol, 0) + 1
        new_streak[symbol] = streak
        if streak >= HEDGE_DELIST_CONFIRM_CHECKS:
            newly_confirmed.append(symbol)
        else:
            first_sighting.append(symbol)

    return DelistCheckResult(
        newly_confirmed=newly_confirmed, first_sighting=first_sighting, delist_streak=new_streak,
        spot_count=next_spot_count, perp_count=next_perp_count,
    )


__all__ = [
    "AGE_PROBE_FETCH_WINDOW_DAYS",
    "AGE_PROBE_TOLERANCE_DAYS",
    "HEDGE_DELIST_ABSOLUTE_FLOOR",
    "HEDGE_DELIST_CHECKED",
    "HEDGE_DELIST_CHECK_INTERVAL_HOURS",
    "HEDGE_DELIST_CONFIRM_CHECKS",
    "HEDGE_DELIST_PREVIOUS_FRACTION",
    "HOURS_PER_YEAR",
    "PERP_EXCHANGE_INFO_PATH",
    "PROBE_FAILURE_FRACTION_LIMIT",
    "SPOT_EXCHANGE_INFO_PATH",
    "TRAILING_CARRY_LOOKBACK_DAYS",
    "UNIVERSE_MEMBER_ADDED",
    "UNIVERSE_MEMBER_REMOVED",
    "UNIVERSE_SCANNED",
    "BinanceSymbols",
    "DelistCheckResult",
    "HysteresisUpdate",
    "ScanProbeDegradedError",
    "ScanResult",
    "UniverseMembership",
    "apply_hysteresis",
    "check_hedge_delisting",
    "fetch_binance_tradeable_symbols",
    "replay_membership",
    "scan_candidates",
]
