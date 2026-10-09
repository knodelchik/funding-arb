"""Dynamic universe scanner tests. Pure unit tests on farb.paper.universe's
functions — no journal/engine machinery here (that integration — soft/hard
removal, the scan running inside a tick, interval gating, pin+dynamic
dedup — is covered in tests/test_paper.py alongside the rest of the paper
executor's own tests).

Each test is designed to DISCRIMINATE a specific plausible bug (add-on
-first-scan instead of after add_after_scans, a unit-mismatch name slipping
through, an age/carry/OI filter reading the wrong direction, a replay that
drops or double-counts state, a cap that starves non-pin candidates behind
pins, a streak that doesn't reset on a gap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

import pytest

from farb.paper import universe
from farb.paper.models import HedgeMode, SymbolSpec, UniverseScanConfig

NOW = datetime(2026, 6, 1, tzinfo=UTC)


# --------------------------------------------------------------------------
# Fakes for the pinned HLClient/BinanceClient interface this module needs
# (duck-typed, matching tests/test_paper.py's own fakes in spirit but kept
# local and minimal to exactly what universe.py's functions call).
# --------------------------------------------------------------------------


@dataclass
class FakeAssetCtx:
    coin: str
    funding_hourly: float
    mark_px: float
    open_interest: float
    oracle_px: float = 0.0


@dataclass
class FakeFundingEvent:
    coin: str
    time: datetime
    rate: float


@dataclass
class FakeHLClient:
    funding: dict[str, list[FakeFundingEvent]] = field(default_factory=dict)
    raise_for_coins: frozenset[str] = frozenset()

    def funding_history(
        self, coin: str, start_ms: int, end_ms: int | None = None
    ) -> list[FakeFundingEvent]:
        if coin in self.raise_for_coins:
            raise RuntimeError(f"simulated HL outage for {coin}")
        events = self.funding.get(coin, [])
        return [
            e for e in events
            if round(e.time.timestamp() * 1000) >= start_ms
            and (end_ms is None or round(e.time.timestamp() * 1000) < end_ms)
        ]


@dataclass
class FakeBinanceClient:
    spot_exchange_info: dict = field(default_factory=lambda: {"symbols": []})
    perp_exchange_info: dict = field(default_factory=lambda: {"symbols": []})
    raise_on_exchange_info: bool = False

    def get(self, base: str, path: str, params: dict) -> dict:
        if self.raise_on_exchange_info:
            raise RuntimeError("simulated Binance outage")
        if path == universe.SPOT_EXCHANGE_INFO_PATH:
            return self.spot_exchange_info
        if path == universe.PERP_EXCHANGE_INFO_PATH:
            return self.perp_exchange_info
        raise AssertionError(f"unexpected exchangeInfo path {path!r}")


def _daily_events(coin: str, start: datetime, days: int, rate: float) -> list[FakeFundingEvent]:
    return [
        FakeFundingEvent(coin=coin, time=start + timedelta(days=i), rate=rate)
        for i in range(days)
    ]


def _rich_history(coin: str, now: datetime, rate: float, *, days_old: int = 95) -> list:
    """Full daily history from `days_old` days back through `now` — easily
    satisfies both the (separate) age probe (min_history_days<=days_old-2)
    and the always-30d carry probe, at whatever `rate`.
    """
    return _daily_events(coin, now - timedelta(days=days_old), days_old + 1, rate)


def _spot_entry(symbol: str, *, status: str = "TRADING", quote: str = "USDT") -> dict:
    return {"symbol": symbol, "status": status, "quoteAsset": quote}


def _perp_entry(
    symbol: str, *, status: str = "TRADING", quote: str = "USDT", contract: str = "PERPETUAL"
) -> dict:
    return {"symbol": symbol, "status": status, "quoteAsset": quote, "contractType": contract}


_PERMISSIVE_CFG = UniverseScanConfig(
    min_open_interest_usd=1.0, min_history_days=1, min_trailing_carry_apr=0.0,
)


# --------------------------------------------------------------------------
# 1. Exact-name-match exclusion + invalid-name drop
# --------------------------------------------------------------------------


def test_scan_candidates_excludes_exact_name_mismatch() -> None:
    """HL's 'KPEPE' must NOT match Binance's real '1000PEPEUSDT' -- the
    exact `hl_coin + "USDT"` construction excludes any unit-mismatch pair
    BY CONSTRUCTION. KPEPE is given FULL, otherwise-qualifying history
    (not just high OI) so that a mutant bypassing the Binance match
    entirely (e.g. defaulting to hedge_mode=SPOT instead of `continue`
    when neither set contains the symbol) would still find it reaching
    -- and passing -- the age/carry probes, and must fail this.
    """
    asset_ctxs = {
        "KPEPE": FakeAssetCtx(coin="KPEPE", funding_hourly=0.001, mark_px=1.0, open_interest=1e12),
    }
    binance_symbols = universe.BinanceSymbols(spot={"1000PEPEUSDT"}, perp=set())
    hl = FakeHLClient(funding={"KPEPE": _rich_history("KPEPE", NOW, 0.0001)})
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1.0, min_history_days=60, min_trailing_carry_apr=0.01,
    )

    result = universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, set(), NOW)

    assert result.qualifying == []


def test_scan_candidates_drops_invalid_symbol_names_and_records_them() -> None:
    """A constructed candidate symbol that fails the [A-Z0-9]{2,20}
    security-boundary check must be dropped (never reach the Binance
    match step) and recorded for audit, not silently vanish.
    """
    asset_ctxs = {
        "foo/bar": FakeAssetCtx(
            coin="foo/bar", funding_hourly=0.001, mark_px=1.0, open_interest=1e12
        ),
    }
    hl = FakeHLClient()

    result = universe.scan_candidates(
        asset_ctxs, universe.BinanceSymbols(), hl, _PERMISSIVE_CFG, set(), NOW
    )

    assert result.qualifying == []
    assert result.dropped_invalid_names == ["foo/barUSDT"]


# --------------------------------------------------------------------------
# 2. Open interest filter
# --------------------------------------------------------------------------


def test_scan_candidates_filters_by_open_interest() -> None:
    """Two otherwise-identical coins, one above min_open_interest_usd and
    one below -- only the former must survive.
    """
    asset_ctxs = {
        "RICH": FakeAssetCtx(
            coin="RICH", funding_hourly=0.0001, mark_px=10.0, open_interest=200_000
        ),
        "THIN": FakeAssetCtx(
            coin="THIN", funding_hourly=0.0001, mark_px=10.0, open_interest=50_000
        ),
    }
    binance_symbols = universe.BinanceSymbols(spot={"RICHUSDT", "THINUSDT"}, perp=set())
    hl = FakeHLClient(
        funding={
            "RICH": _rich_history("RICH", NOW, 0.0001),
            "THIN": _rich_history("THIN", NOW, 0.0001),
        }
    )
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1_000_000, min_history_days=60, min_trailing_carry_apr=0.05,
    )

    result = universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, set(), NOW)

    symbols = {c.symbol for c in result.qualifying}
    assert symbols == {"RICHUSDT"}  # RICH: OI 2,000,000 >= 1,000,000; THIN: 500,000 < 1,000,000


# --------------------------------------------------------------------------
# 3. Age probe (separate, narrower window per the latest fix batch)
# --------------------------------------------------------------------------


def test_scan_candidates_age_probe_rejects_a_young_coin() -> None:
    """A coin whose REAL history only starts 30 days ago must be rejected
    under a 60-day min_history_days, even though it has OTHERWISE-rich
    (OI, carry) data for every day it HAS existed -- the probe's
    "first returned event near the requested start" check is what must
    catch this, not the data's richness.
    """
    young_start = NOW - timedelta(days=30)
    old_start = NOW - timedelta(days=90)
    asset_ctxs = {
        "YOUNG": FakeAssetCtx(coin="YOUNG", funding_hourly=0.0001, mark_px=1.0, open_interest=1e9),
        "OLD": FakeAssetCtx(coin="OLD", funding_hourly=0.0001, mark_px=1.0, open_interest=1e9),
    }
    binance_symbols = universe.BinanceSymbols(spot={"YOUNGUSDT", "OLDUSDT"}, perp=set())
    hl = FakeHLClient(
        funding={
            "YOUNG": _daily_events("YOUNG", young_start, 31, 0.0001),  # only 30d of real history
            "OLD": _daily_events("OLD", old_start, 91, 0.0001),  # 90d of real history
        }
    )
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1.0, min_history_days=60, min_trailing_carry_apr=0.01,
    )

    result = universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, set(), NOW)

    symbols = {c.symbol for c in result.qualifying}
    assert symbols == {"OLDUSDT"}
    assert "YOUNGUSDT" not in symbols


def test_scan_candidates_age_probe_closes_the_low_boundary() -> None:
    """A coin whose real history starts EXACTLY `min_history_days` back
    (not a day more) must still pass -- probing from
    `min_history_days + AGE_PROBE_TOLERANCE_DAYS` and applying the SAME
    tolerance against THAT point means a coin passes once its real
    history reaches back to at least `min_history_days`, not
    `min_history_days - tolerance` (the boundary bug a prior version of
    this probe had).
    """
    exactly_min_start = NOW - timedelta(days=60)
    asset_ctxs = {
        "EDGE": FakeAssetCtx(coin="EDGE", funding_hourly=0.0001, mark_px=1.0, open_interest=1e9),
    }
    binance_symbols = universe.BinanceSymbols(spot={"EDGEUSDT"}, perp=set())
    hl = FakeHLClient(funding={"EDGE": _daily_events("EDGE", exactly_min_start, 61, 0.0001)})
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1.0, min_history_days=60, min_trailing_carry_apr=0.01,
    )

    result = universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, set(), NOW)

    assert {c.symbol for c in result.qualifying} == {"EDGEUSDT"}


def test_scan_candidates_age_probe_rejects_a_59_day_old_coin_under_60_day_minimum() -> None:
    """V12: a coin whose real history starts 59 days back -- ONE DAY
    short of a 60-day min_history_days -- must be REJECTED, pinning the
    exact boundary direction precisely (complementing the "exactly 60
    passes" boundary test above).
    """
    start_59d = NOW - timedelta(days=59)
    asset_ctxs = {
        "EDGE59": FakeAssetCtx(
            coin="EDGE59", funding_hourly=0.0001, mark_px=1.0, open_interest=1e9
        ),
    }
    binance_symbols = universe.BinanceSymbols(spot={"EDGE59USDT"}, perp=set())
    hl = FakeHLClient(funding={"EDGE59": _daily_events("EDGE59", start_59d, 60, 0.0001)})
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1.0, min_history_days=60, min_trailing_carry_apr=0.01,
    )

    result = universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, set(), NOW)

    assert {c.symbol for c in result.qualifying} == set()


# --------------------------------------------------------------------------
# 4. Trailing-carry filter
# --------------------------------------------------------------------------


def test_scan_candidates_min_trailing_carry_apr_filter() -> None:
    """Two coins with identical OI/age, differing only in their trailing
    -30d mean hourly funding -- only the one above min_trailing_carry_apr
    must survive.
    """
    asset_ctxs = {
        "RICHCARRY": FakeAssetCtx(
            coin="RICHCARRY", funding_hourly=0.0001, mark_px=1.0, open_interest=1e9
        ),
        "POORCARRY": FakeAssetCtx(
            coin="POORCARRY", funding_hourly=0.0001, mark_px=1.0, open_interest=1e9
        ),
    }
    binance_symbols = universe.BinanceSymbols(spot={"RICHCARRYUSDT", "POORCARRYUSDT"}, perp=set())
    # RICHCARRY: 0.0001/hr * 8766 = 87.66% APR. POORCARRY: 0.000001/hr * 8766 = 0.8766% APR.
    hl = FakeHLClient(
        funding={
            "RICHCARRY": _rich_history("RICHCARRY", NOW, 0.0001),
            "POORCARRY": _rich_history("POORCARRY", NOW, 0.000001),
        }
    )
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1.0, min_history_days=60, min_trailing_carry_apr=0.05,
    )

    result = universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, set(), NOW)

    symbols = {c.symbol for c in result.qualifying}
    assert symbols == {"RICHCARRYUSDT"}


# --------------------------------------------------------------------------
# 5. hedge_mode preference, ranking (uncapped), and pin exclusion
# --------------------------------------------------------------------------


def test_scan_candidates_prefers_spot_hedge_over_perp_when_both_tradeable() -> None:
    asset_ctxs = {
        "BOTH": FakeAssetCtx(coin="BOTH", funding_hourly=0.0001, mark_px=1.0, open_interest=1e9),
    }
    binance_symbols = universe.BinanceSymbols(spot={"BOTHUSDT"}, perp={"BOTHUSDT"})
    hl = FakeHLClient(funding={"BOTH": _rich_history("BOTH", NOW, 0.0001)})
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1.0, min_history_days=60, min_trailing_carry_apr=0.01,
    )

    result = universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, set(), NOW)

    assert len(result.qualifying) == 1
    assert result.qualifying[0].hedge_mode == HedgeMode.SPOT


def test_scan_candidates_qualifying_is_ranked_desc_and_uncapped() -> None:
    """`qualifying` is sorted by carry APR descending and NEVER capped at
    max_dynamic_members inside scan_candidates -- capping is
    apply_hysteresis's add-time decision, not this function's (an
    existing member's survival must not depend on a cap it has nothing
    to do with). All 3 candidates below qualify and must all appear, in
    carry order, even though max_dynamic_members=2.
    """
    coins = {"LOW": 0.00002, "HIGH": 0.0002, "MID": 0.00008}  # hourly rates, distinct carry ranks
    asset_ctxs = {
        name: FakeAssetCtx(coin=name, funding_hourly=rate, mark_px=1.0, open_interest=1e9)
        for name, rate in coins.items()
    }
    binance_symbols = universe.BinanceSymbols(spot={f"{n}USDT" for n in coins}, perp=set())
    hl = FakeHLClient(funding={n: _rich_history(n, NOW, r) for n, r in coins.items()})
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1.0, min_history_days=60, min_trailing_carry_apr=0.01,
        max_dynamic_members=2,
    )

    result = universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, set(), NOW)

    assert [c.symbol for c in result.qualifying] == ["HIGHUSDT", "MIDUSDT", "LOWUSDT"]


def test_scan_candidates_excludes_pins_before_probing_at_all() -> None:
    """A pinned symbol is excluded BEFORE any filter runs -- never a
    candidate, and never even charged a network probe for it (it has no
    funding history registered at all here; if it weren't excluded
    first, the lookup would just return no data and it would be dropped
    anyway, but `probe_attempts` staying 0 proves it never got that far).
    """
    asset_ctxs = {
        "PINNED": FakeAssetCtx(
            coin="PINNED", funding_hourly=0.0005, mark_px=1.0, open_interest=1e12
        ),
    }
    binance_symbols = universe.BinanceSymbols(spot={"PINNEDUSDT"}, perp=set())
    hl = FakeHLClient()  # no funding history registered for PINNED at all

    result = universe.scan_candidates(
        asset_ctxs, binance_symbols, hl, _PERMISSIVE_CFG, {"PINNEDUSDT"}, NOW
    )

    assert result.qualifying == []
    assert result.probe_attempts == 0


def test_scan_candidates_pins_excluded_before_ranking_reviewer_repro() -> None:
    """Reviewer repro: with a cap of 2 and 2 PINNED symbols that would
    otherwise rank at the very top, the 2 best NON-PIN candidates must
    still get through to `qualifying` (unranked/uncapped here, but this
    proves the pins never occupy a competitive slot at all, which is
    what let them previously starve the genuinely-addable candidates
    once capping+pin-exclusion order was wrong).
    """
    coins = {
        "PINA": 0.0009, "PINB": 0.0008,  # would rank #1 and #2 if NOT excluded
        "BEST": 0.0005, "SECOND": 0.0004, "THIRD": 0.0003,
    }
    asset_ctxs = {
        name: FakeAssetCtx(coin=name, funding_hourly=rate, mark_px=1.0, open_interest=1e9)
        for name, rate in coins.items()
    }
    binance_symbols = universe.BinanceSymbols(spot={f"{n}USDT" for n in coins}, perp=set())
    hl = FakeHLClient(funding={n: _rich_history(n, NOW, r) for n, r in coins.items()})
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1.0, min_history_days=60, min_trailing_carry_apr=0.01,
        max_dynamic_members=2, add_after_scans=1,
    )
    pinned = {"PINAUSDT", "PINBUSDT"}

    scan_result = universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, pinned, NOW)
    assert [c.symbol for c in scan_result.qualifying] == ["BESTUSDT", "SECONDUSDT", "THIRDUSDT"]

    update = universe.apply_hysteresis(universe.UniverseMembership(), scan_result, scan_cfg)
    assert {s.symbol for s in update.to_add} == {"BESTUSDT", "SECONDUSDT"}


# --------------------------------------------------------------------------
# 6. Partial probe failure -> whole scan treated as failed
# --------------------------------------------------------------------------


def test_scan_candidates_raises_when_majority_of_probes_fail() -> None:
    """3 of 5 coins' funding_history probes raise (60% > the 50% limit)
    -- the WHOLE scan must be treated as failed (raises), not just
    silently drop those 3 coins and return a result built from the
    minority that happened to succeed.
    """
    coins = ["A", "B", "C", "D", "E"]
    asset_ctxs = {
        name: FakeAssetCtx(coin=name, funding_hourly=0.0005, mark_px=1.0, open_interest=1e9)
        for name in coins
    }
    binance_symbols = universe.BinanceSymbols(spot={f"{n}USDT" for n in coins}, perp=set())
    hl = FakeHLClient(
        funding={n: _rich_history(n, NOW, 0.0005) for n in coins},
        raise_for_coins=frozenset({"A", "B", "C"}),
    )
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1.0, min_history_days=60, min_trailing_carry_apr=0.01,
    )

    with pytest.raises(universe.ScanProbeDegradedError):
        universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, set(), NOW)


def test_scan_candidates_tolerates_a_minority_of_probe_failures() -> None:
    """Only 1 of 5 probes fails (20% < the 50% limit) -- the scan must
    succeed, just dropping that one coin, exactly as before this fix.
    """
    coins = ["A", "B", "C", "D", "E"]
    asset_ctxs = {
        name: FakeAssetCtx(coin=name, funding_hourly=0.0005, mark_px=1.0, open_interest=1e9)
        for name in coins
    }
    binance_symbols = universe.BinanceSymbols(spot={f"{n}USDT" for n in coins}, perp=set())
    hl = FakeHLClient(
        funding={n: _rich_history(n, NOW, 0.0005) for n in coins},
        raise_for_coins=frozenset({"A"}),
    )
    scan_cfg = UniverseScanConfig(
        min_open_interest_usd=1.0, min_history_days=60, min_trailing_carry_apr=0.01,
    )

    result = universe.scan_candidates(asset_ctxs, binance_symbols, hl, scan_cfg, set(), NOW)

    assert {c.symbol for c in result.qualifying} == {"BUSDT", "CUSDT", "DUSDT", "EUSDT"}


# --------------------------------------------------------------------------
# 7. Add/remove hysteresis
# --------------------------------------------------------------------------


def test_apply_hysteresis_add_requires_consecutive_qualifying_scans() -> None:
    """A single qualifying scan must NOT add a new member -- only after
    `add_after_scans` CONSECUTIVE qualifying scans. A mutant that adds on
    the first qualifying scan (e.g. `streak >= 1` instead of
    `streak >= add_after_scans`) fails the first assertion below.
    """
    spec_a = SymbolSpec(symbol="AUSDT", hl_coin="A", hedge_mode=HedgeMode.SPOT, max_notional=5_000)
    scan_result = universe.ScanResult(qualifying=[spec_a])
    scan_cfg = UniverseScanConfig(add_after_scans=2, remove_after_scans=3)

    membership = universe.UniverseMembership()
    update1 = universe.apply_hysteresis(membership, scan_result, scan_cfg)
    assert update1.to_add == []
    assert update1.qualify_streak == {"AUSDT": 1}

    membership.qualify_streak = update1.qualify_streak
    update2 = universe.apply_hysteresis(membership, scan_result, scan_cfg)
    assert update2.to_add == [spec_a]
    assert update2.qualify_streak == {"AUSDT": 2}


def test_apply_hysteresis_qualify_streak_resets_on_a_failing_scan() -> None:
    """U4: pass, fail, pass must NOT add on the third scan (streak must
    reset to 1 on the gap, not accumulate to 2 across the failing scan in
    between) -- a mutant that MERGES the new snapshot with the old one
    instead of replacing it wholesale would let a stale streak survive
    the gap and add prematurely.
    """
    spec_a = SymbolSpec(symbol="AUSDT", hl_coin="A", hedge_mode=HedgeMode.SPOT, max_notional=5_000)
    scan_cfg = UniverseScanConfig(add_after_scans=2, remove_after_scans=3)
    membership = universe.UniverseMembership()

    # Scan 1: qualifies -> streak 1.
    update1 = universe.apply_hysteresis(
        membership, universe.ScanResult(qualifying=[spec_a]), scan_cfg
    )
    assert update1.qualify_streak == {"AUSDT": 1}
    membership.qualify_streak = update1.qualify_streak

    # Scan 2: does NOT qualify -> dropped from the snapshot entirely.
    update2 = universe.apply_hysteresis(
        membership, universe.ScanResult(qualifying=[]), scan_cfg
    )
    assert update2.qualify_streak == {}
    membership.qualify_streak = update2.qualify_streak

    # Scan 3: qualifies again -> must restart at streak 1, NOT add (which
    # would require reaching add_after_scans=2 via a FRESH streak).
    update3 = universe.apply_hysteresis(
        membership, universe.ScanResult(qualifying=[spec_a]), scan_cfg
    )
    assert update3.qualify_streak == {"AUSDT": 1}
    assert update3.to_add == []


def test_apply_hysteresis_remove_requires_consecutive_disqualifying_scans() -> None:
    """Symmetric to the add case: an existing dynamic member that stops
    qualifying must survive `remove_after_scans - 1` disqualifying scans
    before actually being removed on the scan that reaches the threshold.
    """
    spec_a = SymbolSpec(symbol="AUSDT", hl_coin="A", hedge_mode=HedgeMode.SPOT, max_notional=5_000)
    scan_cfg = UniverseScanConfig(add_after_scans=2, remove_after_scans=3)
    membership = universe.UniverseMembership(dynamic_members={"AUSDT": spec_a})
    empty_scan = universe.ScanResult(qualifying=[])  # A no longer qualifies

    update1 = universe.apply_hysteresis(membership, empty_scan, scan_cfg)
    assert update1.to_remove == []
    membership.disqualify_streak = update1.disqualify_streak

    update2 = universe.apply_hysteresis(membership, empty_scan, scan_cfg)
    assert update2.to_remove == []
    membership.disqualify_streak = update2.disqualify_streak

    update3 = universe.apply_hysteresis(membership, empty_scan, scan_cfg)
    assert update3.to_remove == [("AUSDT", "degraded")]


def test_apply_hysteresis_disqualify_streak_resets_when_member_passes_again() -> None:
    """U5: a member fails one scan (disqualify streak 1), then PASSES the
    next -- it must NOT be removed later just by accumulating a couple
    more disqualifying scans as if the gap never happened; the streak
    must restart from 0/absent on the pass, not resume from 1.
    """
    spec_a = SymbolSpec(symbol="AUSDT", hl_coin="A", hedge_mode=HedgeMode.SPOT, max_notional=5_000)
    scan_cfg = UniverseScanConfig(add_after_scans=2, remove_after_scans=3)
    membership = universe.UniverseMembership(dynamic_members={"AUSDT": spec_a})

    update1 = universe.apply_hysteresis(membership, universe.ScanResult(qualifying=[]), scan_cfg)
    assert update1.disqualify_streak == {"AUSDT": 1}
    membership.disqualify_streak = update1.disqualify_streak

    # Passes again -- dropped from the snapshot entirely.
    update2 = universe.apply_hysteresis(
        membership, universe.ScanResult(qualifying=[spec_a]), scan_cfg
    )
    assert update2.disqualify_streak == {}
    membership.disqualify_streak = update2.disqualify_streak

    # Fails twice more (would be streak 3 if NOT reset, triggering remove
    # at remove_after_scans=3) -- must only reach streak 2, no removal.
    update3 = universe.apply_hysteresis(membership, universe.ScanResult(qualifying=[]), scan_cfg)
    membership.disqualify_streak = update3.disqualify_streak
    update4 = universe.apply_hysteresis(membership, universe.ScanResult(qualifying=[]), scan_cfg)

    assert update4.disqualify_streak == {"AUSDT": 2}
    assert update4.to_remove == []


def test_apply_hysteresis_ranked_order_decides_priority_for_scarce_slots() -> None:
    """When multiple NEW candidates cross their add threshold in the same
    scan but only SOME slots are free, the best-ranked (not an arbitrary
    one) must win -- `scan_result.qualifying` is already carry-sorted,
    and `apply_hysteresis` must walk it in that order.
    """
    spec_best = SymbolSpec(
        symbol="BESTUSDT", hl_coin="BEST", hedge_mode=HedgeMode.SPOT, max_notional=1
    )
    spec_worst = SymbolSpec(
        symbol="WORSTUSDT", hl_coin="WORST", hedge_mode=HedgeMode.SPOT, max_notional=1
    )
    scan_cfg = UniverseScanConfig(add_after_scans=1, max_dynamic_members=1)
    membership = universe.UniverseMembership()

    # qualifying is already ranked desc -- WORST listed first would be a
    # bug in the test fixture itself, so this is deliberately BEST-first.
    update = universe.apply_hysteresis(
        membership, universe.ScanResult(qualifying=[spec_best, spec_worst]), scan_cfg
    )

    assert [s.symbol for s in update.to_add] == ["BESTUSDT"]


def test_apply_hysteresis_does_not_add_past_max_dynamic_members() -> None:
    """A new candidate that crosses its add threshold must still not be
    added if doing so would exceed max_dynamic_members -- existing
    members occupy slots too, not just this scan's own candidate count.
    """
    existing = {
        "AUSDT": SymbolSpec(symbol="AUSDT", hl_coin="A", hedge_mode=HedgeMode.SPOT, max_notional=1),
        "BUSDT": SymbolSpec(symbol="BUSDT", hl_coin="B", hedge_mode=HedgeMode.SPOT, max_notional=1),
    }
    spec_c = SymbolSpec(symbol="CUSDT", hl_coin="C", hedge_mode=HedgeMode.SPOT, max_notional=1)
    membership = universe.UniverseMembership(
        dynamic_members=dict(existing), qualify_streak={"CUSDT": 1}
    )
    scan_cfg = UniverseScanConfig(add_after_scans=2, max_dynamic_members=2)

    update = universe.apply_hysteresis(
        membership, universe.ScanResult(qualifying=[spec_c]), scan_cfg
    )

    assert update.to_add == []  # streak reaches 2, but the cap (2) is already full


def test_apply_hysteresis_existing_member_below_cap_rank_keeps_membership() -> None:
    """A member that still PASSES every filter but would rank, say, 3rd
    in a scan whose cap is 2 must NOT accrue a disqualify streak just
    because of that rank -- membership survival depends only on still
    being in `qualifying`, never on rank or the cap (which only gates
    NEW adds).
    """
    spec_member = SymbolSpec(
        symbol="MEMBERUSDT", hl_coin="MEMBER", hedge_mode=HedgeMode.SPOT, max_notional=1
    )
    spec_a = SymbolSpec(symbol="AUSDT", hl_coin="A", hedge_mode=HedgeMode.SPOT, max_notional=1)
    spec_b = SymbolSpec(symbol="BUSDT", hl_coin="B", hedge_mode=HedgeMode.SPOT, max_notional=1)
    membership = universe.UniverseMembership(dynamic_members={"MEMBERUSDT": spec_member})
    scan_cfg = UniverseScanConfig(max_dynamic_members=2)
    # MEMBERUSDT ranks LAST (3rd) of 3 qualifying candidates this scan.
    qualifying = [spec_a, spec_b, spec_member]

    update = universe.apply_hysteresis(
        membership, universe.ScanResult(qualifying=qualifying), scan_cfg
    )

    assert update.disqualify_streak == {}
    assert update.to_remove == []


# --------------------------------------------------------------------------
# 8. Membership replay (journal round-trip)
# --------------------------------------------------------------------------


def test_replay_membership_reconstructs_bit_exact_from_raw_events() -> None:
    """Replaying a raw event list must exactly reconstruct membership,
    streak snapshots (taken from the LATEST scan, not folded deltas), and
    scan bookkeeping -- the same discipline `journal.replay` applies to
    the paper engine's own events, kept local to this module's own event
    types.
    """
    spec_b = SymbolSpec(symbol="BUSDT", hl_coin="B", hedge_mode=HedgeMode.SPOT, max_notional=5_000)
    events = [
        {
            "type": universe.UNIVERSE_SCANNED,
            "payload": {
                "hour": "2026-06-01T00:00:00+00:00",
                "qualify_streak": {"AUSDT": 1},
                "disqualify_streak": {},
            },
        },
        {
            "type": universe.UNIVERSE_MEMBER_ADDED,
            "payload": {
                "symbol": "BUSDT", "reason": "qualified", "spec": spec_b.model_dump(mode="json"),
            },
        },
        {
            "type": universe.UNIVERSE_SCANNED,
            "payload": {
                "hour": "2026-06-02T00:00:00+00:00",
                "qualify_streak": {"AUSDT": 2},
                "disqualify_streak": {"BUSDT": 1},
            },
        },
        {
            "type": universe.UNIVERSE_MEMBER_REMOVED,
            "payload": {"symbol": "BUSDT", "reason": "degraded"},
        },
    ]

    membership = universe.replay_membership(events)

    assert membership.dynamic_members == {}
    assert membership.qualify_streak == {"AUSDT": 2}
    assert membership.disqualify_streak == {"BUSDT": 1}
    assert membership.last_scan_hour == datetime(2026, 6, 2, tzinfo=UTC)
    assert membership.scan_count == 2


def test_replay_membership_survives_a_restart_with_a_live_member() -> None:
    """A member added and never removed must still be present after
    replaying the whole event list from scratch -- simulating a process
    restart, which is the entire point of journal-sourced state.
    """
    spec_a = SymbolSpec(symbol="AUSDT", hl_coin="A", hedge_mode=HedgeMode.PERP, max_notional=5_000)
    events = [
        {
            "type": universe.UNIVERSE_MEMBER_ADDED,
            "payload": {
                "symbol": "AUSDT", "reason": "qualified", "spec": spec_a.model_dump(mode="json"),
            },
        },
    ]

    membership = universe.replay_membership(events)

    assert membership.dynamic_members == {"AUSDT": spec_a}


def test_replay_membership_reconstructs_delist_check_state() -> None:
    events = [
        {
            "type": universe.HEDGE_DELIST_CHECKED,
            "payload": {
                "hour": "2026-06-01T00:00:00+00:00",
                "delist_streak": {"AUSDT": 1},
                "sanity_breaker_tripped": False,
            },
        },
    ]

    membership = universe.replay_membership(events)

    assert membership.delist_streak == {"AUSDT": 1}
    assert membership.last_delist_check_hour == datetime(2026, 6, 1, tzinfo=UTC)


# --------------------------------------------------------------------------
# 9. Binance exchangeInfo filtering + scanner failure propagation
# --------------------------------------------------------------------------


def test_fetch_binance_tradeable_symbols_filters_status_quote_and_contract_type() -> None:
    binance = FakeBinanceClient(
        spot_exchange_info={
            "symbols": [
                _spot_entry("AUSDT"),
                _spot_entry("BUSDT", status="BREAK"),  # wrong status
                _spot_entry("CBTC", quote="BTC"),  # wrong quote asset
            ]
        },
        perp_exchange_info={
            "symbols": [
                _perp_entry("DUSDT"),
                _perp_entry("EUSDT", contract="CURRENT_QUARTER"),  # not a perpetual
            ]
        },
    )

    result = universe.fetch_binance_tradeable_symbols(binance)

    assert result.spot == {"AUSDT"}
    assert result.perp == {"DUSDT"}


def test_fetch_binance_tradeable_symbols_propagates_a_total_fetch_failure() -> None:
    """A total exchangeInfo fetch failure must PROPAGATE (not be silently
    swallowed into an empty set, which would look identical to "nothing
    is tradeable anymore" -- a mass hard-delisting) so the caller
    (engine.py) can treat it as a whole-scan (or whole-delist-check)
    failure and keep the previous state untouched.
    """
    binance = FakeBinanceClient(raise_on_exchange_info=True)

    with pytest.raises(RuntimeError):
        universe.fetch_binance_tradeable_symbols(binance)


# --------------------------------------------------------------------------
# 10. Hedge-delisting check: consecutive confirmation, venue-specificity,
#     sanity breaker (judges the RESPONSE's own symbol counts, not the
#     fraction of open positions that look delisted).
# --------------------------------------------------------------------------


def _padded_binance_symbols(
    spot: set[str] = frozenset(), perp: set[str] = frozenset(), *, floor: int = 220
) -> universe.BinanceSymbols:
    """A BinanceSymbols with `spot`/`perp` explicitly present, padded with
    enough filler symbols to comfortably clear the hedge-delist sanity
    breaker's HEDGE_DELIST_ABSOLUTE_FLOOR (200) on a cold baseline --
    anything genuinely absent (not in `spot`/`perp` and not filler)
    simulates a real single-symbol delisting without the response itself
    looking truncated.
    """
    filler = {f"FILLERPAD{i}USDT" for i in range(floor)}
    return universe.BinanceSymbols(spot=set(spot) | filler, perp=set(perp) | filler)


def test_check_hedge_delisting_confirms_a_genuine_1_of_1_delisting() -> None:
    """1-of-1 open position genuinely delisted must still confirm and
    force-close after 2 consecutive checks -- the reviewer-confirmed hole
    in the OLD fraction-of-positions breaker (which always tripped at a
    trivial 100% with only one position open, no matter how healthy the
    exchangeInfo response actually was). The response itself stays
    healthy here (padded well past the absolute floor), isolating the fix.
    """
    positions = {"FOOUSDT": HedgeMode.PERP}
    # FOOUSDT absent; response otherwise healthy.
    binance_symbols = _padded_binance_symbols(perp=set())
    membership = universe.UniverseMembership()

    result1 = universe.check_hedge_delisting(positions, binance_symbols, membership)
    assert result1.sanity_breaker_tripped is False
    assert result1.first_sighting == ["FOOUSDT"]
    assert result1.newly_confirmed == []
    membership.delist_streak = result1.delist_streak
    membership.last_delist_spot_count = result1.spot_count
    membership.last_delist_perp_count = result1.perp_count

    result2 = universe.check_hedge_delisting(positions, binance_symbols, membership)
    assert result2.newly_confirmed == ["FOOUSDT"]


def test_check_hedge_delisting_confirms_2_of_3_delisted() -> None:
    """2 of 3 open positions genuinely delisted (67%, would also have
    tripped the OLD fraction-based breaker) must still confirm normally
    when the response itself is healthy.
    """
    positions = {"AUSDT": HedgeMode.SPOT, "BUSDT": HedgeMode.SPOT, "CUSDT": HedgeMode.SPOT}
    binance_symbols = _padded_binance_symbols(spot={"CUSDT"})  # A and B absent, C tradeable
    membership = universe.UniverseMembership()

    result1 = universe.check_hedge_delisting(positions, binance_symbols, membership)
    assert set(result1.first_sighting) == {"AUSDT", "BUSDT"}
    membership.delist_streak = result1.delist_streak
    membership.last_delist_spot_count = result1.spot_count
    membership.last_delist_perp_count = result1.perp_count

    result2 = universe.check_hedge_delisting(positions, binance_symbols, membership)
    assert set(result2.newly_confirmed) == {"AUSDT", "BUSDT"}


def test_check_hedge_delisting_is_venue_specific() -> None:
    """U7 (perp-case): a PERP-hedged position must be checked against the
    PERP tradeable set, never the SPOT set -- checking the WRONG venue
    would wrongly flag a perfectly fine hedge as delisted.
    """
    positions = {"SPOTUSDT": HedgeMode.SPOT, "PERPUSDT": HedgeMode.PERP}
    binance_symbols = _padded_binance_symbols(spot={"SPOTUSDT"}, perp={"PERPUSDT"})
    membership = universe.UniverseMembership()

    result = universe.check_hedge_delisting(positions, binance_symbols, membership)

    assert result.first_sighting == []
    assert result.newly_confirmed == []


def test_check_hedge_delisting_spot_case_fires_first_sighting() -> None:
    """U7 (spot-case): a SPOT-hedged position absent from spot, while the
    SAME symbol sits in perp, must still fire a first_sighting --
    confirms checking the RIGHT (spot) venue specifically. Using the SAME
    symbol in perp (not an unrelated one) is what actually discriminates
    a "union of both venues" mutant: with an unrelated filler symbol in
    perp, the real symbol is absent from BOTH spot and the union alike,
    so a union-based check would (wrongly, but coincidentally) still
    report it delisted and this test would never catch it.
    """
    positions = {"SPOTUSDT": HedgeMode.SPOT}
    binance_symbols = _padded_binance_symbols(spot=set(), perp={"SPOTUSDT"})
    membership = universe.UniverseMembership()

    result = universe.check_hedge_delisting(positions, binance_symbols, membership)

    assert result.first_sighting == ["SPOTUSDT"]


def test_check_hedge_delisting_sanity_breaker_closes_nothing_on_truncated_response() -> None:
    """A response whose spot count COLLAPSES relative to its own last
    known-good baseline is treated as a degenerate/truncated exchangeInfo
    response, not a real mass-delisting event: nothing closes, the
    per-symbol streaks are left exactly as they were, and the baseline
    counts themselves carry forward UNCHANGED (not reset to the bad
    response's collapsed count).
    """
    positions = {"AUSDT": HedgeMode.SPOT}
    membership = universe.UniverseMembership()
    good_symbols = _padded_binance_symbols(spot={"AUSDT"}, floor=500)

    good_result = universe.check_hedge_delisting(positions, good_symbols, membership)
    assert good_result.sanity_breaker_tripped is False
    membership.delist_streak = {"STALE": 1}  # a prior unrelated streak
    membership.last_delist_spot_count = good_result.spot_count
    membership.last_delist_perp_count = good_result.perp_count

    truncated_symbols = universe.BinanceSymbols(spot={"AUSDT"}, perp=set())  # collapsed from 501
    result = universe.check_hedge_delisting(positions, truncated_symbols, membership)

    assert result.sanity_breaker_tripped is True
    assert result.newly_confirmed == []
    assert result.first_sighting == []
    assert result.delist_streak == {"STALE": 1}  # untouched
    assert result.spot_count == membership.last_delist_spot_count  # baseline unchanged, not 1


def test_check_hedge_delisting_sanity_breaker_catches_fractional_drop_above_floor() -> None:
    """Pins the 0.5x-last-known-good HALF of the breaker threshold
    specifically: counts dropping from a baseline of 600 to 250 are
    ABOVE the absolute floor (200) but BELOW half the baseline (300) --
    must still read degenerate. A mutant that only checks the absolute
    floor (dropping the `HEDGE_DELIST_PREVIOUS_FRACTION x baseline` half
    of the max(...) entirely) would see 250 >= 200 and wrongly pass this.
    perp is held constant and healthy throughout so only the SPOT side's
    fraction-vs-floor logic is isolated (an unchanging perp baseline
    would otherwise independently trip -- or not trip -- the breaker on
    its own and mask whether spot's own check is doing the real work).
    """
    positions = {"AUSDT": HedgeMode.SPOT}
    membership = universe.UniverseMembership()
    filler_599 = {f"FILLERPAD{i}USDT" for i in range(599)}
    good_symbols = universe.BinanceSymbols(
        spot={"AUSDT"} | filler_599, perp={"BUSDT"} | filler_599,
    )

    good_result = universe.check_hedge_delisting(positions, good_symbols, membership)
    assert good_result.sanity_breaker_tripped is False
    assert good_result.spot_count == 600
    membership.last_delist_spot_count = good_result.spot_count
    membership.last_delist_perp_count = good_result.perp_count

    dropped_symbols = universe.BinanceSymbols(
        spot={f"DROP{i}USDT" for i in range(249)} | {"AUSDT"},  # 250 -- collapsed from 600
        perp={"BUSDT"} | filler_599,  # unchanged -- stays healthy at 600
    )
    result = universe.check_hedge_delisting(positions, dropped_symbols, membership)

    assert result.sanity_breaker_tripped is True


def test_check_hedge_delisting_no_open_positions_is_a_no_op() -> None:
    membership = universe.UniverseMembership(delist_streak={"STALE": 5})

    result = universe.check_hedge_delisting({}, universe.BinanceSymbols(), membership)

    assert result.sanity_breaker_tripped is False
    assert result.newly_confirmed == []
    assert result.delist_streak == {"STALE": 5}
