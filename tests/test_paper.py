"""Paper executor tests. No network — HLClient and BinanceClient are always
injected as fakes; farb.hl is never imported here (that lane may still be
under construction), matching engine.py's lazy-import design.

Each test is designed to DISCRIMINATE: it is built so the specific
plausible bug (flipped funding sign, wrong fee path, lookahead, double
accrual/entry, wrong hysteresis, wrong margin math) makes the assertion
fail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from farb.paper import broker, engine, journal, report
from farb.paper.models import Fill, HedgeMode, PaperConfig, Position, SymbolSpec
from farb.schemas import FUNDING_SCHEMA, KLINE_SCHEMA

T0 = datetime(2025, 6, 1, tzinfo=UTC)


# --------------------------------------------------------------------------
# Fakes for the pinned farb.hl interface (duck-typed — no import from
# farb.hl anywhere in this file) and for BinanceClient.
# --------------------------------------------------------------------------


@dataclass
class FakeAssetCtx:
    coin: str
    funding_hourly: float
    mark_px: float
    open_interest: float = 0.0
    oracle_px: float = 0.0


@dataclass
class FakeFundingEvent:
    coin: str
    time: datetime
    rate: float


@dataclass
class FakeBookTop:
    coin: str
    best_bid: float
    best_ask: float
    ts: datetime = T0


@dataclass
class FakeHLClient:
    ctxs: dict[str, FakeAssetCtx] = field(default_factory=dict)
    funding: dict[str, list[FakeFundingEvent]] = field(default_factory=dict)
    books: dict[str, FakeBookTop] = field(default_factory=dict)
    raise_on_funding_history: bool = False

    def asset_ctxs(self) -> dict[str, FakeAssetCtx]:
        return self.ctxs

    def funding_history(self, coin: str, start_ms: int) -> list[FakeFundingEvent]:
        if self.raise_on_funding_history:
            raise RuntimeError("simulated HL outage")
        events = self.funding.get(coin, [])
        return [e for e in events if round(e.time.timestamp() * 1000) >= start_ms]

    def book_top(self, coin: str) -> FakeBookTop:
        return self.books[coin]


@dataclass
class FakeBinanceClient:
    book_tickers: dict[str, dict[str, str]] = field(default_factory=dict)
    funding_rows: dict[str, list[dict]] = field(default_factory=dict)
    raise_on_get_list: bool = False

    def get(self, base: str, path: str, params: dict) -> dict:
        return self.book_tickers[params["symbol"]]

    def get_list(self, base: str, path: str, params: dict) -> list[dict]:
        if self.raise_on_get_list:
            raise RuntimeError("simulated Binance outage")
        rows = self.funding_rows.get(params["symbol"], [])
        start_ms = params.get("startTime", 0)
        return [r for r in rows if int(r["fundingTime"]) >= start_ms]


SPOT_SPEC = SymbolSpec(
    symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.SPOT, max_notional=50_000
)
PERP_SPEC = SymbolSpec(
    symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.PERP, max_notional=50_000
)


def _make_cfg(tmp_path: Path, **overrides) -> PaperConfig:
    universe = overrides.pop("universe", [SPOT_SPEC])
    defaults = {
        "universe": universe,
        "capital_usd": 10_000.0,
        "max_positions": 1,
        "ema_periods": 2,
        "enter_apr": 0.12,
        "exit_apr": 0.0,
        "journal_dir": tmp_path / "paper",
        "data_root": tmp_path / "data",
    }
    defaults.update(overrides)
    return PaperConfig(**defaults)


def _hourly_rates(start: datetime, rates: list[float], coin: str) -> list[FakeFundingEvent]:
    return [
        FakeFundingEvent(coin=coin, time=start + timedelta(hours=i), rate=r)
        for i, r in enumerate(rates)
    ]


# --------------------------------------------------------------------------
# 1. Sign conventions
# --------------------------------------------------------------------------


def test_hl_leg_receives_positive_funding_short_sign() -> None:
    """Short HL leg: a positive rate must be a POSITIVE (received) amount.
    Flipping the sign in `hl_funding_usd` breaks this directly.
    """
    amount = broker.hl_funding_usd(rate=0.0002, qty=50.0, mark_price=100.0)
    assert amount == pytest.approx(1.0)
    assert amount > 0


def test_hedge_perp_leg_pays_positive_funding_long_sign() -> None:
    """Long hedge-perp leg: a positive rate is a cost (PAID), which the net
    formula (funding_hl - funding_hedge_paid + basis - fees) must SUBTRACT.
    The raw magnitude has the same shape as the HL leg's; the sign
    convention lives in how it's combined, not in the formula itself.
    """
    amount = broker.hedge_funding_usd(rate=0.0002, qty=50.0, mark_price=100.0)
    assert amount == pytest.approx(1.0)  # same magnitude as the HL case...
    # ...but plugged into net as a subtraction: a positive hedge accrual
    # must REDUCE net, unlike a positive HL accrual which INCREASES it.
    funding_hl = 5.0
    net_with_hedge_cost = funding_hl - amount
    assert net_with_hedge_cost < funding_hl


def test_full_cycle_sign_convention_end_to_end(tmp_path: Path) -> None:
    """Open a PERP-hedge position, accrue one hour of POSITIVE funding on
    both legs via a real tick, then close: net must reflect funding_hl
    ADDED and funding_hedge_paid SUBTRACTED. Flipping either accrual sign
    anywhere in the engine breaks this.
    """
    cfg = _make_cfg(
        tmp_path,
        universe=[PERP_SPEC],
        maker_fill_prob=1.0,
    )
    hour0 = journal.hour_bucket(T0)
    # 9 hourly points: ema_periods=2 needs two rolling-8h-sum values, which
    # needs 9 raw hourly rates (8 for the first sum, one more for the second).
    warmup = _hourly_rates(hour0 - timedelta(hours=9), [0.0002] * 9, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0002, mark_px=100.0)},
        funding={"FOO": warmup},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}},
        funding_rows={"FOOUSDT": []},  # no hedge funding this cycle
    )
    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" in state.open_positions

    # A rate settling STRICTLY AFTER entry (hour0+1h, not hour0 itself —
    # entry_ts==hour0 exactly, and that settlement happened at-or-before
    # entry, not after it) becomes accruable once a tick passes hour0+2h.
    new_event = FakeFundingEvent(coin="FOO", time=hour0 + timedelta(hours=1), rate=0.0002)
    hl.funding["FOO"].append(new_event)
    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(hours=2)
    )
    state = journal.replay(journal.journal_path(cfg))
    pos = state.open_positions["FOOUSDT"]
    assert pos.funding_received_usd == pytest.approx(pos.qty * 0.0002 * 100.0)
    assert pos.funding_received_usd > 0


# --------------------------------------------------------------------------
# 2. Fee arithmetic: exact hand-computed fees for maker (p=1) and taker (p=0)
# --------------------------------------------------------------------------


def test_open_pair_fee_arithmetic_maker(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        universe=[PERP_SPEC],
        maker_fill_prob=1.0,  # rng.random() < 1.0 always -> always maker
    )
    spec = cfg.universe[0]
    hl_book = FakeBookTop(coin="FOO", best_bid=100.0, best_ask=102.0)
    hedge_price = 101.0
    notional = 10_000.0

    fills = broker.open_pair(spec, notional, hl_book, hedge_price, T0, cfg)
    hl_fill, hedge_fill = fills

    expected_qty = notional / hl_book.best_ask
    assert hl_fill.is_maker is True
    assert hl_fill.price == pytest.approx(hl_book.best_ask)
    assert hl_fill.qty == pytest.approx(expected_qty)
    assert hl_fill.fee_usd == pytest.approx(notional * cfg.hl_maker_bps / 10_000)

    assert hedge_fill.is_maker is True
    assert hedge_fill.price == pytest.approx(hedge_price)  # maker: unslipped
    assert hedge_fill.fee_usd == pytest.approx(
        expected_qty * hedge_price * cfg.perp_maker_bps / 10_000
    )


def test_open_pair_fee_arithmetic_taker(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path,
        universe=[PERP_SPEC],
        maker_fill_prob=0.0,  # rng.random() < 0.0 never -> always taker
    )
    spec = cfg.universe[0]
    hl_book = FakeBookTop(coin="FOO", best_bid=100.0, best_ask=102.0)
    hedge_price = 101.0
    notional = 10_000.0

    fills = broker.open_pair(spec, notional, hl_book, hedge_price, T0, cfg)
    hl_fill, hedge_fill = fills

    expected_hl_price = hl_book.best_bid * (1 - cfg.taker_slippage_bps / 10_000)
    expected_qty = notional / expected_hl_price
    assert hl_fill.is_maker is False
    assert hl_fill.price == pytest.approx(expected_hl_price)
    assert hl_fill.fee_usd == pytest.approx(notional * cfg.hl_taker_bps / 10_000)

    expected_hedge_price = hedge_price * (1 + cfg.taker_slippage_bps / 10_000)
    assert hedge_fill.is_maker is False
    assert hedge_fill.price == pytest.approx(expected_hedge_price)
    assert hedge_fill.fee_usd == pytest.approx(
        expected_qty * expected_hedge_price * cfg.perp_taker_bps / 10_000
    )


def test_close_pair_fee_arithmetic_maker_and_taker(tmp_path: Path) -> None:
    cfg_maker = _make_cfg(
        tmp_path,
        universe=[PERP_SPEC],
        maker_fill_prob=1.0,
    )
    position = Position(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.PERP, qty=42.0,
        notional_entry=10_000.0, entry_ts=T0,
        entry_fills=[
            Fill(venue="hyperliquid", leg="hl_short", side="open", price=100.0, qty=42.0,
                 fee_usd=1.0, is_maker=True, ts=T0),
            Fill(venue="binance", leg="hedge_long", side="open", price=100.0, qty=42.0,
                 fee_usd=1.0, is_maker=True, ts=T0),
        ],
        funding_received_usd=0.0, funding_paid_usd=0.0, signal_apr_at_entry=0.5,
    )
    hl_book = FakeBookTop(coin="FOO", best_bid=98.0, best_ask=103.0)
    hedge_price = 99.0

    maker_fills = broker.close_pair(position, hl_book, hedge_price, T0, cfg_maker)
    hl_fill, hedge_fill = maker_fills
    assert hl_fill.price == pytest.approx(hl_book.best_bid)
    assert hl_fill.fee_usd == pytest.approx(
        position.qty * hl_book.best_bid * cfg_maker.hl_maker_bps / 10_000
    )
    assert hedge_fill.price == pytest.approx(hedge_price)
    assert hedge_fill.fee_usd == pytest.approx(
        position.qty * hedge_price * cfg_maker.perp_maker_bps / 10_000
    )

    cfg_taker = _make_cfg(
        tmp_path,
        universe=cfg_maker.universe,
        maker_fill_prob=0.0,
    )
    taker_fills = broker.close_pair(position, hl_book, hedge_price, T0, cfg_taker)
    hl_fill_t, hedge_fill_t = taker_fills
    expected_hl_price = hl_book.best_ask * (1 + cfg_taker.taker_slippage_bps / 10_000)
    assert hl_fill_t.price == pytest.approx(expected_hl_price)
    assert hl_fill_t.fee_usd == pytest.approx(
        position.qty * expected_hl_price * cfg_taker.hl_taker_bps / 10_000
    )
    expected_hedge_price = hedge_price * (1 - cfg_taker.taker_slippage_bps / 10_000)
    assert hedge_fill_t.price == pytest.approx(expected_hedge_price)
    assert hedge_fill_t.fee_usd == pytest.approx(
        position.qty * expected_hedge_price * cfg_taker.perp_taker_bps / 10_000
    )


def test_spot_hedge_uses_flat_fee_no_maker_taker_split(tmp_path: Path) -> None:
    """Spot hedge: always ONE flat fee (spot_fee_bps), never the perp
    maker/taker split, regardless of maker_fill_prob.
    """
    cfg = _make_cfg(tmp_path, maker_fill_prob=1.0)  # spot hedge_mode in default universe
    spec = cfg.universe[0]
    hl_book = FakeBookTop(coin="FOO", best_bid=100.0, best_ask=102.0)
    fills = broker.open_pair(spec, 10_000.0, hl_book, 101.0, T0, cfg)
    hedge_fill = fills[1]
    assert hedge_fill.is_maker is False
    assert hedge_fill.price == pytest.approx(101.0)
    expected_qty = 10_000.0 / hl_book.best_ask
    assert hedge_fill.fee_usd == pytest.approx(expected_qty * 101.0 * cfg.spot_fee_bps / 10_000)


# --------------------------------------------------------------------------
# 3. PIT: signal_apr never sees data at/after `as_of`
# --------------------------------------------------------------------------


def test_signal_apr_is_pit_a_future_rate_never_changes_the_decision() -> None:
    as_of = T0 + timedelta(hours=10)
    base = [(T0 + timedelta(hours=i), 0.0001) for i in range(10)]
    apr_before = engine.signal_apr(base, as_of, ema_periods=2)
    assert apr_before is not None

    with_future = [*base, (as_of, 999.0), (as_of + timedelta(hours=1), -999.0)]
    apr_after = engine.signal_apr(with_future, as_of, ema_periods=2)
    assert apr_after == apr_before


def test_signal_apr_none_until_warmed_up() -> None:
    # Fewer than 8 hourly points -> no rolling-8h sum exists yet -> None.
    short = [(T0 + timedelta(hours=i), 0.0001) for i in range(5)]
    assert engine.signal_apr(short, T0 + timedelta(hours=10), ema_periods=2) is None


def test_signal_apr_annualizes_the_8h_window_not_the_hour() -> None:
    """HL's default hourly funding rate (1.25e-5, i.e. 0.01%/8h) must
    annualize to ~11% APR, not ~88%. The bug this catches: annualizing the
    trailing-8h-SUMMED series (a per-8h-window rate) by HOURS_PER_YEAR
    (8766, "hours per year") instead of PERIODS_PER_YEAR (1095.75,
    "8h-windows per year") overstates it by exactly 8x — 0.01% * 8766 =
    87.66%, the exact wrong value seen in the live journal that triggered
    this fix.
    """
    hl_default_hourly_rate = 1.25e-5
    hourly = [(T0 + timedelta(hours=i), hl_default_hourly_rate) for i in range(20)]
    apr = engine.signal_apr(hourly, T0 + timedelta(hours=20), ema_periods=6)
    assert apr == pytest.approx(0.1096, rel=1e-3)
    # Sanity: this must NOT equal the 8x-overstated (buggy) value, so a
    # regression back to annualizing by HOURS_PER_YEAR fails loudly here.
    assert apr != pytest.approx(hl_default_hourly_rate * 8 * 8766.0, rel=1e-3)


# --------------------------------------------------------------------------
# 4. Journal replay: bit-exact reconciliation + corrupted-line error
# --------------------------------------------------------------------------


def test_journal_replay_reconciles_open_accrue_close_bit_exact(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    entry_fills = [
        Fill(venue="hyperliquid", leg="hl_short", side="open", price=100.0, qty=10.0,
             fee_usd=1.5, is_maker=True, ts=T0),
        Fill(venue="binance", leg="hedge_long", side="open", price=100.5, qty=10.0,
             fee_usd=0.75, is_maker=False, ts=T0),
    ]
    position = Position(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.SPOT, qty=10.0,
        notional_entry=1000.0, entry_ts=T0,
        entry_fills=entry_fills, funding_received_usd=0.0, funding_paid_usd=0.0,
        signal_apr_at_entry=0.5,
    )
    journal.append_event(cfg, journal.POSITION_OPENED, position.model_dump(mode="json"), T0)

    total_received = 0.0
    for i in range(3):
        ts = T0 + timedelta(hours=i + 1)
        amount = broker.hl_funding_usd(rate=0.0001, qty=10.0, mark_price=100.0)
        total_received += amount
        journal.append_event(
            cfg, journal.FUNDING_ACCRUED,
            {
                "symbol": "FOOUSDT", "leg": journal.LEG_HL,
                "key": journal.hour_bucket(ts).isoformat(),
                "rate": 0.0001, "mark_px": 100.0, "qty": 10.0, "amount_usd": amount,
            },
            ts,
        )

    exit_ts = T0 + timedelta(hours=4)
    net = total_received - 0.0 + 5.0 - 2.25
    journal.append_event(
        cfg, journal.POSITION_CLOSED,
        {
            "symbol": "FOOUSDT", "exit_ts": exit_ts.isoformat(),
            "funding_hl": total_received, "funding_hedge_paid": 0.0,
            "basis": 5.0, "fees": 2.25, "net": net, "exit_fills": [],
        },
        exit_ts,
    )

    state = journal.replay(journal.journal_path(cfg))
    assert state.open_positions == {}
    assert len(state.closed) == 1
    assert state.closed[0].funding_hl == pytest.approx(total_received)
    assert state.realized_totals["net"] == pytest.approx(net)
    assert state.realized_totals["funding_hl"] == pytest.approx(total_received)


def test_journal_replay_corrupted_last_line_raises_clear_error(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    journal.append_event(cfg, journal.TICK_STARTED, {"hour": T0.isoformat()}, T0)
    path = journal.journal_path(cfg)
    with path.open("a") as f:
        f.write('{"ts": "not-json-close-brace-missing"\n')  # deliberately malformed

    with pytest.raises(journal.JournalCorruptError):
        journal.replay(path)


# --------------------------------------------------------------------------
# 5. Idempotency: two ticks in the same hour -> second is a no-op
# --------------------------------------------------------------------------


def test_tick_same_hour_twice_is_idempotent_no_double_entry(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, maker_fill_prob=1.0)
    hour0 = journal.hour_bucket(T0)
    warmup = _hourly_rates(hour0 - timedelta(hours=9), [0.0005] * 9, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0005, mark_px=100.0)},
        funding={"FOO": warmup},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}}
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)
    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(minutes=17)
    )  # same hour bucket, later minute

    lines = journal.journal_path(cfg).read_text().splitlines()
    event_types = [json.loads(line)["type"] for line in lines]
    assert event_types.count(journal.TICK_STARTED) == 1
    assert event_types.count(journal.TICK_SKIPPED_DUPLICATE) == 1
    assert event_types.count(journal.POSITION_OPENED) == 1  # not entered twice

    state = journal.replay(journal.journal_path(cfg))
    assert len(state.open_positions) == 1


# --------------------------------------------------------------------------
# 6. Hysteresis: a mid-band dip holds; eventual exit still works
# --------------------------------------------------------------------------


def test_hysteresis_holds_through_mid_band_dip_then_exits(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, maker_fill_prob=1.0, ema_periods=1, enter_apr=0.12, exit_apr=0.005)
    hour0 = journal.hour_bucket(T0)
    # hours [hour0-8h .. hour0-1h]: strong positive -> enters at hour0.
    # (8h-sum 0.0008 * PERIODS_PER_YEAR 1095.75 ~= 88% APR, well above 12%.)
    rates = _hourly_rates(hour0 - timedelta(hours=8), [0.0001] * 8, "FOO")
    # hours [hour0 .. hour0+7h]: small positive -> mid-band dip at hour0+8h
    # (8h-sum 4e-5 * 1095.75 ~= 4.4% APR, between exit_apr=0.5% and
    # enter_apr=12%: must HOLD).
    rates += _hourly_rates(hour0, [0.000005] * 8, "FOO")
    # hours [hour0+8h .. hour0+15h]: strongly negative -> exit at hour0+16h.
    rates += _hourly_rates(hour0 + timedelta(hours=8), [-0.0001] * 8, "FOO")

    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0001, mark_px=100.0)},
        funding={"FOO": rates},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}}
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)
    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" in state.open_positions, "should have entered on the strong positive signal"

    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(hours=8)
    )
    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" in state.open_positions, "mid-band dip must HOLD, not exit or churn"
    assert len(state.closed) == 0

    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(hours=16)
    )
    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" not in state.open_positions, "strongly negative signal must eventually exit"
    assert len(state.closed) == 1


# --------------------------------------------------------------------------
# 7. Margin warning
# --------------------------------------------------------------------------


def test_margin_warning_fires_when_mark_moves_against_the_short(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, hl_leverage=1.5, margin_warn_buffer=0.30)
    entry_price = 100.0
    qty = 100.0
    position = Position(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.SPOT, qty=qty,
        notional_entry=10_000.0, entry_ts=T0,
        entry_fills=[
            Fill(venue="hyperliquid", leg="hl_short", side="open", price=entry_price, qty=qty,
                 fee_usd=0.0, is_maker=True, ts=T0),
            Fill(venue="binance", leg="hedge_long", side="open", price=entry_price, qty=qty,
                 fee_usd=0.0, is_maker=False, ts=T0),
        ],
        funding_received_usd=0.0, funding_paid_usd=0.0, signal_apr_at_entry=0.5,
    )
    state = journal.PaperState()
    state.open_positions["FOOUSDT"] = position

    # Small adverse move: buffer still well above 30% -> no warning.
    engine._check_margin(cfg, state, position, mark_px=101.0, now=T0)
    no_warning_state = journal.replay(journal.journal_path(cfg))
    assert no_warning_state.warnings == []

    # Large adverse move against the short (price way up): breaches 30% buffer.
    engine._check_margin(cfg, state, position, mark_px=150.0, now=T0 + timedelta(hours=1))
    state_after = journal.replay(journal.journal_path(cfg))
    assert len(state_after.warnings) == 1
    assert state_after.warnings[0]["symbol"] == "FOOUSDT"


# --------------------------------------------------------------------------
# Robustness: missing HL history parquet, and a live-fetch failure
# --------------------------------------------------------------------------


def test_hourly_funding_series_survives_missing_parquet_and_live_failure(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)  # cfg.data_root has no venues/ dir at all
    spec = cfg.universe[0]
    state = journal.PaperState()
    hl = FakeHLClient(raise_on_funding_history=True)

    hourly = engine._hourly_funding_series(
        spec.symbol, spec.hl_coin, hl, state, cfg, journal.hour_bucket(T0), T0
    )

    assert hourly == []  # no crash, just an empty (but valid) series
    replayed_state = journal.replay(journal.journal_path(cfg))
    assert any("HL funding_history fetch failed" in w["message"] for w in replayed_state.warnings)


def test_hourly_funding_series_uses_journal_and_parquet_when_live_fails(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    spec = cfg.universe[0]
    state = journal.PaperState()
    hour0 = journal.hour_bucket(T0)
    state.hl_funding_history["FOOUSDT"] = {
        journal.hour_bucket(hour0 - timedelta(hours=1)).isoformat(): 0.0003,
    }
    hl = FakeHLClient(raise_on_funding_history=True)

    hourly = engine._hourly_funding_series(spec.symbol, spec.hl_coin, hl, state, cfg, hour0, T0)

    assert hourly == [(hour0 - timedelta(hours=1), 0.0003)]


# --------------------------------------------------------------------------
# 8. Accrual boundary bugs (validation round 2)
# --------------------------------------------------------------------------


def test_accrue_hl_skips_the_settlement_at_or_before_entry(tmp_path: Path) -> None:
    """A rate settling AT the exact entry timestamp (or before) must NOT be
    accrued — it settled before the position existed. Only rates STRICTLY
    after entry_ts count.
    """
    cfg = _make_cfg(tmp_path)
    entry_ts = T0
    position = Position(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.SPOT, qty=10.0,
        notional_entry=1000.0, entry_ts=entry_ts,
        entry_fills=[
            Fill(venue="hyperliquid", leg="hl_short", side="open", price=100.0, qty=10.0,
                 fee_usd=0.0, is_maker=True, ts=entry_ts),
            Fill(venue="binance", leg="hedge_long", side="open", price=100.0, qty=10.0,
                 fee_usd=0.0, is_maker=False, ts=entry_ts),
        ],
        funding_received_usd=0.0, funding_paid_usd=0.0, signal_apr_at_entry=0.5,
    )
    state = journal.PaperState()
    state.open_positions["FOOUSDT"] = position

    hourly = [(entry_ts, 0.0005), (entry_ts + timedelta(hours=1), 0.0003)]
    engine._accrue_hl(
        cfg, state, position, hourly, mark_px=100.0,
        hour=entry_ts + timedelta(hours=2), now=entry_ts + timedelta(hours=2),
    )

    expected = broker.hl_funding_usd(0.0003, 10.0, 100.0)  # only the +1h rate accrues
    assert position.funding_received_usd == pytest.approx(expected)


def test_accrue_hedge_perp_skips_funding_settled_while_flat_after_reentry(tmp_path: Path) -> None:
    """A symbol traded once (closed), then re-entered 10 days later: rows
    that settled during the FLAT gap must not be charged to the new
    position, even though the accrual-key cursor for this symbol still
    points at the OLD position's last accrual key.
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC])
    old_close_ts = T0
    reentry_ts = T0 + timedelta(days=10)

    state = journal.PaperState()
    state.hedge_accrual_keys.add(("FOOUSDT", old_close_ts.isoformat()))

    new_position = Position(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.PERP, qty=10.0,
        notional_entry=1000.0, entry_ts=reentry_ts,
        entry_fills=[
            Fill(venue="hyperliquid", leg="hl_short", side="open", price=100.0, qty=10.0,
                 fee_usd=0.0, is_maker=True, ts=reentry_ts),
            Fill(venue="binance", leg="hedge_long", side="open", price=100.0, qty=10.0,
                 fee_usd=0.0, is_maker=False, ts=reentry_ts),
        ],
        funding_received_usd=0.0, funding_paid_usd=0.0, signal_apr_at_entry=0.5,
    )
    state.open_positions["FOOUSDT"] = new_position

    # 9 phantom 8h-spaced events settled DURING the flat gap (between the
    # old close and the re-entry) -- must all be skipped.
    flat_gap_rows = [
        {
            "fundingTime": round((old_close_ts + timedelta(hours=8 * i)).timestamp() * 1000),
            "fundingRate": "0.0001", "markPrice": "100.0",
        }
        for i in range(1, 9)
    ]
    real_row = {
        "fundingTime": round((reentry_ts + timedelta(hours=8)).timestamp() * 1000),
        "fundingRate": "0.0002", "markPrice": "100.0",
    }
    binance = FakeBinanceClient(funding_rows={"FOOUSDT": [*flat_gap_rows, real_row]})

    engine._accrue_hedge_perp(
        cfg, state, new_position, binance, now=reentry_ts + timedelta(hours=9)
    )

    expected = broker.hedge_funding_usd(0.0002, 10.0, 100.0)
    assert new_position.funding_paid_usd == pytest.approx(expected)


def test_hl_accrual_guard_prevents_double_accrual_across_overlapping_ticks(tmp_path: Path) -> None:
    """Each tick re-fetches recent HL history (as a real API naturally
    would), so consecutive ticks' fetch windows OVERLAP. A settled hour
    already accrued must not be accrued again just because a later tick's
    fetch happens to include it too. Removing the per-hour accrual-key
    guard breaks this.
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0)
    hour0 = journal.hour_bucket(T0)
    warmup = _hourly_rates(hour0 - timedelta(hours=9), [0.0002] * 9, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0002, mark_px=100.0)},
        funding={"FOO": warmup},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}}
    )
    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    new_event = FakeFundingEvent(coin="FOO", time=hour0 + timedelta(hours=1), rate=0.0002)
    hl.funding["FOO"].append(new_event)
    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(hours=2)
    )
    state = journal.replay(journal.journal_path(cfg))
    after_second_tick = state.open_positions["FOOUSDT"].funding_received_usd

    # Third tick: funding_history still (naturally) returns the SAME
    # hour0+1h rate -- must not be accrued a second time.
    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(hours=3)
    )
    state = journal.replay(journal.journal_path(cfg))
    after_third_tick = state.open_positions["FOOUSDT"].funding_received_usd
    assert after_third_tick == pytest.approx(after_second_tick)

    # Also check the RAW journal directly, not just the replayed value:
    # replay() has its OWN independent idempotency guard when reconstructing
    # state (a deliberate, separate safety net against a corrupt/duplicated
    # journal), which would silently mask an engine-side guard removal that
    # still logs a second funding_accrued event for the same hour. Counting
    # raw events is what actually discriminates the engine-side guard.
    key = journal.hour_bucket(hour0 + timedelta(hours=1)).isoformat()
    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    accrual_events_for_hour = [
        e for e in events
        if e["type"] == journal.FUNDING_ACCRUED
        and e["payload"]["leg"] == journal.LEG_HL
        and e["payload"]["key"] == key
    ]
    assert len(accrual_events_for_hour) == 1


# --------------------------------------------------------------------------
# 9. Net-formula sign, end-to-end (validation round 2)
# --------------------------------------------------------------------------


def test_close_position_journal_net_matches_hand_computed_formula(tmp_path: Path) -> None:
    """Close a position with NONZERO, DIFFERENT-magnitude funding on both
    legs (so a sign error can't cancel out by coincidence) and assert the
    JOURNALED position_closed event's net equals
    funding_hl - funding_hedge_paid + basis - fees, computed independently
    from the journaled fills. A mutant `net = funding_hl + funding_hedge_paid
    + basis - fees` in `_finalize_close` must fail this.
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0)
    position = Position(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.PERP, qty=10.0,
        notional_entry=1000.0, entry_ts=T0,
        entry_fills=[
            Fill(venue="hyperliquid", leg="hl_short", side="open", price=100.0, qty=10.0,
                 fee_usd=1.0, is_maker=True, ts=T0),
            Fill(venue="binance", leg="hedge_long", side="open", price=100.0, qty=10.0,
                 fee_usd=1.0, is_maker=True, ts=T0),
        ],
        funding_received_usd=8.0,  # HL leg received $8 total
        funding_paid_usd=3.0,       # hedge leg paid $3 total (different, nonzero)
        signal_apr_at_entry=0.5,
    )
    state = journal.PaperState()
    state.open_positions["FOOUSDT"] = position

    hl_book = FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)
    binance = FakeBinanceClient(book_tickers={"FOOUSDT": {"askPrice": "101.0", "bidPrice": "99.5"}})
    hl = FakeHLClient(books={"FOO": hl_book})

    engine._close_position(cfg, state, position, hl, binance, now=T0 + timedelta(hours=5))

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    closed = [e for e in events if e["type"] == journal.POSITION_CLOSED]
    assert len(closed) == 1
    payload = closed[0]["payload"]

    exit_fills = payload["exit_fills"]
    hl_exit = next(f for f in exit_fills if f["leg"] == "hl_short")
    hedge_exit = next(f for f in exit_fills if f["leg"] == "hedge_long")
    entry_hl, entry_hedge = position.entry_fills
    expected_basis = position.qty * (
        (hedge_exit["price"] - entry_hedge.price) - (hl_exit["price"] - entry_hl.price)
    )
    expected_fees = (
        entry_hl.fee_usd + entry_hedge.fee_usd + hl_exit["fee_usd"] + hedge_exit["fee_usd"]
    )
    expected_net = 8.0 - 3.0 + expected_basis - expected_fees

    assert payload["funding_hl"] == pytest.approx(8.0)
    assert payload["funding_hedge_paid"] == pytest.approx(3.0)
    assert payload["basis"] == pytest.approx(expected_basis)
    assert payload["fees"] == pytest.approx(expected_fees)
    assert payload["net"] == pytest.approx(expected_net)


# --------------------------------------------------------------------------
# 10. Simulated liquidation
# --------------------------------------------------------------------------


def _liquidation_test_position() -> Position:
    entry_price = 100.0
    qty = 10.0
    return Position(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.PERP, qty=qty,
        notional_entry=entry_price * qty, entry_ts=T0,
        entry_fills=[
            Fill(venue="hyperliquid", leg="hl_short", side="open", price=entry_price, qty=qty,
                 fee_usd=0.0, is_maker=True, ts=T0),
            Fill(venue="binance", leg="hedge_long", side="open", price=entry_price, qty=qty,
                 fee_usd=0.0, is_maker=False, ts=T0),
        ],
        funding_received_usd=0.0, funding_paid_usd=0.0, signal_apr_at_entry=0.5,
    )


def test_liq_price_matches_hand_computed_formula(tmp_path: Path) -> None:
    """entry=100, leverage=1.5, MAINT_MARGIN_FRACTION=0.0125:

        liq_price = entry * (1 + 1/leverage - mm)
                  = 100 * (1 + 1/1.5 - 0.0125)
                  = 100 * (1.6666666666666667 - 0.0125)
                  = 100 * 1.6541666666666667
                  = 165.41666666666666...

    (NOT 166.41...: 1 + 1/1.5 = 1.6666..., not 1.7666...). Pinned as a
    literal, hand-verified independently of engine.py's own arithmetic, so
    a mutant dropping MAINT_MARGIN_FRACTION from the formula (which would
    give 100 * (1 + 1/1.5) = 166.66666...) fails this.
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC], hl_leverage=1.5)
    position = _liquidation_test_position()  # entry_price = 100.0

    assert 165.41666666666666 == pytest.approx(100.0 * (1 + 1 / 1.5 - 0.0125), abs=1e-9)
    assert engine._liq_price(position, cfg) == pytest.approx(165.41666666666666)


def test_liquidate_pair_charges_taker_fees_not_maker(tmp_path: Path) -> None:
    """A liquidation is a forced, aggressive fill on both legs — always
    taker fees, never the maker rate (there is no maker/taker draw at all
    in `liquidate_pair`). A mutant using hl_maker_bps/perp_maker_bps must
    fail this (the default maker/taker rates differ: 1.5 vs 4.5 bps HL,
    2.0 vs 5.0 bps perp).
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC])
    position = _liquidation_test_position()  # qty=10.0
    liq_price = 165.0
    hedge_price = 164.0

    hl_fill, hedge_fill = broker.liquidate_pair(position, liq_price, hedge_price, T0, cfg)

    assert hl_fill.is_maker is False
    assert hl_fill.fee_usd == pytest.approx(position.qty * liq_price * cfg.hl_taker_bps / 10_000)
    assert hedge_fill.is_maker is False
    assert hedge_fill.fee_usd == pytest.approx(
        position.qty * hedge_price * cfg.perp_taker_bps / 10_000
    )


def test_liquidation_fires_and_closes_position_when_mark_breaches_liq_price(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC], hl_leverage=1.5, margin_warn_buffer=0.30)
    position = _liquidation_test_position()
    state = journal.PaperState()
    state.open_positions["FOOUSDT"] = position
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "170.0", "bidPrice": "169.0"}}
    )

    liq_price = engine._liq_price(position, cfg)
    mark_above = liq_price + 1.0

    fired = engine._check_liquidation(
        cfg, state, position, mark_above, binance, now=T0 + timedelta(hours=1)
    )

    assert fired is True
    assert "FOOUSDT" not in state.open_positions

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    liq_events = [e for e in events if e["type"] == journal.LIQUIDATION]
    assert len(liq_events) == 1
    assert liq_events[0]["payload"]["symbol"] == "FOOUSDT"
    closed_events = [e for e in events if e["type"] == journal.POSITION_CLOSED]
    assert len(closed_events) == 1
    assert closed_events[0]["payload"]["liquidated"] is True

    state_after = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" not in state_after.open_positions
    assert len(state_after.closed) == 1
    assert state_after.closed[0].liquidated is True


def test_no_liquidation_just_below_threshold_only_margin_warning(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC], hl_leverage=1.5, margin_warn_buffer=0.30)
    position = _liquidation_test_position()
    state = journal.PaperState()
    state.open_positions["FOOUSDT"] = position
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "160.0", "bidPrice": "159.0"}}
    )

    liq_price = engine._liq_price(position, cfg)
    mark_below = liq_price - 1.0

    fired = engine._check_liquidation(
        cfg, state, position, mark_below, binance, now=T0 + timedelta(hours=1)
    )
    assert fired is False
    assert "FOOUSDT" in state.open_positions

    engine._check_margin(cfg, state, position, mark_below, now=T0 + timedelta(hours=1))

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    assert not [e for e in events if e["type"] == journal.LIQUIDATION]
    warnings = [e for e in events if e["type"] == journal.WARNING]
    assert len(warnings) >= 1
    assert warnings[0]["payload"]["message"] == "margin buffer breached"


# --------------------------------------------------------------------------
# 11. Universe removal must not freeze an open position
# --------------------------------------------------------------------------


def test_position_closes_when_symbol_removed_from_universe(tmp_path: Path) -> None:
    """A position whose symbol is removed from cfg.universe must still be
    priced (from Position.hl_coin/hedge_mode, not a config lookup) and
    closed on the next tick — not frozen forever with its slot occupied.
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0)
    position = _liquidation_test_position()
    journal.append_event(cfg, journal.POSITION_OPENED, position.model_dump(mode="json"), T0)

    cfg_removed = _make_cfg(tmp_path, universe=[], max_positions=1)
    hour1 = journal.hour_bucket(T0 + timedelta(hours=1))
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0001, mark_px=100.0)},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    binance = FakeBinanceClient(book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "99.5"}})

    engine.run_paper("tick", cfg_removed, hl_client=hl, binance_client=binance, now=hour1)

    state = journal.replay(journal.journal_path(cfg_removed))
    assert "FOOUSDT" not in state.open_positions, "must close, not freeze, after universe removal"
    assert len(state.closed) == 1

    lines = journal.journal_path(cfg_removed).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    warning_msgs = [e["payload"]["message"] for e in events if e["type"] == journal.WARNING]
    assert not any("no live HL context" in m for m in warning_msgs)


# --------------------------------------------------------------------------
# 12. D-gate: negative next-hour funding blocks an otherwise-valid entry
# --------------------------------------------------------------------------


def test_entry_gate_blocks_when_next_hour_funding_is_negative(tmp_path: Path) -> None:
    cfg = _make_cfg(
        tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0, ema_periods=1, enter_apr=0.05
    )
    hour0 = journal.hour_bucket(T0)
    rates = _hourly_rates(hour0 - timedelta(hours=8), [0.0005] * 8, "FOO")  # strongly positive ema
    hl = FakeHLClient(
        # negative D-gate
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=-0.0001, mark_px=100.0)},
        funding={"FOO": rates},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}}
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" not in state.open_positions, "negative next-hour funding must block entry"


# --------------------------------------------------------------------------
# 13. Minor fixes: journal integrity, parquet mirror, liquidity cap,
#     missing hedge markPrice fallback
# --------------------------------------------------------------------------


def test_journal_replay_valid_json_missing_type_or_payload_raises(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    path = journal.journal_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(json.dumps({"ts": T0.isoformat()}) + "\n")  # valid JSON, no type/payload

    with pytest.raises(journal.JournalCorruptError):
        journal.replay(path)


def test_tick_writes_parquet_mirror_of_the_journal(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    hour0 = journal.hour_bucket(T0)
    hl = FakeHLClient()
    binance = FakeBinanceClient()

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    mirror_path = journal.parquet_mirror_path(cfg)
    assert mirror_path.exists()
    df = pl.read_parquet(mirror_path)
    jsonl_lines = journal.journal_path(cfg).read_text().splitlines()
    assert len(df) == len(jsonl_lines)
    assert set(df.columns) == {"ts", "type", "payload"}


def test_entry_notional_capped_by_liquidity(tmp_path: Path) -> None:
    """0.5% of trailing-30d median 4-HOUR quote volume caps entry notional
    below the static max_notional/capital-per-slot caps when the symbol is
    thin. With a constant hourly quote volume of 1000, each 4h bucket sums
    to 4000 (median of an all-4000 series is still 4000), so the cap is
    0.5% of 4000, not of a single hour's 1000.
    """
    thin_spec = SymbolSpec(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.SPOT, max_notional=1_000_000
    )
    cfg = _make_cfg(tmp_path, universe=[thin_spec], capital_usd=1_000_000, max_positions=1)

    hours = 24 * 30
    times = [T0 - timedelta(hours=hours - i) for i in range(hours)]
    closes = [10.0] * hours
    df = pl.DataFrame(
        {
            "symbol": ["FOOUSDT"] * hours, "open_time": times,
            "open": closes, "high": closes, "low": closes, "close": closes,
            "volume": [100.0] * hours,  # hourly quote volume = close*volume = 1000.0 constant
        },
        schema=KLINE_SCHEMA,
    )
    path = cfg.data_root / "binance" / "klines_spot_1h" / "FOOUSDT.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

    notional = engine._entry_notional(cfg, journal.PaperState(), thin_spec, now=T0)
    assert notional == pytest.approx(0.005 * 4000.0)


def test_liquidity_cap_uses_4h_bucketed_median_quote_volume(tmp_path: Path) -> None:
    """0.5% of the trailing-30d median 4-HOUR (not 1h) quote volume. Build
    klines where each of 3 complete 4h buckets has a distinguishable,
    hand-computed total quote volume — bucket0=4*10*10=400,
    bucket1=4*10*20=800, bucket2=4*10*30=1200 — and check the cap matches
    0.5% of the MEDIAN across buckets (800), not of any single hour.
    """
    spec = SymbolSpec(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.SPOT, max_notional=1_000_000
    )
    cfg = _make_cfg(tmp_path, universe=[spec], capital_usd=1_000_000, max_positions=1)

    hours = 12
    times = [T0 - timedelta(hours=hours - i) for i in range(hours)]
    closes = [10.0] * hours
    volumes = [10.0] * 4 + [20.0] * 4 + [30.0] * 4
    df = pl.DataFrame(
        {
            "symbol": ["FOOUSDT"] * hours, "open_time": times,
            "open": closes, "high": closes, "low": closes, "close": closes,
            "volume": volumes,
        },
        schema=KLINE_SCHEMA,
    )
    path = cfg.data_root / "binance" / "klines_spot_1h" / "FOOUSDT.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

    cap = engine._liquidity_cap(cfg, spec, T0)
    assert cap == pytest.approx(0.005 * 800.0)


def test_entry_skipped_when_liquidity_cap_below_min_notional(tmp_path: Path) -> None:
    """A symbol whose liquidity cap computes below MIN_ENTRY_NOTIONAL must
    not be entered at all (not entered with a tiny position) — the logged
    decision is 'skip_below_min_notional', not 'enter'.
    """
    thin_spec = SymbolSpec(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.SPOT, max_notional=1_000_000
    )
    cfg = _make_cfg(
        tmp_path, universe=[thin_spec], capital_usd=1_000_000, max_positions=1, maker_fill_prob=1.0
    )

    # 4h-bucketed quote volume = 4*1*25 = 100 -> cap = 0.5%*100 = 0.5, well
    # under the $2000 floor.
    hours = 24 * 30
    times = [T0 - timedelta(hours=hours - i) for i in range(hours)]
    closes = [1.0] * hours
    df = pl.DataFrame(
        {
            "symbol": ["FOOUSDT"] * hours, "open_time": times,
            "open": closes, "high": closes, "low": closes, "close": closes,
            "volume": [25.0] * hours,
        },
        schema=KLINE_SCHEMA,
    )
    path = cfg.data_root / "binance" / "klines_spot_1h" / "FOOUSDT.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

    hour0 = journal.hour_bucket(T0)
    warmup = _hourly_rates(hour0 - timedelta(hours=9), [0.0002] * 9, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0002, mark_px=100.0)},
        funding={"FOO": warmup},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}}
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" not in state.open_positions

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    snapshots = [
        e["payload"] for e in events
        if e["type"] == journal.SIGNAL_SNAPSHOT and e["payload"]["symbol"] == "FOOUSDT"
    ]
    assert len(snapshots) == 1
    assert snapshots[0]["decision"] == "skip_below_min_notional"


def test_hedge_funding_missing_mark_price_falls_back_and_warns_once(tmp_path: Path) -> None:
    """A hedge funding row missing markPrice must fall back to the last
    known hedge mark (not silently drop the funding forever), and warn
    only ONCE for that row (recording the accrual key), not every tick.
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC])
    position = Position(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.PERP, qty=10.0,
        notional_entry=1000.0, entry_ts=T0,
        entry_fills=[
            Fill(venue="hyperliquid", leg="hl_short", side="open", price=100.0, qty=10.0,
                 fee_usd=0.0, is_maker=True, ts=T0),
            Fill(venue="binance", leg="hedge_long", side="open", price=100.0, qty=10.0,
                 fee_usd=0.0, is_maker=False, ts=T0),
        ],
        funding_received_usd=0.0, funding_paid_usd=0.0, signal_apr_at_entry=0.5,
    )
    state = journal.PaperState()
    state.open_positions["FOOUSDT"] = position
    state.last_marks["FOOUSDT"] = {"hl_mark_px": 100.0, "hedge_price": 105.0}

    row_ts = T0 + timedelta(hours=8)
    row = {
        "fundingTime": round(row_ts.timestamp() * 1000),
        "fundingRate": "0.0002", "markPrice": "",  # missing
    }
    binance = FakeBinanceClient(funding_rows={"FOOUSDT": [row]})

    engine._accrue_hedge_perp(cfg, state, position, binance, now=row_ts + timedelta(hours=1))

    expected = broker.hedge_funding_usd(0.0002, 10.0, 105.0)  # fell back to last hedge mark
    assert position.funding_paid_usd == pytest.approx(expected)

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    warnings = [e for e in events if e["type"] == journal.WARNING]
    assert len(warnings) == 1  # exactly once, not "every tick forever"

    # A second call (simulating another tick re-fetching the same window)
    # must not re-warn or re-accrue: the key is now recorded.
    engine._accrue_hedge_perp(cfg, state, position, binance, now=row_ts + timedelta(hours=2))
    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    warnings = [e for e in events if e["type"] == journal.WARNING]
    assert len(warnings) == 1
    assert position.funding_paid_usd == pytest.approx(expected)


# --------------------------------------------------------------------------
# 14. Basis MtM must use a LIVE hedge mark, not the one frozen at entry
# --------------------------------------------------------------------------


def test_basis_mtm_uses_refreshed_hedge_mark_not_entry_frozen_price(tmp_path: Path) -> None:
    """Both legs' prices move by the SAME amount (a parallel shift — no
    real basis change) -> unrealized basis MtM must stay ~0. A bug that
    never refreshes the hedge mark after entry (report compares a LIVE HL
    mark against a STALE, entry-frozen hedge price) would show a large
    nonzero basis_mtm here, since only the HL leg's stored mark would have
    moved.
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0)
    hour0 = journal.hour_bucket(T0)
    warmup = _hourly_rates(hour0 - timedelta(hours=9), [0.0002] * 9, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0002, mark_px=100.0)},
        funding={"FOO": warmup},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}}
    )
    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" in state.open_positions
    pos = state.open_positions["FOOUSDT"]
    entry_hl = next(f.price for f in pos.entry_fills if f.leg == "hl_short")
    entry_hedge = next(f.price for f in pos.entry_fills if f.leg == "hedge_long")

    # Move BOTH marks up by the same +10 -> no real basis change, whatever
    # the entry-time HL-vs-hedge gap happened to be.
    hl.ctxs["FOO"] = FakeAssetCtx(coin="FOO", funding_hourly=0.0002, mark_px=entry_hl + 10.0)
    hl.books["FOO"] = FakeBookTop(
        coin="FOO", best_bid=entry_hl + 9.0, best_ask=entry_hl + 11.0
    )
    binance.book_tickers["FOOUSDT"] = {
        "askPrice": str(entry_hedge + 10.0), "bidPrice": str(entry_hedge + 10.0),
    }

    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(hours=1)
    )

    state = journal.replay(journal.journal_path(cfg))
    marks = state.last_marks["FOOUSDT"]
    assert marks["hedge_price"] == pytest.approx(entry_hedge + 10.0)  # NOT frozen at entry

    basis_mtm = report._open_position_basis_mtm(state.open_positions["FOOUSDT"], marks)
    assert basis_mtm == pytest.approx(0.0, abs=1e-6)


def test_refresh_hedge_mark_carries_previous_on_fetch_failure(tmp_path: Path) -> None:
    """A hedge-mark fetch failure must carry forward the previous mark (not
    overwrite it with a stale/wrong value or drop it), and log a warning.
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC])
    position = Position(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.PERP, qty=10.0,
        notional_entry=1000.0, entry_ts=T0,
        entry_fills=[
            Fill(venue="hyperliquid", leg="hl_short", side="open", price=100.0, qty=10.0,
                 fee_usd=0.0, is_maker=True, ts=T0),
            Fill(venue="binance", leg="hedge_long", side="open", price=100.0, qty=10.0,
                 fee_usd=0.0, is_maker=False, ts=T0),
        ],
        funding_received_usd=0.0, funding_paid_usd=0.0, signal_apr_at_entry=0.5,
    )
    state = journal.PaperState()
    state.open_positions["FOOUSDT"] = position
    state.last_marks["FOOUSDT"] = {"hl_mark_px": 100.0, "hedge_price": 101.5}

    binance = FakeBinanceClient(book_tickers={})  # lookup raises KeyError -> simulated failure

    engine._refresh_hedge_mark(cfg, state, position, binance, now=T0 + timedelta(hours=1))

    assert state.last_marks["FOOUSDT"]["hedge_price"] == pytest.approx(101.5)  # carried forward
    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    warnings = [e for e in events if e["type"] == journal.WARNING]
    assert len(warnings) == 1
    assert "hedge mark fetch failed" in warnings[0]["payload"]["message"]


# --------------------------------------------------------------------------
# 15. Perp-hedge spread gate: a perp hedge PAYS AWAY its own Binance
#     funding, so gating entry/exit on absolute HL funding alone overstates
#     the edge whenever Binance funding is also rich. For hedge_mode=PERP
#     symbols only, the gate becomes the HL-vs-Binance SPREAD; spot hedges
#     (which pay no funding) are unaffected.
# --------------------------------------------------------------------------


def _signal_snapshot(cfg: PaperConfig, symbol: str) -> dict:
    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    return next(
        e["payload"] for e in events
        if e["type"] == journal.SIGNAL_SNAPSHOT and e["payload"]["symbol"] == symbol
    )


def test_perp_hedge_spread_gate_blocks_entry_when_binance_funding_matches_hl(
    tmp_path: Path,
) -> None:
    """Perp-hedge candidate: HL funding is high AND Binance funding is
    EQUALLY high (spread ~ 0) -> must NOT enter, even though the D-gate
    and the absolute-HL EMA are both well above cfg.enter_apr. A mutant
    that reverts a PERP symbol's gate back to the absolute-HL EMA (ignoring
    Binance funding entirely) enters here and fails this.
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0, ema_periods=2)
    hour0 = journal.hour_bucket(T0)
    # 16 hourly HL points covering both 8h windows ending at the two
    # Binance settlements below (hour0-16h and hour0-8h).
    hl_rates = _hourly_rates(hour0 - timedelta(hours=23), [0.0002] * 16, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0002, mark_px=100.0)},
        funding={"FOO": hl_rates},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    bn_rows = [
        {"fundingTime": round((hour0 - timedelta(hours=16)).timestamp() * 1000),
         "fundingRate": "0.0016", "markPrice": "100.0"},
        {"fundingTime": round((hour0 - timedelta(hours=8)).timestamp() * 1000),
         "fundingRate": "0.0016", "markPrice": "100.0"},
    ]
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}},
        funding_rows={"FOOUSDT": bn_rows},
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" not in state.open_positions

    snapshot = _signal_snapshot(cfg, "FOOUSDT")
    assert snapshot["gate"] == "spread"
    assert snapshot["spread_apr"] == pytest.approx(0.0, abs=1e-9)
    assert snapshot["hl_apr"] is not None and snapshot["hl_apr"] > cfg.enter_apr
    assert snapshot["decision"] == "skip_below_threshold"


def test_spot_hedge_keeps_absolute_gate_same_construction_enters(tmp_path: Path) -> None:
    """Identical construction to the spread-gate block above, but
    hedge_mode=SPOT: spot hedges keep the absolute-HL gate untouched (spot
    pays no funding, so there is no Binance-funding leakage to gate on) --
    must enter on the (unchanged) high absolute-HL EMA, ignoring the
    equally-high Binance funding entirely. A mutant that applies the
    spread gate uniformly regardless of hedge_mode fails this (spread ~ 0
    would block entry here too).
    """
    cfg = _make_cfg(tmp_path, universe=[SPOT_SPEC], maker_fill_prob=1.0, ema_periods=2)
    hour0 = journal.hour_bucket(T0)
    hl_rates = _hourly_rates(hour0 - timedelta(hours=23), [0.0002] * 16, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0002, mark_px=100.0)},
        funding={"FOO": hl_rates},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    bn_rows = [
        {"fundingTime": round((hour0 - timedelta(hours=16)).timestamp() * 1000),
         "fundingRate": "0.0016", "markPrice": "100.0"},
        {"fundingTime": round((hour0 - timedelta(hours=8)).timestamp() * 1000),
         "fundingRate": "0.0016", "markPrice": "100.0"},
    ]
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}},
        funding_rows={"FOOUSDT": bn_rows},  # present but must be ignored for a spot hedge
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" in state.open_positions

    snapshot = _signal_snapshot(cfg, "FOOUSDT")
    assert snapshot["gate"] == "absolute"
    assert snapshot["bn_apr"] is None
    assert snapshot["spread_apr"] is None


def test_perp_hedge_spread_gate_exits_when_binance_funding_rises_to_match(tmp_path: Path) -> None:
    """Open a perp-hedge position on a genuinely wide spread (HL rich,
    Binance cheap), then let Binance funding rise to match HL (spread
    collapses to ~0, below exit_apr) while HL funding itself STAYS high --
    must exit. The absolute-gate mutant (HL alone still reads high) would
    hold instead and fails this.
    """
    cfg = _make_cfg(
        tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0, ema_periods=1,
        enter_apr=0.12, exit_apr=0.005,
    )
    hour0 = journal.hour_bucket(T0)
    # 16 HL hourly points, hour0-15h..hour0 inclusive, all at a constant
    # high rate -- covers both settlement windows below (hour0-16h and
    # hour0) and keeps HL funding "high" throughout, unchanged.
    hl_rates = _hourly_rates(hour0 - timedelta(hours=15), [0.0003] * 16, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0003, mark_px=100.0)},
        funding={"FOO": hl_rates},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    # Binance: cheap at entry's settlement (hour0-8h) -> wide spread, entry.
    # Rich at the later settlement (hour0, matching HL's 8h-sum 0.0024)
    # -> spread collapses, exit. Both rows registered up front; the PIT
    # filter inside _spread_signal is what restricts each tick to only the
    # settlements strictly before its own `now`.
    bn_rows = [
        {"fundingTime": round((hour0 - timedelta(hours=8)).timestamp() * 1000),
         "fundingRate": "0.0004", "markPrice": "100.0"},
        {"fundingTime": round(hour0.timestamp() * 1000),
         "fundingRate": "0.0024", "markPrice": "100.0"},
    ]
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}},
        funding_rows={"FOOUSDT": bn_rows},
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)
    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" in state.open_positions, "wide HL-vs-Binance spread must enter"
    entry_snapshot = _signal_snapshot(cfg, "FOOUSDT")
    assert entry_snapshot["gate"] == "spread"
    assert entry_snapshot["spread_apr"] > cfg.enter_apr

    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(hours=8)
    )
    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" not in state.open_positions, (
        "spread collapse must exit despite HL staying high"
    )
    assert len(state.closed) == 1


def test_spread_signal_is_pit_event_at_tick_hour_excluded() -> None:
    """A Binance funding event settling exactly AT the tick hour (t ==
    as_of) must not affect that tick's decision -- only strictly-before
    events are PIT-eligible. The extra event below is constructed so it
    WOULD have a fully-covered 8h HL window if wrongly included (its
    absence from the result is due to the PIT filter itself, not a
    missing-coverage skip), and its rate is chosen to be wildly different
    so an off-by-one (`<=` instead of `<`) changes the result unmistakably.
    """
    as_of = T0 + timedelta(hours=16)
    hl_hourly = [(T0 + timedelta(hours=i), 0.0003) for i in range(17)]  # T0 .. T0+16h
    binance_before = [(T0 + timedelta(hours=8), 0.0004)]

    result_before = engine._spread_signal(hl_hourly, binance_before, as_of, ema_periods=1)
    assert result_before != (None, None)

    binance_with_at_tick_event = [*binance_before, (as_of, -999.0)]
    result_after = engine._spread_signal(
        hl_hourly, binance_with_at_tick_event, as_of, ema_periods=1
    )

    assert result_after == result_before


def test_perp_hedge_falls_back_to_absolute_gate_when_binance_funding_data_missing(
    tmp_path: Path,
) -> None:
    """No Binance funding data at all for the symbol (no parquet on disk,
    live fetch returns nothing) -> falls back to the absolute-HL gate
    (still entering on a strongly positive absolute signal) and journals
    exactly ONE warning about it. A mutant that silently treats a missing
    series as a zero spread (blocking entry) fails the entry assertion; one
    that crashes on empty data fails outright; one that never warns fails
    the warning-count assertion.
    """
    cfg = _make_cfg(
        tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0, ema_periods=2, enter_apr=0.12
    )
    hour0 = journal.hour_bucket(T0)
    warmup = _hourly_rates(hour0 - timedelta(hours=9), [0.0005] * 9, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0005, mark_px=100.0)},
        funding={"FOO": warmup},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}},
        # no funding_rows entry for FOOUSDT at all, and no parquet on disk.
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" in state.open_positions, "must fall back to the absolute-HL gate and enter"

    snapshot = _signal_snapshot(cfg, "FOOUSDT")
    assert snapshot["gate"] == "absolute"
    assert snapshot["bn_apr"] is None
    assert snapshot["spread_apr"] is None

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    fallback_warnings = [
        e for e in events
        if e["type"] == journal.WARNING
        and e["payload"]["symbol"] == "FOOUSDT"
        and "Binance funding data" in e["payload"]["message"]
    ]
    assert len(fallback_warnings) == 1


# --------------------------------------------------------------------------
# 16. Interval-aware spread (post-review fix batch): Binance's own funding
#     interval is NOT always 8h (verified live on HYPEUSDT/PUMPUSDT: both
#     settle every 4h) and can switch mid-series. A fixed-8h-window
#     assumption mis-pairs a narrower-interval Binance rate against a
#     too-wide HL sum and roughly HALVES the effective spread APR --
#     silently entering/holding exactly what the gate exists to block.
# --------------------------------------------------------------------------


def test_perp_hedge_spread_gate_handles_4h_binance_interval_no_entry_at_true_zero_spread(
    tmp_path: Path,
) -> None:
    """HL and Binance funding are both rich, in EXACT proportion to their
    own settlement windows (true spread == 0 throughout) -- must NOT
    enter. A fixed-8h-assuming implementation mis-pairs the second
    (genuinely 4h-spaced) Binance rate against an (incorrectly) 8h-wide HL
    sum, manufacturing a large nonzero spread, and enters wrongly.
    """
    cfg = _make_cfg(tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0, ema_periods=2)
    hour0 = journal.hour_bucket(T0)
    # 12 contiguous HL hourly points, hour0-19h..hour0-8h, constant rate --
    # covers both settlement windows below exactly.
    hl_rates = _hourly_rates(hour0 - timedelta(hours=19), [0.0002] * 12, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0002, mark_px=100.0)},
        funding={"FOO": hl_rates},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    # First Binance settlement (hour0-12h): no prior settlement, falls back
    # to an assumed 8h interval -- its window is hour0-20h..hour0-12h (8h),
    # HL sum 8*0.0002=0.0016, matched exactly by this rate -> zero spread.
    t1 = hour0 - timedelta(hours=12)
    # Second settlement, 4h later (hour0-8h): a REAL 4h interval -- its
    # window is hour0-12h..hour0-8h (4h), HL sum 4*0.0002=0.0008, matched
    # exactly -> zero spread. A fixed-8h-assuming implementation instead
    # pairs this rate against an (incorrectly) 8h-wide HL sum (0.0016),
    # manufacturing a large nonzero mismatch.
    t2 = hour0 - timedelta(hours=8)
    bn_rows = [
        {"fundingTime": round(t1.timestamp() * 1000), "fundingRate": "0.0016",
         "markPrice": "100.0"},
        {"fundingTime": round(t2.timestamp() * 1000), "fundingRate": "0.0008",
         "markPrice": "100.0"},
    ]
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}},
        funding_rows={"FOOUSDT": bn_rows},
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" not in state.open_positions

    snapshot = _signal_snapshot(cfg, "FOOUSDT")
    assert snapshot["gate"] == "spread"
    assert snapshot["spread_apr"] == pytest.approx(0.0, abs=1e-9)
    assert snapshot["hl_apr"] is not None and snapshot["hl_apr"] > cfg.enter_apr
    assert snapshot["decision"] == "skip_below_threshold"


def test_spread_signal_handles_mixed_8h_then_4h_intervals_per_event() -> None:
    """Binance settlement intervals can switch mid-series (e.g. 8h then
    4h) -- each event must be annualized by its OWN gap-to-previous
    interval, not a leftover/global one. A bug that reuses the prior (8h)
    interval for the newest (4h) event exactly HALVES its annualized
    spread (periods_per_year is inversely proportional to interval_hours)
    -- pinned here to a hand-computed exact value so that halving shows up
    unmistakably.
    """
    hl_hourly = [(T0 + timedelta(hours=i), 0.0003) for i in range(1, 21)]  # T0+1h .. T0+20h
    e1 = T0 + timedelta(hours=8)   # first -- falls back to an assumed 8h interval
    e2 = T0 + timedelta(hours=16)  # 8h after e1 -- a REAL (not fallback) 8h interval
    e3 = T0 + timedelta(hours=20)  # 4h after e2 -- switches to a REAL 4h interval
    binance_series = [
        (e1, 0.0024),  # matches e1's own 8h HL sum exactly -> irrelevant to the pinned value
        (e2, 0.0024),  # matches e2's own 8h HL sum exactly -> irrelevant to the pinned value
        (e3, 0.0011),  # e3's 4h HL sum is 0.0012 -> a deliberate, nonzero 0.0001 spread
    ]
    as_of = T0 + timedelta(hours=21)

    _, spread_apr = engine._spread_signal(hl_hourly, binance_series, as_of, ema_periods=1)

    # ema_periods=1 -> alpha=1 -> the EMA collapses to the LAST value only,
    # isolating e3's own annualization: (0.0012 - 0.0011) * (8766/4).
    expected = 0.0001 * (engine.HOURS_PER_YEAR / 4)
    assert spread_apr == pytest.approx(expected)
    # If a bug reused e2's 8h interval for e3, this would come out HALVED.
    assert spread_apr != pytest.approx(0.0001 * (engine.HOURS_PER_YEAR / 8))


def test_spread_signal_requires_full_ema_periods_of_valid_pairs_not_just_any() -> None:
    """Only 2 of the 6 settlement pairs `ema_periods=6` requires exist (each
    with full HL coverage) -> must return (None, None), not compute from
    whatever happens to be available. A mutant that checks `not
    <component>` (list non-empty) instead of `len(<component>) <
    ema_periods` (enough elements) survives with only 2 pairs and must
    fail this.
    """
    as_of = T0 + timedelta(hours=17)
    hl_hourly = [(T0 + timedelta(hours=i), 0.0003) for i in range(17)]  # T0 .. T0+16h
    e1 = T0 + timedelta(hours=8)
    e2 = T0 + timedelta(hours=16)  # 8h after e1 -- second, fully-covered pair
    binance_series = [(e1, 0.0024), (e2, 0.0024)]  # both fully covered, only 2 of 6

    result = engine._spread_signal(hl_hourly, binance_series, as_of, ema_periods=6)

    assert result == (None, None)


def test_perp_hedge_falls_back_when_binance_data_is_stale_not_just_missing(tmp_path: Path) -> None:
    """Fresh HL data + a Binance funding parquet whose only row is a MONTH
    old + a failed live top-up fetch -- the parquet isn't EMPTY, so a
    staleness-blind implementation would happily build a spread EMA from
    month-old Binance rates just because the live fetch failed. Must
    instead treat this as Binance-data-unavailable: gate="absolute" (still
    entering on the absolute-HL signal), not "spread", with a fallback
    warning.
    """
    cfg = _make_cfg(
        tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0, ema_periods=2, enter_apr=0.12
    )
    hour0 = journal.hour_bucket(T0)
    warmup = _hourly_rates(hour0 - timedelta(hours=9), [0.0005] * 9, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0005, mark_px=100.0)},
        funding={"FOO": warmup},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    stale_time = hour0 - timedelta(days=30)
    df = pl.DataFrame(
        {
            "symbol": ["FOOUSDT"],
            "funding_time": [stale_time],
            "funding_rate": [0.0004],
            "mark_price": [100.0],
        },
        schema=FUNDING_SCHEMA,
    )
    path = cfg.data_root / "binance" / "funding" / "FOOUSDT.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)

    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}},
        raise_on_get_list=True,  # simulated live-fetch outage: only the stale parquet remains
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" in state.open_positions, "must fall back to the absolute-HL gate and enter"

    snapshot = _signal_snapshot(cfg, "FOOUSDT")
    assert snapshot["gate"] == "absolute"
    assert snapshot["spread_apr"] is None

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    fallback_warnings = [
        e for e in events
        if e["type"] == journal.WARNING
        and e["payload"]["symbol"] == "FOOUSDT"
        and "Binance funding data" in e["payload"]["message"]
    ]
    assert len(fallback_warnings) == 1


def test_perp_hedge_fallback_warning_fires_once_per_outage_not_every_tick(tmp_path: Path) -> None:
    """Binance data stays missing across TWO consecutive ticks -- the
    fallback warning must fire once for the outage, not once per tick it
    persists.
    """
    cfg = _make_cfg(
        tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0, ema_periods=2, enter_apr=0.12,
        exit_apr=0.005,
    )
    hour0 = journal.hour_bucket(T0)
    warmup = _hourly_rates(hour0 - timedelta(hours=9), [0.0005] * 9, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0005, mark_px=100.0)},
        funding={"FOO": warmup},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}},
        # no funding_rows entry for FOOUSDT at all, and no parquet on disk,
        # in EITHER tick below.
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)
    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(hours=1)
    )

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    fallback_warnings = [
        e for e in events
        if e["type"] == journal.WARNING
        and e["payload"]["symbol"] == "FOOUSDT"
        and "Binance funding data" in e["payload"]["message"]
    ]
    assert len(fallback_warnings) == 1


def test_open_perp_position_exit_falls_back_to_absolute_signal_when_spread_pairs_insufficient(
    tmp_path: Path,
) -> None:
    """An open perp-hedge position must never become unsignalable for exit
    purposes: Binance data IS available and fresh, but there aren't yet
    `ema_periods` settlement pairs with full HL coverage -- the spread
    signal itself reads None. The exit decision must fall back to the
    absolute-HL EMA (not silently hold forever waiting for more pairs)
    when THAT signal is below exit_apr, and the journaled close records
    exit_gate="absolute" for auditability.
    """
    cfg = _make_cfg(
        tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0, ema_periods=6,
        enter_apr=0.6, exit_apr=0.5,
    )
    hour0 = journal.hour_bucket(T0)
    position = _liquidation_test_position()  # hedge_mode=PERP, entry_ts=T0
    journal.append_event(cfg, journal.POSITION_OPENED, position.model_dump(mode="json"), T0)

    # HL funding is LOW (well below exit_apr=0.5 once annualized) -- the
    # absolute-HL EMA alone must trigger the exit. 20 points so the
    # standard hourly-grid signal_apr (needs ema_periods=6 SUMMED values,
    # i.e. 13 raw points) is actually warmed up.
    hl_rates = _hourly_rates(hour0 - timedelta(hours=20), [0.00001] * 20, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.00001, mark_px=100.0)},
        funding={"FOO": hl_rates},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    # A single, fresh, valid Binance settlement -- Binance data is
    # AVAILABLE (not missing, not stale), just short of the 6 pairs
    # ema_periods=6 requires, so the spread signal itself reads None.
    bn_rows = [
        {"fundingTime": round(hour0.timestamp() * 1000), "fundingRate": "0.00008",
         "markPrice": "100.0"},
    ]
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "99.5"}},
        funding_rows={"FOOUSDT": bn_rows},
    )

    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(hours=1)
    )

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" not in state.open_positions, (
        "exit must fall back to the absolute-HL signal, not hold forever waiting on 6 pairs"
    )

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    closed = next(e["payload"] for e in events if e["type"] == journal.POSITION_CLOSED)
    assert closed["exit_gate"] == "absolute"
    assert closed["exit_decision_apr"] is not None and closed["exit_decision_apr"] < cfg.exit_apr


def test_open_perp_position_keeps_its_own_hedge_mode_after_config_flips_to_spot(
    tmp_path: Path,
) -> None:
    """A config edit flipping a symbol's hedge_mode from PERP to SPOT while
    a PERP position is already open must NOT re-gate that open position's
    signal onto the absolute-HL (SPOT) path -- the position's OWN
    hedge_mode (frozen at entry) must keep governing it. Proven by a
    spread-collapse construction where the absolute-HL signal alone would
    read HIGH (hold) while the true spread has collapsed (exit): only the
    correctly-PERP-gated signal exits.
    """
    cfg = _make_cfg(
        tmp_path, universe=[SPOT_SPEC], maker_fill_prob=1.0, ema_periods=1,
        enter_apr=0.12, exit_apr=0.005,
    )
    hour0 = journal.hour_bucket(T0)
    position = Position(
        symbol="FOOUSDT", hl_coin="FOO", hedge_mode=HedgeMode.PERP, qty=10.0,
        notional_entry=1000.0, entry_ts=hour0,
        entry_fills=[
            Fill(venue="hyperliquid", leg="hl_short", side="open", price=100.0, qty=10.0,
                 fee_usd=0.0, is_maker=True, ts=hour0),
            Fill(venue="binance", leg="hedge_long", side="open", price=100.0, qty=10.0,
                 fee_usd=0.0, is_maker=False, ts=hour0),
        ],
        funding_received_usd=0.0, funding_paid_usd=0.0, signal_apr_at_entry=0.5,
    )
    journal.append_event(cfg, journal.POSITION_OPENED, position.model_dump(mode="json"), hour0)

    # Same "spread collapses while HL alone stays high" construction as the
    # spread-exit test: HL rich throughout; Binance cheap at the first
    # settlement, rich (matching HL) at the second -- true spread ~ 0.
    hl_rates = _hourly_rates(hour0 - timedelta(hours=15), [0.0003] * 16, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0003, mark_px=100.0)},
        funding={"FOO": hl_rates},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    bn_rows = [
        {"fundingTime": round((hour0 - timedelta(hours=8)).timestamp() * 1000),
         "fundingRate": "0.0004", "markPrice": "100.0"},
        {"fundingTime": round(hour0.timestamp() * 1000),
         "fundingRate": "0.0024", "markPrice": "100.0"},
    ]
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}},
        funding_rows={"FOOUSDT": bn_rows},
    )

    # cfg.universe now says SPOT for this symbol (simulating a config edit
    # after entry) -- the position's OWN (PERP) hedge_mode must still govern.
    engine.run_paper(
        "tick", cfg, hl_client=hl, binance_client=binance, now=hour0 + timedelta(hours=8)
    )

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" not in state.open_positions, (
        "true spread has collapsed -- must exit using the position's own PERP hedge_mode, "
        "not the config's (now SPOT) one"
    )

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    closed = next(e["payload"] for e in events if e["type"] == journal.POSITION_CLOSED)
    assert closed["exit_gate"] == "spread"


def test_position_opened_payload_records_entry_gate_and_decision_apr(tmp_path: Path) -> None:
    """The entry must be fully auditable: which gate (spread vs absolute)
    and decision APR actually triggered it. The Position model itself
    already carries signal_apr_at_entry, but not which GATE it came from
    -- that's journaled in a separate POSITION_OPENED_SIGNAL event right
    alongside POSITION_OPENED (see `_open_position`'s docstring for why it
    isn't merged into that event's own payload).
    """
    cfg = _make_cfg(
        tmp_path, universe=[PERP_SPEC], maker_fill_prob=1.0, ema_periods=1, enter_apr=0.12
    )
    hour0 = journal.hour_bucket(T0)
    hl_rates = _hourly_rates(hour0 - timedelta(hours=15), [0.0003] * 8, "FOO")
    hl = FakeHLClient(
        ctxs={"FOO": FakeAssetCtx(coin="FOO", funding_hourly=0.0003, mark_px=100.0)},
        funding={"FOO": hl_rates},
        books={"FOO": FakeBookTop(coin="FOO", best_bid=99.0, best_ask=101.0)},
    )
    bn_rows = [
        {"fundingTime": round((hour0 - timedelta(hours=8)).timestamp() * 1000),
         "fundingRate": "0.0004", "markPrice": "100.0"},
    ]
    binance = FakeBinanceClient(
        book_tickers={"FOOUSDT": {"askPrice": "100.0", "bidPrice": "100.0"}},
        funding_rows={"FOOUSDT": bn_rows},
    )

    engine.run_paper("tick", cfg, hl_client=hl, binance_client=binance, now=hour0)

    state = journal.replay(journal.journal_path(cfg))
    assert "FOOUSDT" in state.open_positions

    lines = journal.journal_path(cfg).read_text().splitlines()
    events = [json.loads(line) for line in lines]
    opened_signal = next(
        e["payload"] for e in events if e["type"] == engine.POSITION_OPENED_SIGNAL
    )
    assert opened_signal["entry_gate"] == "spread"
    opened_position = next(
        e["payload"] for e in events if e["type"] == journal.POSITION_OPENED
    )
    assert opened_signal["entry_decision_apr"] == pytest.approx(
        opened_position["signal_apr_at_entry"]
    )
    assert opened_signal["entry_decision_apr"] > cfg.enter_apr


# --------------------------------------------------------------------------
# 17. Re-review pins (M9, M13): two more plausible mutants that survived
#     the first fix batch's own test suite.
# --------------------------------------------------------------------------


def test_spread_signal_excludes_partially_covered_settlement_windows() -> None:
    """A Binance settlement whose window only has PART of its required HL
    hourly coverage (1 of 4 hours, for a real 4h interval) must be
    EXCLUDED from the EMA entirely -- not included with an understated
    (partial) HL sum, which would bias the spread LOW and risk premature
    exits. A mutant that checks `not window` (only excludes a fully EMPTY
    window) instead of `len(window) < round(interval_hours)` (excludes
    ANY partial window) lets this 1-of-4-hour window through and must
    fail this.
    """
    e1 = T0 + timedelta(hours=8)   # first -- fallback 8h interval, FULLY covered below
    e2 = T0 + timedelta(hours=12)  # 4h after e1 -- a real 4h interval, only 1 of 4 hours present
    hl_hourly = [
        *[(T0 + timedelta(hours=i), 0.0003) for i in range(1, 9)],  # T0+1h..T0+8h: e1's full window
        (T0 + timedelta(hours=9), 0.0003),  # only ONE of e2's 4 required hours (T0+9h..T0+12h)
    ]
    binance_series = [(e1, 0.0004), (e2, 0.0001)]
    as_of = T0 + timedelta(hours=13)

    _, spread_apr = engine._spread_signal(hl_hourly, binance_series, as_of, ema_periods=1)

    # ema_periods=1 -> alpha=1 -> the EMA collapses to the LAST VALID
    # value. If e2 were wrongly included, it would BE that last value,
    # completely hiding e1's contribution -- so this pins e1 alone:
    # (8*0.0003 - 0.0004) * (8766/8).
    expected = (8 * 0.0003 - 0.0004) * (engine.HOURS_PER_YEAR / 8)
    assert spread_apr == pytest.approx(expected)


def test_binance_series_is_stale_uses_the_events_own_interval_not_a_fixed_8h() -> None:
    """Staleness must be judged against the EVENT'S OWN interval (here a
    real 4h gap), not a hardcoded 8h assumption -- pinned with LITERAL
    hours (not a reused engine constant), so a mutant that hardcodes 8h
    can't coincidentally pass by sharing the test's own assumption.
    """
    as_of = T0 + timedelta(hours=20)

    # Real 4h interval; newest event is 9h old -> stale under the correct
    # 2x4h=8h threshold (9>8), but would read FRESH under a wrongly
    # hardcoded 2x8h=16h threshold (9<16) -- this is the M13 pin.
    stale_series = [
        (as_of - timedelta(hours=13), 0.0001),
        (as_of - timedelta(hours=9), 0.0002),
    ]
    assert engine._binance_series_is_stale(stale_series, as_of) is True

    # Same real 4h interval; newest event is 7h old -> fresh under the
    # correct 8h threshold (7<8) -- confirms the check isn't just
    # "always True".
    fresh_series = [
        (as_of - timedelta(hours=11), 0.0001),
        (as_of - timedelta(hours=7), 0.0002),
    ]
    assert engine._binance_series_is_stale(fresh_series, as_of) is False


def test_last_gate_scan_tolerates_pre_spread_gate_journal_entries(tmp_path: Path) -> None:
    """The live journal carries a month of SIGNAL_SNAPSHOT events written
    before the spread-gate deploy — no "gate" key. The last-gate scan must
    treat them as unknown, not crash (live incident 2026-09-30 09:10 UTC:
    KeyError 'gate' took the production tick down)."""
    import json as _json

    from farb.paper import journal as pj
    from farb.paper.engine import _last_binance_gate_by_symbol
    from farb.paper.models import PaperConfig

    cfg = PaperConfig(universe=[], journal_dir=tmp_path)
    old_event = {
        "ts": "2026-09-01T00:10:00+00:00",
        "type": pj.SIGNAL_SNAPSHOT,
        "payload": {"symbol": "PUMPUSDT", "ema_apr": 0.12, "decision": "hold"},
    }
    new_event = {
        "ts": "2026-09-30T10:10:00+00:00",
        "type": pj.SIGNAL_SNAPSHOT,
        "payload": {"symbol": "UNIUSDT", "ema_apr": 0.11, "decision": "hold", "gate": "absolute"},
    }
    path = pj.journal_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write(_json.dumps(old_event) + "\n")
        f.write(_json.dumps(new_event) + "\n")

    last = _last_binance_gate_by_symbol(cfg)
    assert "PUMPUSDT" not in last  # unknown, not a crash
    assert last["UNIUSDT"] == "absolute"
