"""Append-only JSONL journal — the paper executor's single source of truth.

The engine never keeps state in memory across runs: every `tick` starts by
calling `replay()` to reconstruct open positions, accrual/idempotency
guards, and realized totals from the journal on disk. This is what makes a
paper-trading process safe to restart, cron-schedule, or run twice by
accident.

Each line is one JSON object: `{"ts": <UTC iso>, "type": <event type>,
"payload": {...}}`. Datetimes inside a payload are always pre-serialized to
ISO-8601 strings by the caller (never left for a `json.dumps(default=...)`
fallback) so replay can parse them back unambiguously.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from farb.paper.models import PaperConfig, Position

# Event type constants — used as string literals everywhere, kept here once
# so a typo in an event type name is a NameError, not a silent mismatch.
TICK_STARTED = "tick_started"
TICK_SKIPPED_DUPLICATE = "tick_skipped_duplicate"
SIGNAL_SNAPSHOT = "signal_snapshot"
POSITION_OPENED = "position_opened"
FUNDING_ACCRUED = "funding_accrued"
POSITION_CLOSED = "position_closed"
LIQUIDATION = "liquidation"
WARNING = "warning"

LEG_HL = "hl"
LEG_HEDGE = "hedge"


def hour_bucket(ts: datetime) -> datetime:
    """Floor `ts` to the UTC hour — the idempotency/accrual bucket unit."""
    return ts.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def journal_path(cfg: PaperConfig) -> Path:
    return cfg.journal_dir / "journal.jsonl"


def parquet_mirror_path(cfg: PaperConfig) -> Path:
    return cfg.journal_dir / "journal.parquet"


def write_parquet_mirror(cfg: PaperConfig) -> None:
    """Rebuild a parquet mirror of the journal: one row per event (ts, type,
    payload as a JSON string).

    Always rebuilt whole from the JSONL (the real source of truth) rather
    than appended incrementally, so the mirror can never drift out of sync
    with it — "simple" per the spec, and correctness here matters more than
    the cost of a full re-read once per tick.
    """
    path = journal_path(cfg)
    rows: list[dict[str, Any]] = []
    if path.exists():
        with path.open() as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                rows.append(
                    {
                        "ts": datetime.fromisoformat(event["ts"]),
                        "type": event["type"],
                        "payload": json.dumps(event["payload"]),
                    }
                )
    df = pl.DataFrame(
        rows, schema={"ts": pl.Datetime("us", "UTC"), "type": pl.Utf8, "payload": pl.Utf8}
    )
    out_path = parquet_mirror_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(out_path)


def append_event(cfg: PaperConfig, event_type: str, payload: dict[str, Any], ts: datetime) -> None:
    """Append one event. `ts` is required (not defaulted to "now") so a tick
    can log every event under the same, single decision timestamp.
    """
    path = journal_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps({"ts": ts.isoformat(), "type": event_type, "payload": payload})
    with path.open("a") as f:
        f.write(line + "\n")


@dataclass
class ClosedPosition:
    """One realized position's decomposition, as logged at `position_closed`."""

    symbol: str
    entry_ts: datetime
    exit_ts: datetime
    notional_entry: float
    funding_hl: float
    funding_hedge_paid: float
    basis: float
    fees: float
    net: float
    liquidated: bool = False


@dataclass
class PaperState:
    """Reconstructed from the journal by `replay()` — the engine's entire
    working memory. Nothing here is ever trusted across a process restart
    except what came out of a replay.
    """

    open_positions: dict[str, Position] = field(default_factory=dict)
    last_tick_hour: datetime | None = None
    # (symbol, iso-hour) already accrued for the HL leg; (symbol, iso-time)
    # already accrued for the hedge-perp leg. Two separate key spaces
    # because HL settles hourly and a perp hedge settles on its own
    # (typically 8h) schedule.
    hl_accrual_keys: set[tuple[str, str]] = field(default_factory=set)
    hedge_accrual_keys: set[tuple[str, str]] = field(default_factory=set)
    # symbol -> last observed marks, updated by funding_accrued and seeded by
    # position_opened; used by `report`/`status` for MtM without a live call.
    last_marks: dict[str, dict[str, float]] = field(default_factory=dict)
    # symbol -> {iso-hour: rate}, rebuilt from HL-leg funding_accrued events —
    # this IS "the journal's stored funding" the signal warmup can draw on.
    hl_funding_history: dict[str, dict[str, float]] = field(default_factory=dict)
    closed: list[ClosedPosition] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    realized_totals: dict[str, float] = field(
        default_factory=lambda: {
            "funding_hl": 0.0,
            "funding_hedge_paid": 0.0,
            "basis": 0.0,
            "fees": 0.0,
            "net": 0.0,
        }
    )


class JournalCorruptError(ValueError):
    """A journal line could not be parsed. Raised loudly rather than
    silently skipped — a corrupted journal must never be replayed as if it
    were a shorter, valid one; that would misstate real position/PnL state.
    """


def _apply_position_opened(state: PaperState, payload: dict[str, Any]) -> None:
    position = Position.model_validate(payload)
    state.open_positions[position.symbol] = position
    hl_fill = next(f for f in position.entry_fills if f.leg == "hl_short")
    hedge_fill = next(f for f in position.entry_fills if f.leg == "hedge_long")
    state.last_marks[position.symbol] = {
        "hl_mark_px": hl_fill.price,
        "hedge_price": hedge_fill.price,
    }


def _apply_funding_accrued(state: PaperState, payload: dict[str, Any]) -> None:
    symbol = payload["symbol"]
    leg = payload["leg"]
    key = (symbol, payload["key"])
    position = state.open_positions.get(symbol)
    if leg == LEG_HL:
        if key in state.hl_accrual_keys:
            return  # already applied — idempotent replay
        state.hl_accrual_keys.add(key)
        if position is not None:
            position.funding_received_usd += payload["amount_usd"]
        state.last_marks.setdefault(symbol, {})["hl_mark_px"] = payload["mark_px"]
        state.hl_funding_history.setdefault(symbol, {})[payload["key"]] = payload["rate"]
    elif leg == LEG_HEDGE:
        if key in state.hedge_accrual_keys:
            return
        state.hedge_accrual_keys.add(key)
        if position is not None:
            position.funding_paid_usd += payload["amount_usd"]
        state.last_marks.setdefault(symbol, {})["hedge_price"] = payload["mark_px"]


def _apply_position_closed(state: PaperState, payload: dict[str, Any]) -> None:
    symbol = payload["symbol"]
    position = state.open_positions.pop(symbol, None)
    entry_ts = (
        position.entry_ts if position is not None else datetime.fromisoformat(payload["exit_ts"])
    )
    notional_entry = position.notional_entry if position is not None else 0.0
    closed = ClosedPosition(
        symbol=symbol,
        entry_ts=entry_ts,
        exit_ts=datetime.fromisoformat(payload["exit_ts"]),
        notional_entry=notional_entry,
        funding_hl=payload["funding_hl"],
        funding_hedge_paid=payload["funding_hedge_paid"],
        basis=payload["basis"],
        fees=payload["fees"],
        net=payload["net"],
        liquidated=payload.get("liquidated", False),
    )
    state.closed.append(closed)
    state.realized_totals["funding_hl"] += closed.funding_hl
    state.realized_totals["funding_hedge_paid"] += closed.funding_hedge_paid
    state.realized_totals["basis"] += closed.basis
    state.realized_totals["fees"] += closed.fees
    state.realized_totals["net"] += closed.net


_APPLIERS = {
    POSITION_OPENED: _apply_position_opened,
    FUNDING_ACCRUED: _apply_funding_accrued,
    POSITION_CLOSED: _apply_position_closed,
}


def replay(path: Path) -> PaperState:
    """Reconstruct a `PaperState` by replaying every event in order.

    A missing journal file is a fresh, empty state (the normal case for a
    brand-new deployment) — not an error. A journal file that exists but
    contains a line that fails to parse IS an error: silently dropping a
    bad line (especially the last one, the shape a mid-write crash leaves
    behind) would understate real position/PnL state without any signal
    that anything was wrong.
    """
    state = PaperState()
    if not path.exists():
        return state

    with path.open() as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise JournalCorruptError(
                    f"{path}:{line_no}: corrupt journal line: {exc}"
                ) from exc

            try:
                event_type = event["type"]
                payload = event["payload"]
            except (KeyError, TypeError) as exc:
                raise JournalCorruptError(
                    f"{path}:{line_no}: journal line missing 'type'/'payload': {exc}"
                ) from exc
            if event_type == TICK_STARTED:
                state.last_tick_hour = datetime.fromisoformat(payload["hour"])
            elif event_type == WARNING:
                state.warnings.append(payload)
            elif event_type in _APPLIERS:
                _APPLIERS[event_type](state, payload)
            # SIGNAL_SNAPSHOT and TICK_SKIPPED_DUPLICATE carry no state to
            # reconstruct — they exist purely as an audit trail.

    return state


__all__ = [
    "FUNDING_ACCRUED",
    "LEG_HEDGE",
    "LEG_HL",
    "LIQUIDATION",
    "POSITION_CLOSED",
    "POSITION_OPENED",
    "SIGNAL_SNAPSHOT",
    "TICK_SKIPPED_DUPLICATE",
    "TICK_STARTED",
    "WARNING",
    "ClosedPosition",
    "JournalCorruptError",
    "PaperState",
    "append_event",
    "hour_bucket",
    "journal_path",
    "parquet_mirror_path",
    "replay",
    "write_parquet_mirror",
]
