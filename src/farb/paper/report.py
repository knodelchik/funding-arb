"""Plain-text status/report rendering for the paper executor.

Both `status` and `report` render purely from a replayed `PaperState` — no
live calls — so they're always available even if HL or Binance are down,
and an open position's basis MtM here always means "as of the last tick",
never "right now".
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from farb.paper.journal import PaperState
    from farb.paper.models import PaperConfig, Position

HOURS_PER_YEAR = 8766.0  # 365.25 * 24, matching engine.py's annualization
# The verified backtest campaign's expected while-deployed APR range
# (research/2026-08-30-campaign/) — the yardstick `report` compares against.
BACKTEST_EXPECTED_APR_RANGE = (0.08, 0.15)


def _open_position_basis_mtm(pos: Position, marks: dict[str, float]) -> float | None:
    entry_hl = next((f.price for f in pos.entry_fills if f.leg == "hl_short"), None)
    entry_hedge = next((f.price for f in pos.entry_fills if f.leg == "hedge_long"), None)
    if entry_hl is None or entry_hedge is None:
        return None
    hl_mark = marks.get("hl_mark_px")
    hedge_mark = marks.get("hedge_price")
    if hl_mark is None or hedge_mark is None:
        return None
    return pos.qty * ((hedge_mark - entry_hedge) - (hl_mark - entry_hl))


def _margin_buffer(pos: Position, cfg: PaperConfig, marks: dict[str, float]) -> float | None:
    entry_hl = next((f.price for f in pos.entry_fills if f.leg == "hl_short"), None)
    hl_mark = marks.get("hl_mark_px")
    hl_margin = pos.notional_entry / cfg.hl_leverage
    if entry_hl is None or hl_mark is None or hl_margin <= 0:
        return None
    unrealized = pos.qty * (entry_hl - hl_mark)  # short: profits when price falls
    return (hl_margin + unrealized) / hl_margin


def render_status(cfg: PaperConfig, state: PaperState) -> str:
    lines: list[str] = [
        f"=== paper status ({datetime.now(UTC).isoformat()}) ===",
        f"last tick: {state.last_tick_hour.isoformat() if state.last_tick_hour else 'never'}",
        f"capital: {cfg.capital_usd:,.2f} USD, max_positions={cfg.max_positions}",
        f"open positions: {len(state.open_positions)}/{cfg.max_positions}",
        "",
    ]
    if not state.open_positions:
        lines.append("(no open positions)")
    for symbol in sorted(state.open_positions):
        pos = state.open_positions[symbol]
        marks = state.last_marks.get(symbol, {})
        buffer = _margin_buffer(pos, cfg, marks)
        buffer_str = f"{buffer:.1%}" if buffer is not None else "n/a"
        lines.append(
            f"  {symbol}: qty={pos.qty:.4f} notional={pos.notional_entry:,.2f} "
            f"entry={pos.entry_ts.isoformat()} funding_recv={pos.funding_received_usd:,.2f} "
            f"funding_paid={pos.funding_paid_usd:,.2f} margin_buffer={buffer_str} "
            f"(as of last tick)"
        )
    if state.warnings:
        lines.append("")
        lines.append(f"warnings logged: {len(state.warnings)} (see journal for detail)")
    return "\n".join(lines)


def render_report(cfg: PaperConfig, state: PaperState) -> str:
    lines: list[str] = [f"=== paper report ({datetime.now(UTC).isoformat()}) ===", ""]

    per_symbol: dict[str, dict[str, float]] = {}
    for c in state.closed:
        agg = per_symbol.setdefault(
            c.symbol,
            {
                "funding_hl": 0.0, "funding_hedge_paid": 0.0, "basis": 0.0,
                "fees": 0.0, "net": 0.0, "trades": 0,
            },
        )
        agg["funding_hl"] += c.funding_hl
        agg["funding_hedge_paid"] += c.funding_hedge_paid
        agg["basis"] += c.basis
        agg["fees"] += c.fees
        agg["net"] += c.net
        agg["trades"] += 1

    lines.append("-- closed positions (realized) --")
    if not per_symbol:
        lines.append("(none closed yet)")
    for symbol in sorted(per_symbol):
        agg = per_symbol[symbol]
        lines.append(
            f"  {symbol}: net={agg['net']:,.2f} funding_hl={agg['funding_hl']:,.2f} "
            f"funding_hedge_paid={agg['funding_hedge_paid']:,.2f} basis={agg['basis']:,.2f} "
            f"fees={agg['fees']:,.2f} trades={int(agg['trades'])}"
        )

    lines.append("")
    lines.append("-- open positions (unrealized, MtM from last journal marks) --")
    if not state.open_positions:
        lines.append("(none open)")
    open_basis_total = 0.0
    for symbol in sorted(state.open_positions):
        pos = state.open_positions[symbol]
        marks = state.last_marks.get(symbol, {})
        basis = _open_position_basis_mtm(pos, marks)
        basis_str = "n/a" if basis is None else f"{basis:,.2f}"
        if basis is not None:
            open_basis_total += basis
        buffer = _margin_buffer(pos, cfg, marks)
        buffer_str = f"{buffer:.1%}" if buffer is not None else "n/a"
        lines.append(
            f"  {symbol}: funding_recv={pos.funding_received_usd:,.2f} "
            f"funding_paid={pos.funding_paid_usd:,.2f} basis_mtm={basis_str} "
            f"margin_buffer={buffer_str} (as of last tick)"
        )

    lines.append("")
    totals = state.realized_totals
    lines.append(
        f"-- totals (realized) -- net={totals['net']:,.2f} funding_hl={totals['funding_hl']:,.2f} "
        f"funding_hedge_paid={totals['funding_hedge_paid']:,.2f} basis={totals['basis']:,.2f} "
        f"fees={totals['fees']:,.2f}"
    )
    lines.append(f"open positions' unrealized basis MtM (as of last tick): {open_basis_total:,.2f}")

    # Dollar-hour-weighted while-deployed APR across all closed trades:
    # sum(notional_i * hours_i) is a dollar-hour basis, converted to a
    # dollar-year basis so net/that reads as an APR — the multi-trade,
    # multi-notional generalization of the backtest's deployed-APR formula.
    dollar_hours = sum(
        c.notional_entry * (c.exit_ts - c.entry_ts).total_seconds() / 3600 for c in state.closed
    )
    dollar_years = dollar_hours / HOURS_PER_YEAR
    lo, hi = BACKTEST_EXPECTED_APR_RANGE
    if dollar_years > 0:
        realized_apr = totals["net"] / dollar_years
        lines.append(
            f"realized while-deployed APR: {realized_apr:.1%} "
            f"(backtest expectation: {lo:.0%}-{hi:.0%})"
        )
    else:
        lines.append(
            f"realized while-deployed APR: n/a (no closed positions yet) "
            f"(backtest expectation: {lo:.0%}-{hi:.0%})"
        )

    return "\n".join(lines)
