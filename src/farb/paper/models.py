"""Shared contracts for the M2 paper executor — the single source of truth
both lanes (HL client lane, paper engine lane) conform to.

Money convention: all cash amounts in USDT/USDC treated as USD-equivalent;
rates are FRACTIONS (0.0001 = 1bp per interval), never percents.
Sign convention: the HL leg is SHORT (receives positive funding); the hedge
leg is LONG (a perp hedge PAYS positive funding; a spot hedge pays nothing).
"""

from __future__ import annotations

import tomllib
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class HedgeMode(StrEnum):
    SPOT = "spot"
    PERP = "perp"


class SymbolSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symbol: str  # with USDT suffix, e.g. "HYPEUSDT"
    hl_coin: str  # HL name, e.g. "HYPE"
    hedge_mode: HedgeMode
    max_notional: float = Field(gt=0)  # per-position cap, USD


class UniverseScanConfig(BaseModel):
    """Daily scanner that adds/retires dynamic universe members.

    Config `universe` entries are PINS the scanner never touches; scanner
    picks extend them. Membership hysteresis (add_after/remove_after
    consecutive scans) exists because churn is fees: a name flapping in and
    out of the list must not flap positions with it.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    scan_interval_hours: float = Field(default=24, ge=1)
    min_open_interest_usd: float = Field(default=7_000_000, gt=0)
    min_history_days: int = Field(default=90, ge=1)  # GRAM lesson: young = untrusted
    min_trailing_carry_apr: float = 0.08  # 30d mean funding, annualized
    add_after_scans: int = Field(default=2, ge=1)
    remove_after_scans: int = Field(default=7, ge=1)
    max_dynamic_members: int = Field(default=12, ge=0)
    default_max_notional: float = Field(default=10_000, gt=0)


class PaperConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    universe: list[SymbolSpec]
    universe_scan: UniverseScanConfig = UniverseScanConfig()
    capital_usd: float = Field(default=100_000, gt=0)
    max_positions: int = Field(default=5, ge=1)

    ema_periods: int = Field(default=6, ge=1)
    enter_apr: float = 0.12
    exit_apr: float = 0.0

    hl_maker_bps: float = 1.5
    hl_taker_bps: float = 4.5
    spot_fee_bps: float = 7.5
    perp_maker_bps: float = 2.0
    perp_taker_bps: float = 5.0
    taker_slippage_bps: float = 1.0
    maker_fill_prob: float = Field(default=0.9, ge=0, le=1)

    hl_leverage: float = Field(default=1.5, gt=0, le=3)
    margin_warn_buffer: float = 0.30

    data_root: Path = Path("data")
    journal_dir: Path = Path("data/paper")

    @model_validator(mode="after")
    def _hysteresis(self) -> PaperConfig:
        if self.enter_apr <= self.exit_apr:
            raise ValueError("enter_apr must be > exit_apr")
        return self


class Fill(BaseModel):
    """One virtual fill on one leg."""

    model_config = ConfigDict(extra="forbid")

    venue: str  # "hyperliquid" | "binance"
    leg: str  # "hl_short" | "hedge_long"
    side: str  # "open" | "close"
    price: float
    qty: float
    fee_usd: float
    is_maker: bool
    ts: datetime


class Position(BaseModel):
    """An open paper position: short HL perp + long hedge, equal qty.

    Carries hl_coin and hedge_mode so the position can be priced, accrued and
    CLOSED even if its symbol is later removed from the config universe — the
    config must never be required to unwind an existing position.
    """

    model_config = ConfigDict(extra="forbid")

    symbol: str
    hl_coin: str
    hedge_mode: HedgeMode
    qty: float
    notional_entry: float
    entry_ts: datetime
    entry_fills: list[Fill]
    funding_received_usd: float = 0.0  # HL leg, cumulative
    funding_paid_usd: float = 0.0  # hedge perp leg, cumulative (0 for spot)
    signal_apr_at_entry: float


def load_paper_config(path: Path) -> PaperConfig:
    with path.open("rb") as f:
        return PaperConfig.model_validate(tomllib.load(f))
