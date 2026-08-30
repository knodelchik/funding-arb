"""Backtest configuration — loaded from TOML, every default documented.

Fees are structural, not optional: the engine takes the whole config and has
no code path that runs without them.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class FeeScenario(BaseModel):
    """One execution-style scenario, costs in basis points per leg (one way)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    spot_bps: float = Field(ge=0)
    perp_bps: float = Field(ge=0)
    slippage_bps: float = Field(ge=0)  # per taker leg, on top of the fee

    @property
    def roundtrip_frac(self) -> float:
        """Total cost of entry+exit across both legs, as a fraction of notional."""
        per_leg = (self.spot_bps + self.perp_bps + 2 * self.slippage_bps) / 10_000
        return 2 * per_leg


class BacktestConfig(BaseModel):
    # extra="forbid": a mistyped key (enter_aprr = ...) must fail loudly, not
    # silently leave the real threshold at its default and misprice the run.
    model_config = ConfigDict(extra="forbid")

    symbols: list[str]
    data_root: Path = Path("data")
    notional_usdt: float = Field(default=10_000, gt=0)

    # Signal: EMA over settled funding rates, annualized by the symbol's interval
    ema_periods: int = Field(default=6, ge=1)
    enter_apr: float = 0.12  # enter when forecast APR above this
    exit_apr: float = 0.02   # exit when forecast APR below this (hysteresis)

    scenarios: list[FeeScenario]

    # Sensitivity grid over (enter_apr, exit_apr) pairs for the report
    sensitivity_enter_apr: list[float] = [0.08, 0.12, 0.20]
    sensitivity_exit_apr: list[float] = [0.0, 0.02, 0.05]

    @model_validator(mode="after")
    def _hysteresis_is_mandatory(self) -> BacktestConfig:
        """enter <= exit would churn on every threshold crossing — the exact
        failure hysteresis exists to prevent; refuse such configs outright."""
        if self.enter_apr <= self.exit_apr:
            raise ValueError(
                f"enter_apr ({self.enter_apr}) must be > exit_apr ({self.exit_apr})"
            )
        bad = [
            (e, x)
            for e in self.sensitivity_enter_apr
            for x in self.sensitivity_exit_apr
            if e <= x
        ]
        if bad:
            raise ValueError(f"sensitivity grid pairs violate enter > exit: {bad}")
        return self


DEFAULT_SCENARIOS = [
    FeeScenario(name="taker", spot_bps=10.0, perp_bps=5.0, slippage_bps=1.0),
    FeeScenario(name="maker_bnb", spot_bps=7.5, perp_bps=2.0, slippage_bps=0.0),
]


def load_config(path: Path) -> BacktestConfig:
    with path.open("rb") as f:
        raw = tomllib.load(f)
    if "scenarios" not in raw:
        raw["scenarios"] = [s.model_dump() for s in DEFAULT_SCENARIOS]
    return BacktestConfig.model_validate(raw)
