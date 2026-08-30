"""Config validation tests: hysteresis and strictness are load-time guards,
so a bad TOML must fail loudly before any number is computed."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from farb.config import DEFAULT_SCENARIOS, BacktestConfig


def _base(**overrides: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "symbols": ["BTCUSDT"],
        "scenarios": [s.model_dump() for s in DEFAULT_SCENARIOS],
    }
    cfg.update(overrides)
    return cfg


def test_enter_apr_must_exceed_exit_apr() -> None:
    with pytest.raises(ValidationError, match="must be > exit_apr"):
        BacktestConfig.model_validate(_base(enter_apr=0.02, exit_apr=0.02))


def test_sensitivity_grid_pairs_must_keep_hysteresis() -> None:
    with pytest.raises(ValidationError, match="sensitivity grid"):
        BacktestConfig.model_validate(
            _base(sensitivity_enter_apr=[0.05], sensitivity_exit_apr=[0.05])
        )


def test_unknown_keys_are_rejected_not_silently_dropped() -> None:
    """extra="forbid": a typo like enter_aprr must not leave the real
    threshold at its default and misprice the whole backtest."""
    with pytest.raises(ValidationError, match="enter_aprr"):
        BacktestConfig.model_validate(_base(enter_aprr=0.99))


def test_valid_config_loads() -> None:
    cfg = BacktestConfig.model_validate(_base())
    assert cfg.enter_apr > cfg.exit_apr
    assert cfg.scenarios[0].roundtrip_frac > 0
