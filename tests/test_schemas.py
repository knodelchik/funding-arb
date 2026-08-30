"""Path-safety tests: symbols come from CLI/config and become file names, so
dataset_path must reject anything that could escape data_root."""

from __future__ import annotations

from pathlib import Path

import pytest

from farb.schemas import dataset_path


@pytest.mark.parametrize(
    "symbol",
    ["../../../etc/passwd", "/tmp/abs", "a", "btcusdt", "BTC-USDT", "BTC USDT", "", "X" * 21],
)
def test_dataset_path_rejects_non_symbol_strings(symbol: str) -> None:
    with pytest.raises(ValueError, match="invalid symbol"):
        dataset_path(Path("data"), "funding", symbol)


def test_dataset_path_accepts_real_symbols() -> None:
    p = dataset_path(Path("data"), "funding", "BTCUSDT")
    assert p == Path("data/binance/funding/BTCUSDT.parquet")
