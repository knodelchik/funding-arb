"""farb CLI: `farb collect` and `farb backtest`.

Imports of the heavy modules are lazy so the CLI parses --help even while
one lane is under construction.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from farb.schemas import DEFAULT_UNIVERSE


def main() -> None:
    parser = argparse.ArgumentParser(prog="farb", description="Binance funding-arb M0 tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_collect = sub.add_parser("collect", help="collect/refresh Binance public data into parquet")
    p_collect.add_argument("--symbols", nargs="*", default=DEFAULT_UNIVERSE)
    p_collect.add_argument("--since", default="2024-12-01", help="UTC date for first collection")
    p_collect.add_argument("--data-root", type=Path, default=Path("data"))

    p_bt = sub.add_parser("backtest", help="run cash-and-carry backtest and write report")
    p_bt.add_argument("--config", type=Path, default=Path("config/backtest.toml"))
    p_bt.add_argument("--reports-dir", type=Path, default=Path("reports"))

    p_paper = sub.add_parser("paper", help="M2 paper executor (HL-hybrid, virtual fills)")
    p_paper.add_argument("action", choices=["tick", "status", "report"])
    p_paper.add_argument("--config", type=Path, default=Path("config/paper.toml"))

    args = parser.parse_args()

    if args.command == "collect":
        from farb.data.collect import run_collect

        since = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=UTC)
        run_collect(symbols=args.symbols, since=since, data_root=args.data_root)
    elif args.command == "backtest":
        from farb.backtest.engine import run_backtest
        from farb.config import load_config

        run_backtest(cfg=load_config(args.config), reports_dir=args.reports_dir)
    elif args.command == "paper":
        from farb.paper.engine import run_paper
        from farb.paper.models import load_paper_config

        run_paper(action=args.action, cfg=load_paper_config(args.config))


if __name__ == "__main__":
    main()
