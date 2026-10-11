"""Command-line interface for the optimization pipeline.

Examples
--------
    building-retrofit-moo --quick
    python scripts/run_pipeline.py --no-figures
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .config import BASE_CSV, FUTURE_CSV, Settings, quick_settings
from .pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="building-retrofit-moo",
        description="Reproduce the multi-objective retrofit optimization study.",
    )
    p.add_argument("--base", type=Path, default=BASE_CSV, help="Baseline-scenario CSV")
    p.add_argument("--future", type=Path, default=FUTURE_CSV, help="Mid-future-scenario CSV")
    p.add_argument("--quick", action="store_true",
                   help="Fast smoke-test budget (small search, low DPI)")
    p.add_argument("--no-figures", action="store_true", help="Skip figure rendering")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-init", type=int, default=None, help="Bayesian init points")
    p.add_argument("--n-iter", type=int, default=None, help="Bayesian iterations")
    p.add_argument("--pop-size", type=int, default=None, help="NSGA-II population")
    p.add_argument("--n-gen", type=int, default=None, help="NSGA-II generations")
    p.add_argument("--dpi", type=int, default=None, help="Figure DPI")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = quick_settings() if args.quick else Settings()

    settings.seed = args.seed
    if args.n_init is not None:
        settings.n_init = args.n_init
    if args.n_iter is not None:
        settings.n_iter = args.n_iter
    if args.pop_size is not None:
        settings.pop_size = args.pop_size
    if args.n_gen is not None:
        settings.n_gen = args.n_gen
    if args.dpi is not None:
        settings.fig_dpi = args.dpi

    run_pipeline(settings=settings, base_csv=args.base, future_csv=args.future,
                 save_figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
