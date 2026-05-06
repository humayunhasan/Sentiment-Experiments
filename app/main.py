"""Optional CLI entrypoint for running the experiment."""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.config import get_settings
from app.services.runner import run_experiment
from app.utils.logger import setup_logging


async def _async_main() -> None:
    parser = argparse.ArgumentParser(description="Run sentiment benchmark experiment.")
    parser.add_argument(
        "--experiment-id",
        default=None,
        help="Fixed experiment id (default: random UUID).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Override SAMPLE_LIMIT / default 10000.",
    )
    args = parser.parse_args()

    setup_logging(logging.INFO)
    get_settings()

    exp_id = await run_experiment(
        experiment_id=args.experiment_id,
        sample_limit=args.sample_limit,
    )
    logging.getLogger(__name__).info("Done. experiment_id=%s", exp_id)


def main() -> None:
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
