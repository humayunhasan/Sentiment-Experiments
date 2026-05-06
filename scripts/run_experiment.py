"""Sample comments and run all sentiment models (full pipeline)."""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.config import get_settings
from app.services.runner import run_experiment
from app.utils.logger import setup_logging


async def _run() -> None:
    parser = argparse.ArgumentParser(
        description="Run full experiment: $sample from youtube_comments, all models, persist results.",
    )
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--sample-limit", type=int, default=None)
    args = parser.parse_args()

    setup_logging(logging.INFO)
    get_settings()

    exp_id = await run_experiment(
        experiment_id=args.experiment_id,
        sample_limit=args.sample_limit,
    )
    logging.getLogger(__name__).info("experiment_id=%s", exp_id)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
