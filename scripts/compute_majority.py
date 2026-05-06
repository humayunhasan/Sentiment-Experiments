"""Recompute majority_label and model_agreement for an experiment."""

from __future__ import annotations

import argparse
import asyncio
import logging

from app.config import get_settings
from app.services.majority import recompute_majority_for_experiment
from app.utils.logger import setup_logging


async def _run() -> None:
    parser = argparse.ArgumentParser(
        description="Recompute majority vote fields from stored per-model results.",
    )
    parser.add_argument("experiment_id", help="Experiment UUID / id string.")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    get_settings()

    n = await recompute_majority_for_experiment(args.experiment_id)
    logging.getLogger(__name__).info("Updated %s documents.", n)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
