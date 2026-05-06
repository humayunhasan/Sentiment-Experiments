"""Print evaluation leaderboard for a completed experiment."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from app.config import get_settings
from app.services.evaluator import evaluate_experiment
from app.utils.logger import setup_logging


async def _run() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models vs majority pseudo-labels.")
    parser.add_argument("experiment_id")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit full JSON report to stdout.",
    )
    args = parser.parse_args()

    setup_logging(logging.INFO)
    get_settings()

    report = await evaluate_experiment(args.experiment_id)
    log = logging.getLogger(__name__)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    log.info("Experiment %s — documents: %s", args.experiment_id, report["total_documents"])
    print(
        "\n=== Leaderboard "
        "(model_name, total, majority_matches, majority_match_rate, "
        "error_count, error_rate, avg_latency_ms) ===\n"
    )
    for row in report.get("leaderboard", []):
        name = row.get("model_name", "")
        total = row.get("total", 0)
        mm = row.get("majority_matches", 0)
        mmr = row.get("majority_match_rate", 0.0)
        ec = row.get("error_count", 0)
        er = row.get("error_rate", 0.0)
        lat = row.get("avg_latency_ms")
        lat_s = f"{lat:.1f}" if lat is not None else "n/a"
        print(
            f"{name:18}  total={total}  maj_match={mm}  maj_rate={mmr:.4f}  "
            f"err={ec}  err_rate={er:.4f}  avg_ms={lat_s}"
        )
    print()


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
