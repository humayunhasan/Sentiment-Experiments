"""Print evaluation leaderboard for a completed experiment."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path

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
        # New cost/token info
        avg_in = row.get("avg_input_tokens_per_request", 0.0)
        avg_out = row.get("avg_output_tokens_per_request", 0.0)
        total_cost = row.get("estimated_cost_for_experiment", 0.0)
        cost_1m = row.get("estimated_cost_per_1m_comments", 0.0)
        print(
            f"  tokens: in={avg_in:.0f} out={avg_out:.0f} | "
            f"cost: total=${total_cost:.4f} per_1m=${cost_1m:.2f}"
        )
    print()

    # Export CSV
    output_dir = Path("exports") / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / "model_cost_accuracy_summary.csv"
    
    rows = report.get("leaderboard", [])
    if rows:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        log.info("Exported summary to %s", csv_file)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
