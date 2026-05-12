"""Export model accuracy and random samples of wrong predictions to CSV."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.db import get_results_collection
from app.utils.logger import setup_logging

logger = logging.getLogger(__name__)

MODELS = [
    "gpt_5_mini",
    "gemini_flash",
    "claude_haiku",
    "deepseek",
    "nltk_vader",
    "hf_tabularisai",
]


async def export_accuracy_summary(
    coll: Any, experiment_id: str, output_dir: Path
) -> list[dict[str, Any]]:
    """Compute accuracy summary for all models and save to CSV."""
    summary_data = []

    for model in MODELS:
        pipeline = [
            {
                "$match": {
                    "experiment_id": experiment_id,
                    "majority_label.eligible_models": {"$gte": 4},
                }
            },
            {
                "$group": {
                    "_id": None,
                    "total_valid": {
                        "$sum": {
                            "$cond": [{"$eq": [f"$results.{model}.error", None]}, 1, 0]
                        }
                    },
                    "total_wrong": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$eq": [f"$model_agreement.{model}", False]},
                                        {"$eq": [f"$results.{model}.error", None]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                    "total_correct": {
                        "$sum": {
                            "$cond": [
                                {
                                    "$and": [
                                        {"$eq": [f"$model_agreement.{model}", True]},
                                        {"$eq": [f"$results.{model}.error", None]},
                                    ]
                                },
                                1,
                                0,
                            ]
                        }
                    },
                    "error_count": {
                        "$sum": {
                            "$cond": [{"$ne": [f"$results.{model}.error", None]}, 1, 0]
                        }
                    },
                    "latency_sum": {"$sum": f"$results.{model}.latency_ms"},
                    "latency_count": {
                        "$sum": {
                            "$cond": [
                                {"$ne": [f"$results.{model}.latency_ms", None]},
                                1,
                                0,
                            ]
                        }
                    },
                }
            },
        ]

        cursor = coll.aggregate(pipeline)
        result_list = await cursor.to_list(length=1)
        
        if not result_list:
            # No data for this experiment/model
            stats = {
                "model_name": model,
                "total_valid": 0,
                "total_wrong": 0,
                "total_correct": 0,
                "accuracy_percent": 0.0,
                "error_count": 0,
                "avg_latency_ms": 0.0,
            }
        else:
            r = result_list[0]
            total_valid = r.get("total_valid", 0)
            total_correct = r.get("total_correct", 0)
            latency_count = r.get("latency_count", 0)
            avg_latency = (
                r.get("latency_sum", 0) / latency_count if latency_count > 0 else 0.0
            )
            accuracy = (total_correct / total_valid * 100) if total_valid > 0 else 0.0

            stats = {
                "model_name": model,
                "total_valid": total_valid,
                "total_wrong": r.get("total_wrong", 0),
                "total_correct": total_correct,
                "accuracy_percent": round(accuracy, 2),
                "error_count": r.get("error_count", 0),
                "avg_latency_ms": round(avg_latency, 2),
            }
        summary_data.append(stats)

    summary_file = output_dir / "model_accuracy_summary.csv"
    with open(summary_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
        writer.writeheader()
        writer.writerows(summary_data)

    logger.info("Exported summary to %s", summary_file)
    return summary_data


async def export_wrong_samples(coll: Any, experiment_id: str, model: str, output_dir: Path):
    """Export up to 100 random wrong predictions for a model."""
    pipeline = [
        {
            "$match": {
                "experiment_id": experiment_id,
                "majority_label.eligible_models": {"$gte": 4},
                f"model_agreement.{model}": False,
                f"results.{model}.error": None,
            }
        },
        {"$sample": {"size": 100}},
        {
            "$project": {
                "comment_id": 1,
                "text": 1,
                "majority_sentiment": "$majority_label.overall_sentiment",
                "majority_votes": "$majority_label.votes",
                "model_sentiment": f"$results.{model}.overall_sentiment",
                "model_confidence": f"$results.{model}.confidence",
                "model_reason": f"$results.{model}.reason",
                "model_entities": f"$results.{model}.entities",
                "raw_response": f"$results.{model}.raw_response",
            }
        },
    ]

    cursor = coll.aggregate(pipeline)
    samples = await cursor.to_list(length=100)

    if not samples:
        logger.info("No wrong samples found for model %s", model)
        return

    sample_file = output_dir / f"wrong_samples_{model}.csv"
    
    # Flatten/stringify complex fields
    for s in samples:
        if "_id" in s:
            del s["_id"]
        if "majority_votes" in s:
            s["majority_votes"] = json.dumps(s["majority_votes"])
        if "model_entities" in s:
            s["model_entities"] = json.dumps(s["model_entities"])

    with open(sample_file, "w", newline="", encoding="utf-8") as f:
        if samples:
            writer = csv.DictWriter(f, fieldnames=samples[0].keys())
            writer.writeheader()
            writer.writerows(samples)

    logger.info("Exported %d wrong samples for %s to %s", len(samples), model, sample_file)


async def main():
    parser = argparse.ArgumentParser(description="Export model errors for an experiment.")
    parser.add_argument("experiment_id", help="The ID of the experiment to export.")
    args = parser.parse_args()

    setup_logging(logging.INFO)
    settings = get_settings()
    coll = get_results_collection(settings)

    output_dir = Path("exports") / args.experiment_id
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting errors for experiment %s to %s", args.experiment_id, output_dir)

    # 1. Export summary
    await export_accuracy_summary(coll, args.experiment_id, output_dir)

    # 2. Export wrong samples for each model
    for model in MODELS:
        await export_wrong_samples(coll, args.experiment_id, model, output_dir)

    logger.info("Done.")


if __name__ == "__main__":
    asyncio.run(main())
