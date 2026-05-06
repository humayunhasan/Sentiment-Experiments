# Sentiment analysis benchmark (YouTube comments)

This project samples YouTube comments from MongoDB, runs several sentiment systems (OpenAI, Gemini, optional Anthropic, DeepSeek, optional Kimi/Moonshot, NLTK VADER, and an optional local Hugging Face classifier), stores per-model outputs in MongoDB, derives a **majority-vote** pseudo-label over `overall_sentiment`, and ranks models by agreement with that majority.

## Setup

1. **Python 3.11+**

2. **Install dependencies** (from this directory, `sentiment_exp/`):

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment**

   Copy `.env.example` to `.env` and set `MONGO_URI` and `DB_NAME`. The example uses **`SAMPLE_LIMIT=100`** so you can validate the pipeline before a 10k run.

   - **Anthropic** is **off by default** (`ENABLE_ANTHROPIC=false`). Set to `true` only if you intend to use Claude, install **`pip install anthropic`**, and set **`ANTHROPIC_API_KEY`** (not listed in the minimal `.env.example`).

   - **Kimi / Moonshot** is optional: set **`KIMI_API_KEY`** (and optionally `KIMI_MODEL`, `KIMI_BASE_URL`). If the key is missing, the runner still records a `kimi` result with an error and does not call the API.

   - **TabularisAI Hugging Face** runs **locally** via `transformers` (`ENABLE_HF_TABULARISAI=true` by default). No Hugging Face API token is required for inference; the model id is **`HF_TABULARISAI_MODEL`** (default `tabularisai/multilingual-sentiment-analysis`).

4. **MongoDB collections**

   - **Input:** `youtube_comments` with fields like `_id`, `text`, `video_id`, `created_at`.
   - **Output:** `sentiment_experiment_results` — one document per sampled comment and `experiment_id`.

## How to run

From `sentiment_exp/` with `PYTHONPATH` including this directory:

**Windows (PowerShell):**

```powershell
cd sentiment_exp
$env:PYTHONPATH = "."
```

**Linux/macOS:**

```bash
cd sentiment_exp
export PYTHONPATH=.
```

### Commands

```bash
python -m scripts.run_experiment
python -m scripts.compute_majority <experiment_id>
python -m scripts.evaluate_models <experiment_id>
python -m scripts.evaluate_models <experiment_id> --json
```

Start with **`SAMPLE_LIMIT=100`** (as in `.env.example`) before scaling to 10,000 comments.

## Experiment logic

1. **Sampling:** `sample_random_comments` uses `$sample` on `youtube_comments`.

2. **Models:** Per comment, active models run concurrently (`asyncio.gather`); across comments, concurrency is capped with **`CONCURRENT_COMMENTS`**.

3. **LLM prompt:** OpenAI, Gemini, Anthropic (if enabled), DeepSeek, and Kimi share the same JSON prompt in `app/models_providers/base.py` (including **entity-level** sentiment where the model returns it).

4. **NLTK VADER:** Overall label from compound score; `entities` is `[]`.

5. **Local HF:** `text-classification` with `HF_TABULARISAI_MODEL`; labels are mapped to `positive` / `negative` / `neutral` (including star-style labels when present).

6. **Persistence:** Upsert per `(comment_id, experiment_id)` after each comment.

7. **Majority:** Votes use `overall_sentiment` only from models with **`error` equal to null** (missing treated as null) and a non-null valid label. Ties resolve to **`neutral`**.

8. **Evaluation:** Leaderboard rows include `model_name`, `total`, `majority_matches`, `majority_match_rate`, `error_count`, `error_rate`, and `avg_latency_ms`.

## Stored result keys

Typical keys: `gpt_5_mini`, `gemini_flash`, `deepseek`, `kimi`, `nltk_vader`, `hf_tabularisai`. When `ENABLE_ANTHROPIC=true`, `claude_haiku` is also stored.

## Notes

- Set **`OPENAI_MODEL`** (and other provider model env vars) if your account does not support the default ids.
- First local HF run downloads weights; NLTK may download the VADER lexicon once.
