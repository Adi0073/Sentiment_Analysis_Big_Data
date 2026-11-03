# Streaming Sentiment Analysis for E-commerce Reviews

Research question: **how much accuracy do you trade for speed when classifying e-commerce
reviews from a live Kafka stream?** Models compared on the same pipeline:
TF-IDF + Logistic Regression (done), DistilBERT, BERT (next).

"Live" data is a **replay**: real historical reviews are published into a real Kafka broker at a
controlled rate and consumed by Spark Structured Streaming. State this honestly in the paper.

```
reviews.csv -> prepare_data -> train / val / stream splits
                                        |
stream.csv -> producer.py (rate-controlled replay) -> Kafka -> Spark Structured Streaming
                                                                   | model.predict + latency stats
                                                              JSONL files or MongoDB
```

## Setup (Python 3.10+, Java 8/11/17 for Spark, Docker for Kafka/Mongo)

```bash
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
docker compose up -d                                    # Kafka (KRaft) on :9092, MongoDB on :27017
```

## Run

```bash
# 1. Prepare data (use your REAL Kaggle file; adjust the column names)
python src/prepare_data.py --input data/raw/reviews.csv \
    --text-col reviewText --rating-col overall --time-col unixReviewTime --max-rows 100000

# 2. Train the baseline (saves models/lr_pipeline.joblib, results/lr_metrics.json)
python src/train_baseline.py

# 3. Streaming: start the consumer first, then the producer (separate terminals)
python src/consumer_spark.py --model lr --sink jsonl
python src/producer.py --rate 100 --limit 5000
```

Outputs per run: `results/stream_lr_predictions.jsonl` and `results/stream_lr_stats.jsonl`
(per-batch size, inference ms, throughput, latency mean/p50/p95/max, batch accuracy).
Use `--sink mongo` to write to MongoDB instead (collections `predictions`, `batch_stats`).

Smoke test without real data: `python scripts/make_sample.py` creates SYNTHETIC reviews
(columns `review`, `rating`, `unix_time`). Never report results from it.

## Experiment plan (what goes in the paper)

Run each model at several rates, e.g. 50, 100, 250, 500, 1000 msgs/s, several repetitions:

| Metric | Where it comes from |
|---|---|
| Macro-F1, per-class F1 (esp. neutral) | `results/*_metrics.json`, predictions file |
| End-to-end latency (mean, p95) | `latency_*_ms` in stats |
| Max sustainable throughput | rate at which p95 latency stops being stable |
| Inference cost per review | `inference_ms / n` |

Clock note: latency = prediction-ready time minus producer send time, so producer and
consumer must run on the same machine.

## Verification status

| Part | Status |
|---|---|
| `prepare_data.py`, `train_baseline.py`, `models.py`, `make_sample.py` | Run end to end on synthetic data |
| `producer.py` pacing | Tested with `--dry-run` (200/s target, 200.5/s achieved) |
| `pipeline_core.py` (scoring, latency, stats, JSONL sink) | Unit-tested |
| `consumer_spark.py`, Kafka producer path, MongoDB sink, `docker-compose.yml` | **Written but NOT run** (no PySpark/Kafka/Docker access in the build sandbox). Expect small fixes on first run. |

## Known limitations / next steps

- Inference runs on the Spark driver (inside `foreachBatch`) so models stay comparable. Moving it to executors is a later optimisation.
- Add DistilBERT / BERT predictors in `src/models.py` (same `predict(texts) -> (labels, confidences)` interface); fine-tune on Colab, then benchmark inference locally. Record your CPU/GPU.
- Add: aggregation/dashboard, results summary script, optional aspect tagging.
- Pandas is pinned `<3` because PySpark 3.5.x `toPandas()` targets pandas 2.x.
