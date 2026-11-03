"""Framework-free core of the streaming job: scoring, latency and batch statistics, sinks.

Kept separate from Spark so it can be unit-tested with plain pandas.

Latency definition (per review): processed_ts_ms - sent_ts_ms, i.e. producer send time to
the moment the prediction is ready. It includes Kafka transit, Spark trigger wait and
inference. Producer and consumer must run on the SAME machine (same clock) for this to be valid.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABELS, MONGO_DB, MONGO_URI, RESULTS_DIR  # noqa: E402


def score_batch(pdf: pd.DataFrame, predictor, now_ms: int = None):
    """Predict a batch. Returns (scored_df, inference_ms)."""
    t0 = time.perf_counter()
    labels, conf = predictor.predict(pdf["text"].tolist())
    infer_ms = (time.perf_counter() - t0) * 1000
    done_ms = int(time.time() * 1000) if now_ms is None else now_ms
    out = pdf.copy()
    out["pred"] = labels
    out["confidence"] = conf
    out["pred_label"] = out["pred"].map(dict(enumerate(LABELS)))
    out["processed_ts_ms"] = done_ms
    out["latency_ms"] = done_ms - out["sent_ts_ms"]
    return out, infer_ms


def batch_stats(out: pd.DataFrame, infer_ms: float, wall_ms: float, batch_id: int, model: str) -> dict:
    n = len(out)
    lat = out["latency_ms"].to_numpy(dtype=float)
    stats = {
        "model": model,
        "batch_id": int(batch_id),
        "n": n,
        "inference_ms": round(infer_ms, 2),
        "batch_wall_ms": round(wall_ms, 2),
        "throughput_per_s": round(n / (wall_ms / 1000), 2) if wall_ms > 0 else None,
        "latency_mean_ms": round(float(lat.mean()), 1),
        "latency_p50_ms": round(float(np.percentile(lat, 50)), 1),
        "latency_p95_ms": round(float(np.percentile(lat, 95)), 1),
        "latency_max_ms": round(float(lat.max()), 1),
        "logged_at_ms": int(time.time() * 1000),
    }
    if "true_label" in out.columns and out["true_label"].notna().all():
        stats["batch_accuracy"] = round(float((out["pred"] == out["true_label"]).mean()), 4)
    return stats


class JsonlSink:
    """Writes to results/stream_<model>_predictions.jsonl and ..._stats.jsonl (no database needed)."""

    def __init__(self, model: str):
        RESULTS_DIR.mkdir(exist_ok=True)
        self.pred_path = RESULTS_DIR / f"stream_{model}_predictions.jsonl"
        self.stats_path = RESULTS_DIR / f"stream_{model}_stats.jsonl"

    def write_predictions(self, df: pd.DataFrame) -> None:
        with open(self.pred_path, "a") as f:
            for rec in df.to_dict("records"):
                f.write(json.dumps(rec, default=str) + "\n")

    def write_stats(self, stats: dict) -> None:
        with open(self.stats_path, "a") as f:
            f.write(json.dumps(stats) + "\n")


class MongoSink:
    def __init__(self, model: str):
        from pymongo import MongoClient  # lazy import

        self.db = MongoClient(MONGO_URI)[MONGO_DB]
        self.db["predictions"].create_index([("model", 1), ("processed_ts_ms", 1)])

    def write_predictions(self, df: pd.DataFrame) -> None:
        if len(df):
            self.db["predictions"].insert_many(df.to_dict("records"))

    def write_stats(self, stats: dict) -> None:
        self.db["batch_stats"].insert_one(dict(stats))


def make_sink(kind: str, model: str):
    return MongoSink(model) if kind == "mongo" else JsonlSink(model)
