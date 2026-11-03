"""Baseline: TF-IDF + Logistic Regression (class-weighted for the neutral-class problem).

Reports macro-F1, per-class F1 and CPU inference cost, and saves the model + metrics.
These numbers are the 'fast model' side of your accuracy-vs-latency comparison.
"""
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABELS, MODELS_DIR, PROC_DIR, RESULTS_DIR, SEED  # noqa: E402


def evaluate(pipe, df: pd.DataFrame, name: str) -> dict:
    pred = pipe.predict(df["text"].tolist())
    macro = f1_score(df["label"], pred, average="macro")
    per_class = f1_score(df["label"], pred, average=None, labels=[0, 1, 2])
    print(f"\n=== {name} ===")
    print(classification_report(df["label"], pred, target_names=LABELS, digits=3, zero_division=0))
    return {
        "macro_f1": round(float(macro), 4),
        "accuracy": round(float((pred == df["label"].to_numpy()).mean()), 4),
        **{f"f1_{LABELS[i]}": round(float(v), 4) for i, v in enumerate(per_class)},
    }


def measure_latency(pipe, texts: list) -> dict:
    rng = np.random.default_rng(SEED)
    sample = [texts[i] for i in rng.choice(len(texts), size=min(1000, len(texts)), replace=False)]
    t0 = time.perf_counter()
    pipe.predict(sample)
    batch_ms = (time.perf_counter() - t0) * 1000
    singles = []
    for t in sample[:200]:
        s = time.perf_counter()
        pipe.predict([t])
        singles.append((time.perf_counter() - s) * 1000)
    return {
        "batch_ms_per_review": round(batch_ms / len(sample), 4),
        "single_ms_p50": round(float(np.percentile(singles, 50)), 4),
        "single_ms_p95": round(float(np.percentile(singles, 95)), 4),
    }


def main() -> None:
    train = pd.read_csv(PROC_DIR / "train.csv")
    val = pd.read_csv(PROC_DIR / "val.csv")
    stream = pd.read_csv(PROC_DIR / "stream.csv")

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    t0 = time.perf_counter()
    pipe.fit(train["text"].tolist(), train["label"])
    train_s = time.perf_counter() - t0
    print(f"Trained on {len(train):,} reviews in {train_s:.1f}s")

    metrics = {
        "model": "tfidf_logreg",
        "train_seconds": round(train_s, 2),
        "val": evaluate(pipe, val, "Validation"),
        "stream_holdout": evaluate(pipe, stream, "Stream hold-out (replayed later)"),
        "latency_cpu": measure_latency(pipe, stream["text"].tolist()),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / "lr_pipeline.joblib")
    (RESULTS_DIR / "lr_metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nSaved models/lr_pipeline.joblib and results/lr_metrics.json")
    print(json.dumps(metrics["latency_cpu"], indent=2))


if __name__ == "__main__":
    main()
