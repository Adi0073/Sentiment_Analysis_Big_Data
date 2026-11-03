"""Predictors share one interface so the streaming code is model-agnostic.

    predictor.predict(list_of_texts) -> (list[int] labels, list[float] confidences)

Add a DistilBERT / BERT predictor here later with the same signature.
"""
import sys
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODELS_DIR  # noqa: E402


class LRPredictor:
    name = "lr"

    def __init__(self, path: Path = MODELS_DIR / "lr_pipeline.joblib"):
        self.pipe = joblib.load(path)

    def predict(self, texts: List[str]) -> Tuple[List[int], List[float]]:
        proba = self.pipe.predict_proba(texts)
        idx = np.argmax(proba, axis=1)
        classes = self.pipe.classes_
        return [int(classes[i]) for i in idx], proba.max(axis=1).round(4).tolist()


def load_predictor(kind: str):
    if kind == "lr":
        return LRPredictor()
    raise ValueError(f"Unknown model '{kind}'. Available: lr (transformers coming next).")
