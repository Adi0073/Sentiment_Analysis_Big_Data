"""Turn a raw reviews CSV into train / val / stream splits with 3-class labels.

Labels: rating 1-2 -> 0 (negative), 3 -> 1 (neutral), 4-5 -> 2 (positive).

Splits:
  * If a timestamp column is given, the split is chronological:
    oldest 80% = history (70% train + 10% val), newest 20% = stream.
    This is the honest setup for a streaming evaluation (no peeking at the future).
  * Otherwise it is a stratified random split with the same proportions.

The stream split keeps its labels so the streaming pipeline can report accuracy.

Example:
  python src/prepare_data.py --input data/raw/reviews.csv \
      --text-col reviewText --rating-col overall --time-col unixReviewTime \
      --max-rows 100000
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import LABELS, PROC_DIR, SEED  # noqa: E402

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")


def rating_to_label(r: float) -> int:
    if r <= 2:
        return 0
    if r == 3:
        return 1
    return 2


def clean_text(s: str) -> str:
    # Deliberately light: transformers work best on near-raw text.
    s = TAG_RE.sub(" ", str(s))
    return WS_RE.sub(" ", s).strip()


def parse_time(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        # unix seconds (ms if values are huge)
        unit = "ms" if series.dropna().median() > 1e11 else "s"
        return pd.to_datetime(series, unit=unit, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--text-col", required=True)
    ap.add_argument("--rating-col", required=True)
    ap.add_argument("--time-col", default=None)
    ap.add_argument("--max-rows", type=int, default=None, help="random subsample for fast iteration")
    ap.add_argument("--min-chars", type=int, default=3)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} rows, columns: {list(df.columns)}")

    keep = [args.text_col, args.rating_col] + ([args.time_col] if args.time_col else [])
    df = df[keep].rename(columns={args.text_col: "text", args.rating_col: "rating"})
    if args.time_col:
        df = df.rename(columns={args.time_col: "ts"})

    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["text", "rating"])
    df = df[df["rating"].between(1, 5)]
    df["text"] = df["text"].map(clean_text)
    df = df[df["text"].str.len() >= args.min_chars].drop_duplicates(subset=["text"])
    df["label"] = df["rating"].map(rating_to_label)

    if args.max_rows and len(df) > args.max_rows:
        df = df.sample(args.max_rows, random_state=SEED)

    if "ts" in df.columns:
        df["ts"] = parse_time(df["ts"])
        df = df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)
        n = len(df)
        hist, stream = df.iloc[: int(n * 0.8)], df.iloc[int(n * 0.8):]
        train, val = hist.iloc[: int(len(hist) * 0.875)], hist.iloc[int(len(hist) * 0.875):]
        split_kind = "chronological"
    else:
        df = df.reset_index(drop=True)
        hist, stream = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=SEED)
        train, val = train_test_split(hist, test_size=0.125, stratify=hist["label"], random_state=SEED)
        split_kind = "stratified random"

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("stream", stream)]:
        part = part.reset_index(drop=True).copy()
        part.insert(0, "id", [f"{name}-{i}" for i in range(len(part))])
        part.to_csv(PROC_DIR / f"{name}.csv", index=False)
        dist = part["label"].value_counts(normalize=True).sort_index()
        dist_s = ", ".join(f"{LABELS[k]} {v:.1%}" for k, v in dist.items())
        print(f"{name:7s} {len(part):>8,} rows | {dist_s}")
    print(f"Split type: {split_kind}. Files written to {PROC_DIR}")


if __name__ == "__main__":
    main()
