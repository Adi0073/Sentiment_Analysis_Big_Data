"""Generate a SYNTHETIC reviews CSV to smoke-test the pipeline plumbing.

This is NOT real data. Never report results from it. Use the real Kaggle
Amazon/Flipkart dataset for anything that goes into your paper.
"""
import argparse
import random
from pathlib import Path

import pandas as pd

POS = ["works great", "love it", "excellent quality", "very happy with this", "fast delivery and perfect fit",
       "totally worth the money", "exceeded my expectations", "battery lasts long", "highly recommend"]
NEG = ["stopped working", "terrible quality", "waste of money", "very disappointed", "arrived broken",
       "does not match the description", "poor build and slow delivery", "would not recommend", "battery drains fast"]
NEU = ["it is okay", "average product", "does the job but nothing special", "decent for the price",
       "some good some bad", "not bad not great", "as expected, nothing more"]
FILL = ["the", "this product", "overall", "honestly", "after a week of use", "for daily use", "I bought it for my family"]


def make_review(rating: int, rng: random.Random) -> str:
    pool = {1: NEG, 2: NEG, 3: NEU, 4: POS, 5: POS}[rating]
    parts = rng.sample(pool, k=2) + rng.sample(FILL, k=2)
    rng.shuffle(parts)
    # a little label noise so metrics are not trivially perfect
    if rng.random() < 0.08:
        parts.append(rng.choice(POS + NEG + NEU))
    return ", ".join(parts).capitalize() + "."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=6000)
    ap.add_argument("--out", default="data/raw/sample_reviews.csv")
    args = ap.parse_args()
    rng = random.Random(42)
    ratings = rng.choices([1, 2, 3, 4, 5], weights=[8, 6, 10, 26, 50], k=args.rows)  # imbalanced like real reviews
    df = pd.DataFrame({
        "review": [make_review(r, rng) + f" (#{i})" for i, r in enumerate(ratings)],
        "rating": ratings,
        "unix_time": [1_600_000_000 + i * 60 for i in range(args.rows)],
    })
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} SYNTHETIC rows to {args.out}")


if __name__ == "__main__":
    main()
