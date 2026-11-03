"""Replay the held-out stream split into Kafka at a controlled rate.

This is the 'live data' substitute: real historical reviews, real Kafka broker,
but you decide the arrival rate, so experiments are repeatable.

Examples:
  python src/producer.py --rate 100 --limit 5000
  python src/producer.py --rate 500 --limit 20000 --loops 3
  python src/producer.py --rate 100 --limit 300 --dry-run     # no Kafka needed
"""
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import KAFKA_BOOTSTRAP, KAFKA_TOPIC, PROC_DIR  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=100, help="target messages per second")
    ap.add_argument("--limit", type=int, default=5000, help="messages per pass")
    ap.add_argument("--loops", type=int, default=1, help="repeat the data to reach larger volumes")
    ap.add_argument("--input", default=str(PROC_DIR / "stream.csv"))
    ap.add_argument("--topic", default=KAFKA_TOPIC)
    ap.add_argument("--dry-run", action="store_true", help="do not contact Kafka; just test pacing")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if "ts" in df.columns:
        df = df.sort_values("ts")  # replay in original chronological order
    df = df.head(args.limit).reset_index(drop=True)

    producer = None
    if not args.dry_run:
        from kafka import KafkaProducer  # lazy import

        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            linger_ms=5,
            acks=1,
        )

    total = len(df) * args.loops
    interval = 1.0 / args.rate
    print(f"Replaying {total:,} reviews at ~{args.rate:g}/s to topic '{args.topic}'"
          f"{' (DRY RUN)' if args.dry_run else ''}")

    start = time.perf_counter()
    sent = 0
    for loop in range(args.loops):
        for row in df.itertuples(index=False):
            target = start + sent * interval
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            msg = {
                "id": f"{row.id}-L{loop}",
                "text": row.text,
                "true_label": int(row.label),
                "sent_ts_ms": int(time.time() * 1000),
            }
            if producer:
                producer.send(args.topic, msg)
            sent += 1
            if sent % max(1, int(args.rate) * 5) == 0:
                print(f"  sent {sent:,}/{total:,}")
    if producer:
        producer.flush()
    elapsed = time.perf_counter() - start
    print(f"Done: {sent:,} messages in {elapsed:.1f}s -> achieved {sent / elapsed:.1f}/s "
          f"(target {args.rate:g}/s)")


if __name__ == "__main__":
    main()
