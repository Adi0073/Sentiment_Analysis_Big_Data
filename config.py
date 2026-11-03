"""Central configuration. Override anything via environment variables."""
import os
from pathlib import Path

class Config:
    ROOT = Path(__file__).resolve().parent
    DATA_DIR = ROOT / "data"
    RAW_DIR = DATA_DIR / "raw"
    PROC_DIR = DATA_DIR / "processed"
    CKPT_DIR = DATA_DIR / "checkpoints"
    MODELS_DIR = ROOT / "models"
    RESULTS_DIR = ROOT / "results"
    BERT_MODEL_NAME="bert-base-uncased"

    # Kafka / Mongo
    KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
    KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "reviews")
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB = os.getenv("MONGO_DB", "ecom_sentiment")

    # Spark Kafka connector. The version MUST match your installed pyspark version
    # (requirements.txt pins pyspark==3.5.1 -> Scala 2.12, Spark 3.5.1).
    SPARK_KAFKA_PACKAGE = os.getenv(
        "SPARK_KAFKA_PACKAGE", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1"
    )

    # Labels: 1-2 stars = negative, 3 = neutral, 4-5 = positive
    LABELS = ["negative", "neutral", "positive"]
    SEED = 42
