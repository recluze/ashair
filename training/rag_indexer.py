#!/usr/bin/env python3
"""
RAG Indexing Script — pre-populates the ASHAIR vector database from dataset CSV files.
Usage:
    python rag_indexer.py --swat path/to/swat.csv --hai path/to/hai.csv --ics-nad path/to/ics_nad.csv
"""
from __future__ import annotations
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import ASHAIRConfig, RAGConfig, DEFAULT_CONFIG
from data.harmonizer import DatasetHarmonizer
from pipeline.rag_engine import RAGEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("rag_indexer")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASHAIR RAG Vector Database Indexer")
    parser.add_argument("--swat", type=str, default=None, help="Path to SWaT dataset CSV")
    parser.add_argument("--hai", type=str, default=None, help="Path to HAI dataset CSV")
    parser.add_argument("--ics-nad", type=str, default=None, dest="ics_nad", help="Path to ICS-NAD feature CSV")
    parser.add_argument("--db-path", type=str, default="data/vector_db", help="Output vector DB directory")
    parser.add_argument("--split", type=str, choices=["train", "test", "all"], default="train",
                        help="Which split to index (train split only is recommended to prevent leakage)")
    parser.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--chunk-size", type=int, default=512)
    return parser.parse_args()


def validate_paths(args: argparse.Namespace) -> None:
    for attr, label in [("swat", "SWaT"), ("hai", "HAI"), ("ics_nad", "ICS-NAD")]:
        path = getattr(args, attr, None)
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"{label} path not found: {path}")
    if not any([args.swat, args.hai, args.ics_nad]):
        raise ValueError("At least one dataset path must be provided")


def build_rag_config(args: argparse.Namespace) -> RAGConfig:
    config = RAGConfig()
    config.vector_db_path = args.db_path
    config.embedding_model = args.embedding_model
    config.chunk_size = args.chunk_size
    return config


def index_dataset(
    rag_engine: RAGEngine,
    harmonizer: DatasetHarmonizer,
    swat_path: str,
    hai_path: str,
    ics_nad_path: str,
    split: str,
) -> int:
    train_samples, test_samples = harmonizer.build_corpus(
        swat_path=swat_path,
        hai_path=hai_path,
        ics_nad_path=ics_nad_path,
    )
    if split == "train":
        samples_to_index = train_samples
    elif split == "test":
        samples_to_index = test_samples
    else:
        samples_to_index = train_samples + test_samples
    records = [s.normalized_record for s in samples_to_index]
    logger.info("Indexing %d records (%s split) into vector database", len(records), split)
    record_ids = rag_engine.index_batch(records, show_progress=True)
    return len(record_ids)


def verify_retrieval(rag_engine: RAGEngine) -> None:
    test_query = "SWaT water treatment tank level anomaly: LIT101=0.148m below nominal range 0.500-0.800m"
    context, similarity = rag_engine.retrieve_context(test_query)
    logger.info("Retrieval verification: top similarity=%.4f", similarity)
    logger.info("Top context excerpt: %s", context[:200])


def main() -> int:
    args = parse_args()
    try:
        validate_paths(args)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Argument validation failed: %s", exc)
        return 1
    rag_config = build_rag_config(args)
    rag_engine = RAGEngine(rag_config)
    existing = rag_engine.load()
    if existing:
        logger.info("Loaded existing vector DB with %d records; appending new records", rag_engine.vector_db.record_count())
    harmonizer = DatasetHarmonizer()
    try:
        total_indexed = index_dataset(
            rag_engine=rag_engine,
            harmonizer=harmonizer,
            swat_path=args.swat,
            hai_path=args.hai,
            ics_nad_path=args.ics_nad,
            split=args.split,
        )
    except Exception as exc:
        logger.error("Indexing failed: %s", exc, exc_info=True)
        return 1
    rag_engine.save()
    logger.info("Indexed %d records. Vector DB saved to %s", total_indexed, args.db_path)
    verify_retrieval(rag_engine)
    logger.info("RAG indexing complete. Total records in DB: %d", rag_engine.vector_db.record_count())
    return 0


if __name__ == "__main__":
    sys.exit(main())
