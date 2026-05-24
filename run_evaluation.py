#!/usr/bin/env python3
"""
ASHAIR Evaluation Runner
Usage:
    python run_evaluation.py --swat data/swat.csv --hai data/hai.csv --ics-nad data/ics_nad.csv
    python run_evaluation.py --swat data/swat.csv --max-samples 500 --no-single-llm
"""
from __future__ import annotations
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import ASHAIRConfig, DEFAULT_CONFIG
from training.harness import EvaluationHarness, EvaluationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/evaluation.log"),
    ],
)
logger = logging.getLogger("run_evaluation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ASHAIR Full Evaluation Harness")
    parser.add_argument("--swat", type=str, default=None)
    parser.add_argument("--hai", type=str, default=None)
    parser.add_argument("--ics-nad", type=str, default=None, dest="ics_nad")
    parser.add_argument("--prompt-injection", type=str, default=None, dest="prompt_injection",
                        help="Path to synthetic prompt injection scenarios JSON")
    parser.add_argument("--results-dir", type=str, default="data/results")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-isolation-forest", action="store_true")
    parser.add_argument("--no-single-llm", action="store_true")
    parser.add_argument("--no-ashair", action="store_true")
    parser.add_argument("--vllm-endpoint", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--vector-db-path", type=str, default=None)
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ASHAIRConfig:
    config = ASHAIRConfig()
    if args.vllm_endpoint:
        config.agents.inference_endpoint = args.vllm_endpoint
    if args.model_name:
        config.agents.model_name = args.model_name
    if args.vector_db_path:
        config.rag.vector_db_path = args.vector_db_path
    config.evaluation.results_path = args.results_dir
    return config


def build_eval_config(args: argparse.Namespace) -> EvaluationConfig:
    return EvaluationConfig(
        swat_path=args.swat,
        hai_path=args.hai,
        ics_nad_path=args.ics_nad,
        results_dir=args.results_dir,
        run_isolation_forest=not args.no_isolation_forest,
        run_single_llm=not args.no_single_llm,
        run_ashair=not args.no_ashair,
        max_test_samples=args.max_samples,
        prompt_injection_scenarios_path=args.prompt_injection,
    )


def ensure_output_directories() -> None:
    for path in ["data", "data/results", "data/vector_db"]:
        os.makedirs(path, exist_ok=True)


def main() -> int:
    args = parse_args()
    ensure_output_directories()
    if not any([args.swat, args.hai, args.ics_nad]):
        logger.error("At least one dataset path must be provided (--swat, --hai, or --ics-nad)")
        return 1
    ashair_config = build_config(args)
    eval_config = build_eval_config(args)
    logger.info("Starting ASHAIR evaluation")
    logger.info("Model: %s @ %s", ashair_config.agents.model_name, ashair_config.agents.inference_endpoint)
    harness = EvaluationHarness(ashair_config=ashair_config, eval_config=eval_config)
    try:
        results = harness.run_full_evaluation()
        logger.info("Evaluation complete")
        return 0
    except Exception as exc:
        logger.error("Evaluation failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
