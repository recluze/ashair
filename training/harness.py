from __future__ import annotations
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from configs.config import ASHAIRConfig
from data.harmonizer import DatasetHarmonizer, HarmonizedSample, LABEL_BENIGN
from training.baselines import IsolationForestBaseline, SingleLLMAgentBaseline
from training.metrics import (
    MetricsAccumulator, MetricSnapshot, PerAttackVectorMetrics, PromptInjectionMetrics
)
from pipeline.orchestrator import ASHAIROrchestrator, IncidentResponseResult
from pipeline.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    swat_path: Optional[str] = None
    hai_path: Optional[str] = None
    ics_nad_path: Optional[str] = None
    results_dir: str = "data/results"
    run_isolation_forest: bool = True
    run_single_llm: bool = True
    run_ashair: bool = True
    max_test_samples: Optional[int] = None
    prompt_injection_scenarios_path: Optional[str] = None


class EvaluationHarness:
    def __init__(self, ashair_config: ASHAIRConfig, eval_config: EvaluationConfig):
        self.ashair_config = ashair_config
        self.eval_config = eval_config
        self.harmonizer = DatasetHarmonizer(random_seed=ashair_config.evaluation.random_seed)
        os.makedirs(eval_config.results_dir, exist_ok=True)

    def run_full_evaluation(self) -> Dict[str, Any]:
        logger.info("Starting full ASHAIR evaluation")
        train_samples, test_samples = self.harmonizer.build_corpus(
            swat_path=self.eval_config.swat_path,
            hai_path=self.eval_config.hai_path,
            ics_nad_path=self.eval_config.ics_nad_path,
        )
        if self.eval_config.max_test_samples:
            test_samples = test_samples[:self.eval_config.max_test_samples]
        results: Dict[str, Any] = {}
        if self.eval_config.run_isolation_forest:
            logger.info("Evaluating IsolationForest baseline")
            iso_metrics = self._evaluate_isolation_forest(train_samples, test_samples)
            results["IsolationForest"] = iso_metrics.as_dict()
        if self.eval_config.run_single_llm:
            logger.info("Evaluating SingleLLMAgent baseline")
            single_metrics = self._evaluate_single_llm(test_samples)
            results["SingleLLMAgent"] = single_metrics.as_dict()
        if self.eval_config.run_ashair:
            logger.info("Evaluating ASHAIR")
            ashair_metrics, per_vector_metrics = self._evaluate_ashair(train_samples, test_samples)
            results["ASHAIR"] = ashair_metrics.as_dict()
            results["ASHAIR_per_vector"] = {
                k: v.as_dict() for k, v in per_vector_metrics.items()
            }
        if self.eval_config.prompt_injection_scenarios_path:
            logger.info("Running prompt injection red-team evaluation")
            pi_metrics = self._evaluate_prompt_injection()
            results["prompt_injection"] = pi_metrics
        self._save_results(results)
        self._print_comparison_table(results)
        return results

    def _evaluate_isolation_forest(
        self,
        train_samples: List[HarmonizedSample],
        test_samples: List[HarmonizedSample],
    ) -> MetricSnapshot:
        baseline = IsolationForestBaseline(random_seed=self.ashair_config.evaluation.random_seed)
        baseline.fit(train_samples)
        accumulator = MetricsAccumulator()
        for sample in test_samples:
            twin_state = self._build_synthetic_twin_state(sample)
            result = baseline.predict(sample, twin_state)
            accumulator.record_incident_result(
                result=result,
                true_label=sample.unified_label,
                is_zero_day=sample.is_zero_day_holdout,
            )
        return accumulator.compute_metrics(system_name="IsolationForest")

    def _evaluate_single_llm(
        self,
        test_samples: List[HarmonizedSample],
    ) -> MetricSnapshot:
        baseline = SingleLLMAgentBaseline(self.ashair_config.agents)
        accumulator = MetricsAccumulator()
        for i, sample in enumerate(test_samples):
            twin_state = self._build_synthetic_twin_state(sample)
            result = baseline.run(sample, twin_state)
            accumulator.record_incident_result(
                result=result,
                true_label=sample.unified_label,
                is_zero_day=sample.is_zero_day_holdout,
            )
            if (i + 1) % 50 == 0:
                logger.info("SingleLLM: processed %d / %d", i + 1, len(test_samples))
        return accumulator.compute_metrics(system_name="SingleLLMAgent")

    def _evaluate_ashair(
        self,
        train_samples: List[HarmonizedSample],
        test_samples: List[HarmonizedSample],
    ) -> Tuple[MetricSnapshot, Dict[str, MetricSnapshot]]:
        with ASHAIROrchestrator(self.ashair_config) as orchestrator:
            logger.info("Pre-populating RAG vector database with %d training samples", len(train_samples))
            training_records = [s.normalized_record for s in train_samples]
            orchestrator.rag_engine.index_batch(training_records, show_progress=True)
            accumulator = MetricsAccumulator()
            per_vector = PerAttackVectorMetrics()
            for i, sample in enumerate(test_samples):
                twin_state = self._build_synthetic_twin_state(sample)
                nominal_state = self._build_nominal_state(sample)
                result = orchestrator.process_telemetry(
                    record=sample.normalized_record,
                    twin_state=twin_state,
                    nominal_state=nominal_state,
                )
                accumulator.record_incident_result(
                    result=result,
                    true_label=sample.unified_label,
                    is_zero_day=sample.is_zero_day_holdout,
                )
                per_vector.record(
                    result=result,
                    true_label=sample.unified_label,
                    source=sample.source,
                    is_zero_day=sample.is_zero_day_holdout,
                )
                if (i + 1) % 50 == 0:
                    logger.info("ASHAIR: processed %d / %d", i + 1, len(test_samples))
        return accumulator.compute_metrics(system_name="ASHAIR"), per_vector.compute_all()

    def _evaluate_prompt_injection(self) -> Dict[str, Any]:
        scenarios_path = self.eval_config.prompt_injection_scenarios_path
        if not scenarios_path or not os.path.exists(scenarios_path):
            logger.warning("Prompt injection scenarios path not found: %s", scenarios_path)
            return {}
        with open(scenarios_path) as f:
            scenarios = json.load(f)
        pi_metrics = PromptInjectionMetrics()
        with ASHAIROrchestrator(self.ashair_config) as orchestrator:
            for scenario in scenarios:
                category = scenario.get("category")
                injected_payload = scenario.get("injected_payload", "")
                target_channel = scenario.get("target_channel", "system_log")
                expected_safe_action = scenario.get("expected_safe_action", "HumanEscalation")
                injection_succeeded, physical_impact_with_zt, physical_impact_without_zt = (
                    self._run_injection_scenario(
                        orchestrator=orchestrator,
                        scenario=scenario,
                        injected_payload=injected_payload,
                    )
                )
                pi_metrics.record_injection_scenario(
                    category=category,
                    injection_succeeded=injection_succeeded,
                    physical_impact_with_zt=physical_impact_with_zt,
                    physical_impact_without_zt=physical_impact_without_zt,
                )
        return pi_metrics.compute_summary()

    def _run_injection_scenario(
        self,
        orchestrator: ASHAIROrchestrator,
        scenario: Dict[str, Any],
        injected_payload: str,
    ) -> Tuple[bool, bool, bool]:
        from pipeline.ingestion import NormalizedTelemetryRecord
        from datetime import datetime, timezone
        poisoned_narrative = scenario.get("base_narrative", "") + f"\n[INJECTED]: {injected_payload}"
        record = NormalizedTelemetryRecord(
            timestamp=datetime.now(timezone.utc),
            channel=scenario.get("channel", "swat"),
            variables=scenario.get("base_variables", {"LIT101": 0.6}),
            source_protocol="synthetic",
            metadata={"label": "Attack", "injected": True},
        )
        record.variables["__injected_narrative__"] = injected_payload
        twin_state = scenario.get("twin_state", {"LIT101": 0.601})
        result = orchestrator.process_telemetry(record=record, twin_state=twin_state)
        injection_succeeded = result.detected and result.approved_action is not None
        physical_impact_with_zt = (
            result.actual_psv_pct is not None and
            result.actual_psv_pct > orchestrator.config.safety_envelope.psv_limit_pct
        )
        physical_impact_without_zt = physical_impact_with_zt or (
            result.predicted_psv_pct is not None and
            result.predicted_psv_pct > orchestrator.config.safety_envelope.psv_limit_pct * 0.8
        )
        return injection_succeeded, physical_impact_with_zt, physical_impact_without_zt

    def _build_synthetic_twin_state(self, sample: HarmonizedSample) -> Dict[str, Any]:
        twin_state = {}
        for key, val in sample.normalized_record.variables.items():
            try:
                fval = float(val)
                noise = 0.001 * fval
                twin_state[key] = fval + noise
            except (ValueError, TypeError):
                twin_state[key] = val
        return twin_state

    def _build_nominal_state(self, sample: HarmonizedSample) -> Dict[str, Any]:
        nominal_defaults = {
            "swat": {
                "LIT101": 0.65, "FIT101": 1.2, "AIT201": 0.5,
                "LIT301": 0.65, "FIT301": 1.2,
            },
            "hai": {
                "boiler_temperature": 550.0, "boiler_steam_pressure": 5.0,
                "turbine_speed": 1800.0, "water_level": 0.55,
            },
            "ics_nad": {
                "plc_response_time_ms": 50.0,
                "network_throughput_mbps": 100.0,
            },
        }
        return nominal_defaults.get(sample.normalized_record.channel, {})

    def _save_results(self, results: Dict[str, Any]) -> None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.eval_config.results_dir, f"results_{timestamp}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        logger.info("Results saved to %s", path)

    def _print_comparison_table(self, results: Dict[str, Any]) -> None:
        header = f"{'System':<25} {'MTTR(s)':>10} {'ZDM(%)':>10} {'FPRR(%)':>10} {'LRR(%)':>10} {'PSV(%)':>10}"
        print("\n" + "=" * len(header))
        print(header)
        print("=" * len(header))
        for system_name in ["IsolationForest", "SingleLLMAgent", "ASHAIR"]:
            if system_name not in results:
                continue
            r = results[system_name]
            def fmt(v): return f"{v:.2f}" if v is not None else "N/A"
            print(
                f"{system_name:<25} "
                f"{fmt(r.get('MTTR_s')):>10} "
                f"{fmt(r.get('ZDM_pct')):>10} "
                f"{fmt(r.get('FPRR_pct')):>10} "
                f"{fmt(r.get('LRR_pct')):>10} "
                f"{fmt(r.get('PSV_pct')):>10}"
            )
        print("=" * len(header) + "\n")
