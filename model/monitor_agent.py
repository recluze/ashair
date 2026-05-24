from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional

from model.base_agent import BaseAgent
from model.prompts import MONITOR_SYSTEM_PROMPT, MONITOR_DETECTION_TEMPLATE
from configs.config import AgentConfig

logger = logging.getLogger(__name__)

IOC_CONFIDENCE_THRESHOLD = 0.85
DIVERGENCE_EMERGENCY_MULTIPLIER = 2.0


class MonitorAgent(BaseAgent):
    def __init__(self, config: AgentConfig, divergence_epsilon: float = 0.05):
        super().__init__(config, "MonitorAgent")
        self.epsilon = divergence_epsilon
        self._inference_time_total = 0.0
        self._call_count = 0

    def run(
        self,
        telemetry_narrative: str,
        rag_context: str,
        divergence: float,
        timestamp: str,
        physical_state: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        user_message = MONITOR_DETECTION_TEMPLATE.format(
            timestamp=timestamp,
            telemetry_narrative=telemetry_narrative,
            divergence=round(divergence, 6),
            epsilon=self.epsilon,
            k=5,
            rag_context=rag_context,
        )
        result, elapsed = self._timed_call(MONITOR_SYSTEM_PROMPT, user_message)
        self._inference_time_total += elapsed
        self._call_count += 1
        result["_inference_time_s"] = elapsed
        result["_agent"] = self.agent_name
        if physical_state is not None and "physical_state_snapshot" not in result:
            result["physical_state_snapshot"] = physical_state
        self._apply_emergency_override(result, divergence)
        logger.info(
            "MonitorAgent: is_incident=%s confidence=%.3f type=%s elapsed=%.3fs",
            result.get("is_incident"),
            result.get("confidence", 0.0),
            result.get("incident_type"),
            elapsed,
        )
        return result

    def _apply_emergency_override(self, result: Dict[str, Any], divergence: float) -> None:
        if divergence > self.epsilon * DIVERGENCE_EMERGENCY_MULTIPLIER:
            if not result.get("is_incident"):
                logger.warning(
                    "MonitorAgent: Emergency override — divergence %.4f exceeds 2x epsilon %.4f",
                    divergence,
                    self.epsilon,
                )
                result["is_incident"] = True
                result["confidence"] = max(result.get("confidence", 0.0), 0.90)
                result["incident_type"] = result.get("incident_type") or "FDI"
                result["reasoning"] = (
                    f"[Emergency override] Divergence {divergence:.4f} > 2*epsilon {self.epsilon:.4f}. "
                    + result.get("reasoning", "")
                )

    def build_incident_report(self, detection_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "affected_subsystem": detection_result.get("affected_subsystem"),
            "incident_type": detection_result.get("incident_type"),
            "anomalous_variables": detection_result.get("anomalous_variables", []),
            "divergence_signal": detection_result.get("divergence_signal"),
            "confidence": detection_result.get("confidence"),
            "physical_state_snapshot": detection_result.get("physical_state_snapshot", {}),
            "rag_precedent_similarity": detection_result.get("rag_precedent_similarity"),
            "reasoning": detection_result.get("reasoning"),
        }

    def mean_inference_time(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._inference_time_total / self._call_count
