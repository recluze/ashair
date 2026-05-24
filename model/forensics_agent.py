from __future__ import annotations
import json
import logging
from typing import Any, Dict, Optional

from model.base_agent import BaseAgent
from model.prompts import FORENSICS_SYSTEM_PROMPT, FORENSICS_INVESTIGATION_TEMPLATE
from configs.config import AgentConfig

logger = logging.getLogger(__name__)


class ForensicsAgent(BaseAgent):
    def __init__(self, config: AgentConfig):
        super().__init__(config, "ForensicsAgent")
        self._inference_time_total = 0.0
        self._call_count = 0

    def run(
        self,
        incident_report: Dict[str, Any],
        physical_state: Dict[str, Any],
        twin_state: Dict[str, Any],
        divergence: float,
        rag_context: str,
    ) -> Dict[str, Any]:
        user_message = FORENSICS_INVESTIGATION_TEMPLATE.format(
            incident_report=json.dumps(incident_report, indent=2),
            twin_state=json.dumps(twin_state, indent=2),
            physical_state=json.dumps(physical_state, indent=2),
            divergence=round(divergence, 6),
            rag_context=rag_context,
        )
        result, elapsed = self._timed_call(FORENSICS_SYSTEM_PROMPT, user_message)
        self._inference_time_total += elapsed
        self._call_count += 1
        result["_inference_time_s"] = elapsed
        result["_agent"] = self.agent_name
        self._ensure_human_escalation_fallback(result)
        logger.info(
            "ForensicsAgent: origin=%s vector=%s candidates=%d elapsed=%.3fs",
            result.get("attack_origin_layer"),
            result.get("attack_vector"),
            len(result.get("candidate_actions", [])),
            elapsed,
        )
        return result

    def _ensure_human_escalation_fallback(self, result: Dict[str, Any]) -> None:
        candidates = result.get("candidate_actions", [])
        has_escalation = any(
            c.get("action_type") == "HumanEscalation" for c in candidates
        )
        if not has_escalation:
            escalation = {
                "rank": len(candidates) + 1,
                "action_type": "HumanEscalation",
                "action_id": "escalate_to_soc",
                "parameters": {"priority": "HIGH", "include_rca": True},
                "rationale": "Safety fallback: escalate if all autonomous actions fail or are unsafe.",
            }
            candidates.append(escalation)
            result["candidate_actions"] = candidates

    def build_remediation_brief(self, forensics_result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "root_cause": forensics_result.get("root_cause"),
            "attack_origin_layer": forensics_result.get("attack_origin_layer"),
            "attack_vector": forensics_result.get("attack_vector"),
            "affected_components": forensics_result.get("affected_components", []),
            "candidate_actions": forensics_result.get("candidate_actions", []),
            "rca_narrative": forensics_result.get("rca_narrative"),
            "rag_precedent_id": forensics_result.get("rag_precedent_id"),
        }

    def mean_inference_time(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._inference_time_total / self._call_count
