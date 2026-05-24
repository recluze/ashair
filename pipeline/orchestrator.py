from __future__ import annotations
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from configs.config import ASHAIRConfig
from model.monitor_agent import MonitorAgent
from model.forensics_agent import ForensicsAgent
from model.admin_agent import AdminAgent
from pipeline.ingestion import NormalizedTelemetryRecord
from pipeline.normalizer import TelemetryNarrativeGenerator
from pipeline.rag_engine import RAGEngine, FeedbackRecord
from pipeline.digital_twin import DigitalTwinSandbox, SandboxSubmission, SandboxResult
from pipeline.safety_envelope import SafetyEnvelope
from pipeline.zero_trust import ZeroTrustOrchestrator, ProofOfIntent, AGENT_ADMIN

logger = logging.getLogger(__name__)

OUTCOME_SUCCESS = "success"
OUTCOME_PARTIAL = "partial"
OUTCOME_FAILURE = "failure"


@dataclass
class IncidentResponseResult:
    incident_id: str
    detected: bool
    mttr_s: Optional[float]
    detection_time_s: float
    forensics_time_s: float
    admin_time_s: float
    sandbox_time_s: float
    total_sandbox_submissions: int
    total_rejected: int
    approved_action: Optional[Dict[str, Any]]
    final_physical_state: Optional[Dict[str, Any]]
    predicted_psv_pct: Optional[float]
    actual_psv_pct: Optional[float]
    lrr: Optional[float]
    outcome: str
    escalated_to_human: bool
    cot_trace: Optional[str]
    audit_entry_id: Optional[str]


class ASHAIROrchestrator:
    def __init__(self, config: ASHAIRConfig):
        self.config = config
        self.narrative_generator = TelemetryNarrativeGenerator()
        self.rag_engine = RAGEngine(config.rag)
        self.safety_envelope = SafetyEnvelope(config.safety_envelope)
        self.sandbox = DigitalTwinSandbox(config.sandbox, self.safety_envelope)
        self.zero_trust = ZeroTrustOrchestrator(config.zero_trust)
        self.monitor_agent = MonitorAgent(config.agents)
        self.forensics_agent = ForensicsAgent(config.agents)
        self.admin_agent = AdminAgent(
            config.agents,
            firewall_api_url=config.firewall_api_url,
            container_api_url=config.container_api_url,
            modbus_host=config.modbus_host,
            modbus_port=config.modbus_port,
            max_reformulation_attempts=config.sandbox.max_reformulation_attempts,
        )
        self._incident_counter = 0

    def process_telemetry(
        self,
        record: NormalizedTelemetryRecord,
        twin_state: Dict[str, Any],
        nominal_state: Optional[Dict[str, Any]] = None,
    ) -> IncidentResponseResult:
        self._incident_counter += 1
        incident_id = f"incident_{self._incident_counter}_{record.timestamp.isoformat()}"
        domain = record.channel
        physical_state = record.variables
        divergence = self._compute_divergence(physical_state, twin_state)
        detect_start = time.perf_counter()
        narrative = self.narrative_generator.generate(record)
        rag_context, rag_similarity = self.rag_engine.retrieve_context(narrative, channel_filter=None)
        detection_result = self.monitor_agent.run(
            telemetry_narrative=narrative,
            rag_context=rag_context,
            divergence=divergence,
            timestamp=record.timestamp.isoformat(),
            physical_state=physical_state,
        )
        detection_time = time.perf_counter() - detect_start
        if not detection_result.get("is_incident"):
            self.rag_engine.index_telemetry_record(record)
            return IncidentResponseResult(
                incident_id=incident_id,
                detected=False,
                mttr_s=None,
                detection_time_s=detection_time,
                forensics_time_s=0.0,
                admin_time_s=0.0,
                sandbox_time_s=0.0,
                total_sandbox_submissions=0,
                total_rejected=0,
                approved_action=None,
                final_physical_state=physical_state,
                predicted_psv_pct=None,
                actual_psv_pct=None,
                lrr=None,
                outcome=OUTCOME_SUCCESS,
                escalated_to_human=False,
                cot_trace=None,
                audit_entry_id=None,
            )
        incident_report = self.monitor_agent.build_incident_report(detection_result)
        logger.info("Incident detected: %s confidence=%.3f", incident_id, detection_result.get("confidence", 0))
        forensics_start = time.perf_counter()
        forensics_rag_context, _ = self.rag_engine.retrieve_context(
            f"RCA query: {incident_report.get('incident_type')} on {incident_report.get('affected_subsystem')}"
        )
        forensics_result = self.forensics_agent.run(
            incident_report=incident_report,
            physical_state=physical_state,
            twin_state=twin_state,
            divergence=divergence,
            rag_context=forensics_rag_context,
        )
        forensics_time = time.perf_counter() - forensics_start
        remediation_brief = self.forensics_agent.build_remediation_brief(forensics_result)
        admin_start = time.perf_counter()
        sandbox_result, approved_admin_result, total_submissions, total_rejected = self._sandbox_gated_execution_loop(
            remediation_brief=remediation_brief,
            physical_state=physical_state,
            twin_state=twin_state,
            domain=domain,
            divergence=divergence,
            incident_context=incident_report,
        )
        admin_time = time.perf_counter() - admin_start
        sandbox_time = self.sandbox.mean_simulation_time() * total_submissions
        mttr = detection_time + forensics_time + admin_time
        is_escalation = (
            approved_admin_result is not None and
            approved_admin_result.get("selected_action", {}).get("action_type") == "HumanEscalation"
        )
        predicted_psv = sandbox_result.envelope_check.max_psv_pct if sandbox_result else None
        actual_psv = None
        lrr = None
        if sandbox_result and sandbox_result.approved and not is_escalation:
            proof = self.zero_trust.generate_proof_of_intent(
                action_type=approved_admin_result["selected_action"]["action_type"],
                agent_name=AGENT_ADMIN,
                incident_context=incident_report,
            )
            execution_success, exec_detail = self.admin_agent.execute_approved_action(
                approved_admin_result, zero_trust_proof=proof.signature
            )
            audit_id = self.zero_trust.write_audit_entry(
                agent_name=AGENT_ADMIN,
                action_type=approved_admin_result["selected_action"]["action_type"],
                approved=True,
                proof=proof,
                cot_trace=approved_admin_result.get("cot_trace"),
                incident_context=incident_report,
                outcome=OUTCOME_SUCCESS if execution_success else OUTCOME_FAILURE,
            )
            actual_psv = self._observe_post_action_psv(physical_state, twin_state, domain, nominal_state)
            lrr = self._observe_load_restoration(physical_state, twin_state, domain)
            outcome = OUTCOME_SUCCESS if execution_success else OUTCOME_FAILURE
        else:
            audit_id = None
            outcome = OUTCOME_FAILURE if sandbox_result is None else OUTCOME_PARTIAL
        self._store_feedback(
            action=approved_admin_result.get("selected_action") if approved_admin_result else None,
            physical_state=physical_state,
            predicted_state=sandbox_result.predicted_physical_state if sandbox_result else {},
            actual_state=twin_state,
            outcome=outcome,
            timestamp=record.timestamp,
        )
        self.rag_engine.index_telemetry_record(record)
        return IncidentResponseResult(
            incident_id=incident_id,
            detected=True,
            mttr_s=mttr,
            detection_time_s=detection_time,
            forensics_time_s=forensics_time,
            admin_time_s=admin_time,
            sandbox_time_s=sandbox_time,
            total_sandbox_submissions=total_submissions,
            total_rejected=total_rejected,
            approved_action=approved_admin_result.get("selected_action") if approved_admin_result else None,
            final_physical_state=physical_state,
            predicted_psv_pct=predicted_psv,
            actual_psv_pct=actual_psv,
            lrr=lrr,
            outcome=outcome,
            escalated_to_human=is_escalation,
            cot_trace=approved_admin_result.get("cot_trace") if approved_admin_result else None,
            audit_entry_id=audit_id,
        )

    def _sandbox_gated_execution_loop(
        self,
        remediation_brief: Dict[str, Any],
        physical_state: Dict[str, Any],
        twin_state: Dict[str, Any],
        domain: str,
        divergence: float,
        incident_context: Dict[str, Any],
    ) -> Tuple[Optional[SandboxResult], Optional[Dict[str, Any]], int, int]:
        max_attempts = self.config.sandbox.max_reformulation_attempts
        sandbox_result: Optional[SandboxResult] = None
        prior_sandbox_result_dict: Optional[Dict[str, Any]] = None
        total_submissions = 0
        total_rejected = 0
        fdi_corrupted = divergence > self.config.agents.max_tokens
        if fdi_corrupted:
            logger.warning("Orchestrator: FDI corruption suspected; forcing HumanEscalation")
            remediation_brief["candidate_actions"] = [
                {
                    "rank": 1,
                    "action_type": "HumanEscalation",
                    "action_id": "emergency_escalate",
                    "parameters": {"priority": "CRITICAL"},
                    "rationale": "FDI corruption detected; autonomous remediation unsafe.",
                }
            ]
        for attempt in range(1, max_attempts + 1):
            admin_result = self.admin_agent.run(
                remediation_brief=remediation_brief,
                physical_state=physical_state,
                twin_state=twin_state,
                sandbox_result=prior_sandbox_result_dict,
                attempt=attempt,
            )
            selected_action = admin_result.get("selected_action", {})
            if selected_action.get("action_type") == "HumanEscalation":
                return None, admin_result, total_submissions, total_rejected
            submission = SandboxSubmission(
                action=selected_action,
                physical_state=physical_state,
                twin_state=twin_state,
                domain=domain,
                divergence_threshold=0.05,
            )
            sandbox_result = self.sandbox.validate(submission)
            total_submissions += 1
            if sandbox_result.approved:
                return sandbox_result, admin_result, total_submissions, total_rejected
            else:
                total_rejected += 1
                prior_sandbox_result_dict = {
                    "approved": False,
                    "failure_report": sandbox_result.failure_report,
                    "predicted_psv_pct": sandbox_result.envelope_check.max_psv_pct,
                }
                logger.warning(
                    "Orchestrator: sandbox rejection %d/%d for incident",
                    attempt, max_attempts,
                )
        logger.error("Orchestrator: max reformulation attempts exceeded; escalating to human")
        admin_escalation = self.admin_agent.run(
            remediation_brief={
                **remediation_brief,
                "candidate_actions": [
                    {"rank": 1, "action_type": "HumanEscalation",
                     "action_id": "max_retry_escalate",
                     "parameters": {"priority": "HIGH"},
                     "rationale": "All autonomous actions failed sandbox validation."}
                ],
            },
            physical_state=physical_state,
            twin_state=twin_state,
            sandbox_result=prior_sandbox_result_dict,
            attempt=max_attempts + 1,
        )
        return None, admin_escalation, total_submissions, total_rejected

    def _compute_divergence(
        self,
        physical_state: Dict[str, Any],
        twin_state: Dict[str, Any],
    ) -> float:
        import numpy as np
        shared = set(physical_state.keys()) & set(twin_state.keys())
        if not shared:
            return 0.0
        diffs = []
        for key in shared:
            try:
                pval = float(physical_state[key])
                tval = float(twin_state[key])
                diffs.append((pval - tval) ** 2)
            except (ValueError, TypeError):
                pass
        return float(np.sqrt(np.mean(diffs))) if diffs else 0.0

    def _observe_post_action_psv(
        self,
        physical_state: Dict[str, Any],
        twin_state: Dict[str, Any],
        domain: str,
        nominal_state: Optional[Dict[str, Any]],
    ) -> float:
        reference = nominal_state if nominal_state else twin_state
        return self.safety_envelope.compute_psv(physical_state, reference, domain)

    def _observe_load_restoration(
        self,
        physical_state: Dict[str, Any],
        twin_state: Dict[str, Any],
        domain: str,
    ) -> float:
        shared = set(physical_state.keys()) & set(twin_state.keys())
        if not shared:
            return 1.0
        ratios = []
        for key in shared:
            try:
                pval = float(physical_state[key])
                tval = float(twin_state[key])
                if abs(tval) < 1e-9:
                    continue
                ratios.append(min(1.0, pval / tval) if tval > 0 else 1.0)
            except (ValueError, TypeError):
                pass
        return float(sum(ratios) / len(ratios)) if ratios else 1.0

    def _store_feedback(
        self,
        action: Optional[Dict[str, Any]],
        physical_state: Dict[str, Any],
        predicted_state: Dict[str, Any],
        actual_state: Dict[str, Any],
        outcome: str,
        timestamp: datetime,
    ) -> None:
        if action is None:
            return
        feedback = FeedbackRecord(
            action=action,
            state_at_execution=physical_state,
            predicted_state=predicted_state,
            actual_state=actual_state,
            outcome=outcome,
            timestamp=timestamp,
        )
        self.rag_engine.index_feedback_record(feedback)

    def close(self):
        self.monitor_agent.close()
        self.forensics_agent.close()
        self.admin_agent.close()
        self.rag_engine.save()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
