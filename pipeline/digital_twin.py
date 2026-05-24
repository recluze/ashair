from __future__ import annotations
import copy
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.config import SandboxConfig
from pipeline.safety_envelope import SafetyEnvelope, EnvelopeCheckResult

logger = logging.getLogger(__name__)


@dataclass
class SandboxSubmission:
    action: Dict[str, Any]
    physical_state: Dict[str, Any]
    twin_state: Dict[str, Any]
    domain: str
    divergence_threshold: float = 0.05


@dataclass
class SandboxResult:
    approved: bool
    predicted_physical_state: Dict[str, Any]
    predicted_divergence: float
    envelope_check: EnvelopeCheckResult
    simulation_time_s: float
    failure_report: Optional[Dict[str, Any]] = None
    action: Optional[Dict[str, Any]] = None


class PhysicsSimulator:
    NETWORK_ISOLATION_LATENCY_MS = 50.0
    PROCESS_RESTART_LATENCY_S = 5.0
    SENSOR_RECALIBRATION_LATENCY_S = 2.0

    def simulate(
        self,
        physical_state: Dict[str, Any],
        twin_state: Dict[str, Any],
        action: Dict[str, Any],
        domain: str,
        timestep_s: float = 1.0,
    ) -> Tuple[Dict[str, Any], float]:
        predicted = copy.deepcopy(physical_state)
        action_type = action.get("action_type")
        parameters = action.get("parameters", {})
        if action_type == "NetworkIsolation":
            predicted = self._simulate_network_isolation(predicted, twin_state, parameters, domain)
        elif action_type == "ProcessManagement":
            predicted = self._simulate_process_management(predicted, twin_state, parameters, domain)
        elif action_type == "ControlLayerAdjustment":
            predicted = self._simulate_control_adjustment(predicted, twin_state, parameters, domain)
        elif action_type == "HumanEscalation":
            pass
        predicted_divergence = self._compute_predicted_divergence(predicted, twin_state)
        return predicted, predicted_divergence

    def _simulate_network_isolation(
        self,
        state: Dict[str, Any],
        twin_state: Dict[str, Any],
        parameters: Dict[str, Any],
        domain: str,
    ) -> Dict[str, Any]:
        segment = parameters.get("segment", "")
        if domain == "ics_nad":
            state["plc_response_time_ms"] = state.get("plc_response_time_ms", 0.0) * 0.3
            state["network_throughput_mbps"] = state.get("network_throughput_mbps", 0.0) * 0.1
        return state

    def _simulate_process_management(
        self,
        state: Dict[str, Any],
        twin_state: Dict[str, Any],
        parameters: Dict[str, Any],
        domain: str,
    ) -> Dict[str, Any]:
        operation = parameters.get("operation", "rollback")
        if operation in ("rollback", "restart"):
            for key in twin_state:
                if key in state:
                    current = state[key]
                    target = twin_state[key]
                    try:
                        state[key] = float(current) * 0.5 + float(target) * 0.5
                    except (ValueError, TypeError):
                        state[key] = target
        return state

    def _simulate_control_adjustment(
        self,
        state: Dict[str, Any],
        twin_state: Dict[str, Any],
        parameters: Dict[str, Any],
        domain: str,
    ) -> Dict[str, Any]:
        register = parameters.get("register")
        target_value = parameters.get("value")
        adjustment_mode = parameters.get("mode", "hard_reset")
        sensor_map = parameters.get("sensor_map", {})
        if register and target_value is not None:
            affected_sensor = sensor_map.get(str(register), register)
            if affected_sensor in state:
                current = state.get(affected_sensor, target_value)
                try:
                    if adjustment_mode == "hard_reset":
                        state[affected_sensor] = float(target_value)
                    elif adjustment_mode == "gradual_ramp":
                        ramp_steps = parameters.get("ramp_steps", 5)
                        state[affected_sensor] = float(current) + (float(target_value) - float(current)) / ramp_steps
                    elif adjustment_mode == "recalibrate":
                        twin_val = twin_state.get(affected_sensor, target_value)
                        state[affected_sensor] = float(twin_val)
                except (ValueError, TypeError):
                    state[affected_sensor] = target_value
        return state

    def _compute_predicted_divergence(
        self,
        predicted_physical: Dict[str, Any],
        twin_state: Dict[str, Any],
    ) -> float:
        shared_keys = set(predicted_physical.keys()) & set(twin_state.keys())
        if not shared_keys:
            return 0.0
        diffs = []
        for key in shared_keys:
            try:
                pval = float(predicted_physical[key])
                tval = float(twin_state[key])
                diffs.append((pval - tval) ** 2)
            except (ValueError, TypeError):
                pass
        if not diffs:
            return 0.0
        return float(np.sqrt(np.mean(diffs)))


class DigitalTwinSandbox:
    def __init__(
        self,
        config: SandboxConfig,
        safety_envelope: SafetyEnvelope,
        simulator: Optional[PhysicsSimulator] = None,
    ):
        self.config = config
        self.safety_envelope = safety_envelope
        self.simulator = simulator or PhysicsSimulator()
        self._simulation_times: List[float] = []

    def validate(self, submission: SandboxSubmission) -> SandboxResult:
        start = time.perf_counter()
        predicted_state, predicted_divergence = self.simulator.simulate(
            physical_state=submission.physical_state,
            twin_state=submission.twin_state,
            action=submission.action,
            domain=submission.domain,
            timestep_s=self.config.simulation_timestep,
        )
        envelope_check = self.safety_envelope.check(
            predicted_state=predicted_state,
            domain=submission.domain,
            divergence=predicted_divergence,
            divergence_threshold=submission.divergence_threshold,
        )
        simulation_time = time.perf_counter() - start
        self._simulation_times.append(simulation_time)
        approved = envelope_check.within_envelope
        failure_report = None
        if not approved:
            failure_report = self._build_failure_report(
                submission.action, predicted_state, envelope_check, predicted_divergence
            )
            logger.warning(
                "Sandbox REJECTED action=%s psv=%.2f%% violations=%d",
                submission.action.get("action_type"),
                envelope_check.max_psv_pct,
                len(envelope_check.violations),
            )
        else:
            logger.info(
                "Sandbox APPROVED action=%s psv=%.2f%% divergence=%.4f",
                submission.action.get("action_type"),
                envelope_check.max_psv_pct,
                predicted_divergence,
            )
        return SandboxResult(
            approved=approved,
            predicted_physical_state=predicted_state,
            predicted_divergence=predicted_divergence,
            envelope_check=envelope_check,
            simulation_time_s=simulation_time,
            failure_report=failure_report,
            action=submission.action,
        )

    def _build_failure_report(
        self,
        action: Dict[str, Any],
        predicted_state: Dict[str, Any],
        envelope_check: EnvelopeCheckResult,
        predicted_divergence: float,
    ) -> Dict[str, Any]:
        violation_details = [
            {
                "variable": v.variable,
                "observed": v.observed_value,
                "bounds": [v.lower_bound, v.upper_bound],
                "deviation_pct": round(v.deviation_pct, 3),
            }
            for v in envelope_check.violations
        ]
        return {
            "rejected_action": action,
            "predicted_psv_pct": round(envelope_check.max_psv_pct, 3),
            "predicted_divergence": round(predicted_divergence, 6),
            "violated_constraints": violation_details,
            "divergence_within_threshold": envelope_check.divergence_within_threshold,
            "reformulation_hint": self._generate_reformulation_hint(action, envelope_check),
        }

    def _generate_reformulation_hint(
        self, action: Dict[str, Any], envelope_check: EnvelopeCheckResult
    ) -> str:
        action_type = action.get("action_type", "")
        violations = envelope_check.violations
        if action_type == "ControlLayerAdjustment" and violations:
            most_violated = max(violations, key=lambda v: v.deviation_pct)
            return (
                f"Variable {most_violated.variable} deviated {most_violated.deviation_pct:.1f}%. "
                f"Consider gradual_ramp mode instead of hard_reset, or target a value closer "
                f"to the safe range [{most_violated.lower_bound}, {most_violated.upper_bound}]."
            )
        return "Consider a less aggressive action or switch to HumanEscalation."

    def mean_simulation_time(self) -> float:
        if not self._simulation_times:
            return 0.0
        return float(np.mean(self._simulation_times))
