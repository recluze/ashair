from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import httpx

from model.base_agent import BaseAgent
from model.prompts import ADMIN_SYSTEM_PROMPT, ADMIN_REMEDIATION_TEMPLATE
from configs.config import AgentConfig

logger = logging.getLogger(__name__)

MODBUS_WRITE_TOOL = "modbus_tcp_write"
FIREWALL_ISOLATE_TOOL = "firewall_isolate_segment"
CONTAINER_ROLLBACK_TOOL = "container_rollback"
SOC_ESCALATE_TOOL = "soc_escalate"
SENSOR_RECALIBRATE_TOOL = "sensor_recalibrate"


class AdminAgent(BaseAgent):
    def __init__(
        self,
        config: AgentConfig,
        firewall_api_url: str,
        container_api_url: str,
        modbus_host: str,
        modbus_port: int,
        max_reformulation_attempts: int = 3,
    ):
        super().__init__(config, "AdminAgent")
        self.firewall_api_url = firewall_api_url
        self.container_api_url = container_api_url
        self.modbus_host = modbus_host
        self.modbus_port = modbus_port
        self.max_reformulation_attempts = max_reformulation_attempts
        self._inference_time_total = 0.0
        self._call_count = 0
        self._http_client = httpx.Client(timeout=30.0)

    def run(
        self,
        remediation_brief: Dict[str, Any],
        physical_state: Dict[str, Any],
        twin_state: Dict[str, Any],
        sandbox_result: Optional[Dict[str, Any]] = None,
        attempt: int = 1,
    ) -> Dict[str, Any]:
        sandbox_result_str = json.dumps(sandbox_result, indent=2) if sandbox_result else "None (first attempt)"
        user_message = ADMIN_REMEDIATION_TEMPLATE.format(
            remediation_brief=json.dumps(remediation_brief, indent=2),
            physical_state=json.dumps(physical_state, indent=2),
            twin_state=json.dumps(twin_state, indent=2),
            sandbox_result=sandbox_result_str,
            attempt=attempt,
            max_attempts=self.max_reformulation_attempts,
        )
        system_prompt = ADMIN_SYSTEM_PROMPT.format(
            max_reformulation_attempts=self.max_reformulation_attempts
        )
        result, elapsed = self._timed_call(system_prompt, user_message)
        self._inference_time_total += elapsed
        self._call_count += 1
        result["_inference_time_s"] = elapsed
        result["_agent"] = self.agent_name
        result["reformulation_attempt"] = attempt
        logger.info(
            "AdminAgent: action_type=%s attempt=%d elapsed=%.3fs",
            result.get("selected_action", {}).get("action_type"),
            attempt,
            elapsed,
        )
        return result

    def execute_approved_action(
        self,
        admin_result: Dict[str, Any],
        zero_trust_proof: str,
    ) -> Tuple[bool, Dict[str, Any]]:
        selected = admin_result.get("selected_action", {})
        action_type = selected.get("action_type")
        parameters = selected.get("parameters", {})
        tool_call = selected.get("tool_call", {})
        start = time.perf_counter()
        try:
            if action_type == "NetworkIsolation":
                success, detail = self._execute_network_isolation(parameters)
            elif action_type == "ProcessManagement":
                success, detail = self._execute_process_management(parameters)
            elif action_type == "ControlLayerAdjustment":
                success, detail = self._execute_control_layer_adjustment(parameters)
            elif action_type == "HumanEscalation":
                success, detail = self._execute_human_escalation(parameters, admin_result)
            else:
                logger.error("AdminAgent: Unknown action_type=%s", action_type)
                return False, {"error": f"Unknown action_type: {action_type}"}
            elapsed = time.perf_counter() - start
            detail["execution_time_s"] = elapsed
            detail["zero_trust_proof"] = zero_trust_proof
            logger.info(
                "AdminAgent: executed %s success=%s in %.3fs",
                action_type, success, elapsed,
            )
            return success, detail
        except Exception as exc:
            logger.error("AdminAgent: execution error for %s: %s", action_type, exc)
            return False, {"error": str(exc)}

    def _execute_network_isolation(self, parameters: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        segment = parameters.get("segment")
        rule_type = parameters.get("rule_type", "block")
        payload = {"segment": segment, "rule": rule_type, "direction": parameters.get("direction", "both")}
        try:
            resp = self._http_client.post(
                f"{self.firewall_api_url}/api/rules",
                json=payload,
            )
            resp.raise_for_status()
            return True, {"firewall_rule_id": resp.json().get("rule_id"), "segment": segment}
        except httpx.HTTPError as exc:
            return False, {"error": str(exc)}

    def _execute_process_management(self, parameters: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        container_id = parameters.get("container_id")
        operation = parameters.get("operation", "rollback")
        snapshot_id = parameters.get("snapshot_id")
        payload = {"container_id": container_id, "operation": operation, "snapshot_id": snapshot_id}
        try:
            resp = self._http_client.post(
                f"{self.container_api_url}/containers/{container_id}/{operation}",
                json=payload,
            )
            resp.raise_for_status()
            return True, {"container_id": container_id, "operation": operation}
        except httpx.HTTPError as exc:
            return False, {"error": str(exc)}

    def _execute_control_layer_adjustment(self, parameters: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        register = parameters.get("register")
        value = parameters.get("value")
        unit_id = parameters.get("unit_id", 1)
        detail = {"register": register, "value": value, "unit_id": unit_id}
        try:
            from pymodbus.client import ModbusTcpClient
            client = ModbusTcpClient(self.modbus_host, port=self.modbus_port)
            client.connect()
            result = client.write_register(register, int(value * 100), slave=unit_id)
            client.close()
            if result.isError():
                return False, {"error": "Modbus write error", **detail}
            return True, detail
        except ImportError:
            logger.warning("pymodbus not installed; simulating Modbus write")
            return True, {"simulated": True, **detail}
        except Exception as exc:
            return False, {"error": str(exc), **detail}

    def _execute_human_escalation(
        self, parameters: Dict[str, Any], admin_result: Dict[str, Any]
    ) -> Tuple[bool, Dict[str, Any]]:
        alert_payload = {
            "priority": parameters.get("priority", "HIGH"),
            "cot_trace": admin_result.get("cot_trace"),
            "reasoning": admin_result.get("reasoning"),
            "include_rca": parameters.get("include_rca", True),
        }
        logger.critical("HUMAN ESCALATION TRIGGERED: %s", json.dumps(alert_payload, indent=2))
        return True, {"escalated": True, "priority": parameters.get("priority")}

    def mean_inference_time(self) -> float:
        if self._call_count == 0:
            return 0.0
        return self._inference_time_total / self._call_count

    def close(self):
        self._http_client.close()
        super().close()
