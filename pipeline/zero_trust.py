from __future__ import annotations
import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Set

from configs.config import ZeroTrustConfig

logger = logging.getLogger(__name__)

AGENT_MONITOR = "MonitorAgent"
AGENT_FORENSICS = "ForensicsAgent"
AGENT_ADMIN = "AdminAgent"

CAPABILITY_PROFILES: Dict[str, Set[str]] = {
    AGENT_MONITOR: {
        "rag_query",
        "divergence_compute",
        "ioc_escalate",
    },
    AGENT_FORENSICS: {
        "rag_query",
        "divergence_compute",
        "twin_state_read",
        "physical_state_read",
        "rca_produce",
        "brief_send",
    },
    AGENT_ADMIN: {
        "sandbox_submit",
        "firewall_isolate",
        "container_rollback",
        "modbus_tcp_write",
        "sensor_recalibrate",
        "soc_escalate",
        "audit_write",
    },
}


@dataclass
class ProofOfIntent:
    action_type: str
    agent_name: str
    incident_context_hash: str
    timestamp_utc: float
    signature: str


@dataclass
class AuditLogEntry:
    entry_id: str
    timestamp_utc: float
    agent_name: str
    action_type: str
    approved: bool
    proof_signature: str
    cot_trace: Optional[str]
    incident_context_hash: str
    outcome: Optional[str] = None


class ZeroTrustOrchestrator:
    def __init__(self, config: ZeroTrustConfig):
        self.config = config
        self._audit_log_path = config.audit_log_path
        self._used_signatures: Set[str] = set()
        self._ensure_audit_log()

    def _ensure_audit_log(self) -> None:
        import os
        os.makedirs(os.path.dirname(self._audit_log_path) or ".", exist_ok=True)
        if not os.path.exists(self._audit_log_path):
            open(self._audit_log_path, "w").close()

    def generate_proof_of_intent(
        self,
        action_type: str,
        agent_name: str,
        incident_context: Dict[str, Any],
    ) -> ProofOfIntent:
        context_hash = self._hash_context(incident_context)
        timestamp = time.time()
        message = f"{action_type}:{agent_name}:{context_hash}:{timestamp:.6f}"
        signature = hmac.new(
            self.config.proof_of_intent_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        return ProofOfIntent(
            action_type=action_type,
            agent_name=agent_name,
            incident_context_hash=context_hash,
            timestamp_utc=timestamp,
            signature=signature,
        )

    def verify_proof_of_intent(
        self,
        proof: ProofOfIntent,
        incident_context: Dict[str, Any],
        agent_name: str,
    ) -> bool:
        if proof.signature in self._used_signatures:
            logger.warning("ZeroTrust: Replayed proof signature rejected for agent=%s", agent_name)
            return False
        age_s = time.time() - proof.timestamp_utc
        if age_s > self.config.token_expiry_seconds:
            logger.warning("ZeroTrust: Expired proof rejected (age=%.1fs) for agent=%s", age_s, agent_name)
            return False
        if proof.agent_name != agent_name:
            logger.warning("ZeroTrust: Agent identity mismatch: proof=%s caller=%s", proof.agent_name, agent_name)
            return False
        context_hash = self._hash_context(incident_context)
        if proof.incident_context_hash != context_hash:
            logger.warning("ZeroTrust: Context hash mismatch for agent=%s", agent_name)
            return False
        message = f"{proof.action_type}:{proof.agent_name}:{proof.incident_context_hash}:{proof.timestamp_utc:.6f}"
        expected_signature = hmac.new(
            self.config.proof_of_intent_secret.encode(),
            message.encode(),
            hashlib.sha256,
        ).hexdigest()
        valid = hmac.compare_digest(proof.signature, expected_signature)
        if valid:
            self._used_signatures.add(proof.signature)
            if len(self._used_signatures) > 10000:
                self._used_signatures = set(list(self._used_signatures)[-5000:])
        else:
            logger.warning("ZeroTrust: Signature verification failed for agent=%s", agent_name)
        return valid

    def check_capability(self, agent_name: str, tool_name: str) -> bool:
        profile = CAPABILITY_PROFILES.get(agent_name, set())
        allowed = tool_name in profile
        if not allowed:
            logger.warning(
                "ZeroTrust: Capability denied agent=%s tool=%s", agent_name, tool_name
            )
        return allowed

    def admit_action(
        self,
        action_type: str,
        tool_name: str,
        agent_name: str,
        proof: ProofOfIntent,
        incident_context: Dict[str, Any],
    ) -> bool:
        if not self.check_capability(agent_name, tool_name):
            return False
        if not self.verify_proof_of_intent(proof, incident_context, agent_name):
            return False
        return True

    def write_audit_entry(
        self,
        agent_name: str,
        action_type: str,
        approved: bool,
        proof: ProofOfIntent,
        cot_trace: Optional[str],
        incident_context: Dict[str, Any],
        outcome: Optional[str] = None,
    ) -> str:
        import uuid
        entry_id = str(uuid.uuid4())
        entry = AuditLogEntry(
            entry_id=entry_id,
            timestamp_utc=time.time(),
            agent_name=agent_name,
            action_type=action_type,
            approved=approved,
            proof_signature=proof.signature,
            cot_trace=cot_trace,
            incident_context_hash=proof.incident_context_hash,
            outcome=outcome,
        )
        with open(self._audit_log_path, "a") as f:
            f.write(json.dumps(entry.__dict__) + "\n")
        return entry_id

    def _hash_context(self, context: Dict[str, Any]) -> str:
        canonical = json.dumps(context, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
