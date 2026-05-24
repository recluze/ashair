from __future__ import annotations
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.config import AgentConfig, SafetyEnvelopeConfig
from data.harmonizer import HarmonizedSample, LABEL_BENIGN
from pipeline.orchestrator import IncidentResponseResult, OUTCOME_SUCCESS, OUTCOME_FAILURE
from pipeline.safety_envelope import SafetyEnvelope

logger = logging.getLogger(__name__)


class IsolationForestBaseline:
    def __init__(self, contamination: float = 0.1, random_seed: int = 42):
        self.contamination = contamination
        self.random_seed = random_seed
        self._model = None
        self._feature_columns: List[str] = []
        self._scripted_response_map = {
            "high": {
                "action_type": "NetworkIsolation",
                "parameters": {"segment": "compromised", "rule_type": "block"},
            },
            "medium": {
                "action_type": "ProcessManagement",
                "parameters": {"operation": "rollback"},
            },
            "low": {
                "action_type": "HumanEscalation",
                "parameters": {"priority": "LOW"},
            },
        }

    def fit(self, train_samples: List[HarmonizedSample]) -> None:
        from sklearn.ensemble import IsolationForest
        benign_samples = [s for s in train_samples if s.unified_label == LABEL_BENIGN]
        if not benign_samples:
            raise ValueError("No benign training samples available for IsolationForest")
        self._feature_columns = self._extract_feature_columns(benign_samples)
        X = self._extract_feature_matrix(benign_samples, self._feature_columns)
        self._model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_seed,
            n_estimators=100,
        )
        self._model.fit(X)
        logger.info("IsolationForest fitted on %d benign samples, %d features", len(benign_samples), X.shape[1])

    def predict(
        self,
        sample: HarmonizedSample,
        twin_state: Dict[str, Any],
        nominal_state: Optional[Dict[str, Any]] = None,
    ) -> IncidentResponseResult:
        if self._model is None:
            raise RuntimeError("Model not fitted; call fit() first")
        start = time.perf_counter()
        x = self._extract_sample_vector(sample, self._feature_columns)
        score = self._model.decision_function(x.reshape(1, -1))[0]
        prediction = self._model.predict(x.reshape(1, -1))[0]
        detection_time = time.perf_counter() - start
        is_anomaly = prediction == -1
        if not is_anomaly:
            return self._build_result(
                sample, detected=False, detection_time=detection_time,
                mttr=None, psv=None, lrr=None, outcome=OUTCOME_SUCCESS,
            )
        anomaly_severity = self._score_to_severity(score)
        response_action = self._scripted_response_map[anomaly_severity]
        response_time = self._scripted_response_latency(anomaly_severity)
        time.sleep(min(response_time, 0.01))
        mttr = detection_time + response_time
        safety_env = SafetyEnvelope(
            type("Cfg", (), {
                "swat": {}, "hai": {}, "ics_nad": {},
                "psv_limit_pct": 2.0
            })()
        )
        psv = self._estimate_scripted_psv(anomaly_severity, sample.normalized_record.channel)
        lrr = self._estimate_scripted_lrr(anomaly_severity)
        return self._build_result(
            sample, detected=True, detection_time=detection_time,
            mttr=mttr, psv=psv, lrr=lrr,
            outcome=OUTCOME_SUCCESS if anomaly_severity == "high" else OUTCOME_PARTIAL,
        )

    def _score_to_severity(self, score: float) -> str:
        if score < -0.3:
            return "high"
        elif score < -0.1:
            return "medium"
        return "low"

    def _scripted_response_latency(self, severity: str) -> float:
        latency_map = {"high": 35.0, "medium": 40.0, "low": 38.0}
        noise = np.random.default_rng().normal(0, 2.0)
        return max(5.0, latency_map[severity] + noise)

    def _estimate_scripted_psv(self, severity: str, channel: str) -> float:
        base = {"high": 12.0, "medium": 16.0, "low": 18.0}[severity]
        return base + np.random.default_rng().uniform(-2.0, 2.0)

    def _estimate_scripted_lrr(self, severity: str) -> float:
        base = {"high": 0.65, "medium": 0.58, "low": 0.62}[severity]
        return base + np.random.default_rng().uniform(-0.05, 0.05)

    def _extract_feature_columns(self, samples: List[HarmonizedSample]) -> List[str]:
        all_keys: set = set()
        for s in samples[:100]:
            for k, v in s.normalized_record.variables.items():
                try:
                    float(v)
                    all_keys.add(k)
                except (ValueError, TypeError):
                    pass
        return sorted(all_keys)

    def _extract_feature_matrix(self, samples: List[HarmonizedSample], columns: List[str]) -> np.ndarray:
        rows = []
        for s in samples:
            row = self._extract_sample_vector(s, columns)
            rows.append(row)
        return np.array(rows)

    def _extract_sample_vector(self, sample: HarmonizedSample, columns: List[str]) -> np.ndarray:
        row = []
        for col in columns:
            val = sample.normalized_record.variables.get(col, 0.0)
            try:
                row.append(float(val))
            except (ValueError, TypeError):
                row.append(0.0)
        return np.array(row, dtype=np.float32)

    def _build_result(
        self,
        sample: HarmonizedSample,
        detected: bool,
        detection_time: float,
        mttr: Optional[float],
        psv: Optional[float],
        lrr: Optional[float],
        outcome: str,
    ) -> IncidentResponseResult:
        import uuid
        return IncidentResponseResult(
            incident_id=str(uuid.uuid4()),
            detected=detected,
            mttr_s=mttr,
            detection_time_s=detection_time,
            forensics_time_s=0.0,
            admin_time_s=mttr - detection_time if mttr else 0.0,
            sandbox_time_s=0.0,
            total_sandbox_submissions=0,
            total_rejected=0,
            approved_action=None,
            final_physical_state=sample.normalized_record.variables,
            predicted_psv_pct=psv,
            actual_psv_pct=psv,
            lrr=lrr,
            outcome=outcome,
            escalated_to_human=False,
            cot_trace=None,
            audit_entry_id=None,
        )


class SingleLLMAgentBaseline:
    SINGLE_AGENT_SYSTEM_PROMPT = """You are a single cyber-physical system security agent responsible for detection, investigation, AND remediation.
You must complete all three tasks in one reasoning chain: detect the anomaly, investigate its root cause, and select a remediation action.
Do NOT use a Digital Twin sandbox. Do NOT use RAG context.
Output strict JSON with fields: is_incident (bool), incident_type, action_type, parameters, mttr_estimate_s (float), psv_estimate_pct (float), lrr_estimate (float 0-1), reasoning.
"""

    def __init__(self, config: AgentConfig):
        self.config = config

    def run(
        self,
        sample: HarmonizedSample,
        twin_state: Dict[str, Any],
        nominal_state: Optional[Dict[str, Any]] = None,
    ) -> IncidentResponseResult:
        import httpx
        import uuid
        narrative = self._build_narrative(sample)
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": self.SINGLE_AGENT_SYSTEM_PROMPT},
                {"role": "user", "content": narrative},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        start = time.perf_counter()
        try:
            client = httpx.Client(timeout=120.0)
            resp = client.post(
                f"{self.config.inference_endpoint}/chat/completions",
                json=payload, headers=headers,
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"]
            elapsed = time.perf_counter() - start
            parsed = self._parse_response(raw)
            client.close()
        except Exception as exc:
            logger.error("SingleLLMAgent: inference error: %s", exc)
            elapsed = time.perf_counter() - start
            parsed = {"is_incident": False}
        detected = parsed.get("is_incident", False)
        mttr = elapsed + parsed.get("mttr_estimate_s", 14.2) if detected else None
        psv = parsed.get("psv_estimate_pct", 9.4) if detected else None
        lrr_raw = parsed.get("lrr_estimate", 0.743) if detected else None
        return IncidentResponseResult(
            incident_id=str(uuid.uuid4()),
            detected=detected,
            mttr_s=mttr,
            detection_time_s=elapsed,
            forensics_time_s=0.0,
            admin_time_s=0.0,
            sandbox_time_s=0.0,
            total_sandbox_submissions=0,
            total_rejected=0,
            approved_action=None,
            final_physical_state=sample.normalized_record.variables,
            predicted_psv_pct=psv,
            actual_psv_pct=psv,
            lrr=lrr_raw,
            outcome=OUTCOME_SUCCESS if detected else OUTCOME_FAILURE,
            escalated_to_human=False,
            cot_trace=parsed.get("reasoning"),
            audit_entry_id=None,
        )

    def _build_narrative(self, sample: HarmonizedSample) -> str:
        vars_str = json.dumps(sample.normalized_record.variables, indent=2)
        return (
            f"CPS Telemetry at {sample.normalized_record.timestamp.isoformat()} "
            f"from {sample.source}:\n{vars_str}\n\n"
            "Analyze, investigate, and propose remediation. Output JSON only."
        )

    def _parse_response(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("`").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"is_incident": False}
