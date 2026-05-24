from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from configs.config import SafetyEnvelopeConfig

logger = logging.getLogger(__name__)


@dataclass
class EnvelopeViolation:
    variable: str
    observed_value: float
    lower_bound: float
    upper_bound: float
    deviation_pct: float


@dataclass
class EnvelopeCheckResult:
    within_envelope: bool
    violations: List[EnvelopeViolation]
    max_psv_pct: float
    divergence: float
    divergence_within_threshold: bool


class SafetyEnvelope:
    def __init__(self, config: SafetyEnvelopeConfig):
        self.config = config
        self._domain_bounds: Dict[str, Dict[str, Tuple[float, float]]] = {
            "swat": config.swat,
            "hai": config.hai,
            "ics_nad": config.ics_nad,
        }

    def check(
        self,
        predicted_state: Dict[str, Any],
        domain: str,
        divergence: float,
        divergence_threshold: float,
    ) -> EnvelopeCheckResult:
        bounds = self._domain_bounds.get(domain, {})
        violations = []
        max_psv = 0.0
        for variable, (lo, hi) in bounds.items():
            value = predicted_state.get(variable)
            if value is None:
                continue
            try:
                fval = float(value)
            except (ValueError, TypeError):
                continue
            nominal_mid = (lo + hi) / 2.0
            if fval < lo:
                dev = abs(lo - fval) / (nominal_mid + 1e-9) * 100.0
                violations.append(EnvelopeViolation(variable, fval, lo, hi, dev))
                max_psv = max(max_psv, dev)
            elif fval > hi:
                dev = abs(fval - hi) / (nominal_mid + 1e-9) * 100.0
                violations.append(EnvelopeViolation(variable, fval, lo, hi, dev))
                max_psv = max(max_psv, dev)

        divergence_ok = divergence <= divergence_threshold
        within_envelope = len(violations) == 0 and divergence_ok
        return EnvelopeCheckResult(
            within_envelope=within_envelope,
            violations=violations,
            max_psv_pct=max_psv,
            divergence=divergence,
            divergence_within_threshold=divergence_ok,
        )

    def compute_psv(
        self,
        observed_state: Dict[str, Any],
        nominal_state: Dict[str, Any],
        domain: str,
    ) -> float:
        bounds = self._domain_bounds.get(domain, {})
        max_psv = 0.0
        for variable in bounds:
            observed = observed_state.get(variable)
            nominal = nominal_state.get(variable)
            if observed is None or nominal is None:
                continue
            try:
                fobs = float(observed)
                fnom = float(nominal)
            except (ValueError, TypeError):
                continue
            if abs(fnom) < 1e-9:
                continue
            psv = abs(fobs - fnom) / abs(fnom) * 100.0
            max_psv = max(max_psv, psv)
        return max_psv

    def get_bounds(self, domain: str) -> Dict[str, Tuple[float, float]]:
        return self._domain_bounds.get(domain, {})

    def add_custom_bounds(self, domain: str, bounds: Dict[str, Tuple[float, float]]) -> None:
        if domain not in self._domain_bounds:
            self._domain_bounds[domain] = {}
        self._domain_bounds[domain].update(bounds)
