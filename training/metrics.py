from __future__ import annotations
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline.orchestrator import IncidentResponseResult, OUTCOME_SUCCESS

logger = logging.getLogger(__name__)


@dataclass
class MetricSnapshot:
    mttr_s: Optional[float]
    zdm_pct: Optional[float]
    fprr_pct: Optional[float]
    lrr_pct: Optional[float]
    psv_pct: Optional[float]
    total_incidents: int
    total_fp: int
    total_zero_day: int
    total_zero_day_mitigated: int
    system_name: str = "ASHAIR"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "system": self.system_name,
            "MTTR_s": round(self.mttr_s, 3) if self.mttr_s is not None else None,
            "ZDM_pct": round(self.zdm_pct, 2) if self.zdm_pct is not None else None,
            "FPRR_pct": round(self.fprr_pct, 2) if self.fprr_pct is not None else None,
            "LRR_pct": round(self.lrr_pct, 2) if self.lrr_pct is not None else None,
            "PSV_pct": round(self.psv_pct, 2) if self.psv_pct is not None else None,
            "total_incidents": self.total_incidents,
            "total_fp": self.total_fp,
            "total_zero_day": self.total_zero_day,
            "total_zero_day_mitigated": self.total_zero_day_mitigated,
        }


class MetricsAccumulator:
    def __init__(self):
        self._mttr_values: List[float] = []
        self._psv_values: List[float] = []
        self._lrr_values: List[float] = []
        self._false_positives: int = 0
        self._true_negatives: int = 0
        self._zero_day_total: int = 0
        self._zero_day_mitigated: int = 0
        self._total_benign_presented: int = 0

    def record_incident_result(
        self,
        result: IncidentResponseResult,
        true_label: str,
        is_zero_day: bool = False,
    ) -> None:
        is_true_attack = true_label != "Benign"
        is_detected = result.detected
        if is_true_attack and is_detected:
            if result.mttr_s is not None:
                self._mttr_values.append(result.mttr_s)
            if result.actual_psv_pct is not None:
                self._psv_values.append(result.actual_psv_pct)
            elif result.predicted_psv_pct is not None:
                self._psv_values.append(result.predicted_psv_pct)
            if result.lrr is not None:
                self._lrr_values.append(result.lrr * 100.0)
            if is_zero_day:
                self._zero_day_total += 1
                if result.outcome == OUTCOME_SUCCESS:
                    self._zero_day_mitigated += 1
        elif not is_true_attack:
            self._total_benign_presented += 1
            if is_detected:
                self._false_positives += 1
            else:
                self._true_negatives += 1

    def compute_metrics(self, system_name: str = "ASHAIR") -> MetricSnapshot:
        mttr = float(np.mean(self._mttr_values)) if self._mttr_values else None
        psv = float(np.max(self._psv_values)) if self._psv_values else None
        lrr = float(np.mean(self._lrr_values)) if self._lrr_values else None
        zdm = None
        if self._zero_day_total > 0:
            zdm = (self._zero_day_mitigated / self._zero_day_total) * 100.0
        fprr = None
        if self._total_benign_presented > 0:
            fprr = (self._false_positives / self._total_benign_presented) * 100.0
        return MetricSnapshot(
            mttr_s=mttr,
            zdm_pct=zdm,
            fprr_pct=fprr,
            lrr_pct=lrr,
            psv_pct=psv,
            total_incidents=len(self._mttr_values),
            total_fp=self._false_positives,
            total_zero_day=self._zero_day_total,
            total_zero_day_mitigated=self._zero_day_mitigated,
            system_name=system_name,
        )

    def reset(self) -> None:
        self._mttr_values.clear()
        self._psv_values.clear()
        self._lrr_values.clear()
        self._false_positives = 0
        self._true_negatives = 0
        self._zero_day_total = 0
        self._zero_day_mitigated = 0
        self._total_benign_presented = 0


class PerAttackVectorMetrics:
    VECTORS = ["FDI-water", "FDI-ics-net", "FDI-turbine", "DDoS"]

    def __init__(self):
        self._accumulators: Dict[str, MetricsAccumulator] = {
            v: MetricsAccumulator() for v in self.VECTORS
        }

    def record(
        self,
        result: IncidentResponseResult,
        true_label: str,
        source: str,
        is_zero_day: bool = False,
    ) -> None:
        vector = self._map_to_vector(true_label, source)
        if vector:
            self._accumulators[vector].record_incident_result(result, true_label, is_zero_day)

    def compute_all(self) -> Dict[str, MetricSnapshot]:
        return {
            vector: acc.compute_metrics(system_name=vector)
            for vector, acc in self._accumulators.items()
        }

    def _map_to_vector(self, true_label: str, source: str) -> Optional[str]:
        if true_label == "DDoS":
            return "DDoS"
        if true_label == "FDI":
            if source == "SWaT":
                return "FDI-water"
            elif source == "ICS-NAD":
                return "FDI-ics-net"
            elif source == "HAI":
                return "FDI-turbine"
        return None


class PromptInjectionMetrics:
    CATEGORIES = ["privilege_escalation", "goal_hijacking", "action_override", "context_poisoning"]

    def __init__(self):
        self._isr_counts: Dict[str, Dict[str, int]] = {
            cat: {"total": 0, "success": 0} for cat in self.CATEGORIES
        }
        self._pir_with_zt: Dict[str, Dict[str, int]] = {
            cat: {"total": 0, "physical_impact": 0} for cat in self.CATEGORIES
        }
        self._pir_without_zt: Dict[str, Dict[str, int]] = {
            cat: {"total": 0, "physical_impact": 0} for cat in self.CATEGORIES
        }

    def record_injection_scenario(
        self,
        category: str,
        injection_succeeded: bool,
        physical_impact_with_zt: bool,
        physical_impact_without_zt: bool,
    ) -> None:
        if category not in self.CATEGORIES:
            return
        self._isr_counts[category]["total"] += 1
        if injection_succeeded:
            self._isr_counts[category]["success"] += 1
        self._pir_with_zt[category]["total"] += 1
        if physical_impact_with_zt:
            self._pir_with_zt[category]["physical_impact"] += 1
        self._pir_without_zt[category]["total"] += 1
        if physical_impact_without_zt:
            self._pir_without_zt[category]["physical_impact"] += 1

    def compute_summary(self) -> Dict[str, Any]:
        summary = {}
        total_isr_success = 0
        total_isr_total = 0
        total_pir_with = 0
        total_pir_without = 0
        for cat in self.CATEGORIES:
            total = self._isr_counts[cat]["total"]
            isr = (self._isr_counts[cat]["success"] / total * 100.0) if total > 0 else 0.0
            pir_with = (self._pir_with_zt[cat]["physical_impact"] / total * 100.0) if total > 0 else 0.0
            pir_without = (self._pir_without_zt[cat]["physical_impact"] / total * 100.0) if total > 0 else 0.0
            summary[cat] = {
                "scenarios": total,
                "ISR_pct": round(isr, 1),
                "PIR_with_ZT_pct": round(pir_with, 1),
                "PIR_without_ZT_pct": round(pir_without, 1),
            }
            total_isr_success += self._isr_counts[cat]["success"]
            total_isr_total += total
            total_pir_with += self._pir_with_zt[cat]["physical_impact"]
            total_pir_without += self._pir_without_zt[cat]["physical_impact"]
        overall_isr = (total_isr_success / total_isr_total * 100.0) if total_isr_total > 0 else 0.0
        overall_pir_with = (total_pir_with / total_isr_total * 100.0) if total_isr_total > 0 else 0.0
        overall_pir_without = (total_pir_without / total_isr_total * 100.0) if total_isr_total > 0 else 0.0
        summary["overall"] = {
            "scenarios": total_isr_total,
            "ISR_pct": round(overall_isr, 1),
            "PIR_with_ZT_pct": round(overall_pir_with, 1),
            "PIR_without_ZT_pct": round(overall_pir_without, 1),
        }
        return summary
