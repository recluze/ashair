from __future__ import annotations
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline.ingestion import (
    CSVBatchIngester, NormalizedTelemetryRecord, RawTelemetryRecord, TelemetryNormalizer
)

logger = logging.getLogger(__name__)

LABEL_FDI = "FDI"
LABEL_DDOS = "DDoS"
LABEL_BENIGN = "Benign"

SWAT_ATTACK_KEYWORDS = ["attack"]
ICS_NAD_DDOS_CATEGORIES = ["dos", "ddos", "denial"]
ICS_NAD_FDI_CATEGORIES = ["fdi", "false data injection", "injection"]
HAI_FDI_ATTACK_PRIMITIVES = ["sp", "pv", "cv", "cp"]


@dataclass
class HarmonizedSample:
    normalized_record: NormalizedTelemetryRecord
    unified_label: str
    original_label: str
    source: str
    is_zero_day_holdout: bool = False


class LabelNormalizer:
    def normalize_swat_label(self, raw_label: str, attack_type: str = "") -> str:
        if raw_label.strip().lower() in ("attack", "1"):
            return LABEL_FDI
        return LABEL_BENIGN

    def normalize_hai_label(self, raw_label: str, attack_type: str = "") -> str:
        if raw_label.strip().lower() in ("attack", "1") or raw_label.strip() != "0":
            for primitive in HAI_FDI_ATTACK_PRIMITIVES:
                if primitive in attack_type.lower():
                    return LABEL_FDI
            return LABEL_FDI
        return LABEL_BENIGN

    def normalize_ics_nad_label(self, raw_label: str, attack_category: str = "") -> str:
        cat_lower = attack_category.lower()
        if any(kw in cat_lower for kw in ICS_NAD_DDOS_CATEGORIES):
            return LABEL_DDOS
        if any(kw in cat_lower for kw in ICS_NAD_FDI_CATEGORIES):
            return LABEL_FDI
        label_lower = raw_label.lower()
        if label_lower not in ("benign", "normal", "0"):
            return LABEL_FDI
        return LABEL_BENIGN


class TemporalSplitter:
    def __init__(self, test_ratio: float = 0.2, zero_day_ratio: float = 0.2, random_seed: int = 42):
        self.test_ratio = test_ratio
        self.zero_day_ratio = zero_day_ratio
        self.random_seed = random_seed

    def split(
        self, samples: List[HarmonizedSample]
    ) -> Tuple[List[HarmonizedSample], List[HarmonizedSample]]:
        samples_sorted = sorted(samples, key=lambda s: s.normalized_record.timestamp)
        split_idx = int(len(samples_sorted) * (1.0 - self.test_ratio))
        train_samples = samples_sorted[:split_idx]
        test_samples = samples_sorted[split_idx:]
        self._mark_zero_day_holdout(test_samples, train_samples)
        return train_samples, test_samples

    def _mark_zero_day_holdout(
        self,
        test_samples: List[HarmonizedSample],
        train_samples: List[HarmonizedSample],
    ) -> None:
        train_primitives = set(
            s.original_label.strip().lower()
            for s in train_samples
            if s.unified_label != LABEL_BENIGN
        )
        for sample in test_samples:
            if sample.unified_label == LABEL_BENIGN:
                sample.is_zero_day_holdout = False
                continue
            original_lower = sample.original_label.strip().lower()
            sample.is_zero_day_holdout = original_lower not in train_primitives


class DatasetHarmonizer:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.ingester = CSVBatchIngester()
        self.normalizer = TelemetryNormalizer()
        self.label_normalizer = LabelNormalizer()
        self.splitter = TemporalSplitter(random_seed=random_seed)

    def load_and_harmonize_swat(self, filepath: str) -> List[HarmonizedSample]:
        raw_records = self.ingester.ingest_swat(filepath)
        samples = []
        for raw in raw_records:
            norm = self.normalizer.normalize(raw)
            original_label = str(norm.metadata.get("raw_label", "Normal"))
            unified_label = self.label_normalizer.normalize_swat_label(original_label)
            samples.append(HarmonizedSample(
                normalized_record=norm,
                unified_label=unified_label,
                original_label=original_label,
                source="SWaT",
            ))
        logger.info("Loaded SWaT: %d samples", len(samples))
        return samples

    def load_and_harmonize_hai(self, filepath: str) -> List[HarmonizedSample]:
        raw_records = self.ingester.ingest_hai(filepath)
        samples = []
        for raw in raw_records:
            norm = self.normalizer.normalize(raw)
            original_label = str(norm.metadata.get("label", "Normal"))
            attack_type = str(norm.metadata.get("attack_type", ""))
            unified_label = self.label_normalizer.normalize_hai_label(original_label, attack_type)
            samples.append(HarmonizedSample(
                normalized_record=norm,
                unified_label=unified_label,
                original_label=original_label,
                source="HAI",
            ))
        logger.info("Loaded HAI: %d samples", len(samples))
        return samples

    def load_and_harmonize_ics_nad(self, filepath: str) -> List[HarmonizedSample]:
        raw_records = self.ingester.ingest_ics_nad(filepath)
        samples = []
        for raw in raw_records:
            norm = self.normalizer.normalize(raw)
            original_label = str(norm.metadata.get("label", "Benign"))
            attack_category = str(norm.metadata.get("attack_category", ""))
            unified_label = self.label_normalizer.normalize_ics_nad_label(original_label, attack_category)
            samples.append(HarmonizedSample(
                normalized_record=norm,
                unified_label=unified_label,
                original_label=original_label,
                source="ICS-NAD",
            ))
        logger.info("Loaded ICS-NAD: %d samples", len(samples))
        return samples

    def build_corpus(
        self,
        swat_path: Optional[str] = None,
        hai_path: Optional[str] = None,
        ics_nad_path: Optional[str] = None,
    ) -> Tuple[List[HarmonizedSample], List[HarmonizedSample]]:
        all_samples: List[HarmonizedSample] = []
        if swat_path and os.path.exists(swat_path):
            all_samples.extend(self.load_and_harmonize_swat(swat_path))
        if hai_path and os.path.exists(hai_path):
            all_samples.extend(self.load_and_harmonize_hai(hai_path))
        if ics_nad_path and os.path.exists(ics_nad_path):
            all_samples.extend(self.load_and_harmonize_ics_nad(ics_nad_path))
        all_samples = self._deduplicate(all_samples)
        all_samples = self._balance_benign_samples(all_samples)
        train_samples, test_samples = self.splitter.split(all_samples)
        self._log_corpus_stats(train_samples, test_samples)
        return train_samples, test_samples

    def _deduplicate(self, samples: List[HarmonizedSample]) -> List[HarmonizedSample]:
        seen_timestamps = set()
        deduplicated = []
        for sample in samples:
            ts_key = (
                sample.normalized_record.timestamp.isoformat(),
                sample.source,
                sample.unified_label,
            )
            if ts_key not in seen_timestamps:
                seen_timestamps.add(ts_key)
                deduplicated.append(sample)
        removed = len(samples) - len(deduplicated)
        if removed:
            logger.info("Deduplication removed %d samples", removed)
        return deduplicated

    def _balance_benign_samples(self, samples: List[HarmonizedSample]) -> List[HarmonizedSample]:
        attack_count = sum(1 for s in samples if s.unified_label != LABEL_BENIGN)
        benign_samples = [s for s in samples if s.unified_label == LABEL_BENIGN]
        attack_samples = [s for s in samples if s.unified_label != LABEL_BENIGN]
        if len(benign_samples) > attack_count * 2:
            rng = np.random.default_rng(self.random_seed)
            indices = rng.choice(len(benign_samples), size=attack_count * 2, replace=False)
            benign_samples = [benign_samples[i] for i in sorted(indices)]
            logger.info(
                "Balanced benign samples: %d → %d (attack count: %d)",
                len(benign_samples) + (sum(1 for s in samples if s.unified_label == LABEL_BENIGN) - len(benign_samples)),
                len(benign_samples),
                attack_count,
            )
        return attack_samples + benign_samples

    def _log_corpus_stats(
        self,
        train: List[HarmonizedSample],
        test: List[HarmonizedSample],
    ) -> None:
        for split_name, split in [("Train", train), ("Test", test)]:
            counts: Dict[str, int] = {}
            for s in split:
                counts[s.unified_label] = counts.get(s.unified_label, 0) + 1
            zero_day = sum(1 for s in split if s.is_zero_day_holdout)
            logger.info(
                "%s split: %s | zero-day holdouts: %d",
                split_name,
                str(counts),
                zero_day,
            )
