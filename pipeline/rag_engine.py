from __future__ import annotations
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from configs.config import RAGConfig
from pipeline.ingestion import NormalizedTelemetryRecord
from pipeline.normalizer import TelemetryNarrativeGenerator

logger = logging.getLogger(__name__)


@dataclass
class VectorRecord:
    record_id: str
    narrative: str
    embedding: np.ndarray
    metadata: Dict[str, Any]
    timestamp: datetime
    channel: str
    label: Optional[str] = None


@dataclass
class FeedbackRecord:
    action: Dict[str, Any]
    state_at_execution: Dict[str, Any]
    predicted_state: Dict[str, Any]
    actual_state: Dict[str, Any]
    outcome: str
    timestamp: datetime
    embedding: Optional[np.ndarray] = None
    record_id: Optional[str] = None


class EmbeddingModel:
    def __init__(self, model_name: str, embedding_dim: int):
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            logger.info("Loaded embedding model: %s", self.model_name)
        except ImportError:
            logger.warning("sentence-transformers not installed; using random embeddings")

    def embed(self, text: str) -> np.ndarray:
        self._load_model()
        if self._model is None:
            rng = np.random.default_rng(abs(hash(text)) % (2**32))
            vec = rng.standard_normal(self.embedding_dim).astype(np.float32)
            return vec / (np.linalg.norm(vec) + 1e-9)
        embedding = self._model.encode(text, normalize_embeddings=True)
        return embedding.astype(np.float32)

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        self._load_model()
        if self._model is None:
            return np.array([self.embed(t) for t in texts])
        embeddings = self._model.encode(texts, normalize_embeddings=True, batch_size=32)
        return embeddings.astype(np.float32)


class VectorDatabase:
    def __init__(self, config: RAGConfig):
        self.config = config
        self._records: List[VectorRecord] = []
        self._feedback_records: List[FeedbackRecord] = []
        self._embeddings: Optional[np.ndarray] = None
        self._dirty = True
        os.makedirs(config.vector_db_path, exist_ok=True)

    def add_record(self, record: VectorRecord) -> None:
        self._records.append(record)
        self._dirty = True

    def add_feedback_record(self, feedback: FeedbackRecord) -> None:
        self._feedback_records.append(feedback)
        self._dirty = True

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        channel_filter: Optional[str] = None,
    ) -> List[Tuple[VectorRecord, float]]:
        if not self._records:
            return []
        if self._dirty or self._embeddings is None:
            self._rebuild_index()
        scores = np.dot(self._embeddings, query_embedding)
        if channel_filter:
            mask = np.array([r.channel == channel_filter for r in self._records])
            scores = np.where(mask, scores, -np.inf)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self._records[i], float(scores[i])) for i in top_indices if scores[i] > -np.inf]

    def _rebuild_index(self) -> None:
        if not self._records:
            return
        self._embeddings = np.stack([r.embedding for r in self._records])
        norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        self._embeddings = self._embeddings / (norms + 1e-9)
        self._dirty = False

    def save(self, filename: str = "vector_db.pkl") -> None:
        path = os.path.join(self.config.vector_db_path, filename)
        with open(path, "wb") as f:
            pickle.dump({"records": self._records, "feedback": self._feedback_records}, f)
        logger.info("VectorDatabase saved: %d records", len(self._records))

    def load(self, filename: str = "vector_db.pkl") -> bool:
        path = os.path.join(self.config.vector_db_path, filename)
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._records = data.get("records", [])
        self._feedback_records = data.get("feedback", [])
        self._dirty = True
        logger.info("VectorDatabase loaded: %d records", len(self._records))
        return True

    def record_count(self) -> int:
        return len(self._records)


class RAGEngine:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.embedding_model = EmbeddingModel(config.embedding_model, config.embedding_dim)
        self.vector_db = VectorDatabase(config)
        self.narrative_generator = TelemetryNarrativeGenerator()
        self._record_counter = 0

    def index_telemetry_record(self, record: NormalizedTelemetryRecord) -> str:
        narrative = self.narrative_generator.generate(record)
        embedding = self.embedding_model.embed(narrative)
        record_id = f"{record.channel}_{record.timestamp.isoformat()}_{self._record_counter}"
        self._record_counter += 1
        vector_record = VectorRecord(
            record_id=record_id,
            narrative=narrative,
            embedding=embedding,
            metadata=record.metadata,
            timestamp=record.timestamp,
            channel=record.channel,
            label=record.metadata.get("label"),
        )
        self.vector_db.add_record(vector_record)
        return record_id

    def index_batch(self, records: List[NormalizedTelemetryRecord], show_progress: bool = True) -> List[str]:
        narratives = [self.narrative_generator.generate(r) for r in records]
        embeddings = self.embedding_model.embed_batch(narratives)
        record_ids = []
        for i, (record, narrative, embedding) in enumerate(zip(records, narratives, embeddings)):
            record_id = f"{record.channel}_{record.timestamp.isoformat()}_{self._record_counter}"
            self._record_counter += 1
            vector_record = VectorRecord(
                record_id=record_id,
                narrative=narrative,
                embedding=embedding,
                metadata=record.metadata,
                timestamp=record.timestamp,
                channel=record.channel,
                label=record.metadata.get("label"),
            )
            self.vector_db.add_record(vector_record)
            record_ids.append(record_id)
            if show_progress and (i + 1) % 1000 == 0:
                logger.info("RAGEngine: indexed %d / %d records", i + 1, len(records))
        return record_ids

    def retrieve_context(
        self,
        query_narrative: str,
        top_k: Optional[int] = None,
        channel_filter: Optional[str] = None,
    ) -> Tuple[str, float]:
        k = top_k or self.config.top_k
        query_embedding = self.embedding_model.embed(query_narrative)
        results = self.vector_db.search(query_embedding, top_k=k, channel_filter=channel_filter)
        if not results:
            return "No historical precedents found.", 0.0
        context_parts = []
        max_similarity = 0.0
        for rank, (record, similarity) in enumerate(results, 1):
            max_similarity = max(max_similarity, similarity)
            label_str = f" [Label: {record.label}]" if record.label else ""
            context_parts.append(
                f"[Precedent {rank} | similarity={similarity:.4f}{label_str}]\n{record.narrative}"
            )
        return "\n\n".join(context_parts), max_similarity

    def index_feedback_record(self, feedback: FeedbackRecord) -> None:
        action_str = json.dumps(feedback.action)
        state_str = json.dumps(feedback.state_at_execution)
        narrative = (
            f"Feedback record: action={action_str}, "
            f"outcome={feedback.outcome}, "
            f"state={state_str}"
        )
        embedding = self.embedding_model.embed(narrative)
        feedback.embedding = embedding
        feedback.record_id = f"feedback_{feedback.timestamp.isoformat()}_{self._record_counter}"
        self._record_counter += 1
        self.vector_db.add_feedback_record(feedback)
        indexed_record = VectorRecord(
            record_id=feedback.record_id,
            narrative=narrative,
            embedding=embedding,
            metadata={"outcome": feedback.outcome, "action_type": feedback.action.get("action_type")},
            timestamp=feedback.timestamp,
            channel="feedback",
            label=feedback.outcome,
        )
        self.vector_db.add_record(indexed_record)

    def save(self) -> None:
        self.vector_db.save()

    def load(self) -> bool:
        return self.vector_db.load()
