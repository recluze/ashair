from __future__ import annotations
import json
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import httpx

from configs.config import AgentConfig

logger = logging.getLogger(__name__)


class LLMInferenceError(Exception):
    pass


class BaseAgent(ABC):
    def __init__(self, config: AgentConfig, agent_name: str):
        self.config = config
        self.agent_name = agent_name
        self._client = httpx.Client(timeout=120.0)

    def _call_llm(self, system_prompt: str, user_message: str) -> str:
        payload = {
            "model": self.config.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        endpoint = f"{self.config.inference_endpoint}/chat/completions"
        try:
            response = self._client.post(endpoint, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise LLMInferenceError(f"HTTP {exc.response.status_code}: {exc.response.text}") from exc
        except Exception as exc:
            raise LLMInferenceError(str(exc)) from exc

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip().rstrip("```").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.error("%s: JSON parse error: %s\nRaw: %s", self.agent_name, exc, raw)
            raise

    def _timed_call(self, system_prompt: str, user_message: str) -> tuple[Dict[str, Any], float]:
        start = time.perf_counter()
        raw = self._call_llm(system_prompt, user_message)
        elapsed = time.perf_counter() - start
        parsed = self._parse_json_response(raw)
        return parsed, elapsed

    @abstractmethod
    def run(self, *args, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
