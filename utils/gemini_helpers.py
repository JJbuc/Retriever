from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
import threading
from typing import Any, Dict, List, Tuple

import google.generativeai as genai

LOGGER = logging.getLogger(__name__)


def _usage_to_dict(usage: Any | None) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    result: Dict[str, Any] = {}
    for key in ("prompt_token_count", "candidates_token_count", "total_token_count"):
        if hasattr(usage, key):
            result[key] = getattr(usage, key)
    return result


def _extract_json(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("{"):
            cleaned = "\n".join(lines)
        else:
            cleaned = "\n".join(lines[1:])
    match = cleaned.find("{")
    if match != -1:
        cleaned = cleaned[match:]
    return json.loads(cleaned)


def _response_text(response: genai.types.GenerateContentResponse) -> str:
    if getattr(response, "text", None):
        return response.text
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if content and getattr(content, "parts", None):
            texts = [part.text for part in content.parts if getattr(part, "text", None)]
            if texts:
                return "\n".join(texts)
    return ""


@dataclass
class GenerationResult:
    text: str
    latency_seconds: float
    prompt_tokens: int
    completion_tokens: int
    raw: Any | None = None


@dataclass
class EvaluationScores:
    context_relevance: float
    answer_relevance: float
    rationale: str
    prompt_tokens: int
    completion_tokens: int
    raw: Any | None = None


@dataclass
class Triple:
    subject: str
    relation: str
    object: str
    source_chunk_id: str
    confidence: float
    metadata: Dict[str, Any]


class GeminiService:
    def __init__(
        self,
        api_key: str | None,
        generator_model: str,
        judge_model: str,
        max_calls_per_minute: int,
        min_call_interval: float,
    ) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is required when using GeminiService.")
        genai.configure(api_key=api_key)
        self.generator_model_name = generator_model
        self.judge_model_name = judge_model
        self.generator = genai.GenerativeModel(generator_model)
        try:
            self.judge = genai.GenerativeModel(judge_model)
            self._judge_name = judge_model
            self._judge_fallback = False
        except Exception as exc:  # pragma: no cover - depends on external service configs
            LOGGER.warning(
                "Failed to initialise Gemini judge model %s (%s). Falling back to generator model %s.",
                judge_model,
                exc,
                generator_model,
            )
            self.judge = self.generator
            self._judge_name = generator_model
            self._judge_fallback = True

        self._max_calls_per_minute = max(0, max_calls_per_minute)
        self._min_call_interval = max(0.0, min_call_interval)
        self._rate_lock = threading.Lock()
        self._call_timestamps: deque[float] = deque()
        self._last_call_at: float | None = None

    def generate_answer(self, question: str, context: str) -> GenerationResult:
        prompt = (
            "You are a concise assistant. Use the provided context strictly to answer the question.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        start = time.perf_counter()
        response = self._call_model(self.generator, prompt)
        latency = time.perf_counter() - start
        text = _response_text(response).strip()
        usage = _usage_to_dict(getattr(response, "usage_metadata", None))
        return GenerationResult(
            text=text,
            latency_seconds=latency,
            prompt_tokens=usage.get("prompt_token_count", 0),
            completion_tokens=usage.get("candidates_token_count", 0),
            raw=response,
        )

    def evaluate_relevance(self, question: str, context: str, answer: str) -> EvaluationScores:
        prompt = (
            "You are scoring retrieval quality on a 1-5 scale.\n"
            "Context Relevance: Does the retrieved context contain the information to answer the question?\n"
            "Answer Relevance: Does the answer correctly solve the question using the context?\n"
            "Respond in JSON with keys: context_score, answer_score, rationale.\n\n"
            f"Question: {question}\n\nContext:\n{context}\n\nAnswer:\n{answer}"
        )
        response = self._generate_with_judge(prompt)
        usage = _usage_to_dict(getattr(response, "usage_metadata", None))
        try:
            payload = _extract_json(_response_text(response))
        except Exception as exc:  # pragma: no cover - fallback logging
            LOGGER.error("Failed to parse Gemini judge response: %s\n%s", exc, _response_text(response))
            payload = {"context_score": 0, "answer_score": 0, "rationale": "Parsing failure"}
        return EvaluationScores(
            context_relevance=float(payload.get("context_score", 0)),
            answer_relevance=float(payload.get("answer_score", 0)),
            rationale=str(payload.get("rationale", "")),
            prompt_tokens=usage.get("prompt_token_count", 0),
            completion_tokens=usage.get("candidates_token_count", 0),
            raw=response,
        )

    def extract_triples(self, chunk_id: str, text: str) -> Tuple[List[Triple], Dict[str, Any]]:
        prompt = (
            "Extract entities and relationships from the text.\n"
            "Return a JSON object with `triples` (list of {subject, relation, object, confidence, metadata}) "
            "and `entities` (list of unique entity names).\n"
            "Text:\n"
            f"{text}\n"
        )
        response = self._generate_with_judge(prompt)
        usage = _usage_to_dict(getattr(response, "usage_metadata", None))
        try:
            payload = _extract_json(_response_text(response))
        except Exception as exc:  # pragma: no cover - protective logging
            LOGGER.error("Graph extraction parse error: %s -> %s", exc, _response_text(response))
            payload = {"triples": [], "entities": []}

        triples: List[Triple] = []
        for item in payload.get("triples", []):
            triples.append(
                Triple(
                    subject=str(item.get("subject", "")).strip(),
                    relation=str(item.get("relation", "")).strip(),
                    object=str(item.get("object", "")).strip(),
                    source_chunk_id=chunk_id,
                    confidence=float(item.get("confidence", 1.0) or 1.0),
                    metadata=item.get("metadata") or {},
                )
            )

        metadata = {
            "entities": payload.get("entities", []),
            "usage": usage,
            "raw": response,
        }
        return triples, metadata

    def _generate_with_judge(self, prompt: str):
        try:
            return self._call_model(self.judge, prompt)
        except Exception as exc:  # pragma: no cover - depends on external service availability
            if self._judge_fallback:
                raise
            LOGGER.warning(
                "Gemini judge model %s failed with %s; retrying with generator model %s.",
                self._judge_name,
                exc,
                self.generator_model_name,
            )
            self.judge = self.generator
            self._judge_fallback = True
            return self._call_model(self.judge, prompt)

    def _call_model(self, model, prompt: str):
        self._before_call()
        try:
            return model.generate_content(prompt)
        finally:
            self._after_call()

    def _before_call(self) -> None:
        if self._max_calls_per_minute <= 0 and self._min_call_interval <= 0:
            return
        wait_time = 0.0
        while True:
            with self._rate_lock:
                now = time.perf_counter()
                window = 60.0
                while self._call_timestamps and now - self._call_timestamps[0] >= window:
                    self._call_timestamps.popleft()
                if self._min_call_interval > 0 and self._last_call_at is not None:
                    elapsed = now - self._last_call_at
                    if elapsed < self._min_call_interval:
                        wait_time = max(wait_time, self._min_call_interval - elapsed)
                if self._max_calls_per_minute > 0 and len(self._call_timestamps) >= self._max_calls_per_minute:
                    earliest = self._call_timestamps[0]
                    wait_time = max(wait_time, window - (now - earliest))
                if wait_time == 0.0:
                    return
            time.sleep(wait_time)
            wait_time = 0.0

    def _after_call(self) -> None:
        if self._max_calls_per_minute <= 0 and self._min_call_interval <= 0:
            return
        with self._rate_lock:
            now = time.perf_counter()
            window = 60.0
            if self._max_calls_per_minute > 0:
                while self._call_timestamps and now - self._call_timestamps[0] >= window:
                    self._call_timestamps.popleft()
                self._call_timestamps.append(now)
            self._last_call_at = now

