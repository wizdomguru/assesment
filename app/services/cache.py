import hashlib
import json
from dataclasses import asdict
from typing import Any

from redis import Redis

from app.models.answers import AnswerCitation, AnswerResult


class AnswerCache:
    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        collection_name: str,
        retrieval_config: str,
        llm_config: str,
        client: Any = None,
    ):
        if ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be greater than zero")
        self.ttl_seconds = ttl_seconds
        self.collection_name = collection_name
        self.retrieval_config = retrieval_config
        self.llm_config = llm_config
        self.client = client or Redis.from_url(redis_url, decode_responses=True)

    def _version_key(self) -> str:
        return f"rag:collection-version:{self.collection_name}"

    def _cache_key(
        self, question: str, version: int, retrieval_config: str | None = None
    ) -> str:
        material = json.dumps(
            {
                "collection": self.collection_name,
                "version": version,
                "question": question,
                "retrieval": retrieval_config or self.retrieval_config,
                "llm": self.llm_config,
            },
            sort_keys=True,
        ).encode("utf-8")
        digest = hashlib.sha256(material).hexdigest()
        return f"rag:answer:{digest}"

    def collection_version(self) -> int:
        value = self.client.get(self._version_key())
        return int(value) if value is not None else 0

    def bump_collection_version(self) -> int:
        return int(self.client.incr(self._version_key()))

    def get(
        self, question: str, retrieval_config: str | None = None
    ) -> AnswerResult | None:
        key = self._cache_key(
            question, self.collection_version(), retrieval_config
        )
        raw = self.client.get(key)
        if raw is None:
            return None
        value = json.loads(raw)
        return AnswerResult(
            answer=value["answer"],
            grounded=value["grounded"],
            citations=[AnswerCitation(**citation) for citation in value["citations"]],
        )

    def set(
        self,
        question: str,
        result: AnswerResult,
        retrieval_config: str | None = None,
    ) -> None:
        if not result.grounded:
            return
        key = self._cache_key(
            question, self.collection_version(), retrieval_config
        )
        self.client.setex(key, self.ttl_seconds, json.dumps(asdict(result)))