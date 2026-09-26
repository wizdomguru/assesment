from app.models.answers import AnswerCitation, AnswerResult
from app.services.cache import AnswerCache


class FakeRedis:
    def __init__(self) -> None:
        self.values = {}
        self.expirations = {}

    def get(self, key):
        return self.values.get(key)

    def incr(self, key):
        value = int(self.values.get(key, 0)) + 1
        self.values[key] = str(value)
        return value

    def setex(self, key, ttl, value):
        self.values[key] = value
        self.expirations[key] = ttl


def make_cache() -> AnswerCache:
    return AnswerCache(
        "unused",
        60,
        "chunks",
        "top_k=5;threshold=.7",
        "model=test-model",
        client=FakeRedis(),
    )


def test_cache_round_trips_successful_answer_and_citation() -> None:
    cache = make_cache()
    result = AnswerResult(
        "Use OAuth.", [AnswerCitation("guide.md", "doc-1-0", 2, "Auth")], True
    )

    cache.set("How?", result)
    cached = cache.get("How?")

    assert cached == result
    assert cache.client.expirations[next(iter(cache.client.expirations))] == 60


def test_cache_does_not_store_fallback_answers() -> None:
    cache = make_cache()

    cache.set("Unknown?", AnswerResult("Not enough", [], False))

    assert cache.get("Unknown?") is None


def test_collection_version_invalidates_old_key() -> None:
    cache = make_cache()
    result = AnswerResult("Use OAuth.", [], True)

    cache.set("How?", result)
    assert cache.get("How?") == result
    assert cache.bump_collection_version() == 1
    assert cache.get("How?") is None


def test_cache_separates_per_request_retrieval_settings() -> None:
    cache = make_cache()
    result = AnswerResult("Use OAuth.", [], True)
    request_config = "top_k=3;threshold=0.8"

    cache.set("How?", result, retrieval_config=request_config)

    assert cache.get("How?", retrieval_config=request_config) == result
    assert cache.get("How?", retrieval_config="top_k=5;threshold=0.7") is None