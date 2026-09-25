from app.models.documents import DocumentChunk, RetrievedChunk
from app.services.answer_generation import (
    FALLBACK_ANSWER,
    AnswerGenerationService,
    build_grounded_messages,
)
from app.services.llm import OpenAICompatibleClient
from app.services.retrieval import RetrievalDecision


class FakeLLM:
    def __init__(self) -> None:
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        return "Use OAuth 2.0."


class FakeResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": "grounded answer"}}]}


class FakeHTTPClient:
    def __init__(self) -> None:
        self.request = None

    def post(self, path, **kwargs):
        self.request = (path, kwargs)
        return FakeResponse()


def make_decision(sufficient: bool = True) -> RetrievalDecision:
    result = RetrievedChunk(
        DocumentChunk("Use OAuth 2.0.", "doc-1", "guide.md", "doc-1-0", "Auth", 4),
        0.9,
    )
    return RetrievalDecision(sufficient, [result] if sufficient else [], "test")


def test_grounded_answer_has_metadata_citations() -> None:
    llm = FakeLLM()
    result = AnswerGenerationService(llm).answer("How?", make_decision())

    assert result.answer == "Use OAuth 2.0."
    assert result.grounded is True
    assert result.citations[0].filename == "guide.md"
    assert result.citations[0].page_number == 4
    assert "Do not use outside knowledge" in llm.messages[0]["content"]


def test_insufficient_context_declines_without_calling_llm() -> None:
    llm = FakeLLM()
    result = AnswerGenerationService(llm).answer("Unknown?", make_decision(False))

    assert result.answer == FALLBACK_ANSWER
    assert result.grounded is False
    assert result.citations == []
    assert llm.messages is None


def test_prompt_contains_only_supplied_context() -> None:
    messages = build_grounded_messages("How?", make_decision().chunks)

    assert "Use OAuth 2.0." in messages[1]["content"]
    assert "unprovided fact" not in messages[1]["content"]


def test_openai_compatible_client_posts_grounded_request() -> None:
    transport = FakeHTTPClient()
    client = OpenAICompatibleClient(
        "https://llm.example/v1", "secret", "test-model", client=transport
    )

    assert client.complete([{"role": "user", "content": "Hello"}]) == "grounded answer"
    path, request = transport.request
    assert path == "/chat/completions"
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert request["json"]["model"] == "test-model"
    assert request["json"]["temperature"] == 0