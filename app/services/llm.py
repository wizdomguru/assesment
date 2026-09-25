from collections.abc import Sequence
import logging
import time
from typing import Any

import httpx


logger = logging.getLogger(__name__)


class OpenAICompatibleClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout_seconds: float = 30.0,
        client: Any = None,
    ):
        if not api_key:
            raise ValueError("LLM API key is required")
        self.model = model
        self.client = client or httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
        )
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def complete(self, messages: Sequence[dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0,
        }
        for attempt in range(3):
            try:
                response = self.client.post(
                    "/chat/completions",
                    headers=self.headers,
                    json=payload,
                )
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 503 or attempt == 2:
                    raise RuntimeError("LLM request failed") from exc
                delay = 2**attempt
                logger.warning(
                    "LLM provider returned 503; retrying in %ss (attempt %d/3)",
                    delay,
                    attempt + 2,
                )
                time.sleep(delay)
            except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
                raise RuntimeError("LLM request failed") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM returned an empty answer")
        return content.strip()