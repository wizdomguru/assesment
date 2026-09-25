from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Axentra RAG Service"
    app_version: str = "0.1.0"
    environment: str = "development"
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "document_chunks"
    retrieval_top_k: int = 5
    retrieval_score_threshold: float | None = None
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    llm_enabled: bool = True
    llm_timeout_seconds: float = 30.0
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()