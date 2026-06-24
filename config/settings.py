"""
Centralized configuration management for AI Recruiter Assistant.
Loads from environment variables with sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

SKILLS_JSON_PATH = BASE_DIR / "config" / "skills.json"


@dataclass
class EmbeddingConfig:
    backend: str = os.getenv("EMBEDDING_BACKEND", "ollama")  # "ollama" | "huggingface"
    ollama_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    hf_model: str = os.getenv("HF_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = 768


@dataclass
class LLMConfig:
    backend: str = os.getenv("LLM_BACKEND", "groq")  # "groq" | "ollama" | "openai"
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    ollama_model: str = os.getenv("OLLAMA_LLM_MODEL", "llama3")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_prompt_chars: int = int(os.getenv("LLM_MAX_PROMPT_CHARS", "6000"))


@dataclass
class CacheConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true")
    dir: Path = field(default_factory=lambda: BASE_DIR / ".cache")
    ttl_seconds: int = field(default_factory=lambda: int(os.getenv("CACHE_TTL_SECONDS", "3600")))


@dataclass
class SkillConfig:
    fuzzy_threshold: int = int(os.getenv("SKILL_FUZZY_THRESHOLD", "85"))
    embedding_similarity_threshold: float = float(
        os.getenv("SKILL_EMBED_THRESHOLD", "0.80")
    )
    use_embeddings: bool = os.getenv("SKILL_USE_EMBEDDINGS", "true").lower() == "true"
    db_path: Path = field(default_factory=lambda: BASE_DIR / "config" / "skills.json")


@dataclass
class AppConfig:
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    skill: SkillConfig = field(default_factory=SkillConfig)

    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")
    faiss_index_path: Path = field(default_factory=lambda: BASE_DIR / "vectorstore" / "index.faiss")
    faiss_meta_path: Path = field(default_factory=lambda: BASE_DIR / "vectorstore" / "meta.pkl")

    def __post_init__(self) -> None:
        self.cache.dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_index_path.parent.mkdir(parents=True, exist_ok=True)


# Singleton
settings = AppConfig()
