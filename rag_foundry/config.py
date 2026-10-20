"""Configuration loading and defaults.

Unknown keys are preserved so a newer configuration file stays

A single :class:`RagConfig` dataclass carries every tunable knob. It loads
from an optional JSON file and ignores unknown keys so older/newer configs
remain forward-compatible.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


@dataclass
class RagConfig:
    """All tunables for the pipeline. Defaults work out of the box."""

    index_path: str = "ragf_index.json"
    chunk_strategy: str = "fixed"  # "fixed" or "sentence"
    chunk_size: int = 800
    chunk_overlap: int = 120

    # retrieval
    top_k: int = 5
    rrf_k: int = 60
    min_score: float = 0.1

    # BM25
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # hashed vectorizer
    vector_dims: int = 4096

    # LLM (OpenAI-compatible chat completions)
    llm_base_url: str = "https://api.openai.com"
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.2
    llm_timeout: float = 60.0
    api_key_env: str = "RAGF_API_KEY"

    def api_key(self) -> Optional[str]:
        """Read the API key from the configured environment variable."""
        return os.environ.get(self.api_key_env)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the config to a plain dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RagConfig":
        """Build a config from a dict, ignoring unknown keys."""
        if not data:
            return cls()
        if not isinstance(data, dict):
            raise ValueError("config data must be a JSON object")
        valid = {f.name for f in dataclasses.fields(cls)}
        kwargs = {k: v for k, v in data.items() if k in valid}
        return cls(**kwargs)


def load_config(path: Optional[str]) -> RagConfig:
    """Load a RagConfig from a JSON file, or return defaults when path is None."""
    if path is None:
        return RagConfig()
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return RagConfig.from_dict(data)
