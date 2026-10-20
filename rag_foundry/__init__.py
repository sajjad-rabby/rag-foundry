"""Retrieval-augmented generation over local documents.

The package ingests PDF, Markdown, text and HTML, indexes chunks with BM25 and a
character n-gram vector index, fuses the two with reciprocal rank fusion, and
answers questions with citations, falling back to extractive answers when no
LLM is configured. Evaluation compares generated answers to reference answers with an LLM judge.

Ingests documents (PDF/DOCX/MD/TXT/HTML), chunks them, indexes them with
hybrid BM25 + hashed-vector search (fused via Reciprocal Rank Fusion),
answers questions against an OpenAI-compatible chat endpoint, and scores
answers with an LLM judge.
"""

from rag_foundry.config import RagConfig, load_config
from rag_foundry.chunk import Chunk, chunk_text
from rag_foundry.index import InMemoryIndex, build_index_from_paths
from rag_foundry.retrieve import hybrid_search
from rag_foundry.generate import Answer, Refusal, LLMError, answer_question

__version__ = "0.1.0"

__all__ = [
    "RagConfig",
    "load_config",
    "Chunk",
    "chunk_text",
    "InMemoryIndex",
    "build_index_from_paths",
    "hybrid_search",
    "Answer",
    "Refusal",
    "LLMError",
    "answer_question",
    "__version__",
]
