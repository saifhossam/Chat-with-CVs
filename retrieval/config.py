#retrieval/config.py
from dataclasses import dataclass, field
import os

from dotenv import load_dotenv

load_dotenv()

AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
LLM_MODEL = "gpt-4.1-nano"

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "cv_chunks"

RETRIEVAL_MODE = "Hybrid"
RRF_K = 60

@dataclass
class RAGConfig:
    qdrant_url: str = field(default_factory=lambda: os.getenv("QDRANT_URL", ""))
    qdrant_collection: str = field(default_factory=lambda: os.getenv("QDRANT_COLLECTION", "cv_chunks"))
    qdrant_api_key: str = field(default_factory=lambda: os.getenv("QDRANT_API_KEY", ""))

    openrouter_api_key: str = field(default_factory=lambda: os.getenv("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(default_factory=lambda: os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    gemini_embed_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBED_MODEL", "text-embedding-3-small"))

    dense_topn: int = field(default_factory=lambda: int(os.getenv("DENSE_TOPN", "20")))
    bm25_topn: int = field(default_factory=lambda: int(os.getenv("BM25_TOPN", "20")))
    rrf_k: int = field(default_factory=lambda: int(os.getenv("RRF_K", "60")))
    hybrid_topk: int = field(default_factory=lambda: int(os.getenv("HYBRID_TOPK", "8")))
    final_topk: int = field(default_factory=lambda: int(os.getenv("FINAL_TOPK", "5")))