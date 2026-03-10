#retrieval/embeddings.py
from __future__ import annotations

from typing import List
from openai import AzureOpenAI
from retrieval.config import AZURE_API_KEY, AZURE_ENDPOINT, EMBEDDING_MODEL


_client_instance = None


def get_embeddings_client() -> AzureOpenAI:
    global _client_instance
    if _client_instance is None:
        _client_instance = AzureOpenAI(
            api_key=AZURE_API_KEY,
            api_version="2024-02-01",
            azure_endpoint=AZURE_ENDPOINT,
        )
    return _client_instance


def embed_query(text: str) -> List[float]:
    """Embed a single query string (retrieval time)."""
    client = get_embeddings_client()
    response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def embed_documents(texts: List[str]) -> List[List[float]]:
    """Embed a batch of texts (index time). Batches to stay within API limits."""
    client = get_embeddings_client()
    batch_size = 512
    vectors = []
    for i in range(0, len(texts), batch_size):
        response = client.embeddings.create(
            input=texts[i : i + batch_size],
            model=EMBEDDING_MODEL,
        )
        # API returns data sorted by index — preserve order explicitly
        batch = sorted(response.data, key=lambda x: x.index)
        vectors.extend(item.embedding for item in batch)
    return vectors


def get_embeddings():
    """
    Compatibility shim — returns an object with embed_query / embed_documents
    methods so qdrant_indexing.py can call get_embeddings().embed_query(...)
    without changes.
    """
    class _Shim:
        embed_query     = staticmethod(embed_query)
        embed_documents = staticmethod(embed_documents)
    return _Shim()

