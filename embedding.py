from typing import List
from config import azure_client, EMBEDDING_MODEL


def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Converts a list of strings into a list of 1536-dimensional vectors."""
    if not texts:
        return []
    response = azure_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]
