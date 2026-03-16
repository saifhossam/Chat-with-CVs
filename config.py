import os
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv
load_dotenv()

COLLECTION = "cv_chunks"
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

EMBED_DIM = 1536
LLM = "gpt-4.1-nano"
EMBEDDING_MODEL = "text-embedding-3-small"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

azure_client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://gbgacademy-genai3.openai.azure.com/",
    api_key=AZURE_OPENAI_API_KEY,
)

qdrant_client = QdrantClient(
    url = QDRANT_URL,
    api_key = QDRANT_API_KEY
)
