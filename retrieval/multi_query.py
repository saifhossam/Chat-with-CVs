from openai import AzureOpenAI
from pydantic import BaseModel, Field
from typing import List
import os
from dotenv import load_dotenv
from textblob import TextBlob

load_dotenv()

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://gbgacademy-genai3.openai.azure.com/",
    api_key=AZURE_OPENAI_API_KEY,
)

class MultiQueryResponse(BaseModel):
    queries: List[str] = Field(
        description="List of semantically similar search queries generated from the user question."
    )

def correct_spelling(text):
    return str(TextBlob(text).correct())

def generate_multi_queries(question, model="gpt-4.1-nano", num_queries=2):
    question = correct_spelling(question)
    system_prompt = f"""
    You are an expert in search query generation for a Retrieval-Augmented Generation (RAG) system.

    Your task is to generate exactly {num_queries} alternative search queries based on the user's question.
    These queries will be used to retrieve relevant documents from a vector database.

    Guidelines:
    - Preserve the core intent of the original question.
    - Generate queries using different wording, keywords, or perspectives.
    - Avoid simple grammatical paraphrases; aim for meaningful variation.
    - Include a mix of:
      1- short keyword-style queries
      2- natural language queries
      3- document-oriented queries (e.g., how information might appear in a document or CV).
    - Do NOT answer the question.

    If the user's question does not contain meaningful information related to candidate CVs, return an empty list of queries.

    Return ONLY valid JSON in this format:

    {{
      "queries": ["query1", "query2"]
    }}
    """
    response = client.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        response_format=MultiQueryResponse,
        temperature=0.2
    )

    queries = response.choices[0].message.parsed.queries

    return list(set(queries + [question]))

