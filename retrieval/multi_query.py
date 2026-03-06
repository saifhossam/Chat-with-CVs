# multi query generation
from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = AzureOpenAI(
    api_version="2024-12-01-preview",
    azure_endpoint="https://gbgacademy-genai3.openai.azure.com/",
    api_key=OPENAI_API_KEY,
)

def generate_multi_queries(question, model="gpt-4.1-mini", num_queries=2):

    system_prompt = f"""
    Generate {num_queries} search queries that could help find relevant information
    from candidate CV documents.

    Return each query on a new line.
    """

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.2
    )

    queries = response.choices[0].message.content.split("\n")
    queries = [q.strip("- ").strip() for q in queries if q.strip()]

    return list(set(queries + [question]))