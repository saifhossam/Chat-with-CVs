from pydantic import BaseModel, Field
from typing import List
from textblob import TextBlob
from prompts import query_generator_prompt
from config import LLM, azure_client

class MultiQueryResponse(BaseModel):
    queries: List[str] = Field(
        description="List of semantically similar search queries generated from the user question."
    )

def correct_spelling(text):
    return str(TextBlob(text).correct())


def generate_multi_queries(question, model = LLM, num_queries=2):

    system_prompt = query_generator_prompt.format(num_queries=num_queries)

    question = correct_spelling(question)
    response = azure_client.chat.completions.parse(
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

