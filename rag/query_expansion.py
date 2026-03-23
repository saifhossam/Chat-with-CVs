from pydantic import BaseModel, Field
from typing import List
from prompts import query_generator_prompt
from config import LLM, azure_client

class MultiQueryResponse(BaseModel):
    queries: List[str] = Field(
        description="List of semantically similar search queries generated from the user question."
    )

def generate_multi_queries(question: str,*,num_queries: int = 2,model: str = LLM,):

    system_prompt = query_generator_prompt.format(num_queries=num_queries)

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

