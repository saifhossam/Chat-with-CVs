"""
generator.py
────────────
Generates a natural-language answer from retrieved CV chunks using the
Azure OpenAI LLM defined in config.py.

Usage
-----
    from generator import generate_answer

    answer = generate_answer(
        question="Who has experience with Python?",
        context_chunks=pipeline_results,   # list[dict] from retrieve_pipeline()
        candidate_name="John Doe",          # optional filter label shown in prompt
    )
"""

from __future__ import annotations

from typing import List, Optional

from config import azure_client, LLM
from prompts import generator_system_prompt, generator_user_template


def _build_context(context_chunks: List[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    parts = []
    for item in context_chunks:
        chunk = item.get("chunk", item)          # support both raw chunk dicts and pipeline dicts
        name    = chunk.get("candidate_name", "Unknown")
        section = chunk.get("section", "")
        text    = chunk.get("text", "").strip()

        header = f"[Candidate: {name}  |  Section: {section}]"
        parts.append(f"{header}\n{text}")

    return "\n\n---\n\n".join(parts)


def generate_answer(
    question: str,
    context_chunks: List[dict],
    candidate_name: Optional[str] = None,
) -> str:
    """
    Generate an answer for *question* grounded in *context_chunks*.

    Args:
        question:        The recruiter's free-text question.
        context_chunks:  List of result dicts from retrieve_pipeline()
                         (each has a nested "chunk" dict) or raw chunk dicts.
        candidate_name:  Optional — used to personalise the prompt when the
                         user is asking about a specific candidate.

    Returns:
        The LLM's answer as a plain string.
    """
    if not context_chunks:
        return (
            "I couldn't find any relevant information in the indexed CVs "
            "to answer your question. Please make sure the CVs have been "
            "uploaded and indexed."
        )

    context = _build_context(context_chunks)

    scope_note = (
        f" Focus your answer on **{candidate_name}**." if candidate_name else ""
    )

    user_message = generator_user_template.format(
        context=context,
        question=question + scope_note,
    )

    response = azure_client.chat.completions.create(
        model=LLM,
        messages=[
            {"role": "system", "content": generator_system_prompt},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()