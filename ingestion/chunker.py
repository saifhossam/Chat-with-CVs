from __future__ import annotations

import os
from typing import Dict, List, Optional

from langchain_core.documents import Document

from sentence_transformers import SentenceTransformer
from sentence_transformers import util as st_util

from docling.document_converter import DocumentConverter

from config import azure_client, LLM, KNOWN_SECTIONS, NER_MODEL, HEADERS_EMBEDDING_MODEL



_classifier_model      = None
_classifier_embeddings = None


def _get_classifier():
    """Load bge-large-en-v1.5 and precompute KNOWN_SECTIONS embeddings once."""
    global _classifier_model, _classifier_embeddings
    if _classifier_model is None:
        _classifier_model = SentenceTransformer(HEADERS_EMBEDDING_MODEL)
        _classifier_embeddings = _classifier_model.encode(
            KNOWN_SECTIONS,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
    return _classifier_model, _classifier_embeddings


def classify_section(heading: str, threshold: float = 0.80) -> Optional[str]:
    """
    Map a raw CV heading to a recognised section label via cosine similarity.

    Returns the original heading in Title Case if score >= threshold,
    or None if the heading is not a recognised CV section.
    """

    heading_upper = heading.upper().strip()
    if len(heading_upper.split()) > 6:
        return None

    model, section_embeddings = _get_classifier()
    embedding = model.encode(
        heading_upper, convert_to_tensor=True, normalize_embeddings=True
    )
    scores = st_util.cos_sim(embedding, section_embeddings)[0]

    if scores.max().item() >= threshold:
        return heading.strip().title()  # preserve the actual CV wording

    return None


# =============================================================================
# HEADER DETECTION via Docling
# =============================================================================

def detect_headers(pdf_path: str) -> Dict[str, str]:
    """
    Identify CV section headers using Docling's structural analysis.

    Docling parses the PDF layout and labels elements as section_header,
    list_item, text, etc. We take only level-1 section_header elements
    and validate them with the semantic classifier.

    Args:
        pdf_path: Absolute path to a CV PDF file.

    Returns:
        header_lookup: {UPPERCASE_TEXT: "Title Case Label"}
        e.g. {"EDUCATION": "Education", "TRAINING": "Training"}
    """

    converter = DocumentConverter()
    result    = converter.convert(pdf_path)
    texts     = result.document.model_dump(mode="python").get("texts", [])

    header_lookup: Dict[str, str] = {}
    for t in texts:
        label = t.get("label", "")
        level = t.get("level")
        text  = t.get("text", "").strip()
        if not text:
            continue
        try:
            level_int = level.value if hasattr(level, "value") else int(level)
        except (TypeError, ValueError):
            level_int = None

        if label == "section_header" and level_int == 1:
            section_label = classify_section(text)
            if section_label is not None:
                header_lookup[text.upper()] = section_label

    return header_lookup


# =============================================================================
# CANDIDATE NAME EXTRACTION via LLM
# =============================================================================

# def extract_candidate_name(lines: List[str]) -> str:
#     """
#     Extract the candidate's full name from the first 50 lines of a CV
#     using the Azure OpenAI LLM defined in config.py.

#     Falls back to 'Unknown Candidate' if the LLM cannot identify a name.
#     """
#     header_text = "\n".join(lines[:50])

#     try:
#         response = azure_client.chat.completions.create(
#             model=LLM,
#             messages=[
#                 {
#                     "role": "system",
#                     "content": (
#                         "You are a CV parser. Extract the full name of the candidate "
#                         "from the top of the CV text provided. "
#                         "Return ONLY the full name — no explanation, no punctuation, "
#                         "no extra words. If you cannot find a name, return exactly: "
#                         "Unknown Candidate"
#                     ),
#                 },
#                 {
#                     "role": "user",
#                     "content": header_text,
#                 },
#             ],
#             temperature=0,
#             max_tokens=20,
#         )
#         name = response.choices[0].message.content.strip()
#         return name if name else "Unknown Candidate"

#     except Exception:
#         return "Unknown Candidate"


# =============================================================================
# CANDIDATE NAME EXTRACTION via NER
# =============================================================================

_ner_pipeline = None


def _get_ner():
    global _ner_pipeline
    if _ner_pipeline is None:
        _ner_pipeline = hf_pipeline(
            "ner",
            model=NER_MODEL,
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def extract_candidate_name(lines: List[str]) -> str:
    """
    Extract the candidate's full name from the first lines of a CV via NER.

    Runs on the first 30 lines (name is always near the top). Requires
    at least 2 tokens (first + last name). Falls back to 'Unknown Candidate'.
    """
    ner    = _get_ner()
    # Strip emails, URLs, phones so NER focuses on the name
    header = " ".join(lines[:30])
    header = re.sub(r"\S+@\S+", " ", header)
    header = re.sub(r"http\S+|www\S+|\S*\.com\S+", " ", header)
    header = re.sub(r"\+?\d[\d\s\-]{7,}", " ", header)

    entities   = ner(header[:500], aggregation_strategy="simple")
    candidates = []
    for e in entities:
        if e["entity_group"] != "PER":
            continue
        name = e["word"].strip()
        if len(name.split()) < 2:
            continue
        score = e.get("score", 0)
        if name.islower():
            score -= 0.2
        score += max(0, (500 - e.get("start", 500)) / 500)
        candidates.append((score, name))

    if not candidates:
        return "Unknown Candidate"

    candidates.sort(reverse=True)
    return candidates[0][1]


# =============================================================================
# MAIN CHUNKING FUNCTION
# =============================================================================

def chunk_cv(
    pdf_path: str,
    lines: List[str],
    candidate_name: str,
    original_filename: str = "",
) -> List[Document]:
    """
    Split a CV's extracted text lines into one Document per section.

    Args:
        pdf_path:          Source PDF path (temp path used for Docling).
        lines:             Text lines from ingestion/loader.load_pdf().
        candidate_name:    From extract_candidate_name().
        original_filename: The original uploaded filename to store in metadata.

    Returns:
        List of Documents — one per section — with metadata:
            candidate_name, section, source_cv.
    """
    header_lookup = detect_headers(pdf_path)
    filename      = original_filename or os.path.basename(pdf_path)

    sections: Dict[str, List[str]] = {}
    current_section = "Header & Personal Information"
    sections[current_section] = []

    for line in lines:
        matched = header_lookup.get(line.upper())
        if matched:
            current_section = matched
            if current_section not in sections:
                sections[current_section] = []
            continue

        if line.isupper() and 1 <= len(line.split()) <= 5:
            fallback = classify_section(line)
            if fallback is not None:
                current_section = fallback
                if current_section not in sections:
                    sections[current_section] = []
                continue

        if current_section not in sections:
            sections[current_section] = []
        sections[current_section].append(line)

    chunks: List[Document] = []
    for section_label, section_lines in sections.items():
        content = "\n".join(section_lines).strip()
        if not content:
            continue
        chunks.append(Document(
            page_content=(
                f"Candidate: {candidate_name} | Section: {section_label}\n\n"
                f"{content}"
            ),
            metadata={
                "candidate_name": candidate_name,
                "section":        section_label,
                "source_cv":      filename,
            },
        ))

    return chunks
