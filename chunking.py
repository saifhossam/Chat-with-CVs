from pathlib import Path
from tqdm import tqdm
import re

from langchain_community.document_loaders import PyPDFLoader

# Header detection pattern (capitalized lines, typical CV headers)
HEADER_PATTERN = re.compile(r"^[A-Z][A-Za-z0-9 &\-]+$")

# Maximum characters per semantic chunk (adjust for embeddings / LLM context)
MAX_CHUNK_SIZE = 500  # e.g., ~500 characters per chunk
CHUNK_OVERLAP = 50    # overlap between chunks to preserve context


def split_into_sections(text):
    """
    Header-aware section splitting.
    Returns a dict: {section_name: text}
    """
    sections = {}
    current_section = "general"
    sections[current_section] = []

    for line in text.split("\n"):
        clean = line.strip()
        if HEADER_PATTERN.match(clean):
            current_section = clean.lower()
            sections[current_section] = []
        else:
            sections[current_section].append(line)

    # Join lines
    return {sec: "\n".join(lines).strip() for sec, lines in sections.items() if "\n".join(lines).strip()}


def semantic_chunk_text(text, max_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """
    Split text into semantic chunks for embeddings / RAG.
    Splits by paragraphs first, then by size if too long.
    """
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) + 1 <= max_size:
            current_chunk += (p + "\n")
        else:
            # Save current chunk
            chunks.append(current_chunk.strip())
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
            current_chunk = overlap_text + p + "\n"

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def run_cv_rag_chunking(folder_path):
    """
    Processes all PDFs in folder_path into header-aware + semantic chunks.
    Returns list of dicts with page_content and metadata.
    """
    all_chunks = []
    pdf_files = list(Path(folder_path).glob("*.pdf"))

    if not pdf_files:
        print("⚠️ No PDFs found.")
        return []

    for pdf_path in tqdm(pdf_files, desc="Processing CVs"):
        print(f"\n📄 Processing {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        # Merge all pages into one text
        full_text = "\n".join([d.page_content for d in docs])

        # Header-aware section splitting
        sections = split_into_sections(full_text)

        for section_name, section_text in sections.items():
            # Semantic chunking for each section
            section_chunks = semantic_chunk_text(section_text)
            for chunk_text in section_chunks:
                all_chunks.append({
                    "page_content": chunk_text,
                    "metadata": {
                        "candidate_name": pdf_path.stem,
                        "section": section_name,
                        "source_file": pdf_path.name
                    }
                })

    return all_chunks


def inspect_chunks(chunks, preview_chars=300):
    print("\n" + "="*50)
    print(f"Total Chunks: {len(chunks)}")
    print("="*50)

    for i, chunk in enumerate(chunks[:5]):
        print(f"\n--- CHUNK {i+1} ---")
        print("Metadata:", chunk["metadata"])
        print("Content Preview:\n", chunk["page_content"][:preview_chars])
        print("-"*30)


# Example usage
if __name__ == "__main__":
    folder = "cv" 
    chunks = run_cv_rag_chunking(folder)
    inspect_chunks(chunks)