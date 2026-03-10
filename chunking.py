from pathlib import Path
from tqdm import tqdm
import re
import json

from langchain_community.document_loaders import PyPDFLoader


# Detect common CV headers
HEADER_PATTERN = re.compile(
    r"^(SUMMARY|PROFESSIONAL SUMMARY|PROFILE|ABOUT ME|OBJECTIVE|CAREER OBJECTIVE|"
    r"EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT HISTORY|WORK HISTORY|"
    r"EDUCATION|ACADEMIC BACKGROUND|EDUCATIONAL BACKGROUND|"
    r"SKILLS|SOFT SKILLS|HARD SKILLS|TECHNICAL SKILLS|CORE SKILLS|KEY SKILLS|SKILL SET|COMPETENCIES|CORE COMPETENCIES|"
    r"PROJECTS|PERSONAL PROJECTS|ACADEMIC PROJECTS|PROJECT EXPERIENCE|"
    r"CERTIFICATIONS|LICENSES|CERTIFICATES|PROFESSIONAL CERTIFICATIONS|"
    r"PUBLICATIONS|RESEARCH|RESEARCH EXPERIENCE|"
    r"ACHIEVEMENTS|KEY ACHIEVEMENTS|ACCOMPLISHMENTS|AWARDS|HONORS|"
    r"LEADERSHIP|LEADERSHIP EXPERIENCE|"
    r"VOLUNTEERING|VOLUNTEER EXPERIENCE|COMMUNITY SERVICE|"
    r"INTERNSHIPS|INTERNSHIP EXPERIENCE|"
    r"LANGUAGES|LANGUAGE SKILLS|"
    r"INTERESTS|HOBBIES|PASSIONS|EXTRACURRICULAR ACTIVITIES|"
    r"PROFESSIONAL AFFILIATIONS|MEMBERSHIPS|"
    r"REFERENCES)$",
    re.IGNORECASE
)


def split_into_sections(text):


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

    return {
        sec: "\n".join(lines).strip()
        for sec, lines in sections.items()
        if "\n".join(lines).strip()
    }

def split_section_entries(section_text):


    entries = []
    buffer = []

    lines = section_text.split("\n")

    for line in lines:

        clean = line.strip()

        if not clean:
            if buffer:
                entries.append("\n".join(buffer).strip())
                buffer = []
            continue

        # bullet point
        if clean.startswith(("•", "-", "*")):

            bullet_text = clean.lstrip("•-* ").strip()

            if bullet_text:
                if buffer:
                    entries.append("\n".join(buffer).strip())
                    buffer = []

                entries.append(bullet_text)

            continue

        # detect job / entry titles (common CV patterns)
        if re.search(r"( at | - | — | \| )", clean) and len(clean) < 120:
            if buffer:
                entries.append("\n".join(buffer).strip())
                buffer = []

            entries.append(clean)
            continue

        buffer.append(clean)

    if buffer:
        entries.append("\n".join(buffer).strip())

    # remove very short noise chunks
    return [e for e in entries if len(e) > 3]


def run_cv_rag_chunking(folder_path):

    all_chunks = []

    pdf_files = list(Path(folder_path).glob("*.pdf"))

    if not pdf_files:
        print("⚠️ No PDFs found in folder.")
        return []

    for pdf_path in tqdm(pdf_files, desc="Processing CVs"):

        print(f"📄 Processing: {pdf_path.name}")

        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        full_text = "\n".join([d.page_content for d in docs])

        sections = split_into_sections(full_text)

        chunk_id = 0

        for section_name, section_text in sections.items():

            entries = split_section_entries(section_text)

            for entry in entries:

                chunk_id += 1

                chunk = {
                    "page_content": entry,
                    "metadata": {
                        "candidate_name": pdf_path.stem,
                        "source_file": pdf_path.name,
                        "section": section_name,
                        "chunk_id": chunk_id,
                        "chunk_type": "cv_entry"
                    }
                }

                all_chunks.append(chunk)

    return all_chunks


def save_chunks_to_json(chunks, output_file="cv_chunks.json"):
    """
    Save all chunks into a formatted JSON file.
    """

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print("\n✅ Chunks saved successfully!")
    print(f"📁 File: {output_file}")
    print(f"🧠 Total Chunks: {len(chunks)}")


if __name__ == "__main__":

    folder = "cv"

    chunks = run_cv_rag_chunking(folder)

    save_chunks_to_json(chunks)
