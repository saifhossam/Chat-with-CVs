from pathlib import Path
from tqdm import tqdm
import re
import json

from langchain_community.document_loaders import PyPDFLoader


# -------------------------------------------------
# SECTION HEADER DETECTION
# -------------------------------------------------

HEADER_PATTERN = re.compile(
    r"^(SUMMARY|PROFESSIONAL SUMMARY|PROFILE|ABOUT ME|OBJECTIVE|CAREER OBJECTIVE|"
    r"EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE|EMPLOYMENT HISTORY|WORK HISTORY|"
    r"EDUCATION|ACADEMIC BACKGROUND|EDUCATIONAL BACKGROUND|"
    r"SKILLS|SOFT SKILLS|HARD SKILLS|TECHNICAL SKILLS|CORE SKILLS|KEY SKILLS|"
    r"PROJECTS|PERSONAL PROJECTS|ACADEMIC PROJECTS|PROJECT EXPERIENCE|"
    r"CERTIFICATIONS|LICENSES|CERTIFICATES|PROFESSIONAL CERTIFICATIONS|"
    r"PUBLICATIONS|RESEARCH|RESEARCH EXPERIENCE|"
    r"ACHIEVEMENTS|ACCOMPLISHMENTS|AWARDS|HONORS|"
    r"LEADERSHIP|LEADERSHIP EXPERIENCE|"
    r"VOLUNTEERING|VOLUNTEER EXPERIENCE|"
    r"INTERNSHIPS|TRAINING|TRAININGS|"
    r"LANGUAGES|LANGUAGE SKILLS|"
    r"INTERESTS|HOBBIES|PASSIONS|"
    r"REFERENCES)$",
    re.IGNORECASE
)


# -------------------------------------------------
# CLEAN PDF TEXT
# -------------------------------------------------

def clean_text(text):

    lines = text.split("\n")
    cleaned = []
    seen_links = set()

    for line in lines:

        line = line.strip()

        if not line:
            continue

        # remove mailto / tel
        line = re.sub(r"mailto:.*", "", line)
        line = re.sub(r"tel:.*", "", line)

        # remove duplicate links
        if "http" in line or "github.com" in line or "linkedin.com" in line:
            if line in seen_links:
                continue
            seen_links.add(line)

        cleaned.append(line)

    return "\n".join(cleaned)


# -------------------------------------------------
# SPLIT INTO SECTIONS
# -------------------------------------------------

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
            sections[current_section].append(clean)

    return {
        sec: "\n".join(lines).strip()
        for sec, lines in sections.items()
        if "\n".join(lines).strip()
    }


# -------------------------------------------------
# ENTRY TITLE DETECTION
# -------------------------------------------------

def is_entry_title(line):

    if len(line) > 120:
        return False

    if re.search(r"\|", line):
        return True

    if re.search(r"( internship| engineer| developer| analyst| assistant)", line.lower()):
        return True

    if re.search(r"( project)", line.lower()):
        return True

    if re.search(r"( at )", line.lower()):
        return True

    return False


# -------------------------------------------------
# DATE DETECTION (to avoid splitting entries)
# -------------------------------------------------

def is_date_or_location(line):

    if re.search(r"\d{4}", line):
        return True

    if re.search(r"(present)", line.lower()):
        return True

    if re.search(r"(cairo|egypt|city|village)", line.lower()):
        return True

    return False


# -------------------------------------------------
# SPLIT SECTIONS INTO ENTRIES
# -------------------------------------------------

def split_section_entries(section_text, section_name):

    lines = [l.strip() for l in section_text.split("\n") if l.strip()]

    section_name = section_name.lower()

    # Sections that should stay ONE chunk
    single_chunk_sections = [
        "skills",
        "languages",
        "interests",
        "hobbies",
        "profile",
        "summary"
    ]

    if section_name in single_chunk_sections:
        return ["\n".join(lines)]

    entries = []
    buffer = []

    for line in lines:

        # bullet points
        if line.startswith(("•", "-", "*")):
            buffer.append(line)
            continue

        # date / location should stay with entry
        if is_date_or_location(line):
            buffer.append(line)
            continue

        # new entry title
        if is_entry_title(line):

            if buffer:
                entries.append("\n".join(buffer))
                buffer = []

            buffer.append(line)
            continue

        buffer.append(line)

    if buffer:
        entries.append("\n".join(buffer))

    return entries


# -------------------------------------------------
# MAIN PIPELINE
# -------------------------------------------------

def run_cv_rag_chunking(folder_path):

    all_chunks = []

    pdf_files = list(Path(folder_path).glob("*.pdf"))

    if not pdf_files:
        print("⚠️ No PDFs found.")
        return []

    for pdf_path in tqdm(pdf_files, desc="Processing CVs"):

        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        full_text = "\n".join([d.page_content for d in docs])

        full_text = clean_text(full_text)

        sections = split_into_sections(full_text)

        chunk_id = 0

        for section_name, section_text in sections.items():

            entries = split_section_entries(section_text, section_name)

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


# -------------------------------------------------
# SAVE JSON
# -------------------------------------------------

def save_chunks_to_json(chunks, output_file="cv_chunks.json"):

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=4, ensure_ascii=False)

    print("\n✅ Chunks saved")
    print(f"📁 File: {output_file}")
    print(f"🧠 Total chunks: {len(chunks)}")


# -------------------------------------------------
# RUN
# -------------------------------------------------

if __name__ == "__main__":

    folder = "cv"

    chunks = run_cv_rag_chunking(folder)

    save_chunks_to_json(chunks)
