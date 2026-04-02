# 📄 CV Intelligence Platform

> A production-ready Retrieval-Augmented Generation (RAG) pipeline for intelligent CV analysis — upload candidate PDFs, ask questions in natural language, and get grounded, recruiter-friendly answers powered by Azure OpenAI and Qdrant.

---

## ✨ Features

- **PDF Ingestion** — Upload one or more CV PDFs with automatic deduplication via SHA-256 hashing; re-uploading the same file is a no-op.
- **Hybrid Retrieval** — Combines dense vector search (Qdrant) with sparse BM25 keyword search for high-recall, high-precision retrieval.
- **Multi-Query Expansion** — Automatically generates alternative phrasings of your question to improve retrieval coverage.
- **Cross-Encoder Reranking** — Retrieved chunks are reranked for relevance before being passed to the LLM.
- **Candidate Filtering** — Narrow your question to a specific candidate or query across all indexed CVs at once.
- **Grounded Answers** — The LLM answers strictly from retrieved context, citing candidate name and CV section for every claim.
- **Streamlit UI** — Clean, interactive chat interface with expandable source chunk citations.

---

## 🗂️ Project Structure

```
.
├── app.py                  # Streamlit frontend — upload, chat, filter
├── config.py               # Azure OpenAI + Qdrant client initialisation
├── embedding.py            # Text → vector via Azure text-embedding-3-small
├── generator.py            # LLM answer generation (Azure GPT-4.1-nano)
├── prompts.py              # System & user prompt templates
├── requirements.txt        # Python dependencies
├── .env                    # Runtime secrets (not committed — see env_template.txt)
├── env_template.txt        # .env template
│
├── ingestion/
│   └── ingest.py           # PDF parsing, chunking, and Qdrant upsert
│
├── rag/
│   ├── retrieval.py        # BM25 index builder + chunk loader from Qdrant
│   └── pipeline.py         # Full hybrid retrieve → rerank pipeline
│
└── db/
    └── qdrant_client.py    # Qdrant helpers (collection setup, dedup check)
```

---

## ⚙️ Architecture

```
PDF Upload
    │
    ▼
[Ingestion]  pdfplumber → text chunks → embeddings → Qdrant upsert
    │
    ▼
[Query]  User question
    │
    ├─► [Query Expansion]      multi-query generation via LLM
    │
    ├─► [Dense Retrieval]      Azure embedding → Qdrant ANN search
    │
    ├─► [Sparse Retrieval]     BM25 keyword search on in-memory index
    │
    ├─► [Hybrid Fusion]        RRF score combination
    │
    ├─► [Cross-Encoder Rerank] sentence-transformers reranker
    │
    └─► [Generation]           Azure GPT-4.1-nano → grounded answer
```

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/cv-intelligence.git
cd cv-intelligence
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the template and fill in your credentials:

```bash
cp env_template.txt .env
```

```dotenv
# Azure OpenAI
AZURE_API_KEY=your_azure_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4.1-nano

# Qdrant Cloud
QDRANT_URL=https://your-cluster-id.region.aws.cloud.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_COLLECTION=cv_chunks
```

> ⚠️ Never commit `.env` to version control. Add it to `.gitignore`.

### 4. Run the Application

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

---

## 🖥️ Usage

1. **Upload CVs** — Use the sidebar to drag-and-drop or browse for PDF files. Already-indexed CVs are skipped automatically.
2. **Ask Questions** — Type any natural language question in the chat box, for example:
   - *"Who has the most Python experience?"*
   - *"Which candidates have worked at a FAANG company?"*
   - *"Summarise John Doe's educational background."*
3. **Filter by Candidate** — Use the dropdown above the chat to scope your question to a single candidate.
4. **View Sources** — Expand the *Source Chunks* panel under any answer to see the exact CV excerpts used.
5. **Refresh Index** — Click *Refresh Index* in the sidebar to sync the BM25 index after external changes to Qdrant.

---

## 🔧 Configuration Reference

All runtime configuration lives in `config.py` and is driven by environment variables.

| Variable | Description | Default |
|---|---|---|
| `AZURE_API_KEY` | Azure OpenAI API key | — |
| `QDRANT_URL` | Qdrant cluster URL | — |
| `QDRANT_API_KEY` | Qdrant API key | — |
| `QDRANT_COLLECTION` | Collection name | `cv_chunks` |

| Constant | Value | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Azure embedding deployment |
| `LLM` | `gpt-4.1-nano` | Azure chat completion deployment |
| `EMBED_DIM` | `1536` | Embedding vector dimension |

---

## 🤖 Model Stack

| Component | Model | Provider |
|---|---|---|
| Embeddings | `text-embedding-3-small` | Azure OpenAI |
| Generation | `gpt-4.1-nano` | Azure OpenAI |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace |
| Vector Store | Qdrant Cloud | Qdrant |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI |
| `openai` | Azure OpenAI client |
| `qdrant-client` | Vector store |
| `pdfplumber` | PDF text extraction |
| `rank-bm25` | Sparse keyword retrieval |
| `sentence-transformers` | Cross-encoder reranking |
| `faiss-cpu` | Local vector operations |
| `langchain` | Document chunking utilities |
| `python-dotenv` | Environment variable loading |

---

## 🔐 Security Notes

- Store all API keys in `.env` — never hardcode them in source files.
- Use `python-dotenv` locally; use proper secrets management (e.g. Azure Key Vault, GitHub Secrets) in production.
- The Qdrant collection stores CV text chunks — ensure your Qdrant cluster has appropriate access controls enabled.

---

## 🗺️ Roadmap

- [ ] Support for `.docx` and `.txt` CV formats
- [ ] Batch candidate comparison reports (PDF export)
- [ ] Role-based filtering (e.g. "find candidates for a senior backend role")
- [ ] Conversation memory across sessions
- [ ] Authentication layer for multi-user deployments

---

## 🤝 Team Members
- Saif Hossam
- Ahmed Essam
- Fatma Badr
- Merna Hany
- Rawan Nagy
