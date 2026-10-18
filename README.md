# PDF Chat — RAG with Local Embeddings & Claude

A Retrieval-Augmented Generation (RAG) pipeline that ingests PDF documents, stores them as vector embeddings in ChromaDB, and answers questions using Claude with page-level citations.

## How It Works

```
PDF files  →  pypdf extracts text  →  chunk (800 chars, 100 overlap)
    →  all-MiniLM-L6-v2 embeds locally  →  stored in ChromaDB

Question  →  embed  →  retrieve top 5 chunks  →  Claude answers with citations
```

### Pipeline

1. **Ingest** (`ingest.py`) — Reads every PDF in `data/`, splits text into overlapping chunks of ~800 characters, embeds each chunk using the all-MiniLM-L6-v2 sentence-transformer model running locally, and stores the embeddings in a persistent ChromaDB collection along with file name and page number metadata.

2. **Query** (`query.py`) — Takes a natural-language question, embeds it with the same model, retrieves the 5 most similar chunks from ChromaDB, and sends them as context to Claude (Sonnet) with an instruction to answer only from the provided context and cite page numbers.

## Example

```
$ python query.py "What is the transformer architecture?"

The Transformer is a model architecture that eschews recurrence and instead
relies entirely on an attention mechanism to draw global dependencies between
input and output. It uses stacked self-attention and point-wise, fully connected
layers for both the encoder and decoder. (page 2, page 3)
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file with your Anthropic API key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Place PDF files in the `data/` directory, then:

```bash
python ingest.py       # Ingest PDFs into ChromaDB
python query.py "your question here"
```

## Project Structure

```
├── ingest.py           # PDF → chunks → embeddings → ChromaDB
├── query.py            # Question → retrieve → Claude answers
├── requirements.txt
├── data/               # Place PDF files here
│   └── *.pdf
├── chroma_db/          # Persistent vector store (gitignored)
└── all-MiniLM-L6-v2/  # Local embedding model (gitignored)
```

## Tech Stack

- **pypdf** — PDF text extraction
- **sentence-transformers** (all-MiniLM-L6-v2) — Local embeddings, no API needed
- **ChromaDB** — Persistent vector database
- **Anthropic Claude API** — LLM for question answering with citations
- **python-dotenv** — Environment variable management

## Key Design Decisions

- **Local embeddings** — all-MiniLM-L6-v2 runs on CPU with no external API calls, keeping embedding costs at zero and avoiding network dependencies
- **Overlapping chunks** — 100-character overlap prevents context loss at chunk boundaries
- **Page-level metadata** — Every chunk carries its source file and page number, enabling precise citations
- **Context-only answering** — Claude's system prompt constrains it to answer strictly from retrieved context, reducing hallucination
