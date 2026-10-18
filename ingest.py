"""PDF -> chunks -> embeddings -> Chroma."""

from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

DATA_DIR = Path(__file__).parent / "data"
CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "pdf_chunks"

MODEL_NAME = "all-MiniLM-L6-v2"
# If the model can't be downloaded from the Hugging Face Hub (e.g. blocked
# network), download it manually and place it here instead.
LOCAL_MODEL_DIR = Path(__file__).parent / MODEL_NAME

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def load_pdf_chunks(pdf_path: Path):
    reader = PdfReader(str(pdf_path))
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        for chunk in chunk_text(text):
            if chunk.strip():
                yield chunk, page_number


def main():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    model_path = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_NAME
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_path)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
    )

    documents = []
    metadatas = []
    ids = []

    pdf_paths = sorted(DATA_DIR.glob("*.pdf"))
    for pdf_path in pdf_paths:
        for i, (chunk, page_number) in enumerate(load_pdf_chunks(pdf_path)):
            documents.append(chunk)
            metadatas.append({"file_name": pdf_path.name, "page": page_number})
            ids.append(f"{pdf_path.name}:{page_number}:{i}")

    if not documents:
        print(f"No PDF chunks found in {DATA_DIR}")
        return

    collection.add(documents=documents, metadatas=metadatas, ids=ids)
    print(f"Ingested {len(documents)} chunks from {len(pdf_paths)} PDF(s) into '{COLLECTION_NAME}'")


if __name__ == "__main__":
    main()
