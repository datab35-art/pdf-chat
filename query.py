"""Question -> retrieve -> Claude answers with citations."""

import sys

import chromadb
from anthropic import Anthropic
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv

from ingest import CHROMA_DIR, COLLECTION_NAME, LOCAL_MODEL_DIR, MODEL_NAME

load_dotenv()

N_RESULTS = 5
CLAUDE_MODEL = "claude-sonnet-5"

SYSTEM_PROMPT = (
    "Answer the question using only the provided context. "
    "Cite the page number(s) you used for each part of your answer. "
    "If the context does not contain the answer, say so instead of guessing."
)


def get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    model_path = str(LOCAL_MODEL_DIR) if LOCAL_MODEL_DIR.exists() else MODEL_NAME
    embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_path)
    return client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_function)


def build_context(documents: list[str], metadatas: list[dict]) -> str:
    parts = []
    for doc, meta in zip(documents, metadatas):
        parts.append(f"[{meta['file_name']}, page {meta['page']}]\n{doc}")
    return "\n\n".join(parts)


def answer_question(question: str) -> str:
    collection = get_collection()
    results = collection.query(query_texts=[question], n_results=N_RESULTS)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = build_context(documents, metadatas)

    client = Anthropic()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            }
        ],
    )
    return response.content[0].text


def main():
    question = " ".join(sys.argv[1:]) or input("Question: ")
    print(answer_question(question))


if __name__ == "__main__":
    main()
