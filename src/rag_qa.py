import argparse
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


# Paths & config
PROCESSED_DIR = Path("data/processed")
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
INDEX_FILE = PROCESSED_DIR / "faiss_index.bin"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(chunks_file: Path = CHUNKS_FILE):
    """
    Load chunk metadata from chunks.jsonl into a list of dicts.
    Each dict has: chunk_id, paper_id, page_num, text.
    """
    if not chunks_file.exists():
        raise FileNotFoundError(f"{chunks_file} not found. Run build_index.py first.")

    chunks = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def load_index(index_file: Path = INDEX_FILE):
    """Load FAISS index from disk."""
    if not index_file.exists():
        raise FileNotFoundError(f"{index_file} not found. Run build_index.py first.")
    index = faiss.read_index(str(index_file))
    return index


def get_embedding_model(model_name: str = EMBED_MODEL_NAME):
    """Load the same SentenceTransformer model used for indexing."""
    model = SentenceTransformer(model_name)
    return model


def retrieve_chunks(question: str, top_k: int = 5, paper_id: str | None = None):
    """
    Given a question string, return top_k most similar chunks.

    If paper_id is provided, we will prefer chunks from that paper only:
    - ask FAISS for more neighbours (top_k * 10)
    - then filter down to those whose chunk["paper_id"] == paper_id

    Returns a list of (distance, chunk_dict), sorted by increasing distance.
    Lower distance = closer / more similar.
    """
    print(f"\n[QUERY] {question}")
    if paper_id:
        print(f"[FILTER] Restricting to paper_id = {paper_id}")

    # Load resources
    chunks = load_chunks()
    index = load_index()
    model = get_embedding_model()

    # Encode the question
    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")

    # If we plan to filter by paper_id, ask for more neighbours
    search_k = top_k * 10 if paper_id is not None else top_k

    distances, indices = index.search(q_emb, search_k)
    distances = distances[0]
    indices = indices[0]

    results = []
    for dist, idx in zip(distances, indices):
        if idx == -1:
            continue

        chunk = chunks[int(idx)]

        # Apply paper filter if requested
        if paper_id is not None and chunk.get("paper_id") != paper_id:
            continue

        results.append((float(dist), chunk))

        if len(results) >= top_k:
            break

    return results



def pretty_print_results(results):
    """Print retrieved chunks in a readable way."""
    if not results:
        print("No results found.")
        return

    print("Top retrieved chunks:\n")
    for rank, (dist, chunk) in enumerate(results, start=1):
        paper_id = chunk["paper_id"]
        page_num = chunk["page_num"]
        text = chunk["text"]

        preview = text[:300].replace("\n", " ")
        if len(text) > 300:
            preview += "..."

        print(f"#{rank}")
        print(f"  Paper: {paper_id} | Page: {page_num}")
        print(f"  Distance: {dist:.4f}")
        print(f"  Text: {preview}")
        print("-" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Simple retrieval over PDF chunks."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="*",
        help="Question to ask about the documents (if empty, you will be prompted).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve.",
    )
    args = parser.parse_args()

    if args.question:
        question = " ".join(args.question)
    else:
        question = input("Enter your question: ").strip()

    if not question:
        print("No question provided.")
        return

    results = retrieve_chunks(question, top_k=args.top_k)
    pretty_print_results(results)


if __name__ == "__main__":
    main()
