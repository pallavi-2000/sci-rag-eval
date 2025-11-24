import argparse
import json
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
INDEX_FILE = PROCESSED_DIR / "faiss_index.bin"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_chunks(chunks_file: Path = CHUNKS_FILE):
    if not chunks_file.exists():
        raise FileNotFoundError(f"{chunks_file} not found. Run build_index.py first.")

    chunks = []
    with chunks_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def load_index(index_file: Path = INDEX_FILE):
    if not index_file.exists():
        raise FileNotFoundError(f"{index_file} not found. Run build_index.py first.")
    index = faiss.read_index(str(index_file))
    return index


def load_embeddings(emb_file: Path = EMBEDDINGS_FILE):
    if not emb_file.exists():
        raise FileNotFoundError(f"{emb_file} not found. Run build_index.py first.")
    return np.load(emb_file)


def get_embedding_model(model_name: str = EMBED_MODEL_NAME):
    model = SentenceTransformer(model_name)
    return model


def retrieve_chunks(
    question: str,
    top_k: int = 5,
    paper_id: str | None = None,
) -> List[Tuple[float, dict]]:
    """
    Retrieve top_k chunks for a question.
    If paper_id is given, will re-rank and keep only chunks from that paper.
    """
    print(f"\n[QUERY] {question}")
    if paper_id:
        print(f"[FILTER] Prefer paper_id = {paper_id}")

    chunks = load_chunks()
    index = load_index()
    model = get_embedding_model()

    q_emb = model.encode([question], convert_to_numpy=True).astype("float32")

    # Get more neighbours then optionally filter
    search_k = top_k * 10
    distances, indices = index.search(q_emb, search_k)
    distances = distances[0]
    indices = indices[0]

    results = []
    for dist, idx in zip(distances, indices):
        if idx == -1:
            continue
        chunk = chunks[int(idx)]

        if paper_id is not None and chunk.get("paper_id") != paper_id:
            continue

        results.append((float(dist), chunk))
        if len(results) >= top_k:
            break

    return results


def pretty_print_results(results):
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
    parser = argparse.ArgumentParser(description="Simple retrieval over PDF chunks.")
    parser.add_argument(
        "question",
        type=str,
        nargs="*",
        help="Question to ask (if empty, you'll be prompted).",
    )
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks.")
    parser.add_argument("--paper_id", type=str, default=None, help="Optional paper filter.")
    args = parser.parse_args()

    if args.question:
        question = " ".join(args.question)
    else:
        question = input("Enter your question: ").strip()

    if not question:
        print("No question provided.")
        return

    results = retrieve_chunks(question, top_k=args.top_k, paper_id=args.paper_id)
    pretty_print_results(results)


if __name__ == "__main__":
    main()
