from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss


# Paths
PROCESSED_DIR = Path("data/processed")
RAW_FILE = PROCESSED_DIR / "papers_raw.jsonl"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
INDEX_FILE = PROCESSED_DIR / "faiss_index.bin"


def load_raw_records():
    """Load page-level records from papers_raw.jsonl."""
    if not RAW_FILE.exists():
        raise FileNotFoundError(f"{RAW_FILE} does not exist. Run ingest_papers.py first.")

    records = []
    with RAW_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def chunk_text(text: str, max_chars: int = 800, min_chars: int = 200):
    """
    Naive character-based chunking:
    - Splits text into windows of up to max_chars.
    - Skips chunks that are too short (to avoid tiny noisy chunks).
    """
    chunks = []
    text = text.strip().replace("\n", " ")
    n = len(text)

    for start in range(0, n, max_chars):
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if len(chunk) < min_chars:
            continue
        chunks.append(chunk)

    return chunks


def build_chunks(records):
    """
    From page-level records, build a list of chunk dicts:
    {
      "chunk_id": int,
      "paper_id": str,
      "page_num": int,
      "text": str
    }
    """
    chunks = []
    chunk_id = 0

    for rec in records:
        paper_id = rec["paper_id"]
        page_num = rec["page_num"]
        text = rec["text"]

        page_chunks = chunk_text(text)
        for ch in page_chunks:
            chunks.append({
                "chunk_id": chunk_id,
                "paper_id": paper_id,
                "page_num": page_num,
                "text": ch
            })
            chunk_id += 1

    return chunks


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw records from {RAW_FILE} ...")
    records = load_raw_records()
    print(f"Loaded {len(records)} page-level records.")

    print("Building chunks ...")
    chunks = build_chunks(records)
    print(f"Created {len(chunks)} chunks.")

    if not chunks:
        print("No chunks created. Check your input PDFs or chunking settings.")
        return

    # Save chunks metadata as JSONL
    print(f"Saving chunks to {CHUNKS_FILE} ...")
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Load SentenceTransformer model
    print("Loading embedding model (all-MiniLM-L6-v2) ...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Compute embeddings
    texts = [c["text"] for c in chunks]
    print("Computing embeddings (this may take a bit) ...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    print(f"Embeddings shape: {embeddings.shape}")
    np.save(EMBEDDINGS_FILE, embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dim} ...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(INDEX_FILE))

    print(f"\n✅ Done!")
    print(f"  - Saved chunks:      {CHUNKS_FILE}")
    print(f"  - Saved embeddings:  {EMBEDDINGS_FILE}")
    print(f"  - Saved FAISS index: {INDEX_FILE}")


if __name__ == "__main__":
    main()
