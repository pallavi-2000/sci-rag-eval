from pathlib import Path
import json
import re

from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_FILE = PROCESSED_DIR / "papers_raw.jsonl"
CHUNKS_FILE = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_FILE = PROCESSED_DIR / "embeddings.npy"
INDEX_FILE = PROCESSED_DIR / "faiss_index.bin"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# --------------------
# Load page-level data
# --------------------
def load_raw_records():
    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"{RAW_FILE} does not exist. Run ingest_papers.py first."
        )

    records = []
    with RAW_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------
# Sentence splitting
# --------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[\.!?])\s+")


def sentence_split(text: str):
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if not text:
        return []
    return _SENT_SPLIT_RE.split(text)


def make_chunks_from_page(
    text: str,
    max_chars: int = 900,
    min_chars: int = 200,
    overlap_chars: int = 150,
):
    """
    Sentence-based chunking with overlapping windows.
    """
    sentences = sentence_split(text)
    if not sentences:
        return []

    chunks = []
    current = ""

    for sent in sentences:
        if not sent:
            continue

        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if len(current) >= min_chars:
                chunks.append(current)

            if overlap_chars > 0 and current:
                overlap = current[-overlap_chars:]
                current = (overlap + " " + sent).strip()
            else:
                current = sent.strip()

    if len(current) >= min_chars:
        chunks.append(current)

    return chunks


# --------------------
# Back-matter detection
# --------------------
def is_back_matter_page(text: str, page_num: int, total_pages: int | None) -> bool:
    """
    Very conservative heuristic:
    - Only consider last 1–2 pages as potential back-matter.
    - Then look for strong 'References' / 'Acknowledgements' signatures.
    """
    if total_pages is None:
        return False

    if page_num < total_pages - 1:
        return False  # only last 2 pages

    lines = [ln.strip().lower() for ln in text.split("\n") if ln.strip()]
    first_line = lines[0] if lines else ""

    keywords = [
        "references",
        "acknowledgements",
        "acknowledgments",
        "bibliography",
        "all authors and affiliations",
        "all authors and aﬃliations",
    ]

    if any(first_line.startswith(kw) for kw in keywords):
        return True

    lowered = text.lower()
    if lowered.count("doi") > 8 or lowered.count("arxiv.org") > 8 or lowered.count("https://") > 10:
        return True

    return False


# --------------------
# Build chunks
# --------------------
def build_chunks(records):
    chunks = []
    chunk_id = 0

    for rec in records:
        paper_id = rec["paper_id"]
        page_num = rec["page_num"]
        total_pages = rec.get("total_pages")
        text = rec["text"]

        if not text or len(text.strip()) < 50:
            continue

        if is_back_matter_page(text, page_num, total_pages):
            continue

        page_chunks = make_chunks_from_page(text)

        for ch in page_chunks:
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "paper_id": paper_id,
                    "page_num": page_num,
                    "text": ch,
                }
            )
            chunk_id += 1

    print(f"Created {len(chunks)} chunks after conservative back-matter filtering.")
    return chunks


# --------------------
# Main
# --------------------
def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading raw records from {RAW_FILE} ...")
    records = load_raw_records()
    print(f"Loaded {len(records)} page-level records.")

    print("Building chunks ...")
    chunks = build_chunks(records)
    if not chunks:
        print("No chunks created – something is wrong with the input PDFs.")
        return

    print(f"Created {len(chunks)} chunks.")
    print(f"Saving chunks to {CHUNKS_FILE} ...")
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Loading embedding model '{EMBED_MODEL_NAME}' ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    texts = [c["text"] for c in chunks]
    print("Computing embeddings (this may take a bit) ...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    print(f"Embeddings shape: {embeddings.shape}")

    np.save(EMBEDDINGS_FILE, embeddings)

    dim = embeddings.shape[1]
    print(f"Building FAISS index with dimension {dim} ...")
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings.astype("float32"))
    faiss.write_index(index, str(INDEX_FILE))

    print("\n✅ Done!")
    print(f"  - Saved chunks:      {CHUNKS_FILE}")
    print(f"  - Saved embeddings:  {EMBEDDINGS_FILE}")
    print(f"  - Saved FAISS index: {INDEX_FILE}")


if __name__ == "__main__":
    main()
