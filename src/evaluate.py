import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from rag_answer_local import generate_answer_local

# Paths
EVAL_FILE = Path("eval/qa_set.jsonl")
OUT_FILE = Path("eval/eval_results.csv")

# Embedding model used for scoring similarity
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

ID_MAP = {
    "supernova": "paper2",
    "nsbh": "paper1",
    "paper1": "paper1",
    "paper2": "paper2",
}



def load_eval_set(eval_file: Path = EVAL_FILE):
    """
    Load evaluation questions from a JSONL file.

    Each line should be a JSON object with at least:
      - "question": str
      - "gold_answer": str

    Optional fields:
      - "paper_id": str
      - "notes": str
    """
    if not eval_file.exists():
        raise FileNotFoundError(
            f"{eval_file} not found. Create eval/qa_set.jsonl first."
        )

    items = []
    with eval_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    print(f"Loaded {len(items)} eval questions from {eval_file}.")
    return items


def compute_similarity(model: SentenceTransformer, text1: str, text2: str) -> float:
    """
    Compute cosine similarity between two texts using sentence-transformer embeddings.
    """
    emb = model.encode([text1, text2], convert_to_numpy=True)
    v1, v2 = emb[0], emb[1]

    num = float(np.dot(v1, v2))
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return num / den


def run_evaluation(top_k: int = 5, model_name: str = "google/flan-t5-small"):
    """
    Run evaluation over all QA pairs in eval/qa_set.jsonl.

    For each item:
      - call the local RAG pipeline (generate_answer_local)
      - compute similarity between predicted and gold answer
      - save everything into eval/eval_results.csv
    """
    # 1. Load eval questions
    eval_items = load_eval_set()

    if not eval_items:
        print("No eval items found.")
        return

    # 2. Load embedding model for scoring
    print(f"Loading embedding model for evaluation: {EMBED_MODEL_NAME}")
    emb_model = SentenceTransformer(EMBED_MODEL_NAME)

    results = []

    for i, item in enumerate(eval_items, start=1):
        raw_paper_id = item.get("paper_id", "")
        paper_id = ID_MAP.get(raw_paper_id, raw_paper_id or None)
        pred_answer, _ = generate_answer_local(
            question=question,
            top_k=top_k,
            model_name=model_name,
            paper_id=paper_id,
            )



        if pred_answer is None:
            pred_answer = ""
            similarity = 0.0
        else:
            # 4. Compute similarity between gold and predicted
            similarity = compute_similarity(emb_model, gold_answer, pred_answer)

        results.append({
            "paper_id": paper_id,
            "question": question,
            "gold_answer": gold_answer,
            "pred_answer": pred_answer,
            "similarity": similarity,
            "notes": notes,
        })

    # 5. Save results as CSV
    df = pd.DataFrame(results)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8")

    print(f"\n✅ Evaluation complete! Saved results to {OUT_FILE}")
    print("Preview:")
    print(df.head(min(5, len(df))))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate local RAG pipeline on a gold Q&A set."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of chunks to retrieve for each question.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-small",
        help="HF model name used in rag_answer_local.",
    )
    args = parser.parse_args()

    run_evaluation(top_k=args.top_k, model_name=args.model)


if __name__ == "__main__":
    main()
