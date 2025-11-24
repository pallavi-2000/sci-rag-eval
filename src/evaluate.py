import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from rag_answer_local import generate_answer_local

BASE_DIR = Path(__file__).resolve().parent.parent

EVAL_FILE = BASE_DIR / "eval" / "qa_set.jsonl"
OUT_FILE = BASE_DIR / "eval" / "eval_results.csv"

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def load_eval_set(eval_file: Path = EVAL_FILE):
    if not eval_file.exists():
        raise FileNotFoundError(f"{eval_file} not found.")

    items = []
    with eval_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    print(f"Loaded {len(items)} eval questions.")
    return items


def compute_similarity(model: SentenceTransformer, text1: str, text2: str) -> float:
    emb = model.encode([text1, text2], convert_to_numpy=True)
    v1, v2 = emb[0], emb[1]

    num = float(np.dot(v1, v2))
    den = float(np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return num / den


def run_evaluation(top_k: int = 8, model_name: str = "google/flan-t5-large"):
    eval_items = load_eval_set()

    print(f"Loading embedding model for evaluation: {EMBED_MODEL_NAME}")
    emb_model = SentenceTransformer(EMBED_MODEL_NAME)

    results = []

    for i, item in enumerate(eval_items, start=1):
        question = item["question"]
        gold_answer = item["gold_answer"]
        notes = item.get("notes", "")
        paper_id = item.get("paper_id", None)

        print(f"\n[{i}/{len(eval_items)}] Q: {question}")

        pred_answer, _ = generate_answer_local(
            question=question,
            top_k=top_k,
            model_name=model_name,
            paper_id=paper_id,
        )

        if not pred_answer:
            pred_answer = "Not enough information in the documents to answer this confidently."

        similarity = compute_similarity(emb_model, gold_answer, pred_answer)

        results.append(
            {
                "paper_id": paper_id,
                "question": question,
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "similarity": similarity,
                "notes": notes,
            }
        )

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
    parser.add_argument("--top_k", type=int, default=8, help="Number of chunks to retrieve.")
    parser.add_argument(
        "--model",
        type=str,
        default="google/flan-t5-large",
        help="HF model name (must match rag_answer_local).",
    )
    args = parser.parse_args()

    run_evaluation(top_k=args.top_k, model_name=args.model)


if __name__ == "__main__":
    main()
