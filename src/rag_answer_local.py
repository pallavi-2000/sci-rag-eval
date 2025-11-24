import argparse
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from rag_qa import retrieve_chunks, pretty_print_results

MODEL_NAME = "google/flan-t5-large"

# Cache for reuse
_tokenizer = None
_model = None
_device = None


def load_local_model(model_name: str = MODEL_NAME):
    global _tokenizer, _model, _device

    if _tokenizer is not None:
        return _tokenizer, _model, _device

    print(f"Loading local model: {model_name} ...")
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model.to(_device)

    return _tokenizer, _model, _device


def build_context_block(results: List[Tuple[float, dict]], max_chars: int = 4000) -> str:
    pieces = []
    total_len = 0

    for _, chunk in results:
        paper_id = chunk["paper_id"]
        page_num = chunk["page_num"]
        text = chunk["text"].strip().replace("\n", " ")

        header = f"[Source: {paper_id}, page {page_num}]\n"
        block = header + text + "\n\n"

        if total_len + len(block) > max_chars:
            break

        pieces.append(block)
        total_len += len(block)

    return "\n".join(pieces)


def build_prompt(question: str, context: str) -> str:
    return (
        "You are a careful scientific assistant. "
        "Answer the question using ONLY the context below. "
        "If the answer is not clearly supported by the context, say: "
        "'Not enough information in the documents to answer this confidently.'\n\n"
        "Write your answer as one or two complete sentences. "
        "Include any numerical values and units explicitly.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def generate_answer_local(
    question: str,
    top_k: int = 5,
    model_name: str = MODEL_NAME,
    max_new_tokens: int = 256,
    paper_id: str | None = None,
):
    """
    Full RAG pipeline:
    1. Retrieve chunks (optionally filtered by paper_id).
    2. Build prompt.
    3. Generate answer with FLAN-T5.
    """
    results = retrieve_chunks(question, top_k=top_k, paper_id=paper_id)

    if not results:
        print("⚠️ Retrieval returned no chunks — cannot answer.")
        return "Not enough information in the documents to answer this confidently.", []

    context = build_context_block(results)
    prompt = build_prompt(question, context)

    tokenizer, model, device = load_local_model(model_name)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    if not answer:
        answer = "Not enough information in the documents to answer this confidently."

    return answer, results


def main():
    parser = argparse.ArgumentParser(
        description="RAG QA with a local FLAN-T5 model (no API, no cost)."
    )
    parser.add_argument("question", type=str, nargs="*", help="Question to ask.")
    parser.add_argument("--top_k", type=int, default=5, help="Number of chunks to retrieve.")
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="HF model name.")
    parser.add_argument("--paper_id", type=str, default=None, help="Optional paper filter.")
    args = parser.parse_args()

    question = " ".join(args.question) if args.question else input("Enter your question: ").strip()

    if not question:
        print("No question provided.")
        return

    print(f"\n[QUESTION] {question}\n")

    answer, results = generate_answer_local(
        question=question,
        top_k=args.top_k,
        model_name=args.model,
        paper_id=args.paper_id,
    )

    print("===== LOCAL MODEL ANSWER =====\n")
    print(answer)
    print("\n==============================\n")

    print("===== RETRIEVED CHUNKS USED =====")
    pretty_print_results(results)


if __name__ == "__main__":
    main()
