import argparse
from typing import List, Tuple

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from rag_qa import retrieve_chunks, pretty_print_results


MODEL_NAME = "google/flan-t5-small"


def load_local_model(model_name: str = MODEL_NAME):
    """
    Load a small local text2text model (FLAN-T5) for QA.
    Runs on CPU by default, no API key needed.
    """
    print(f"Loading local model: {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model, device


def build_context_block(results: List[Tuple[float, dict]], max_chars: int = 4000) -> str:
    """
    Turn retrieved chunks into a single context string with source headers.
    """
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
    """
    Build a prompt for FLAN-T5.
    FLAN-T5 works well with instruction-style prompts.
    """
    prompt = (
        "You are a careful scientific assistant. "
        "Answer the question using ONLY the context below. "
        "If the answer is not clearly supported by the context, say: "
        "'Not enough information in the documents to answer this confidently.'\n\n"
        "Write your answer as one or two complete sentences. "
        "Include any numerical values and units explicitly. "
        "Do not answer with a single word like 'sensitivity' or 'GW200105' "
        "unless that single word is itself the full correct answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
    return prompt



def generate_answer_local(
    question: str,
    top_k: int = 5,
    model_name: str = MODEL_NAME,
    max_new_tokens: int = 256,
    paper_id: str | None = None,
):
    """
    Full RAG pipeline with a local model:
    1. Retrieve top_k chunks (optionally restricted to a given paper_id).
    2. Build a prompt with those chunks as context.
    3. Generate an answer using FLAN-T5 locally.
    """
    # 1. Retrieve chunks (optionally filtered)
    results = retrieve_chunks(question, top_k=top_k, paper_id=paper_id)

    if not results:
        print("No retrieval results found. Cannot answer.")
        return None, []

    # 2. Build context + prompt
    context = build_context_block(results)
    prompt = build_prompt(question, context)

    # 3. Load model & tokenizer
    tokenizer, model, device = load_local_model(model_name)

    # 4. Tokenize & generate
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    answer = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    return answer, results


def main():
    parser = argparse.ArgumentParser(
        description="RAG QA with a local FLAN-T5 model (no API, no cost)."
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
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        help="Hugging Face model name to use.",
    )
    args = parser.parse_args()

    if args.question:
        question = " ".join(args.question)
    else:
        question = input("Enter your question: ").strip()

    if not question:
        print("No question provided.")
        return

    print(f"\n[QUESTION] {question}\n")

    answer, results = generate_answer_local(
        question=question,
        top_k=args.top_k,
        model_name=args.model,
    )

    if answer is None:
        return

    print("===== LOCAL MODEL ANSWER =====\n")
    print(answer)
    print("\n==============================\n")

    print("===== RETRIEVED CHUNKS USED =====")
    pretty_print_results(results)


if __name__ == "__main__":
    main()