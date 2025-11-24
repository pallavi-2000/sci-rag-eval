from pathlib import Path
import json

import fitz  # PyMuPDF


BASE_DIR = Path(__file__).resolve().parent.parent
PAPERS_DIR = BASE_DIR / "data" / "papers"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUT_FILE = PROCESSED_DIR / "papers_raw.jsonl"


def extract_paper(pdf_path: Path):
    """
    Extract plain text per page from a PDF using PyMuPDF.
    Returns a list of records with fields:
      - paper_id     (stem of filename)
      - page_num     (1-based)
      - total_pages
      - text
    """
    doc = fitz.open(pdf_path)
    total_pages = doc.page_count
    paper_id = pdf_path.stem

    records = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text")
        text = text.replace("\r", " ").strip()
        records.append(
            {
                "paper_id": paper_id,
                "page_num": i,
                "total_pages": total_pages,
                "text": text,
            }
        )
    doc.close()
    return records


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not PAPERS_DIR.exists():
        raise FileNotFoundError(
            f"{PAPERS_DIR} does not exist. Put your PDFs in data/papers/"
        )

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"No PDFs found in {PAPERS_DIR}")

    print(f"Found {len(pdf_files)} PDF(s):")
    for pdf in pdf_files:
        print(f"  - {pdf.name}")

    all_records = []
    for pdf in pdf_files:
        print(f"\nExtracting from {pdf.name} ...")
        recs = extract_paper(pdf)
        print(f"  -> {len(recs)} pages extracted.")
        all_records.extend(recs)

    print(f"\nTotal pages extracted across all papers: {len(all_records)}")

    print(f"Writing JSONL to {OUT_FILE} ...")
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("\n✅ Done! Page-level records written to data/processed/papers_raw.jsonl")


if __name__ == "__main__":
    main()
