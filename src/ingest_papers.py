from pathlib import Path
import json
from pypdf import PdfReader


# Folders
RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
OUT_FILE = OUT_DIR / "papers_raw.jsonl"


def extract_pdf(pdf_path: Path):
    """
    Read one PDF and return a list of records:
    {
      "paper_id": <filename without .pdf>,
      "page_num": <1-based page index>,
      "text": <page text>
    }
    """
    reader = PdfReader(str(pdf_path))
    records = []

    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text()
        except Exception as e:
            print(f"Error reading page {i+1} of {pdf_path.name}: {e}")
            continue

        if not text or not text.strip():
            # skip empty pages
            continue

        records.append({
            "paper_id": pdf_path.stem,
            "page_num": i + 1,
            "text": text,
        })

    return records


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(RAW_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {RAW_DIR.resolve()}")
        return

    all_records = []

    print(f"Found {len(pdf_files)} PDF(s) in {RAW_DIR}...")
    for pdf in pdf_files:
        print(f"\nProcessing {pdf.name} ...")
        recs = extract_pdf(pdf)
        print(f"  -> {len(recs)} page(s) with text")
        all_records.extend(recs)

    if not all_records:
        print("No text extracted from any PDFs.")
        return

    # Write everything to a JSONL file
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"\n✅ Done! Saved {len(all_records)} records to {OUT_FILE}")


if __name__ == "__main__":
    main()
