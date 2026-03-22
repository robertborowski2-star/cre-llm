import os
import json
import pdfplumber
from docx import Document
from pathlib import Path

# ---- Paths ----
RAW_DIR     = Path("data/raw")
CLEANED_DIR = Path("data/cleaned")

# ---- PDF Extractor ----
def extract_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()

# ---- Word Extractor ----
def extract_docx(path):
    doc = Document(path)
    text = ""
    for para in doc.paragraphs:
        if para.text.strip():
            text += para.text.strip() + "\n"
    return text.strip()

# ---- Text Cleaner ----
def clean_text(text):
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        line = line.strip()
        # skip empty lines and garbage
        if not line:
            continue
        if len(line) < 3:
            continue
        # skip lines that are just numbers (page numbers etc)
        if line.replace(".", "").replace("-", "").isdigit():
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

# ---- Main Processor ----
def process_document(path):
    path = Path(path)
    suffix = path.suffix.lower()

    print(f"Processing: {path.name}")

    # extract based on file type
    if suffix == ".pdf":
        raw_text = extract_pdf(path)
    elif suffix in [".docx", ".doc"]:
        raw_text = extract_docx(path)
    elif suffix == ".txt":
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
    else:
        print(f"  Skipping unsupported format: {suffix}")
        return None

    if not raw_text:
        print(f"  Warning: no text extracted from {path.name}")
        return None

    # clean the text
    cleaned = clean_text(raw_text)

    # save to cleaned folder
    out_name = path.stem + ".txt"
    out_path = CLEANED_DIR / out_name
    out_path.write_text(cleaned, encoding="utf-8")

    # save metadata
    meta = {
        "filename":   path.name,
        "source":     str(path),
        "characters": len(cleaned),
        "words":      len(cleaned.split()),
        "type":       suffix
    }

    print(f"  ✓ {meta['words']:,} words → {out_name}")
    return meta

# ---- Run on all files in data/raw ----
def run():
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    files = list(RAW_DIR.iterdir())
    if not files:
        print("No files found in data/raw/")
        print("Drop some PDFs or Word docs in there and run again.")
        return

    all_meta = []
    for f in files:
        if f.is_file():
            meta = process_document(f)
            if meta:
                all_meta.append(meta)

    # save index of all processed docs
    index_path = CLEANED_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(all_meta, f, indent=2)

    total_words = sum(m["words"] for m in all_meta)
    print(f"\n✓ Processed {len(all_meta)} documents")
    print(f"✓ Total words: {total_words:,}")
    print(f"✓ Index saved to {index_path}")

if __name__ == "__main__":
    run()
