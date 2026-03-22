import os
from pathlib import Path
from supabase import create_client
from dotenv import load_dotenv

# ---- Load credentials ----
load_dotenv()
url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
supabase = create_client(url, key)

# ---- Config ----
BUCKETS   = ["cbre-colliers-reports", "local-reports"]
RAW_DIR   = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# ---- Download all files from a bucket (recursive) ----
def download_bucket(bucket_name, folder=""):
    path = folder if folder else ""
    files = supabase.storage.from_(bucket_name).list(path)
    
    if not files:
        return 0

    downloaded = 0
    for f in files:
        filename = f["name"]
        full_path = f"{folder}/{filename}" if folder else filename

        # if it looks like a folder (no extension) go recursive
        if "." not in filename:
            print(f"  → Entering folder: {full_path}")
            downloaded += download_bucket(bucket_name, full_path)
            continue

        # skip non-documents
        if not filename.endswith((".pdf", ".docx", ".doc", ".txt")):
            print(f"  Skipping: {filename}")
            continue

        # flatten folder structure into filename
        safe_name = full_path.replace("/", "_")
        out_path  = RAW_DIR / safe_name

        if out_path.exists():
            print(f"  Already exists: {safe_name}")
            continue

        print(f"  Downloading: {full_path}...")
        data = supabase.storage.from_(bucket_name).download(full_path)
        out_path.write_bytes(data)
        print(f"  ✓ {safe_name} ({len(data)/1024:.0f} KB)")
        downloaded += 1

    return downloaded

# ---- Run ----
total = 0
for bucket in BUCKETS:
    total += download_bucket(bucket)

print(f"\n✓ Downloaded {total} new files to data/raw/")
print("Run python3 extractor.py to process them")
