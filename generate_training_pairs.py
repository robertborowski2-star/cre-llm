import os
import json
import anthropic
from pathlib import Path
from dotenv import load_dotenv

# ---- Config ----
load_dotenv()
CLEANED_DIR  = Path("data/anonymized")
INDEX_DIR    = Path("data/cleaned")
TRAINING_DIR = Path("data/training")
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ---- System prompt ----
# This instructs Claude to generate training pairs from CRE documents
SYSTEM_PROMPT = """You are a training data generator for a Canadian commercial real estate AI model.

Given a CRE document, generate 10 high quality question/answer pairs that test:
- Specific facts from the document (cap rates, prices, terms, locations)
- Domain reasoning (what does this metric mean, why does it matter)
- Canadian market context (how does this compare to market norms)
- Underwriting judgment (what are the risks, what would a lender focus on)

Rules:
- Questions must be answerable from the document content
- Answers must be specific, accurate, and use proper CRE terminology
- Include Canadian context where relevant (CMHC, GoC spreads, provincial nuances)
- Vary question types: factual, analytical, comparative, risk-focused
- Answers should be 2-5 sentences, thorough but concise

Respond ONLY with a JSON array, no preamble, no markdown backticks.
Format:
[
  {"prompt": "question here", "completion": "answer here"},
  ...
]"""

# ---- Generate pairs for one document ----
def generate_pairs(doc_path, max_chars=8000):
    text = doc_path.read_text(encoding="utf-8")
    
    if len(text) > max_chars:
        text = text[:max_chars]
    
    print(f"  Generating pairs for: {doc_path.name} ({len(text):,} chars)")
    
    try:
        # scale pairs by document length
        word_count = len(text.split())
        if word_count > 5000:
            num_pairs = 20
        elif word_count > 2000:
            num_pairs = 15
        else:
            num_pairs = 10

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=8000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"Generate {num_pairs} training pairs from this CRE document:\n\n{text}"
            }]
        )
        
        raw = response.content[0].text.strip()
        
        # strip markdown backticks if present
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        
        # aggressive control character cleaning
        # keep only printable ascii + newlines
        cleaned = ""
        for char in raw:
            if char == '\n' or char == '\t':
                cleaned += " "
            elif ord(char) >= 32 and ord(char) < 127:
                cleaned += char
            elif ord(char) > 127:
                # keep unicode letters (accented chars etc)
                cleaned += char
        
        pairs = json.loads(cleaned)
        print(f"  ✓ Generated {len(pairs)} pairs")
        return pairs
        
    except json.JSONDecodeError as e:
        print(f"  ✗ JSON parse failed: {e}")
        print(f"  Raw response preview: {raw[:200]}")
        return []
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return []

# ---- Run on all cleaned documents ----
def run():
    # load index
    index_path = INDEX_DIR / "index.json"
    if not index_path.exists():
        print("No index found - run extractor.py first")
        return
    
    index = json.loads(index_path.read_text())
    
    # check which docs already have training pairs
    done_path = TRAINING_DIR / "generated.json"
    if done_path.exists():
        already_done = set(json.loads(done_path.read_text()))
    else:
        already_done = set()
    
    # load existing pairs
    pairs_path = TRAINING_DIR / "pairs.jsonl"
    existing_count = 0
    if pairs_path.exists():
        existing_count = sum(1 for _ in pairs_path.open())
    
    print(f"Corpus: {len(index)} documents")
    print(f"Already processed: {len(already_done)} documents")
    print(f"Existing pairs: {existing_count:,}")
    print(f"To process: {len(index) - len(already_done)} documents")
    print()
    
    new_pairs = []
    newly_done = []
    
    for doc in index:
        filename = doc["filename"]
        stem     = Path(filename).stem
        
        if stem in already_done:
            continue
        
        doc_path = CLEANED_DIR / (stem + ".txt")
        if not doc_path.exists():
            # fallback to cleaned if anonymized version doesn't exist
            doc_path = INDEX_DIR / (stem + ".txt")
        
        pairs = generate_pairs(doc_path)
        
        if pairs:
            new_pairs.extend(pairs)
            newly_done.append(stem)
    
    # append new pairs to jsonl file
    if new_pairs:
        with open(pairs_path, "a") as f:
            for pair in new_pairs:
                f.write(json.dumps(pair) + "\n")
    
    # update done list
    all_done = list(already_done) + newly_done
    done_path.write_text(json.dumps(all_done, indent=2))
    
    total_pairs = existing_count + len(new_pairs)
    print(f"\n✓ Generated {len(new_pairs)} new pairs from {len(newly_done)} documents")
    print(f"✓ Total training pairs: {total_pairs:,}")
    print(f"✓ Saved to {pairs_path}")

if __name__ == "__main__":
    run()
