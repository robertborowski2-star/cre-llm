import re
import json
import spacy
from pathlib import Path

# ---- Load spaCy model ----
print("Loading NLP model...")
nlp = spacy.load("en_core_web_sm")

CLEANED_DIR = Path("data/cleaned")
ANON_DIR    = Path("data/anonymized")
ANON_DIR.mkdir(exist_ok=True)

# ---- Replacement tokens ----
REPLACEMENTS = {
    "PERSON":   "[PERSON]",
    "ORG":      "[COMPANY]",
    "GPE":      "[CITY]",      # cities, countries
    "LOC":      "[LOCATION]",  # non-GPE locations
}

# ---- Regex patterns for things spaCy misses ----
REGEX_PATTERNS = [
    # email addresses
    (r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '[EMAIL]'),
    # phone numbers
    (r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]'),
    # URLs
    (r'https?://\S+', '[URL]'),
    # signatures like "Sincerely, John Smith"
    (r'(?i)(sincerely|regards|best regards|yours truly),?\s*\n\s*[A-Z][a-z]+ [A-Z][a-z]+', '[SIGNATURE]'),
]

# ---- Known CRE terms to preserve (don't anonymize these) ----
# spaCy sometimes flags lender names, institutions as ORGs we want to keep
PRESERVE = {
    'cmhc', 'osfi', 'rbc', 'td', 'bmo', 'cibc', 'scotiabank',
    'national bank', 'equitable bank', 'laurentian', 'home trust',
    'first national', 'mcap', 'cmls', 'romspen', 'trez', 'kingsett',
    'canada', 'ontario', 'bc', 'alberta', 'quebec', 'toronto',
    'vancouver', 'calgary', 'montreal', 'ottawa', 'edmonton',
    'canada mortgage', 'housing corporation', 'acm', 'eqb', 'equitable',
}

def should_preserve(text):
    return text.lower().strip() in PRESERVE

def anonymize_text(text):
    # ---- Step 1: regex patterns first ----
    for pattern, replacement in REGEX_PATTERNS:
        text = re.sub(pattern, replacement, text)

    # ---- Step 2: spaCy NER ----
    # process in chunks to handle long documents
    chunk_size = 100000
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    anonymized_chunks = []
    for chunk in chunks:
        doc = nlp(chunk)
        
        # build replacement map - go in reverse to preserve positions
        replacements = []
        for ent in doc.ents:
            if ent.label_ in REPLACEMENTS:
                if not should_preserve(ent.text):
                    replacements.append((ent.start_char, ent.end_char, REPLACEMENTS[ent.label_]))
        
        # apply replacements in reverse order
        result = chunk
        for start, end, replacement in sorted(replacements, reverse=True):
            result = result[:start] + replacement + result[end:]
        
        anonymized_chunks.append(result)
    
    return ''.join(anonymized_chunks)

def run():
    # load index
    index_path = CLEANED_DIR / "index.json"
    index = json.loads(index_path.read_text())
    
    # check what's already anonymized
    already_done = {f.stem for f in ANON_DIR.glob("*.txt")}
    
    to_process = [d for d in index if Path(d['filename']).stem not in already_done]
    
    print(f"Total documents:     {len(index)}")
    print(f"Already anonymized:  {len(already_done)}")
    print(f"To process:          {len(to_process)}")
    print()
    
    total_replacements = 0
    
    for doc in to_process:
        stem     = Path(doc['filename']).stem
        src_path = CLEANED_DIR / (stem + ".txt")
        
        if not src_path.exists():
            continue
        
        print(f"Anonymizing: {doc['filename']}")
        
        original = src_path.read_text(encoding='utf-8', errors='ignore')
        anonymized = anonymize_text(original)
        
        # count replacements made
        replacements = sum(anonymized.count(r) for r in [
            '[PERSON]', '[COMPANY]', '[EMAIL]', '[PHONE]', '[SIGNATURE]'
        ])
        total_replacements += replacements
        
        # save anonymized version
        out_path = ANON_DIR / (stem + ".txt")
        out_path.write_text(anonymized, encoding='utf-8')
        
        print(f"  ✓ {replacements} replacements made")
    
    print(f"\n✓ Anonymized {len(to_process)} documents")
    print(f"✓ Total replacements: {total_replacements:,}")
    print(f"✓ Saved to {ANON_DIR}")

if __name__ == "__main__":
    run()