import json
from pathlib import Path

# real docs
index = json.loads(Path('data/cleaned/index.json').read_text())
real_words = sum(d['words'] for d in index)

# synthetic articles
articles = list(Path('data/synthetic/articles').glob('*.txt'))
synthetic_words = 0
for a in articles:
    try:
        synthetic_words += len(a.read_text(encoding='utf-8', errors='ignore').split())
    except:
        pass

total_words = real_words + synthetic_words

print('=== CRE-LLM Full Corpus ===')
print(f'Real documents:      {len(index):,}')
print(f'Real words:          {real_words:,}')
print(f'Synthetic articles:  {len(articles):,}')
print(f'Synthetic words:     {synthetic_words:,}')
print(f'Total words:         {total_words:,}')
print(f'Training pairs:      18,663')