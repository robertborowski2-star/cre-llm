import os
import json
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ---- Paths ----
NEWS_DIR     = Path("data/raw/news")
PAIRS_PATH   = Path("data/training/pairs.jsonl")
DONE_PATH    = Path("data/synthetic/scraped_urls.json")
NEWS_DIR.mkdir(parents=True, exist_ok=True)

# ---- Load done URLs ----
if DONE_PATH.exists():
    done_urls = set(json.loads(DONE_PATH.read_text()))
else:
    done_urls = set()

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

# ---- Scrapers (reused from Klaus scraper) ----
def scrape_renx():
    articles = []
    try:
        response = requests.get('https://renx.ca', headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        cards = soup.find_all('div', class_='article-card', limit=30)
        for card in cards:
            link_elem = card.find('a', href=True)
            if not link_elem:
                continue
            title = link_elem.get('title', '').strip()
            link  = link_elem.get('href', '').strip()
            if not link.startswith('http'):
                link = f"https://renx.ca{link}"
            if title and link:
                articles.append({'title': title, 'url': link, 'source': 'RENx'})
    except Exception as e:
        print(f"RENx error: {e}")
    return articles

def scrape_storeys():
    articles = []
    try:
        response = requests.get('https://storeys.com/feed/', headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.content, 'lxml-xml')
        items = soup.find_all('item', limit=30)
        for item in items:
            title_elem = item.find('title')
            link_elem  = item.find('link')
            desc_elem  = item.find('description')
            if not title_elem or not link_elem:
                continue
            title = title_elem.text.strip()
            link  = link_elem.text.strip()
            # get content directly from RSS — no need to fetch URL
            content = desc_elem.text.strip() if desc_elem else ""
            # strip HTML tags from RSS content
            content_soup = BeautifulSoup(content, 'html.parser')
            clean_text = content_soup.get_text(separator=' ', strip=True)
            if title and link and len(clean_text) > 200:
                articles.append({
                    'title':   title,
                    'url':     link,
                    'source':  'Storeys',
                    'content': clean_text[:10000]  # pre-fetched content
                })
    except Exception as e:
        print(f"Storeys RSS error: {e}")
    return articles

def scrape_commercial_realestate_ca():
    articles = []
    try:
        response = requests.get('https://www.commercialrealestate.ca/news/', headers=HEADERS, timeout=15)
        soup = BeautifulSoup(response.content, 'html.parser')
        cards = soup.find_all('article', limit=30)
        for card in cards:
            title_elem = card.find('h2') or card.find('h3')
            if not title_elem:
                continue
            link_elem = title_elem.find('a')
            if not link_elem:
                continue
            title = title_elem.get_text(strip=True)
            link  = link_elem.get('href', '')
            if not link.startswith('http'):
                link = f"https://www.commercialrealestate.ca{link}"
            if title and link:
                articles.append({'title': title, 'url': link, 'source': 'CommercialRealEstate.ca'})
    except Exception as e:
        print(f"CommercialRealEstate.ca error: {e}")
    return articles

# ---- Fetch full article text ----
def fetch_article_text(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            tag.decompose()
        # try article body first
        body = soup.find('article') or soup.find('div', class_='post-content') or soup.find('div', class_='entry-content')
        if body:
            text = body.get_text(separator=' ', strip=True)
        else:
            text = soup.get_text(separator=' ', strip=True)
        # clean up whitespace
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        text = '\n'.join(lines)
        return text[:10000]
    except Exception as e:
        return None

# ---- Generate training pairs from article ----
def generate_pairs(title, text, source):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": f"""Generate 8 training pairs from this Canadian CRE news article for training an AI model.

Title: {title}
Source: {source}

Article:
{text[:6000]}

Rules:
- Questions must be specific to facts in this article
- Answers must be accurate and use proper CRE terminology
- Focus on market data, deal details, trends, and lending implications
- Answers should be 2-4 sentences

Respond ONLY with a JSON array, no preamble:
[{{"prompt": "...", "completion": "..."}}, ...]"""
        }]
    )

    raw = response.content[0].text.strip()
    if "```" in raw:
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    # clean control characters
    cleaned = ""
    for char in raw:
        if char in ['\n', '\t']:
            cleaned += " "
        elif ord(char) >= 32:
            cleaned += char

    return json.loads(cleaned)

# ---- Main ----
def run():
    print("Scraping Canadian CRE news sources...")

    all_articles = []
    all_articles.extend(scrape_renx())
    all_articles.extend(scrape_storeys())
    all_articles.extend(scrape_commercial_realestate_ca())

    print(f"Found {len(all_articles)} articles")
    new_articles = [a for a in all_articles if a['url'] not in done_urls]
    print(f"New (not yet processed): {len(new_articles)}")

    new_pairs  = 0
    newly_done = []

    for article in new_articles:
        print(f"\n{article['source']}: {article['title'][:60]}...")

        # use pre-fetched content if available (RSS), otherwise fetch
        text = article.get('content') or fetch_article_text(article['url'])
        if not text or len(text) < 200:
            print("  ✗ Too short or failed to fetch")
            continue

        # save raw article text
        safe_name = article['title'].lower().replace(' ', '_')[:60] + '.txt'
        safe_name = ''.join(c for c in safe_name if c.isalnum() or c in '_-.')
        article_path = NEWS_DIR / safe_name
        article_path.write_text(f"{article['title']}\n{article['source']}\n\n{text}", encoding='utf-8')

        # generate pairs
        try:
            pairs = generate_pairs(article['title'], text, article['source'])
            with open(PAIRS_PATH, 'a') as f:
                for pair in pairs:
                    f.write(json.dumps(pair) + '\n')
            new_pairs += len(pairs)
            newly_done.append(article['url'])
            print(f"  ✓ {len(pairs)} pairs saved")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

        # save progress
        all_done = list(done_urls) + newly_done
        DONE_PATH.write_text(json.dumps(all_done, indent=2))

        time.sleep(1)

    total_pairs = sum(1 for _ in open(PAIRS_PATH))
    print(f"\n✓ {new_pairs} new pairs from {len(newly_done)} articles")
    print(f"✓ Total training pairs: {total_pairs:,}")

if __name__ == "__main__":
    run()