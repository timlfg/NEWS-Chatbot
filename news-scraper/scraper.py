# Mehmet Kahya - <https://github.com/mehmetkahya0>
# 30.08.2025

import requests
from bs4 import BeautifulSoup
from newspaper import Article
import json
import os
from datetime import datetime
from utils import clean_text, is_turkish
from urllib.parse import urljoin

# Scrape edilecek siteler (örnek: NTV, CNN Türk)
NEWS_SITES = [
    "https://www.ntv.com.tr/",
    "https://www.cnnturk.com/",
    "https://www.haberturk.com/",
    "https://www.mynet.com/"

]

def get_links(url: str):
    """Ana sayfadan haber linklerini toplar."""
    html = requests.get(url, timeout=10).text
    soup = BeautifulSoup(html, "html.parser")
    links = [a["href"] for a in soup.find_all("a", href=True) if a["href"]]
    links = [urljoin(url, l) for l in links]
    print(f"Found {len(links)} links")
    print(links[:5])
    # Filtre: haber linklerini seç
    links = [l for l in links if "haber" in l or "news" in l or "son-dakika" in l]
    print(f"Filtered to {len(links)} links")
    print(links[:5])
    return list(set(links))

def scrape_article(url: str):
    """Tek bir haber sayfasını indirip temizler."""
    try:
        article = Article(url, language="tr")
        article.download()
        article.parse()
        title = clean_text(article.title)
        text = clean_text(article.text)

        if len(text.split()) < 50:  # çok kısa haberleri atla
            return None
        if not is_turkish(text):
            return None

        return {
            "url": url,
            "title": title,
            "content": text,
            "date": str(article.publish_date) if article.publish_date else None
        }
    except Exception as e:
        print(f"Hata: {url} -> {e}")
        return None

def main():
    # Dataset klasörü
    os.makedirs("datasets", exist_ok=True)
    filename = f"datasets/news_{datetime.now().strftime('%Y%m%d')}.jsonl"

    # Mevcut haberleri oku ve URL'leri set'e ekle
    existing_urls = set()
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    existing_urls.add(data["url"])
                except:
                    pass

    all_articles = []
    for site in NEWS_SITES:
        print(f"⏳ {site} taranıyor...")
        links = get_links(site)
        print(f"🔗 {len(links)} link bulundu.")
        for link in links[:200]:  # scrape up to 200 articles per site
            if link in existing_urls:
                continue  # aynı haber varsa atla
            article = scrape_article(link)
            if article:
                all_articles.append(article)
                existing_urls.add(article["url"])

    # JSONL olarak ekle (append mode)
    with open(filename, "a", encoding="utf-8") as f:
        for item in all_articles:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ {len(all_articles)} yeni haber eklendi -> {filename}")

if __name__ == "__main__":
    main()
