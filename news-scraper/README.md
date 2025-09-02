# News Scraper

A Python-based web scraper for collecting Turkish news articles from popular news websites. It uses the `newspaper3k` library to extract article content, titles, and publication dates, and filters for Turkish language articles.

## Features

- Scrapes news articles from multiple Turkish news sites
- Filters articles by language (Turkish only)
- Cleans and normalizes text content
- Saves data in JSONL format
- Handles errors gracefully

## Supported Sites

- NTV (https://www.ntv.com.tr/)
- CNN Türk (https://www.cnnturk.com/)
- HaberTürk (https://www.haberturk.com/)
- Mynet (https://www.mynet.com/)

## Installation

1. Ensure you have Python 3.8+ installed.

2. Clone the repository and navigate to the news-scraper directory:
   ```bash
   cd news-scraper
   ```

3. Create a virtual environment:
   ```bash
   python -m venv ../.venv
   ```

4. Activate the virtual environment:
   ```bash
   source ../.venv/bin/activate  # On Windows: ../.venv/Scripts/activate
   ```

5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the scraper:
```bash
python scraper.py
```

The script will:
- Fetch links from the homepage of each supported site
- Filter for news-related URLs
- Scrape up to 50 articles per site
- Save the results to `datasets/news_YYYYMMDD.jsonl`

## Output Format

The output is a JSONL file where each line is a JSON object with the following fields:
- `url`: The URL of the article
- `title`: The cleaned title of the article
- `content`: The cleaned text content of the article
- `date`: The publication date (if available)

Example:
```json
{"url": "https://example.com/article", "title": "Example Title", "content": "Example content...", "date": "2023-08-30"}
```

## Dependencies

- requests: For HTTP requests
- beautifulsoup4: For HTML parsing
- newspaper3k: For article extraction
- lxml: For XML/HTML processing
- langdetect: For language detection

## Configuration

To add more sites, edit the `NEWS_SITES` list in `scraper.py`.

To change the number of articles scraped per site, modify the slice in the main loop (e.g., `links[:50]`).

## Notes

- The scraper respects a timeout of 10 seconds per request.
- Articles shorter than 50 words are skipped.
- Only articles detected as Turkish are included.
- Some sites may have anti-scraping measures; the script handles common errors.
