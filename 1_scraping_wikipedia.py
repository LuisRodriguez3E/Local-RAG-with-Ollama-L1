import json
import os
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
from dotenv import load_dotenv


WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/api.php"
REQUEST_TIMEOUT_SECONDS = 30


load_dotenv()


def load_keywords(file_path: str) -> pd.DataFrame:
    keywords = pd.read_excel(file_path)

    if "Keyword" not in keywords.columns:
        raise ValueError("keywords.xlsx must contain a 'Keyword' column.")

    if "Pages" not in keywords.columns:
        keywords["Pages"] = 1

    keywords = keywords.dropna(subset=["Keyword"]).copy()
    keywords["Keyword"] = keywords["Keyword"].astype(str).str.strip()
    keywords = keywords[keywords["Keyword"] != ""]
    keywords["Pages"] = (
        pd.to_numeric(keywords["Pages"], errors="coerce")
        .fillna(1)
        .astype(int)
        .clip(lower=1, upper=10)
    )

    return keywords


def search_titles(session: requests.Session, keyword: str, limit: int) -> list[str]:
    response = session.get(
        WIKIPEDIA_API_URL,
        params={
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": keyword,
            "srlimit": limit,
            "utf8": 1,
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    results = response.json().get("query", {}).get("search", [])
    titles = [result["title"] for result in results if result.get("title")]

    return titles or [keyword]


def fetch_page(session: requests.Session, title: str, keyword: str) -> dict | None:
    response = session.get(
        WIKIPEDIA_API_URL,
        params={
            "action": "query",
            "format": "json",
            "formatversion": 2,
            "prop": "extracts|info",
            "titles": title,
            "redirects": 1,
            "explaintext": 1,
            "inprop": "url",
        },
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()

    pages = response.json().get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    raw_text = (page.get("extract") or "").strip()

    if page.get("missing") or not raw_text:
        return None

    resolved_title = page.get("title", title)
    resolved_url = page.get("fullurl") or (
        "https://en.wikipedia.org/wiki/" + quote(resolved_title.replace(" ", "_"))
    )

    return {
        "keyword": keyword,
        "pageid": page.get("pageid"),
        "title": resolved_title,
        "url": resolved_url,
        "raw_text": raw_text,
    }


def main() -> None:
    dataset_dir = Path(os.getenv("DATASET_STORAGE_FOLDER", "datasets"))
    output_file = dataset_dir / "data.txt"
    keywords = load_keywords("keywords.xlsx")

    session = requests.Session()
    session.headers.update(
        {"User-Agent": "Local-RAG-with-Ollama-L1/1.0 (Wikipedia ingestion)"}
    )

    dataset_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    seen_page_ids: set[int] = set()

    for row in keywords.itertuples(index=False):
        titles = search_titles(session, row.Keyword, row.Pages)

        for title in titles:
            record = fetch_page(session, title, row.Keyword)
            if not record:
                continue

            page_id = record.get("pageid")
            if page_id in seen_page_ids:
                continue

            if page_id is not None:
                seen_page_ids.add(page_id)

            records.append(record)
            print(f"Fetched: {record['title']} ({record['url']})")

    with output_file.open("w", encoding="utf-8") as file_handle:
        for record in records:
            file_handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} Wikipedia articles to {output_file}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        raise SystemExit(f"Wikipedia API request failed: {exc}") from exc
