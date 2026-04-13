import json
import os
import shutil
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


def process_json_lines(file_path: Path) -> list[dict]:
    extracted: list[dict] = []

    with file_path.open(encoding="utf-8") as file_handle:
        for line in file_handle:
            line = line.strip()
            if not line:
                continue
            extracted.append(json.loads(line))

    return extracted


def main() -> None:
    dataset_file = Path(os.getenv("DATASET_STORAGE_FOLDER", "datasets")) / "data.txt"
    database_location = Path(os.getenv("DATABASE_LOCATION", "chroma_db"))

    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_file}. Run 1_scraping_wikipedia.py first."
        )

    file_content = process_json_lines(dataset_file)
    if not file_content:
        raise RuntimeError(f"No records found in {dataset_file}.")

    embeddings = OllamaEmbeddings(model=os.getenv("EMBEDDING_MODEL"))

    if database_location.exists():
        shutil.rmtree(database_location)

    vector_store = Chroma(
        collection_name=os.getenv("COLLECTION_NAME"),
        embedding_function=embeddings,
        persist_directory=str(database_location),
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    total_chunks = 0

    for record in file_content:
        raw_text = (record.get("raw_text") or "").strip()
        if not raw_text:
            continue

        title = record.get("title") or "Untitled"
        source = record.get("url") or record.get("source") or "Unknown source"

        documents = text_splitter.create_documents(
            [raw_text],
            metadatas=[{"source": source, "title": title}],
        )
        ids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=ids)

        total_chunks += len(documents)
        print(f"Ingested: {title}")

    print(
        f"Embedded {len(file_content)} articles into {total_chunks} vector chunks at "
        f"{database_location}"
    )


if __name__ == "__main__":
    main()
