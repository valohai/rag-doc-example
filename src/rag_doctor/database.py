from __future__ import annotations

import json
import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import valohai.config
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_voyageai import VoyageAIEmbeddings

from rag_doctor.consts import (
    ANTHROPIC_EMBEDDING_MODEL,
    ANTHROPIC_EMBEDDING_LENGTH,
    EMBEDDING_MODEL,
    EMBEDDING_LENGTH,
    CONTENT_COLUMN,
    PROVIDER,
    SOURCE_COLUMN,
    COLLECTION_NAME,
)

log = logging.getLogger(__name__)


def create_database(
    *,
    documentation_dir: str,
    content_column_index: int,
    source_column_index: int,
    header_row_skip: int,
    database_dir: str,
    provider: str = PROVIDER,
):
    documents = gather_documentation(
        documentation_dir=documentation_dir,
        content_column_index=content_column_index,
        source_column_index=source_column_index,
        header_row_skip=header_row_skip,
    )

    Path(database_dir).mkdir(exist_ok=True)
    db_client = QdrantClient(path=database_dir)

    create_qdrant_collection(db_client, provider)  
    vectorize_documents(db_client, documents, provider)  

    if valohai.config.is_running_in_valohai():
        package_database(database_dir)

    log.info("Vector database creation complete!")


def prepare_database(database_dir: Path | str) -> QdrantClient:
    database_dir_path = Path(database_dir)

    if not database_dir_path.is_dir():
        raise ValueError(f"Database directory {database_dir_path} does not exist")

    # if it doesn't look like a Qdrant database, find the first zip and unzip it
    if not (database_dir_path / "meta.json").is_file():
        try:
            first_zip = next(database_dir_path.glob("*.zip"))
        except StopIteration:
            raise ValueError(f"Directory {database_dir_path} does not contain a Qdrant database")
        with zipfile.ZipFile(first_zip) as zip_file:
            zip_file.extractall(database_dir_path)

    return QdrantClient(path=str(database_dir_path))


def gather_documentation(
    *,
    documentation_dir: str,
    content_column_index: int,
    source_column_index: int,
    header_row_skip: int,
) -> pd.DataFrame:
    log.info("Gathering technical documentation...")

    documents = []
    for doc_path in Path(documentation_dir).iterdir():
        if not doc_path.is_file():
            continue
        log.info(f"Loading {doc_path.name}...")
        doc_frame = pd.read_csv(
            doc_path,
            names=[CONTENT_COLUMN, SOURCE_COLUMN],
            usecols=[content_column_index, source_column_index],
            skiprows=header_row_skip,
        )
        log.info(f"Preview:\n{doc_frame.head(3)}")
        documents.append(doc_frame)

    merged_documents = pd.concat(documents, ignore_index=True)

    msg = f"Gathered {len(merged_documents)} pieces of technical documentation"
    if len(documents) > 1:
        msg += f" across {len(documents)} files"
    log.info(msg)

    return merged_documents


def create_qdrant_collection(db_client: QdrantClient, provider: str = PROVIDER) -> None:
    log.info("Creating Qdrant collection for the embeddings...")
    collections = db_client.get_collections().collections
    if not any(collection.name == COLLECTION_NAME for collection in collections):
        if provider == "anthropic":
            embedding_length = ANTHROPIC_EMBEDDING_LENGTH
        else:
            embedding_length = EMBEDDING_LENGTH
            
        config = models.VectorParams(size=embedding_length, distance=models.Distance.COSINE)
        db_client.create_collection(collection_name=COLLECTION_NAME, vectors_config=config)


def vectorize_documents(db_client: QdrantClient, documents: pd.DataFrame, provider: str = PROVIDER) -> None:
    batch_size = 100
    
    if provider == "anthropic":
        embeddings = VoyageAIEmbeddings(
            model=ANTHROPIC_EMBEDDING_MODEL,
            batch_size=20,
        )
    else:  # default to openai
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            chunk_size=20,
        )
    batch_count = (len(documents) // batch_size) + 1

    log.info("Vectorizing documents...")
    for i in range(0, len(documents), batch_size):
        batch_number = i // batch_size + 1
        log.info(f"Vectorizing batch {batch_number}/{batch_count}")

        if valohai.config.is_running_in_valohai():
            progress = min(1.0, batch_number / batch_count)
            label = "Progress" if progress < 1.0 else "Complete"
            color = "blue" if progress < 1.0 else "green"
            gauge = {"type": "gauge", "label": label, "value": progress, "color": color}
            valohai.set_status_detail(json.dumps(gauge))

        batch = documents.iloc[i : i + batch_size]
        texts = batch[CONTENT_COLUMN].tolist()
        embedding_vectors = embeddings.embed_documents(texts)

        points = []
        for j, (_, row) in enumerate(batch.iterrows()):
            points.append(
                models.PointStruct(
                    id=i + j,
                    vector=embedding_vectors[j],
                    payload={
                        CONTENT_COLUMN: row[CONTENT_COLUMN],
                        SOURCE_COLUMN: row[SOURCE_COLUMN],
                    },
                )
            )

        db_client.upsert(collection_name=COLLECTION_NAME, points=points)


def package_database(database_dir: Path | str):
    log.info("Packaging database for upload...")

    database_dir_path = Path(database_dir)

    if not database_dir_path.is_dir():
        raise ValueError(f"Qdrant database directory {database_dir_path} does not exist.")

    timestamp = datetime.now().astimezone(timezone.utc).strftime("%Y%m%d-%H%M%S")
    zip_path = Path(valohai.outputs().path(f"doc-embeddings-{timestamp}.zip"))
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zip_file:
        for file_path in database_dir_path.rglob("*"):
            zip_file.write(file_path, file_path.relative_to(database_dir_path))

    log.info(f"Database packaged for upload: {zip_path}")
