from __future__ import annotations

import logging
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import valohai.config
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http import models

from rag_doctor.consts import (
    EMBEDDING_MODEL,
    EMBEDDING_LENGTH,
    CONTENT_COLUMN,
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
):
    documents = gather_documentation(
        documentation_dir=documentation_dir,
        content_column_index=content_column_index,
        source_column_index=source_column_index,
        header_row_skip=header_row_skip,
    )

    Path(database_dir).mkdir(exist_ok=True)
    db_client = QdrantClient(path=database_dir)

    create_qdrant_collection(db_client)
    vectorize_documents(db_client, documents)

    if valohai.config.is_running_in_valohai():
        package_database(database_dir)

    log.info("Vector database creation complete!")


def get_qdrant_client(database_dir: Path | str) -> QdrantClient:
    database_dir_path = Path(database_dir)

    if not database_dir_path.is_dir():
        raise ValueError(f"Database directory {database_dir_path} does not exist.")

    # TODO: if database_dir_path doesn't look like a Qdrant directory, find the first zip and unzip it

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


def create_qdrant_collection(db_client: QdrantClient) -> None:
    log.info("Creating Qdrant collection for the embeddings...")
    collections = db_client.get_collections().collections
    if not any(collection.name == COLLECTION_NAME for collection in collections):
        config = models.VectorParams(size=EMBEDDING_LENGTH, distance=models.Distance.COSINE)
        db_client.create_collection(collection_name=COLLECTION_NAME, vectors_config=config)


def vectorize_documents(db_client: QdrantClient, documents: pd.DataFrame) -> None:
    batch_size = 100
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    batch_count = (len(documents) // batch_size) + 1

    log.info("Vectorizing documents...")
    for i in range(0, len(documents), batch_size):
        batch_number = i // batch_size + 1
        log.info(f"Vectorizing batch {batch_number}/{batch_count}")

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
