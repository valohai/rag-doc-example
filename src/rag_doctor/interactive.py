from pathlib import Path

from qdrant_client import QdrantClient

from rag_doctor.query import create_rag_chain, log


def start_chat(*, database_dir: str) -> None:
    greeting = "RAG Doctor is in office! Ask your technical questions or type 'quit' to exit"
    print()
    print(greeting)
    print("-" * len(greeting))

    # TODO: if database_dir doesn't look like a Qdrant directory, find the first zip and unzip it

    db_dir_path = Path(database_dir)
    client = QdrantClient(path=str(db_dir_path))

    rag_chain = create_rag_chain(client)

    while True:
        query = input("\nYou:\n> ").strip()
        if query.lower() == "quit":
            break
        if query:
            try:
                message = rag_chain(query)
                print("\nDoctor:")
                print(message.content)
            except Exception as e:
                log.exception(e)
