from qdrant_client import QdrantClient

from rag_doctor.consts import PROVIDER
from rag_doctor.query import create_rag_chain, log


def start_chat(*, db_client: QdrantClient, provider: str = PROVIDER) -> None:
    greeting = "🥼️ RAG Doctor is in office! Ask your technical questions or type 'quit' to exit"
    print()
    print(greeting)
    print("-" * len(greeting))

    rag_chain = create_rag_chain(db_client, provider=provider)
    while True:
        query = input("\n👤 You:\n> ").strip()
        if query.lower() == "quit":
            break
        if query:
            try:
                message, _ = rag_chain(query)
                print("\n🥼 Doctor:")
                print(message.content)
            except Exception as e:
                log.exception(e)
