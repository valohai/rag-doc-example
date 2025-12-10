import logging
from typing import List, Callable

import tiktoken
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.messages import BaseMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import Tokenizer, split_text_on_tokens
from qdrant_client import QdrantClient

from rag_doctor.consts import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    PROMPT_MODEL,
    CONTENT_COLUMN,
    SOURCE_COLUMN,
    PROMPT_MAX_TOKENS,
)

log = logging.getLogger(__name__)


def create_rag_chain(db_client: QdrantClient) -> Callable[[str], BaseMessage]:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    def retrieve_related_documents(query: str) -> List[Document]:
        query_vector = embeddings.embed_query(query)
        results = db_client.search(
            collection_name=COLLECTION_NAME, query_vector=query_vector, limit=3
        )
        documents = []
        for result in results:
            documents.append(
                Document(
                    page_content=result.payload[CONTENT_COLUMN],
                    metadata={SOURCE_COLUMN: result.payload[SOURCE_COLUMN]},
                )
            )
        return documents

    template = """You are a helpful AI assistant that answers questions about technical documentation.
    Use the following documentation excerpts to answer the question. If you don't know the answer,
    just say you don't know. Include relevant sources in your answer and make sure they are full URLs.

    Documentation excerpts:
    {context}

    Question: {question}

    Answer: """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    prompt_model = ChatOpenAI(model=PROMPT_MODEL, temperature=0)
    prompt_chain = prompt | prompt_model

    token_encoder = tiktoken.encoding_for_model(model_name=PROMPT_MODEL)

    def count_tokens(text: str) -> int:
        return len(token_encoder.encode(text))

    template_token_count = count_tokens(template.format(context="", question=""))

    def rag_chain(question: str) -> BaseMessage:
        documents = retrieve_related_documents(question)

        remaining_tokens = PROMPT_MAX_TOKENS
        log.debug(f"tokens at the start:    {remaining_tokens}")

        remaining_tokens -= template_token_count
        log.debug(f"tokens after template:  {remaining_tokens}")

        sources: list[str] = [doc.metadata[SOURCE_COLUMN] for doc in documents]
        sources_bullet_points = "\n".join([f"- Source: {source}" for source in sources])
        remaining_tokens -= count_tokens(sources_bullet_points)
        log.debug(f"tokens after sources:   {remaining_tokens}")

        remaining_tokens -= count_tokens(question)
        log.debug(f"tokens after question:  {remaining_tokens}")

        separator = "\n\n"
        remaining_tokens -= count_tokens(separator)
        log.debug(f"tokens after separator: {remaining_tokens}")

        documentation = "\n\n".join(doc.page_content for doc in documents)
        tokenizer = Tokenizer(
            chunk_overlap=0,
            decode=token_encoder.decode,
            encode=lambda text: token_encoder.encode(text),
            tokens_per_chunk=remaining_tokens,
        )

        try:
            truncated_content = split_text_on_tokens(text=documentation, tokenizer=tokenizer)[0]
        except IndexError:
            log.exception("Failed to truncated content, skip augmentation:")
            truncated_content = ""

        remaining_tokens -= count_tokens(truncated_content)
        log.debug(f"tokens after docs:      {remaining_tokens}")

        context = f"{truncated_content}{separator}{sources_bullet_points}"
        message = prompt_chain.invoke(input={"context": context, "question": question})
        return message

    return rag_chain
