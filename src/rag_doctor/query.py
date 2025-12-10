import logging
from typing import Callable, List

import tiktoken
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import Tokenizer, split_text_on_tokens
from qdrant_client import QdrantClient

from rag_doctor.consts import (
    ANTHROPIC_PROMPT_MAX_TOKENS,
    ANTHROPIC_PROMPT_MODEL,
    COLLECTION_NAME,
    CONTENT_COLUMN,
    EMBEDDING_MODEL,
    PROMPT_MAX_TOKENS,
    PROMPT_MODEL,
    PROVIDER,
    SOURCE_COLUMN,
)

log = logging.getLogger(__name__)


def create_rag_chain(
    db_client: QdrantClient,
    provider: str = PROVIDER,
) -> Callable[[str], BaseMessage]:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    if provider == "anthropic":
        prompt_model = ChatAnthropic(model=ANTHROPIC_PROMPT_MODEL, temperature=0)
        max_tokens = ANTHROPIC_PROMPT_MAX_TOKENS
        model_name = ANTHROPIC_PROMPT_MODEL
    else:
        prompt_model = ChatOpenAI(model=PROMPT_MODEL, temperature=0)
        max_tokens = PROMPT_MAX_TOKENS
        model_name = PROMPT_MODEL

    def retrieve_related_documents(query: str) -> List[Document]:
        query_vector = embeddings.embed_query(query)
        results = db_client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=3,
        )
        documents = []
        retrieved_contents = []
        for result in results:
            documents.append(
                Document(
                    page_content=result.payload[CONTENT_COLUMN],
                    metadata={SOURCE_COLUMN: result.payload[SOURCE_COLUMN]},
                ),
            )
            retrieved_contents.append(result.payload[CONTENT_COLUMN])
        return documents, retrieved_contents

    template = """You are a helpful AI assistant that answers questions about technical documentation.
    Use the following documentation excerpts to answer the question. If you don't know the answer,
    just say you don't know. Include relevant sources in your answer and make sure they are full URLs.

    Documentation excerpts:
    {context}

    Question: {question}

    Answer: """

    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    prompt_chain = prompt | prompt_model

    if provider == "anthropic":

        def count_tokens(text: str) -> int:
            return len(text) // 4
    else:
        token_encoder = tiktoken.encoding_for_model(model_name=model_name)

        def count_tokens(text: str) -> int:
            return len(token_encoder.encode(text))

    template_token_count = count_tokens(template.format(context="", question=""))

    def rag_chain(question: str) -> tuple[BaseMessage, List[int]]:
        documents, retrieved_contents = retrieve_related_documents(question)

        remaining_tokens = max_tokens
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

        if provider == "anthropic":
            max_chars = remaining_tokens * 4
            if len(documentation) > max_chars:
                truncated_content = documentation[:max_chars]
            else:
                truncated_content = documentation
        else:
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
        return message, retrieved_contents

    return rag_chain
