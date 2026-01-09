import logging
from dataclasses import dataclass
from typing import Callable, List

import tiktoken
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
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
    SOURCE_COLUMN,
)

log = logging.getLogger(__name__)


@dataclass
class ProviderConfig:
    model: BaseChatModel
    max_tokens: int
    count_tokens: Callable[[str], int]
    truncate: Callable[[str, int], str]


def get_provider_config(
    provider: str,
) -> ProviderConfig:
    """Return (model, max_tokens, count_tokens_fn, truncate_fn) for the given provider."""

    if provider == "anthropic":
        model = ChatAnthropic(model=ANTHROPIC_PROMPT_MODEL, temperature=0)
        max_tokens = ANTHROPIC_PROMPT_MAX_TOKENS

        # Rough token count estimate for Anthropic models: 4 characters ≈ 1 token.
        # This is a heuristic and may not be accurate for all text types.
        def count_tokens(text: str) -> int:
            return len(text) // 4

        def truncate(text: str, limit: int) -> str:
            max_chars = limit * 4
            return text[:max_chars] if len(text) > max_chars else text

    elif provider == "openai":
        model = ChatOpenAI(model=PROMPT_MODEL, temperature=0)
        max_tokens = PROMPT_MAX_TOKENS
        encoder = tiktoken.encoding_for_model(model_name=PROMPT_MODEL)

        def count_tokens(text: str) -> int:
            return len(encoder.encode(text))

        def truncate(text: str, limit: int) -> str:
            tokenizer = Tokenizer(
                chunk_overlap=0,
                decode=encoder.decode,
                encode=lambda t: encoder.encode(t),
                tokens_per_chunk=limit,
            )
            return split_text_on_tokens(text=text, tokenizer=tokenizer)[0]
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return ProviderConfig(model, max_tokens, count_tokens, truncate)


def create_rag_chain(
    db_client: QdrantClient,
    provider: str,
) -> Callable[[str], tuple[BaseMessage, list[str]]]:
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    config = get_provider_config(provider)

    def retrieve_related_documents(query: str) -> tuple[list[Document], list[str]]:
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
    prompt_chain = prompt | config.model

    template_token_count = config.count_tokens(template.format(context="", question=""))

    def rag_chain(question: str) -> tuple[BaseMessage, List[str]]:
        documents, retrieved_contents = retrieve_related_documents(question)

        remaining_tokens = config.max_tokens
        log.debug(f"tokens at the start:    {remaining_tokens}")

        remaining_tokens -= template_token_count
        log.debug(f"tokens after template:  {remaining_tokens}")

        sources: list[str] = [doc.metadata[SOURCE_COLUMN] for doc in documents]
        sources_bullet_points = "\n".join([f"- Source: {source}" for source in sources])
        remaining_tokens -= config.count_tokens(sources_bullet_points)
        log.debug(f"tokens after sources:   {remaining_tokens}")

        remaining_tokens -= config.count_tokens(question)
        log.debug(f"tokens after question:  {remaining_tokens}")

        separator = "\n\n"
        remaining_tokens -= config.count_tokens(separator)
        log.debug(f"tokens after separator: {remaining_tokens}")

        documentation = "\n\n".join(doc.page_content for doc in documents)
        truncated_content = config.truncate(documentation, remaining_tokens)
        remaining_tokens -= config.count_tokens(truncated_content)
        log.debug(f"tokens after docs:      {remaining_tokens}")

        context = f"{truncated_content}{separator}{sources_bullet_points}"
        message = prompt_chain.invoke(input={"context": context, "question": question})
        return message, retrieved_contents

    return rag_chain
