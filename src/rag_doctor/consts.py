EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_LENGTH = 1_536  # the dimensions of a "text-embedding-ada-002" embedding vector

PROMPT_MODEL = "gpt-4o-mini"
PROMPT_MAX_TOKENS = 128_000  # model "context window" from https://platform.openai.com/docs/models

COLLECTION_NAME = "docs"
CONTENT_COLUMN = "content"
SOURCE_COLUMN = "source"

PROVIDER = "openai"  

ANTHROPIC_PROMPT_MODEL = "claude-3-5-sonnet-20241022"
ANTHROPIC_PROMPT_MAX_TOKENS = 200_000