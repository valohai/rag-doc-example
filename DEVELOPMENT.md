# Development Documentation

## Setup

1. Install `uv`:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install all development dependencies to a new virtual environment:

   ```bash
   uv sync --group lint
   ```

3. Set up your environment variables:

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` and add e.g. your OpenAI API key.

## Usage

Create the vector database and fill it with the text embeddings for search:

```bash
uv run rag-doctor create-database
```

Start an interactive session to query the documentation:

```bash
uv run rag-doctor chat
```
