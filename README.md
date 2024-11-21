# 🧑‍⚕️ RAG Doctor - Technical Documentation Assistant

This project implements a Retrieval-Augmented Generation (RAG) system for querying technical
documentation.

## Overview

- Creates a Qdrant vector database for embeddings from the given CSV file(s)
- Provides an interactive interface for querying the documentation using natural language
- Uses OpenAI's embeddings for fast similarity search and GPT models for high-quality responses
- Each query retrieves the 3 most relevant documentation snippets for context
- Includes source links in responses for reference
