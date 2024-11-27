# Retrieval-Augmented Generation with Documentation

#### <i>aka. RAG Doctor</i> 🧑‍⚕️

This repository showcases a Retrieval-Augmented Generation (**RAG**) system for interacting with
documentation that uses natural language queries to retrieve and summarize relevant information.

[](https://github.com/user-attachments/assets/462ee24c-6798-42b3-8dea-80f37cb49e5f)

[valohai-website]: https://valohai.com/

[valohai-app]: https://app.valohai.com/

[valohai-templates]: https://app.valohai.com/projects/import-tutorial/

[qdrant-website]: https://qdrant.tech/

[openai-platform]: https://platform.openai.com/

[openai-api-keys]: https://platform.openai.com/docs/quickstart#create-and-export-an-api-key

## Overview

- Creates a [Qdrant](qdrant-website) vector database for embeddings from the given CSV file(s)
    - _The vector database is used for fast similarity search to find document snippets_
    - _We use a CSV based on Hugging Face documentation as an example_
- Uses OpenAI's embeddings for fast similarity search and GPT models for high-quality responses
- Provides an interactive interface for querying the documentation using natural language
- Each query retrieves the most relevant documentation snippets for context
- Answers include source links for reference

## Prerequisites

- [Valohai](valohai-website) account to run the pipelines
- [OpenAI](openai-platform) account to use their APIs
- Less than $5 in OpenAI credits

## Setup

If you can't find this project in your [Valohai Templates](valohai-templates), you can set it up
manually:

1. Create a new project on [Valohai](valohai-app)
2. Set the project repository to: `https://github.com/valohai/rag-doc-example`

   ![](.github/repository-setup.png)

3. Save the settings and click "Fetch Repository"

   ![](.github/repository-fetch.png)</br>
   This makes sure the project is up to date

4. 🔑 [Create an OpenAI API key](openai-api-keys) for this project
    - We will need the API key next so record it down

5. Assign the API key to this project:

   ![](.github/project-api-key.png)</br>
   You will see _✅ Changes to OPENAI_API_KEY saved_ if everything went correctly.

And now you are ready to run the pipelines!

## Usage

![](.github/pipeline-create.png)

1. Navigate to the "Pipelines" tab
2. Click the "Create Pipeline" button
3. Select the "assistant-pipeline" pipeline template
4. Click the "Create pipeline from template" button
5. Feel free to look around and finally click the "Create pipeline" button

This will start the pipeline:

![](.github/pipeline-pending.png)</br>
_Feel free to explore around while it runs._

When it finishes, the last step will contain qualitative results to review:

![](.github/query-output.png)</br>
_This manual evaluation is a simplification how to validate the quality of the generated
responses. "LLM evals" is a large topic outside the scope of this particular example._

Now you have a mini-pipeline that maintains a RAG vector database and allows you to ask questions
about the documentation. You can ask your own questions by creating new executions based on the
"do-query" step.

## Next Steps

### Automatic Deployment

The repository also contains a pipeline "assistant-pipeline-with-deployment" which deploys the RAG
system to an HTTP endpoint after a manual human validation of the "manual-evaluation" pipeline step.

<details>
<summary>🤩 Show Me!</summary>

1. Create a Valohai Deployment to tell where the HTTP endpoint should be hosted:

   ![](.github/deployment-create.png)</br>
   _You can use **Valohai Public Cloud** and **valohai.cloud** as the target when trialing out. Make
   sure to name the deployment `public`_

2. Create a pipeline as we did before, but use the "assistant-pipeline-with-deployment" template.

   ![](.github/deployment-pipeline.png)</br>
   _The pipeline should look something like this._

3. The pipeline will halt to a "⏳️ Pending Approval" state, where you can click the "Approve" button
   to proceed.

   ![](.github/deployment-approval.png)

4. After approval, the pipeline will build and deploy the endpoint.

   ![](.github/deployment-running.png)

5. You can use the "Test Deployment" button to run a test queries against the endpoint.

   ![](.github/deployment-test-query.png)
   ![](.github/deployment-test-result.png)

</details>

### Using Other Models

This example uses OpenAI for both the embedding and query models.

Either could be changed to a different provider or a local model.

<details>
<summary>🤩 Show Me!</summary>

Changing models inside the OpenAI ecosystem is a matter of changing constants in
`src/rag_doctor/consts.py`:

```python
EMBEDDING_MODEL = "text-embedding-ada-002"
EMBEDDING_LENGTH = 1_536  # the dimensions of a "text-embedding-ada-002" embedding vector

PROMPT_MODEL = "gpt-4o-mini"
PROMPT_MAX_TOKENS = 128_000  # model "context window" from https://platform.openai.com/docs/models
```

Further modifying the chat model involves reimplementing the query logic in
`src/rag_doctor/query.py`.

Similarly, modifying the embedding model is a matter of reimplementing the embedding logic in both
`src/rag_doctor/database.py` and `src/rag_doctor/query.py`.

**If you decide to change the embedding model, remember to recreate the vector database.**

</details>

### Using Your Own Documentation

You can take a look at the input file given to the "embedding" node and create a similar CSV from
your own documentation and replace the input with that CSV.
