from typing import Annotated

from fastapi import FastAPI, HTTPException, Form
from pydantic import BaseModel
from qdrant_client import QdrantClient

from rag_doctor.consts import PROVIDER
from rag_doctor.query import create_rag_chain


class ResponsePayload(BaseModel):
    answer: str


def create_app(db_client: QdrantClient, provider: str = PROVIDER) -> FastAPI:
    app = FastAPI(title="RAG Doctor API")
    rag_chain = create_rag_chain(db_client, provider)

    @app.post("/{full_path:path}", response_model=ResponsePayload)
    async def solo_handler(full_path: str, question: Annotated[str, Form()]) -> ResponsePayload:
        message = rag_chain(question)

        if not message or not message.content:
            raise HTTPException(status_code=500, detail="No response from the model")

        return ResponsePayload(answer=message.content)

    return app
