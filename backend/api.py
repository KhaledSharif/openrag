from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")

app = FastAPI()


class DocumentType(str, Enum):
    pdf = "pdf"
    html = "html"
    txt = "txt"


class Document(BaseModel):
    id: int
    name: str  # filename or url
    type: DocumentType
    content: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "id": 1,
                    "name": "https://github.com/cyberbotics/webots",
                    "type": "html",
                    "content": "<html></html>",
                }
            ]
        }
    }


docs = [Document(id=1, name="dna.txt", content="DNA is cool", type=DocumentType.txt)]


@app.get("/docs/{doc_id}", response_model=Document)
async def get_document(doc_id: int):
    return docs[0]


@app.post("/docs", response_model=int)
async def create_document(doc_type: DocumentType):
    return docs[0].id

