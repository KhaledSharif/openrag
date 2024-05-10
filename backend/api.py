from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum

app = FastAPI()

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

class Document(BaseModel):
    name: str
    content: str | None = None

    model_config = {
        "json_schema_extra": {
            "examples": [{"name": "DNA", "content": "A very nice document"}]
        }
    }


docs = [Document(name="DNA", content="DNA is cool")]




@app.get("/docs/{doc_id}", response_model=Document)
async def get_document(doc_id: int):
    return docs[0]

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}
