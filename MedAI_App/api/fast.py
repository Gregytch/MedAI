import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing import List, Dict, Any


app = FastAPI()


# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
#app.state.model = load_model()

@app.get("/")
def root():
    return {
    'greeting': 'Hello, this is the first test'
    }

class DataRow(BaseModel):
    data: List[Dict[str, Any]] = Field(..., description="A list of dictionaries, where each dictionary represents a row with column names as keys and numeric values.")

@app.post('/predict_disease')
def get_prediction(row: DataRow):
    df = pd.DataFrame(row.model_dump()["data"])

# Do some processing (modify as needed)

    res1 = df.sum(axis=1, numeric_only=True)
    res2 = df.mean(axis=1, numeric_only=True)
    return {"result1": f"Here is prediction 1: {res1[0]}", "result2": f"Here is prediction2: {res2[0]}"}
