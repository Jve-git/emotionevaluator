import sys
import os
import logging

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)  # to be able to include src

from fastapi import FastAPI
from pydantic import BaseModel
from src.sentiment_prediction import perform_sentiment_analysis

app = FastAPI()


# Define input model using Pydantic
class TextInput(BaseModel):
    text: str


@app.post("/sentiment")
def analyze_sentiment(input: TextInput):
    result = perform_sentiment_analysis(reviews=[input.text])
    sentiment = result[0][0]
    return {"sentiment": sentiment["label"], "score": sentiment["score"]}
