from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import sys

# Import your sentiment function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.sentiment_prediction import perform_sentiment_analysis

app = FastAPI()

# Serve static files (like index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    return FileResponse("static/index.html")

# API input model
class TextInput(BaseModel):
    text: str

@app.post("/sentiment")
def analyze_sentiment(input: TextInput):
    result = perform_sentiment_analysis(reviews=[input.text])
    sentiment = result[0][0]
    return {"sentiment": sentiment["label"], "score": sentiment["score"]}
