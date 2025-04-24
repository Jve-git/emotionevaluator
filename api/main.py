from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import os
import sys

# Import your sentiment function
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.sentiment_prediction import perform_sentiment_analysis
except ImportError as e:
    raise ImportError(f"Error importing sentiment analysis function: {e}")

app = FastAPI()

# Serve static files (like index.html)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_index():
    try:
        return FileResponse("static/index.html")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading index page: {e}")

# API input model
class TextInput(BaseModel):
    text: str

@app.post("/sentiment")
def analyze_sentiment(input: TextInput):
    try:
        result = perform_sentiment_analysis(reviews=[input.text])
        sentiment = result[0][0]
        return {"sentiment": sentiment["label"], "score": sentiment["score"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {e}")
