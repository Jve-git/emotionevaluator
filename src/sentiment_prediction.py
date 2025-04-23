import pandas as pd
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score
import re

def read_data(input_data) -> pd.DataFrame:
    '''
    Read the reviews and their respective sentiments.

    Args:
        input_data: the CSV file that you want to read.

    '''
    data = pd.read_csv(input_data, sep=";",  encoding='latin-1')
    return data

def preprocessing_text(text) -> str():
    # Remove unneeded whitespace(s)
    text = text.strip()

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)
    return text

def perform_sentiment_analysis(model_name, reviews):
    # Load sentiment analysis pipeline
    classifier = pipeline("sentiment-analysis", model_name)

    # Get predictions
    results = classifier(reviews, truncation=True)

    sentiments = [r['label'] for r in results]
    return sentiments

def generate_output(reviews, sentiments, model_name, filename="sentiment_output.csv") -> None:
    df = pd.DataFrame({
        "review": reviews,
        "sentiment": sentiments,
        "model": model_name
    })
    df.to_csv(filename, index=False)

def benchmark(models, reviews, labels):
    report = []
    for model_name in models:
        sentiments = perform_sentiment_analysis(model_name, reviews)
        preds = [1 if s.lower() == "positive" else 0 for s in sentiments]
        accuracy = sum([p == label for p, label in zip(preds, labels)]) / len(labels)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)
        report.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
    return report

# def create_curves(ROC)