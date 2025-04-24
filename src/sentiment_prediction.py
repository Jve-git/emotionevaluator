import pandas as pd
from transformers import pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import re
import numpy as np
import matplotlib.pyplot as plt
import time

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

    # Start timing
    start_time = time.time()

    # Get predictions
    results = classifier(reviews, truncation=True)

    # End timing
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"[{model_name}] Inference Time: {inference_time:.2f} seconds for {len(reviews)} reviews")

    sentiments = [r['label'] for r in results]
    return results, classifier, sentiments

def generate_output(reviews, sentiments, model_name, filename="sentiment_output.csv") -> None:
    df = pd.DataFrame({
        "review": reviews,
        "sentiment": sentiments,
        "model": model_name
    })
    df.to_csv(filename, index=False)

def benchmark(models, reviews, labels, sentiments_list):
    report = []
    for model_name, sentiments in zip(models, sentiments_list):
        # sentiments = perform_sentiment_analysis(model_name, reviews)
        preds = [1 if s.lower()[:3] == "pos" else 0 for s in sentiments]

        # accuracy
        accuracy = sum([p == label for p, label in zip(preds, labels)]) / len(labels)
        # precision
        precision = precision_score(labels, preds)
        # recall
        recall = recall_score(labels, preds)
        # f1 score
        f1 = f1_score(labels, preds)
        report.append({
            "model": model_name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })
        # confusion matrix
        cm = confusion_matrix(labels, preds)
        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1])
        cm_display.plot(cmap='YlOrRd') # Colours of Sopra Steria
        plt.savefig(f"output/confusion_matrix_{model_name.split('/')[0]}.png")
        plt.close()
        # lift curve
        
        sorted_indices = np.argsort(preds)[::-1]
        labels_array = np.array(labels) 
        labels_sorted = labels_array[sorted_indices]
        # preds_sorted = preds[sorted_indices]

        cumulative_positives = np.cumsum(labels_sorted)
        percentage_of_population = np.arange(1, len(labels_sorted) + 1) / len(labels_sorted)
        lift = cumulative_positives / (np.sum(labels_sorted) * percentage_of_population)

        plt.figure(figsize=(10, 6))
        plt.plot(percentage_of_population, lift, label="Lift Curve", color='orange')
        plt.plot([0, 1], [1, 1], 'k--', label="Baseline (Random Model)")
        plt.xlabel("Percentage of Sample")
        plt.ylabel("Lift")
        plt.title("Lift Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"output/lift_curve_{model_name.split('/')[0]}.png")
        plt.close()
    return report