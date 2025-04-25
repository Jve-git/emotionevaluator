import logging
import pandas as pd
from sentiment_prediction import (
    read_data,
    preprocessing_text,
    perform_sentiment_analysis,
    generate_output,
    benchmark,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info("Loading data...")
dataset = read_data("data/IMDB-movie-reviews.csv")
reviews = dataset.iloc[:, 0].tolist()
reviews = [preprocessing_text(review) for review in reviews]
labels = dataset.iloc[:, 1].tolist()
labels = [1 if label.lower() == "positive" else 0 for label in labels]
logging.info("Performing sentiment analysis...")
models = [
    "distilbert-base-uncased-finetuned-sst-2-english",  # Default model for sentiment-analysis pipeline
    "siebert/sentiment-roberta-large-english",  # Large RoBERTa model fine-tuned on diverse sentiment data High performance on general-purpose sentiment tasks
    "aychang/roberta-base-imdb",  # Specifically trained on the IMDB dataset
]

all_results = []
sentiments_list = []
pred_dict = {}
for model_name in models:
    pred_list = []
    logging.info(f"Evaluating {model_name}...")
    results, classifier, sentiments = perform_sentiment_analysis(model_name, reviews)
    sentiments_list.append(sentiments)
    # for the lift curve
    for result in results:
        if result["label"][:3].lower() == "pos":
            pred = result["score"]
        else:
            pred = 1 - result["score"]
        pred_list.append(pred)
    pred_dict[model_name] = pred_list
    generate_output(
        reviews, sentiments, model_name, f"output/output_{model_name.split('/')[0]}.csv"
    )

logging.info("Benchmarking...")
report = benchmark(models, reviews, labels, sentiments_list, pred_dict)
pd.DataFrame(report).to_markdown("output/benchmark_report.md", index=False)
