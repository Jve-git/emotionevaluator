import pandas as pd
from sentiment_prediction import read_data, preprocessing_text, perform_sentiment_analysis, generate_output, benchmark

print("Loading data...")
dataset = read_data("data/IMDB-movie-reviews.csv")
reviews = dataset.iloc[:, 0].tolist()
reviews = [preprocessing_text(review) for review in reviews]
labels = dataset.iloc[:, 1].tolist()
labels = [1 if label.lower() == "positive" else 0 for label in labels]

print("Performing sentiment analysis...")
models = [
        "distilbert-base-uncased-finetuned-sst-2-english", # Default model for sentiment-analysis pipeline
        "siebert/sentiment-roberta-large-english", # Large RoBERTa model fine-tuned on diverse sentiment data High performance on general-purpose sentiment tasks
        "aychang/roberta-base-imdb" # Specifically trained on the IMDB dataset
    ]

all_results = []
sentiments_list = []
for model_name in models:
    print(f"Evaluating {model_name}...")
    results, classifier, sentiments = perform_sentiment_analysis(model_name, reviews)
    sentiments_list.append(sentiments)
    generate_output(reviews, sentiments, model_name, f"output/output_{model_name.split('/')[0]}.csv")

print("Benchmarking...")
report = benchmark(models, reviews, labels, sentiments_list)
pd.DataFrame(report).to_markdown("output/benchmark_report.md", index=False)