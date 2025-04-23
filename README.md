# emotionevaluator
This repository serves as a sentiment evaluator for text, classifying it as either positive or negative.

It makes used of three different pretrained models (distilbert-base-uncased-finetuned-sst-2-english, siebert/sentiment-roberta-large-english and aychang/roberta-base-imdb) and compares these models against each other by making use of different metrics such as accuracy, precision and recall.

For each of the models predictions, Shapley values are calculated in order to explain which features contibute towards the models' predictions.