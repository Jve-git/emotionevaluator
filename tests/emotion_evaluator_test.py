import pytest
import pandas as pd
import os
import sys

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from src.sentiment_prediction import (
    read_data,
    preprocessing_text,
    perform_sentiment_analysis,
    generate_output,
    benchmark,
)
except ImportError as e:
    raise ImportError(f"Error importing sentiment_prediction file: {e}")

# --- Fixtures ---

@pytest.fixture
def sample_csv(tmp_path):
    file = tmp_path / "sample.csv"
    file.write_text("review;label\nGreat movie!;lpositive\nTerrible film!;lnegative\n")
    return str(file)

@pytest.fixture
def sample_reviews():
    return ["I loved it!", "Worst ever."]

@pytest.fixture
def sample_labels():
    return ["positive", "negative"]

@pytest.fixture
def dummy_sentiments_list():
    return [["positive", "negative"], ["positive", "positive"], ["negative", "negative"]]

# --- Tests ---

def test_read_data_valid(sample_csv):
    df = read_data(sample_csv)
    assert not df.empty
    assert "review" in df.columns
    assert "label" in df.columns

def test_preprocessing_text_basic():
    assert preprocessing_text("  Hello!!! ") == "Hello!!!"
    assert preprocessing_text("") == ""

def test_perform_sentiment_analysis_output_length(sample_reviews):
    model = "distilbert-base-uncased-finetuned-sst-2-english"
    results, classifier, sentiments = perform_sentiment_analysis(model, sample_reviews)
    assert len(sentiments) == len(sample_reviews)
    assert isinstance(sentiments[0], str)

def test_generate_output_creates_file(tmp_path):
    file_path = tmp_path / "output.csv"
    generate_output(["Test review"], ["positive"], "test_model", str(file_path))
    assert file_path.exists()
    df = pd.read_csv(file_path)
    assert "review" in df.columns
    assert "sentiment" in df.columns