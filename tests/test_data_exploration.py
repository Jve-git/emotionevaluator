import pandas as pd

reviews = pd.read_csv("data/IMDB-movie-reviews.csv", sep=";", encoding="latin-1")
first_five = reviews.head(5)

print(f"first column: {first_five.iloc[:, 0]}")
print(f"second column: {first_five.iloc[:, 1]}")

num_rows = len(reviews)
print(f"total number of rows is {num_rows}")

second_col = reviews.iloc[:, 1]
value_counts = second_col.value_counts()
print(value_counts)


def test_dummy():
    assert True
