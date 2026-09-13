# Script to train machine learning model.

import pickle
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from starter.starter.ml.data import process_data
from starter.starter.ml.model import (
    compute_model_metrics,
    inference,
    train_model,
)

DATA_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "census.csv"
)
MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "model" / "census_model.pkl"
)
SLICE_OUTPUT_PATH = Path(__file__).resolve().parents[2] / "slice_output.txt"

# Clean spaces from column names and values in the CSV.
data = pd.read_csv(DATA_PATH)
data.columns = [column.strip() for column in data.columns]
for column in data.columns:
    if pd.api.types.is_object_dtype(data[column]):
        data[column] = data[column].str.strip()

# Optional enhancement: use K-fold cross validation instead of a split.
train, test = train_test_split(
    data,
    test_size=0.20,
    random_state=42,
)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True,
)

X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
)

model = train_model(X_train, y_train)

# Save the trained model, encoder, and label binarizer to the API artifact.
MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
with MODEL_PATH.open("wb") as f:
    pickle.dump({"model": model, "encoder": encoder, "lb": lb}, f)

# Evaluate each category in the held-out data and save the results.
with SLICE_OUTPUT_PATH.open("w") as f:
    for feature in cat_features:
        f.write(f"Feature: {feature}\n")
        for value in sorted(test[feature].unique()):
            mask = test[feature].to_numpy() == value
            precision, recall, fbeta = compute_model_metrics(
                y_test[mask], inference(model, X_test[mask])
            )
            f.write(
                f"{value}: precision={precision:.4f}, "
                f"recall={recall:.4f}, f1={fbeta:.4f}\n"
            )
