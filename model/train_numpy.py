import argparse
import csv
import json
from pathlib import Path

import numpy as np

from model.linear_regression import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Resources" / "clean_melbourne_housing.csv"
MODEL_PATH = ROOT / "model" / "linear_regression_numpy.json"
NUMERIC = ["Rooms", "Distance", "Bathroom", "Car"]
CATEGORICAL = ["Type", "Region"]


def train(data_path=DATA_PATH, output=MODEL_PATH, loss="mse", method="batch", learning_rate=0.03, epochs=500):
    with open(data_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = set(NUMERIC + CATEGORICAL + ["Price"])
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset is missing columns: {', '.join(sorted(missing))}")
        rows = list(reader)
    if len(rows) < 2:
        raise ValueError("Dataset must contain at least two rows")

    numeric = np.array([[float(row[column]) for column in NUMERIC] for row in rows])
    numeric_mean, numeric_scale = numeric.mean(axis=0), numeric.std(axis=0)
    numeric_scale[numeric_scale == 0] = 1
    categories = {column: sorted({row[column] for row in rows}) for column in CATEGORICAL}
    encoded = np.array([[float(row[column] == category) for column in CATEGORICAL for category in categories[column]] for row in rows])
    X = np.column_stack(((numeric - numeric_mean) / numeric_scale, encoded))
    target = np.array([float(row["Price"]) for row in rows])
    if not np.isfinite(X).all() or not np.isfinite(target).all() or (target <= 0).any():
        raise ValueError("Dataset features and positive Price targets must be finite numeric values")
    target_mean, target_scale = float(target.mean()), float(target.std()) or 1.0
    model = LinearRegression().fit(X, (target - target_mean) / target_scale, loss=loss, method=method, learning_rate=learning_rate, epochs=epochs)

    artifact = {
        "numeric_features": NUMERIC,
        "categorical_features": CATEGORICAL,
        "categories": categories,
        "numeric_mean": numeric_mean.tolist(),
        "numeric_scale": numeric_scale.tolist(),
        "weights": model.weights.tolist(),
        "bias": model.bias,
        "target_mean": target_mean,
        "target_scale": target_scale,
        "loss": loss,
        "method": method,
        "history": model.history,
        "rows_trained": len(rows),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact), encoding="utf-8")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NumPy linear regression on the repository's cleaned Melbourne dataset")
    parser.add_argument("--loss", choices=("mse", "mae", "huber"), default="mse")
    parser.add_argument("--method", choices=("batch", "sgd"), default="batch")
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()
    result = train(loss=args.loss, method=args.method, learning_rate=args.learning_rate, epochs=args.epochs)
    print(f"Trained on {result['rows_trained']:,} rows; final standardized loss: {result['history'][-1]:.5f}")
