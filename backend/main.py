"""HTTP API for training and querying the NumPy housing model."""

import json
import os
import threading

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from model.train_numpy import CATEGORICAL, DATA_PATH, MODEL_PATH, NUMERIC, train

MODEL_LOCK = threading.Lock()

app = FastAPI(title="Melbourne Housing ML API", version="1.0.0")
origins = [
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGIN", "http://localhost:5173,http://127.0.0.1:5173").split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class TrainingRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    loss: str = Field(default="mse", pattern="^(mse|mae|huber)$")
    method: str = Field(default="batch", pattern="^(batch|sgd)$")
    learning_rate: float = Field(default=0.03, gt=0, le=1)
    epochs: int = Field(default=300, ge=1, le=2000)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)

    Rooms: float = Field(gt=0, le=30)
    Distance: float = Field(ge=0, le=500)
    Bathroom: float = Field(ge=0, le=30)
    Car: float = Field(ge=0, le=30)
    Type: str = Field(min_length=1, max_length=80)
    Region: str = Field(min_length=1, max_length=120)


def load_model() -> dict:
    try:
        return json.loads(MODEL_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail="Train the model before requesting a prediction.") from exc
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=500, detail="The trained model artifact could not be read.") from exc


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/options")
def options() -> dict:
    categories = {column: set() for column in CATEGORICAL}
    try:
        with DATA_PATH.open(newline="", encoding="utf-8-sig") as file:
            import csv

            for row in csv.DictReader(file):
                for column in CATEGORICAL:
                    categories[column].add(row[column])
    except OSError as exc:
        raise HTTPException(status_code=500, detail="The housing dataset could not be read.") from exc

    labels = {"h": "House", "t": "Townhouse", "u": "Unit / apartment"}
    return {
        "regions": sorted(categories["Region"]),
        "property_types": [
            {"value": value, "label": labels.get(value, value)}
            for value in sorted(categories["Type"])
        ],
        "rows": sum(1 for _ in DATA_PATH.open(encoding="utf-8-sig")) - 1,
        "model_ready": MODEL_PATH.exists(),
        "numeric_features": NUMERIC,
    }


@app.post("/api/train")
def train_model(request: TrainingRequest) -> dict:
    try:
        with MODEL_LOCK:
            result = train(
                loss=request.loss,
                method=request.method,
                learning_rate=request.learning_rate,
                epochs=request.epochs,
            )
    except (ValueError, OSError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    history = result["history"]
    return {
        "loss": result["loss"],
        "method": result["method"],
        "epochs": len(history),
        "rows_trained": result["rows_trained"],
        "first_epoch_loss": history[0],
        "final_loss": history[-1],
        "history": history,
    }


@app.post("/api/predict")
def predict(request: PredictionRequest) -> dict:
    values = request.model_dump()
    with MODEL_LOCK:
        artifact = load_model()
        categories = artifact["categories"]
        for column in CATEGORICAL:
            if values[column] not in categories[column]:
                raise HTTPException(status_code=422, detail=f"Unknown {column.lower()} value.")

        numeric = np.array([values[column] for column in NUMERIC], dtype=float)
        scaled_numeric = (numeric - np.asarray(artifact["numeric_mean"])) / np.asarray(artifact["numeric_scale"])
        encoded = np.array(
            [float(values[column] == category) for column in CATEGORICAL for category in categories[column]],
            dtype=float,
        )
        features = np.concatenate((scaled_numeric, encoded))
        standardized_prediction = float(features @ np.asarray(artifact["weights"]) + artifact["bias"])
        price = standardized_prediction * artifact["target_scale"] + artifact["target_mean"]

    return {"price": round(float(price), 2), "currency": "AUD", "loss": artifact["loss"], "method": artifact["method"]}
