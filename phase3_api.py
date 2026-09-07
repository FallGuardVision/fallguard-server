"""
Phase 3 -- Single-window inference endpoint.

Run: uvicorn phase3_api:app --reload
Test: POST a (30,132) window as JSON to /predict

This does NOT do streaming/buffering yet -- just proves the model can be
served correctly over HTTP with the right preprocessing.
"""

import numpy as np
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

MODEL_PATH = r"Z:\College\FallDetectionFYP\models\bilstm_v2\fall_detection_bilstm_v2.keras"
MEAN_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels\feat_mean_v2.npy"
STD_PATH = r"Z:\College\FallDetectionFYP\processed_data\labels\feat_std_v2.npy"

CLASS_NAMES = ["STABLE", "HIGH_RISK", "FALL", "IMMOBILE"]

app = FastAPI()

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
feat_mean = np.load(MEAN_PATH)
feat_std = np.load(STD_PATH)
print("Model loaded.")


def preprocess_window(window_flat_132: np.ndarray) -> np.ndarray:
    T = window_flat_132.shape[0]
    reshaped = window_flat_132.reshape(T, 33, 4)
    normalized = (reshaped - feat_mean) / feat_std
    return normalized.reshape(T, 132)


class WindowRequest(BaseModel):
    # window: 30 frames x 132 flattened keypoint values, raw (unnormalized)
    window: List[List[float]]


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    class_probabilities: dict


@app.get("/")
def health_check():
    return {"status": "ok", "model_input_shape": str(model.input_shape)}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: WindowRequest):
    window = np.array(req.window, dtype=np.float32)  # (30, 132)

    if window.shape != (30, 132):
        return {
            "predicted_class": "ERROR",
            "confidence": 0.0,
            "class_probabilities": {
                "error": f"expected shape (30,132), got {window.shape}"
            },
        }

    normalized = preprocess_window(window)
    batch = np.expand_dims(normalized, axis=0)  # (1, 30, 132)
    pred = model.predict(batch, verbose=0)[0]   # (4,)

    predicted_idx = int(np.argmax(pred))
    return PredictionResponse(
        predicted_class=CLASS_NAMES[predicted_idx],
        confidence=float(pred[predicted_idx]),
        class_probabilities={
            CLASS_NAMES[i]: float(pred[i]) for i in range(len(CLASS_NAMES))
        },
    )