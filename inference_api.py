from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from audio_utils import extract_mfcc_features, calculate_mel_difference


DATASET_ROOT = Path(".").resolve()
MODELS_DIR = DATASET_ROOT / "models"
FRONTEND_DIR = DATASET_ROOT / "frontend"
MACHINE_TYPES = ["fan", "pump", "slider", "ToyCar", "ToyConveyor"]


class PredictionResponse(BaseModel):
    machine_type: str
    anomaly_score: float
    is_anomaly: bool


class MelDifferenceResponse(BaseModel):
    original: List[List[float]]
    reconstructed: List[List[float]]
    difference: List[List[float]]
    time_axis: List[float]
    freq_axis: List[int]
    max_difference: float
    anomaly_regions: List[Dict[str, float]]  # Regions with high difference


app = FastAPI(title="Acoustic Anomaly Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model(machine_type: str):
    model_path = MODELS_DIR / f"{machine_type}_isolation_forest.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found for machine '{machine_type}' at {model_path}")
    bundle: Dict = joblib.load(model_path)
    return bundle["model"], bundle["feature_dim"]


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
async def predict_anomaly(
    machine_type: str = Form(...),
    file: UploadFile = File(...),
):
    if machine_type not in MACHINE_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid machine_type. Must be one of {MACHINE_TYPES}")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    # Save uploaded file temporarily to disk for librosa to read
    temp_path = DATASET_ROOT / "temp_upload.wav"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        features = extract_mfcc_features(str(temp_path))
        X = features.reshape(1, -1)

        model, expected_dim = load_model(machine_type)
        if X.shape[1] != expected_dim:
            raise HTTPException(
                status_code=500,
                detail=f"Feature dimension mismatch: expected {expected_dim}, got {X.shape[1]}",
            )

        # IsolationForest: decision_function -> higher = more normal, lower = more anomalous
        score = float(model.decision_function(X)[0])
        # Convert to an anomaly score where higher = more anomalous
        anomaly_score = float(-score)
        is_anomaly = bool(model.predict(X)[0] == -1)

        return PredictionResponse(
            machine_type=machine_type,
            anomaly_score=anomaly_score,
            is_anomaly=is_anomaly,
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


@app.post("/mel-difference", response_model=MelDifferenceResponse)
async def get_mel_difference(
    machine_type: str = Form(...),
    file: UploadFile = File(...),
    reconstruction_method: str = Form("smooth"),  # "smooth", "mean", "median"
):
    """
    Calculate mel spectrogram difference for XAI visualization.
    Returns |Original Mel - Reconstructed Mel| heatmap.
    """
    if machine_type not in MACHINE_TYPES:
        raise HTTPException(status_code=400, detail=f"Invalid machine_type. Must be one of {MACHINE_TYPES}")

    if not file.filename.lower().endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    if reconstruction_method not in ["smooth", "mean", "median"]:
        reconstruction_method = "smooth"

    temp_path = DATASET_ROOT / "temp_mel_diff.wav"
    try:
        with open(temp_path, "wb") as f:
            f.write(await file.read())

        # Calculate mel difference
        result = calculate_mel_difference(
            str(temp_path),
            reconstruction_method=reconstruction_method
        )

        # Convert numpy arrays to lists for JSON
        original = result["original"].tolist()
        reconstructed = result["reconstructed"].tolist()
        difference = result["difference"].tolist()
        time_axis = result["time_axis"].tolist()
        freq_axis = result["freq_axis"].tolist()

        # Find anomaly regions (high difference areas)
        max_diff = float(np.max(result["difference"]))
        threshold = max_diff * 0.7  # Top 30% of differences
        
        # Find time regions with high differences
        time_diffs = np.max(result["difference"], axis=0)  # Max difference per time frame
        anomaly_regions = []
        in_anomaly = False
        start_time = 0
        
        for i, diff in enumerate(time_diffs):
            if diff > threshold and not in_anomaly:
                in_anomaly = True
                start_time = result["time_axis"][i]
            elif diff <= threshold and in_anomaly:
                in_anomaly = False
                anomaly_regions.append({
                    "start": float(start_time),
                    "end": float(result["time_axis"][i]),
                    "max_difference": float(np.max(result["difference"][:, max(0, i-10):i+1]))
                })
        
        if in_anomaly:
            anomaly_regions.append({
                "start": float(start_time),
                "end": float(result["time_axis"][-1]),
                "max_difference": float(np.max(result["difference"][:, -10:]))
            })

        return MelDifferenceResponse(
            original=original,
            reconstructed=reconstructed,
            difference=difference,
            time_axis=time_axis,
            freq_axis=freq_axis,
            max_difference=max_diff,
            anomaly_regions=anomaly_regions,
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# Serve frontend static files (mount at the end so API routes take precedence)
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="static")

