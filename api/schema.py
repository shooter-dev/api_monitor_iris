# api/schema.py
from pydantic import BaseModel
from typing import List

class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class PredictionResponse(BaseModel):
    prediction: str
    prediction_name: str
    probabilities: List[float]
    confidence: float
    model_version: str

class ModelInfoResponse(BaseModel):
    model_type: str
    features: List[str]
    target_names: List[str]
    target_mapping: dict
    training_samples: int
    accuracy: float
    model_loaded: bool

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    timestamp: str

class PredictionStatsResponse(BaseModel):
    total_predictions: int
    class_distribution: dict
    average_confidence: float
    last_prediction: dict = None

class SampleDataResponse(BaseModel):
    setosa: dict
    versicolor: dict
    virginica: dict