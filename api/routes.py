# api/routes.py
from fastapi import APIRouter, HTTPException
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os
from .schema import (
    IrisFeatures, PredictionResponse, ModelInfoResponse, 
    HealthResponse, PredictionStatsResponse, SampleDataResponse
)

logger = logging.getLogger(__name__)

# Router principal
router = APIRouter()

# Variables globales (seront injectées depuis app.py)
model = None
model_metadata = {}
PREDICTIONS_LOG = "monitoring/predictions_log.csv"

@router.post("/predict", response_model=PredictionResponse)
async def predict(features: IrisFeatures):
    """
    Effectue une prédiction sur les caractéristiques d'une fleur Iris
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Modèle non chargé")
    
    try:
        # Préparation des données
        input_data = pd.DataFrame([[
            features.sepal_length,
            features.sepal_width,
            features.petal_length,
            features.petal_width
        ]], columns=model_metadata['features'])
        
        # Prédiction
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = np.max(probabilities)
        prediction_name = model_metadata['target_mapping'][str(prediction)]
        
        # Enregistrement pour monitoring
        await log_prediction(features, prediction, prediction_name, confidence)
        
        return PredictionResponse(
            prediction=int(prediction),
            prediction_name=prediction_name,
            probabilities=[float(p) for p in probabilities],
            confidence=float(confidence),
            model_version="1.0.0"
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def log_prediction(features: IrisFeatures, prediction: int, prediction_name: str, confidence: float):
    """Enregistre les prédictions pour le monitoring"""
    try:
        new_log = pd.DataFrame([{
            'timestamp': datetime.now(),
            'sepal_length': features.sepal_length,
            'sepal_width': features.sepal_width,
            'petal_length': features.petal_length,
            'petal_width': features.petal_width,
            'prediction': prediction,
            'prediction_name': prediction_name,
            'confidence': confidence
        }])
        
        # Ajout au fichier de log (version asynchrone)
        new_log.to_csv(PREDICTIONS_LOG, mode='a', header=False, index=False)
        logger.info(f"Prédiction enregistrée: {prediction_name} (confiance: {confidence:.3f})")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'enregistrement du log: {e}")

@router.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Retourne les informations du modèle"""
    if not model_metadata:
        raise HTTPException(status_code=500, detail="Métadonnées non chargées")
    
    return ModelInfoResponse(
        model_type=model_metadata.get('model_type'),
        features=model_metadata.get('features'),
        target_names=model_metadata.get('target_names'),
        target_mapping=model_metadata.get('target_mapping'),
        training_samples=model_metadata.get('training_samples'),
        accuracy=model_metadata.get('accuracy'),
        model_loaded=model is not None
    )

@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy", 
        model_loaded=model is not None,
        timestamp=datetime.now().isoformat()
    )

@router.get("/", include_in_schema=False)
async def root():
    return {
        "message": "Iris Classification API",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "model_info": "/model-info",
            "predict": "/predict",
            "sample": "/generate-sample",
            "stats": "/prediction-stats"
        }
    }

@router.get("/generate-sample", response_model=SampleDataResponse)
async def generate_sample():
    """Génère un échantillon de données Iris typiques"""
    return SampleDataResponse(
        setosa={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        },
        versicolor={
            "sepal_length": 6.0,
            "sepal_width": 2.7,
            "petal_length": 4.2,
            "petal_width": 1.3
        },
        virginica={
            "sepal_length": 6.7,
            "sepal_width": 3.1,
            "petal_length": 5.6,
            "petal_width": 2.4
        }
    )

@router.get("/prediction-stats", response_model=PredictionStatsResponse)
async def prediction_stats():
    """Retourne les statistiques des prédictions récentes"""
    try:
        if os.path.exists(PREDICTIONS_LOG):
            df = pd.read_csv(PREDICTIONS_LOG)
            last_prediction = df.iloc[-1].to_dict() if len(df) > 0 else None
            
            return PredictionStatsResponse(
                total_predictions=len(df),
                class_distribution=df['prediction_name'].value_counts().to_dict(),
                average_confidence=float(df['confidence'].mean()) if len(df) > 0 else 0.0,
                last_prediction=last_prediction
            )
        else:
            return PredictionStatsResponse(
                total_predictions=0,
                class_distribution={},
                average_confidence=0.0
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des stats: {e}")

def set_model_globals(model_instance, metadata, predictions_log_path):
    """Fonction pour injecter les variables globales depuis app.py"""
    global model, model_metadata, PREDICTIONS_LOG
    model = model_instance
    model_metadata = metadata
    PREDICTIONS_LOG = predictions_log_path