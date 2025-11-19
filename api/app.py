# api/app.py
from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib
import json
import pandas as pd
import logging
import os
from .config import MODEL_PATH, MODEL_SCALERX_PATH, MODEL_SCALERY_PATH, PREDICTIONS_LOG
from .routes import router, set_model_globals
from .monitoring_prometheus import setup_monitoring

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("🚀 Starting up Iris Classification API...")
    
    # Chargement du modèle
    global model, model_scaler_X, model_scaler_y
    
   
    model = joblib.load(MODEL_PATH)
    model_scaler_X = joblib.load(MODEL_SCALERX_PATH)
    model_scaler_y = joblib.load(MODEL_SCALERY_PATH) 
    
    # Initialisation du fichier de log
    os.makedirs("logfiles", exist_ok=True)
    if not os.path.exists(PREDICTIONS_LOG):
        df_log = pd.DataFrame(columns=[
            'timestamp', 'sepal_length', 'sepal_width', 
            'petal_length', 'petal_width', 'prediction',
            'prediction_name', 'confidence'
        ])
        df_log.to_csv(PREDICTIONS_LOG, index=False)
        logger.info("📝 Fichier de log des prédictions initialisé")
    
    # Injection des variables globales dans les routes
    set_model_globals(model, model_scaler_X, model_scaler_y, PREDICTIONS_LOG)

    
    yield  # L'application est en cours d'exécution
    
    # Shutdown
    logger.info("🛑 Shutting down Iris Classification API...")
    # Nettoyage si nécessaire

# Création de l'application FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="API pour la classification des fleurs Iris avec RandomForest",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configuration du monitoring prometheus
setup_monitoring(app)


# Inclusion des routes
app.include_router(router)

# Middleware pour les logs
@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"📨 Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"📤 Response: {response.status_code}")
    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True,
        log_level="info"
    )