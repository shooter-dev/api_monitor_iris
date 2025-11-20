# api/app.py - VERSION CORRIGÉE AVEC INSTRUMENTATOR
from fastapi import FastAPI
from contextlib import asynccontextmanager
from starlette.middleware.base import BaseHTTPMiddleware
import joblib
import pandas as pd
import logging
import os
from .config import MODEL_PATH, MODEL_SCALERX_PATH, MODEL_SCALERY_PATH, PREDICTIONS_LOG
from .routes import router, set_model_globals
from .monitoring_prometheus import prometheus_middleware, setup_metrics_endpoint

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gère le cycle de vie de l'application"""
    # ========== STARTUP ==========
    logger.info("🚀 Starting up Iris Classification API...")
    
    # Chargement du modèle
    global model, model_scaler_X, model_scaler_y
    
    try:
        model = joblib.load(MODEL_PATH)
        model_scaler_X = joblib.load(MODEL_SCALERX_PATH)
        model_scaler_y = joblib.load(MODEL_SCALERY_PATH)
        logger.info("✅ Modèles chargés avec succès")
    except Exception as e:
        logger.error(f"❌ Erreur lors du chargement des modèles: {e}")
        raise
    
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
    logger.info("✅ Variables globales injectées dans les routes")
    logger.info("✅ Démarrage de l'API terminé")

    yield
    
    # ========== SHUTDOWN ==========
    logger.info("🛑 Shutting down Iris Classification API...")


# Création de l'application FastAPI
app = FastAPI(
    title="Iris Classification API",
    description="API pour la classification des fleurs Iris avec RandomForest",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# ⬇️ 1. CONFIGURATION DE PROMETHEUS /metrics (DOIT être fait AVANT les middlewares)
setup_metrics_endpoint(app)

# ⬇️ 2. MIDDLEWARE Prometheus
app.add_middleware(BaseHTTPMiddleware, dispatch=prometheus_middleware)

# ⬇️ 3. ROUTES
app.include_router(router)

# ⬇️ 4. MIDDLEWARE de logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware pour logger toutes les requêtes"""
    logger.info(f"📨 Request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"📤 Response: {response.status_code}")
    return response


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )