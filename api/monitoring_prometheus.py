# monitoring_prometheus.py - VERSION FINALE CORRIGÉE
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import time

# ========== MÉTRIQUES CUSTOMISÉES ==========
IRIS_PREDICTION_COUNT = Counter(
    'iris_prediction_requests_total',
    'Total number of Iris prediction requests',
    ['prediction_class', 'status']
)

PREDICTION_LATENCY = Histogram(
    'iris_prediction_latency_seconds',
    'Iris prediction request latency',
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
)

PREDICTION_CONFIDENCE = Histogram(
    'iris_prediction_confidence',
    'Confidence of Iris predictions',
    ['prediction_class'],
    buckets=[0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
)

ACTIVE_REQUESTS = Gauge(
    'iris_active_requests',
    'Number of active requests to the Iris API'
)

REQUEST_BY_ENDPOINT = Counter(
    'iris_requests_by_endpoint_total',
    'Total requests by endpoint',
    ['method', 'endpoint', 'status_code']
)


# ========== MIDDLEWARE (défini au niveau module) ==========
async def prometheus_middleware(request, call_next):
    """
    Middleware pour les métriques Prometheus
    IMPORTANT : Doit être ajouté AVANT le démarrage de l'app
    """
    start_time = time.time()
    
    # Traitement spécial pour /predict
    if request.url.path == "/predict":
        ACTIVE_REQUESTS.inc()
        
        try:
            # Appeler l'endpoint et obtenir la réponse
            response = await call_next(request)
            
            # Mesurer la latence
            process_time = time.time() - start_time
            PREDICTION_LATENCY.observe(process_time)
            
            # Compter les prédictions selon le status code
            if response.status_code == 200:
                IRIS_PREDICTION_COUNT.labels(
                    prediction_class="success",
                    status="200"
                ).inc()
            else:
                IRIS_PREDICTION_COUNT.labels(
                    prediction_class="error",
                    status=str(response.status_code)
                ).inc()
            
            return response
            
        except Exception as e:
            # En cas d'erreur, compter comme erreur
            IRIS_PREDICTION_COUNT.labels(
                prediction_class="error",
                status="500"
            ).inc()
            raise e
            
        finally:
            # Toujours décrémenter le compteur de requêtes actives
            ACTIVE_REQUESTS.dec()
    
    else:
        # Pour les autres endpoints, monitoring simple
        try:
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Compter par endpoint
            REQUEST_BY_ENDPOINT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            return response
            
        except Exception as e:
            raise e


# ========== SETUP FUNCTION (appelée dans le lifespan) ==========
def setup_metrics_endpoint(app):
    """
    Configure l'endpoint /metrics pour Prometheus
    À appeler dans le lifespan de FastAPI
    """
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_respect_env_var=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[".*admin.*", "/metrics"],
        env_var_name="ENABLE_METRICS",
        inprogress_name="iris_inprogress",
        inprogress_labels=True,
    )
    
    # Instrumenter et exposer /metrics
    instrumentator.instrument(app).expose(app)
    
    return instrumentator