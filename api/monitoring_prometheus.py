from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import time

# Métriques minimales pour commencer
IRIS_PREDICTION_COUNT = Counter(
    'iris_prediction_requests_total',
    'Total number of Iris prediction requests',
    ['status']
)

PREDICTION_LATENCY = Histogram(
    'iris_prediction_latency_seconds',
    'Iris prediction request latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0]
)

ACTIVE_REQUESTS = Gauge(
    'iris_active_requests', 
    'Number of active requests to the Iris API'
)

def setup_monitoring(app):
    # Instrumentation de base seulement
    Instrumentator().instrument(app).expose(app)

    @app.middleware("http")
    async def simple_iris_monitoring(request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        ACTIVE_REQUESTS.inc()
        start_time = time.time()
        
        try:
            response = await call_next(request)
            
            # Seulement pour /predict
            if request.url.path == "/predict":
                latency = time.time() - start_time
                PREDICTION_LATENCY.observe(latency)
                
                if response.status_code == 200:
                    IRIS_PREDICTION_COUNT.labels(status="success").inc()
                else:
                    IRIS_PREDICTION_COUNT.labels(status="error").inc()
            
            return response
            
        except Exception:
            if request.url.path == "/predict":
                IRIS_PREDICTION_COUNT.labels(status="error").inc()
            raise
        finally:
            ACTIVE_REQUESTS.dec()