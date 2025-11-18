# monitoring_prometheus.py
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import time

# Métriques customisées pour Iris
IRIS_PREDICTION_COUNT = Counter(
    'iris_prediction_requests_total',
    'Total number of Iris prediction requests',
    ['prediction_class', 'status']
)

PREDICTION_LATENCY = Histogram(
    'iris_prediction_latency_seconds',
    'Iris prediction request latency',
    buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
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

def setup_monitoring(app):
    # Instrumentation de base
    instrumentator = Instrumentator()
    
    instrumentator.instrument(app).expose(app)
    
    # Middleware personnalisé pour les métriques Iris
    @app.middleware("http")
    async def monitor_iris_requests(request, call_next):
        if request.url.path == "/predict":
            start_time = time.time()
            ACTIVE_REQUESTS.inc()
            
            try:
                response = await call_next(request)
                process_time = time.time() - start_time
                PREDICTION_LATENCY.observe(process_time)
                
                # Compter les prédictions
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
                    
            except Exception as e:
                IRIS_PREDICTION_COUNT.labels(
                    prediction_class="error", 
                    status="500"
                ).inc()
                raise e
            finally:
                ACTIVE_REQUESTS.dec()
            
            return response
        else:
            # Monitoring pour les autres endpoints
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            REQUEST_BY_ENDPOINT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()
            
            return response