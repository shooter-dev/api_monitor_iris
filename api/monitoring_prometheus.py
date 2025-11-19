from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import time
import json

# --------- Iris-specific metrics ----------
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

# --------- Setup ----------
def setup_monitoring(app):

    # Base FastAPI instrumentation
    Instrumentator(
        should_group_status_codes=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=["/metrics"]
    ).instrument(app).expose(app)

     # --------- Simple Iris metrics middleware ----------
    @app.middleware("http")
    async def collect_iris_metrics(request, call_next):
        # Skip /metrics endpoint
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()
        ACTIVE_REQUESTS.inc()

        try:
            response = await call_next(request)
            latency = time.time() - start_time

            # Record generic request data
            REQUEST_BY_ENDPOINT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()

            # --- Iris /predict endpoint ---
            if request.url.path == "/predict":
                PREDICTION_LATENCY.observe(latency)
                
                # Simple counting without reading response body
                if response.status_code == 200:
                    IRIS_PREDICTION_COUNT.labels(status="success").inc()
                else:
                    IRIS_PREDICTION_COUNT.labels(status="error").inc()

            return response

        except Exception:
            # Count errors for /predict endpoint
            if request.url.path == "/predict":
                IRIS_PREDICTION_COUNT.labels(status="error").inc()
            raise
        finally:
            ACTIVE_REQUESTS.dec()