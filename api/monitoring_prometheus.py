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

    # --------- Custom Iris metrics middleware ----------
    @app.middleware("http")
    async def collect_iris_metrics(request, call_next):

        # Do not monitor /metrics
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
                status_code=str(response.status_code)
            ).inc()

            # --- Iris /predict endpoint ---
            if request.url.path == "/predict":
                PREDICTION_LATENCY.observe(latency)

                try:
                    body = b"".join([chunk async for chunk in response.body_iterator])
                    response.body_iterator = iter([body])
                    data = json.loads(body.decode())

                    pred_class = str(data.get("prediction_name", "unknown"))
                    confidence = float(data.get("confidence", 0.0))

                    IRIS_PREDICTION_COUNT.labels(
                        prediction_class=pred_class,
                        status=str(response.status_code),
                    ).inc()

                    PREDICTION_CONFIDENCE.labels(prediction_class=pred_class).observe(confidence)

                except Exception:
                    IRIS_PREDICTION_COUNT.labels(
                        prediction_class="error",
                        status="500",
                    ).inc()

            return response

        finally:
            ACTIVE_REQUESTS.dec()
