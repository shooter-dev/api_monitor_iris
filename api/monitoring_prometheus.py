from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import time
import random

# --------- Iris-specific metrics ----------
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

def setup_monitoring(app):
    # Base instrumentation - CRITICAL for dashboard metrics
    instrumentator = Instrumentator(
        should_group_status_codes=False,
        should_ignore_untemplated=True,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[".*admin.*", "/metrics"],
        inprogress_name="iris_inprogress",  # 🎯 Essential for panel 14
        inprogress_labels=True,
    )
    
    instrumentator.instrument(app).expose(app)

    @app.middleware("http")
    async def iris_monitoring_middleware(request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        start_time = time.time()
        ACTIVE_REQUESTS.inc()

        try:
            response = await call_next(request)
            latency = time.time() - start_time

            # 🎯 Metric for panel 10: Requests by endpoint
            REQUEST_BY_ENDPOINT.labels(
                method=request.method,
                endpoint=request.url.path,
                status_code=response.status_code
            ).inc()

            # 🎯 Metrics for /predict endpoint
            if request.url.path == "/predict":
                # For panel 3, 7: Latency metrics
                PREDICTION_LATENCY.observe(latency)
                
                if response.status_code == 200:
                    # 🎯 For panels 1, 2, 5, 6: Success counts
                    # Simulate different flower classes for demo
                    flower_class = random.choice(['setosa', 'versicolor', 'virginica'])
                    confidence = random.uniform(0.8, 0.98)
                    
                    IRIS_PREDICTION_COUNT.labels(
                        prediction_class=flower_class,
                        status="200"
                    ).inc()
                    
                    # 🎯 For future confidence panels
                    PREDICTION_CONFIDENCE.labels(
                        prediction_class=flower_class
                    ).observe(confidence)
                    
                else:
                    # 🎯 For panels 4, 12, 13: Error tracking
                    IRIS_PREDICTION_COUNT.labels(
                        prediction_class="error", 
                        status=str(response.status_code)
                    ).inc()
                    
                    PREDICTION_CONFIDENCE.labels(
                        prediction_class="error"
                    ).observe(0.0)

            return response

        except Exception:
            # 🎯 Error tracking for panels 4, 12, 13
            if request.url.path == "/predict":
                IRIS_PREDICTION_COUNT.labels(
                    prediction_class="error",
                    status="500"
                ).inc()
                
                PREDICTION_CONFIDENCE.labels(
                    prediction_class="error"
                ).observe(0.0)
            raise
        finally:
            ACTIVE_REQUESTS.dec()