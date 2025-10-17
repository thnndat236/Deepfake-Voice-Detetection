from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from core.config import settings
import logging
from utils.tracer import setup_tracing, remove_tracing
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from prometheus_fastapi_instrumentator import Instrumentator
from api.routes.dfv_router import router as dfv_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

tracer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting DeepFake Voice Detection API...")
    
    # Setup tracing on startup
    global tracer
    tracer = setup_tracing()

    # Instrument OpenTelemetry only if tracing is enabled
    tracing_enabled = settings.TRACING_ENABLE.lower() == "true"
    if tracing_enabled:
        logger.info("Instrumenting FastAPI and Requests with OpenTelemetry...")
        FastAPIInstrumentor.instrument_app(app)
        RequestsInstrumentor().instrument()
    else:
        logger.info("Skipping OpenTelemetry instrumentation (tracing disabled)")

    # Expose metrics on startup
    logger.info("Exposing metrics with Prometheus Fastapi Instrumentator...")
    instrumentator.expose(app)
    logger.info("Exposing metrics successfully")

    yield

    # Shutdown
    logger.info("Cleaning up resources...")
    
    # Uninstrument app
    remove_tracing()
    logger.info("Telemetry resources cleaned up")


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    lifespan=lifespan,
    root_path="/api"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus metrics endpoint
instrumentator = Instrumentator(excluded_handlers=["/metrics"]).instrument(app)

@app.get("/")
def main():
    return {"message": "Welcome to DeepFake Voice Detection"}

@app.get("/health")
async def health_check():
    global tracer
    tracing_enabled = settings.TRACING_ENABLE.lower() == "true"

    if tracing_enabled:
        with tracer.start_as_current_span("health_check") as span:
            span.set_attribute("health.status", "ok")
            span.set_attribute("tracing", "enable")
        return {"status": "healthy", "tracing": "enable"}
    return {"status": "healthy", "tracing": "disable"}

# Include DFV router
app.include_router(dfv_router)