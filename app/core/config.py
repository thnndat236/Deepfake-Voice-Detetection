from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv


load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "Deepfake Voice Detector"
    VERSION: str = "1.0.0"
    DFV_TFLITE_MODEL: str = "model/deepfakevoice/deepfakevoice.tflite"
    DFV_SCALER: str = "model/deepfakevoice/scaler.joblib"

    # Direct connect to Jaeger Collector
    TRACING_ENABLE: str = os.getenv("TRACING_ENABLE", "False")
    JAEGER_COLLECTOR_ENDPOINT: str = os.getenv("JAEGER_COLLECTOR_ENDPOINT", "http://localhost:4317")
    JAEGER_COLLECTOR_INSECURE: str = os.getenv("JAEGER_COLLECTOR_INSECURE", "True")
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "deepfakevoice-service")
    JAEGER_HOSTNAME: str = os.getenv("JAEGER_HOSTNAME", "deepfakevoice-service")

settings = Settings()