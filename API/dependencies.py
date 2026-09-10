"""FastAPI Dependency Injections cho các dịch vụ Backend."""

from __future__ import annotations

from API.config import CONFIG, APIConfig
from API.services.model_loader import LoadedModel, load_phishguard_model
from API.services.predictor import PredictorService

# Model được nạp một lần cho toàn bộ process; không cache URL để tránh giữ dữ liệu
# duyệt web trong bộ nhớ và để mọi request đi cùng một luồng xử lý.
_LOADED_MODEL_INSTANCE: LoadedModel | None = None
_PREDICTOR_INSTANCE: PredictorService | None = None


def get_config() -> APIConfig:
    return CONFIG


def get_loaded_model() -> LoadedModel:
    global _LOADED_MODEL_INSTANCE
    if _LOADED_MODEL_INSTANCE is None:
        _LOADED_MODEL_INSTANCE = load_phishguard_model(CONFIG.model_path, CONFIG.metadata_path)
    return _LOADED_MODEL_INSTANCE


def get_predictor() -> PredictorService:
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        model = get_loaded_model()
        _PREDICTOR_INSTANCE = PredictorService(loaded_model=model)
    return _PREDICTOR_INSTANCE
