"""FastAPI Dependency Injections cho các dịch vụ Backend."""

from __future__ import annotations

from API.config import CONFIG, APIConfig
from API.services.cache import PredictionCache
from API.services.model_loader import LoadedModel, load_phishguard_model
from API.services.predictor import PredictorService

# Global Singleton instances
_CACHE_INSTANCE = PredictionCache(capacity=CONFIG.cache_size)
_LOADED_MODEL_INSTANCE: LoadedModel | None = None
_PREDICTOR_INSTANCE: PredictorService | None = None


def get_config() -> APIConfig:
    return CONFIG


def get_cache() -> PredictionCache:
    return _CACHE_INSTANCE


def get_loaded_model() -> LoadedModel:
    global _LOADED_MODEL_INSTANCE
    if _LOADED_MODEL_INSTANCE is None:
        _LOADED_MODEL_INSTANCE = load_phishguard_model(CONFIG.model_path, CONFIG.metadata_path)
    return _LOADED_MODEL_INSTANCE


def get_predictor() -> PredictorService:
    global _PREDICTOR_INSTANCE
    if _PREDICTOR_INSTANCE is None:
        model = get_loaded_model()
        cache = get_cache()
        _PREDICTOR_INSTANCE = PredictorService(loaded_model=model, cache=cache)
    return _PREDICTOR_INSTANCE
