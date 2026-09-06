"""Ứng dụng FastAPI chính PhishGuard ML API Backend."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError

from API.config import CONFIG
from API.errors import (
    PhishGuardAPIException,
    generic_exception_handler,
    phishguard_exception_handler,
    validation_exception_handler,
)
from API.routes.prediction import router as prediction_router
from API.routes.system import router as system_router
from phishguard import __version__

logging.basicConfig(
    level=getattr(logging, CONFIG.log_level.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("phishguard.api")


def create_app() -> FastAPI:
    """Khởi tạo ứng dụng FastAPI với đầy đủ routers, validation và error handlers."""
    app = FastAPI(
        title="PhishGuard ML API",
        description="REST API phân loại rủi ro URL dựa trên 25 đặc trưng lexical-v2 và mô hình XGBoost Native JSON.",
        version=__version__,
        contact={"name": "PhishGuard ML Maintainers"},
        license_info={"name": "MIT"},
    )

    # Đăng ký Exception Handlers chuẩn hóa
    app.add_exception_handler(PhishGuardAPIException, phishguard_exception_handler) # type: ignore
    app.add_exception_handler(RequestValidationError, validation_exception_handler) # type: ignore
    app.add_exception_handler(Exception, generic_exception_handler) # type: ignore

    # Đăng ký Routers
    app.include_router(system_router)
    app.include_router(prediction_router)

    LOGGER.info("PhishGuard ML API App đã khởi tạo thành công (Version %s)", __version__)
    return app


app = create_app()
