"""Chuẩn hóa cấu trúc và mã lỗi (Error Contract) cho PhishGuard ML API."""

from __future__ import annotations

import uuid

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    error: ErrorDetail


class PhishGuardAPIException(Exception):
    """Exception chuẩn của hệ thống API."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = status.HTTP_400_BAD_REQUEST,
    ) -> None:
        self.code = code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


def build_error_response(code: str, message: str, status_code: int) -> JSONResponse:
    request_id = str(uuid.uuid4())[:8]
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "code": code,
                "message": message,
                "request_id": request_id,
            }
        },
    )


async def phishguard_exception_handler(_request: Request, exc: PhishGuardAPIException) -> JSONResponse:
    return build_error_response(exc.code, exc.message, exc.status_code)


async def validation_exception_handler(_request: Request, exc: RequestValidationError) -> JSONResponse:
    # Lấy thông báo lỗi đầu tiên ngắn gọn
    errors = exc.errors()
    first_msg = errors[0].get("msg", "Dữ liệu yêu cầu không hợp lệ") if errors else "Yêu cầu không hợp lệ"
    return build_error_response("INVALID_URL", first_msg, status.HTTP_422_UNPROCESSABLE_ENTITY)


async def generic_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    return build_error_response("PREDICTION_FAILED", f"Lỗi xử lý nội bộ: {exc!s}", status.HTTP_500_INTERNAL_SERVER_ERROR)
