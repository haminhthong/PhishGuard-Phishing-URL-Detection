"""
Điểm khởi chạy máy chủ REST API PhishGuard ML trong môi trường phát triển (Development Mode).
"""

from __future__ import annotations

import uvicorn

from API.config import CONFIG


def start() -> None:
    """Khởi chạy máy chủ Uvicorn server tại địa chỉ localhost (127.0.0.1:5000)."""
    uvicorn.run("API.app:app", host=CONFIG.host, port=CONFIG.port, reload=CONFIG.reload_enabled)


if __name__ == "__main__":
    print("=" * 65)
    print(" 🚀 PhishGuard ML API Backend đang khởi động...")
    print(f" 🌐 Địa chỉ API local:    http://{CONFIG.host}:{CONFIG.port}")
    print(f" 📚 Tài liệu Swagger UI:  http://{CONFIG.host}:{CONFIG.port}/docs")
    print(f" 🩺 Kiểm tra Health:      http://{CONFIG.host}:{CONFIG.port}/health")
    print(f" 🧠 Thông tin model:       http://{CONFIG.host}:{CONFIG.port}/model-info")
    print("=" * 65)
    start()
