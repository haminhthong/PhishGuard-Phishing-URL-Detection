"""Cấu hình runtime tập trung cho PhishGuard ML API Backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class APIConfig:
    """Đọc cấu hình từ biến môi trường hoặc dùng giá trị mặc định an toàn."""

    project_root: Path = Path(__file__).resolve().parents[1]
    # Không trỏ vào bản sao model trong API; loader dùng artifacts/ làm nguồn duy nhất.
    model_path: Path | None = (
        Path(os.environ["PHISHGUARD_MODEL_PATH"]) if os.getenv("PHISHGUARD_MODEL_PATH") else None
    )
    metadata_path: Path | None = (
        Path(os.environ["PHISHGUARD_METADATA_PATH"])
        if os.getenv("PHISHGUARD_METADATA_PATH")
        else None
    )
    host: str = os.getenv("PHISHGUARD_HOST", "127.0.0.1")
    port: int = int(os.getenv("PHISHGUARD_PORT", "5000"))
    reload_enabled: bool = os.getenv("PHISHGUARD_RELOAD", "false").lower() == "true"
    max_url_length: int = int(os.getenv("PHISHGUARD_MAX_URL_LENGTH", "2048"))
    max_batch_size: int = int(os.getenv("PHISHGUARD_MAX_BATCH_SIZE", "50"))
    request_timeout_seconds: float = float(os.getenv("PHISHGUARD_TIMEOUT", "3.5"))
    log_level: str = os.getenv("PHISHGUARD_LOG_LEVEL", "INFO")


CONFIG = APIConfig()
