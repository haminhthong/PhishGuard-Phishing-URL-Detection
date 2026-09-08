"""Compatibility entry point cho lifecycle candidate mới.

Artifact được đóng gói bất biến ngay trong ``scripts/train.py``. Không còn
copy model sang API/ hoặc artifacts/ vì các bản sao đó tạo source-of-truth giả.
"""

from __future__ import annotations


def main() -> None:
    """Thông báo command thay thế để không âm thầm ghi đè release."""
    raise RuntimeError(
        "export_model.py đã được thay bằng: train -> evaluate -> security_stress -> promote_release"
    )


if __name__ == "__main__":
    main()
