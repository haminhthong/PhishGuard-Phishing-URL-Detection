"""Hợp đồng đầu vào giữa pipeline huấn luyện và dịch vụ dự đoán."""

FEATURE_CONTRACT_VERSION = "lexical-v1"

# Không đổi tên hoặc thứ tự nếu chưa huấn luyện và xuất lại mô hình.
FEATURE_COLUMNS: tuple[str, ...] = (
    "Having_IP", "Tiny_URL", "TLD_Length", "Digit_Count", "Dot_Count",
    "At_Count", "Hyphen_Count", "Per_Count", "Equal_Count", "Redirection",
    "Depth", "FD_Length",
)
