# Đóng góp cho PhishGuard ML

1. Tạo nhánh như `feature/domain-reputation` hoặc `fix/url-validation`.
2. Cài bằng `python -m pip install -r requirements-dev.txt`.
3. Giữ hàm nhỏ, tên rõ nghĩa; chú thích tiếng Việt tập trung giải thích lý do.
4. Chạy `ruff check API tests` và `python -m unittest discover -s tests -v`.
5. Cập nhật README/Model Card khi thay đổi hành vi hoặc mô hình.

Không đổi thứ tự `FEATURE_COLUMNS` khi chưa huấn luyện và version lại mô hình. Không commit URL phishing đang hoạt động, bí mật hoặc dữ liệu cá nhân. Mọi thay đổi mô hình phải kèm số liệu đánh giá có thể tái lập.
