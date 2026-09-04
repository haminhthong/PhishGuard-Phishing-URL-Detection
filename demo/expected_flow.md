# Kịch bản Demo An toàn (Deterministic Safe Demo Flow) — PhishGuard ML

Kịch bản thử nghiệm trình diễn PhishGuard ML Chrome Extension & API mà **KHÔNG** sử dụng các trang lừa đảo thật đang hoạt động trên Internet.

## Quy trình 8 bước Trình diễn

1. **Khởi chạy API Backend:**
   ```powershell
   .venv\Scripts\python.exe -m API.main
   ```
2. **Kiểm tra Health Check:** Truy cập `http://127.0.0.1:5000/health` và xác nhận trạng thái `"status": "ok"`, `model_version: "3.0.0"`.
3. **Mở Chrome Extension:** Load tiện ích `Extension/` vào Chrome (`chrome://extensions`).
4. **Quét URL An toàn (`demo/safe_urls.json`):**
   - Truy cập `https://github.com`.
   - Quan sát Icon Badge chuyển sang **SAFE** (màu xanh lá cây).
5. **Quét URL Nghi ngờ (`demo/suspicious_urls.json`):**
   - Mở URL mô phỏng `http://192.168.1.1/admin/login.php` hoặc `http://paypal.com-verify-account.security-update.info/login`.
   - Quan sát Icon Badge chuyển sang **WARN** (màu đỏ) và trang cảnh báo PhishGuard warning UI xuất hiện.
6. **Thử nghiệm "Cho phép 1 lần" (Allow Once):**
   - Nhấn nút "Tiếp tục truy cập (Chỉ 1 lần)" trên trang cảnh báo.
   - Trang mở thành công, lần truy cập tiếp theo sẽ được kiểm tra lại.
7. **Thử nghiệm Whitelist:**
   - Thêm `192.168.1.1` vào Whitelist từ Popup UI.
   - Tải lại trang, Badge hiển thị **SAFE** ngay lập tức.
8. **Xử lý Offline Graceful Handling:**
   - Tắt API server (Ctrl+C).
   - Truy cập URL mới, Badge hiển thị **?** (màu vàng) báo lỗi kết nối mà KHÔNG tự động đánh dấu URL là an toàn.
