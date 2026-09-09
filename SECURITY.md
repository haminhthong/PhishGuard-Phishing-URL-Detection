# Chính Sách Bảo Mật & Threat Model — PhishGuard ML

## 1. Triết Lý & Cam Kết Bảo Mật

PhishGuard ML được thiết kế theo triết lý **Local-First & Privacy-Preserving**. Hệ thống bảo vệ người dùng trước các URL lừa đảo (Phishing) trực tiếp trên máy cục bộ, không gửi lịch sử duyệt web hoặc URL truy cập ra bất kỳ dịch vụ đám mây hay bên thứ ba nào.

---

## 2. Mô Hình Đe Dọa (Threat Model)

### 2.1. Mối Đe Dọa Nằm Trong Phạm Vi Xử Lý (In-Scope Threats)
Hệ thống tập trung phát hiện các kỹ thuật lừa đảo phổ biến dựa trên phân tích cấu trúc URL:
- **Cấu trúc URL Lừa đảo (Deceptive URL Structure):** URL có độ dài bất thường, lạm dụng ký tự số, ký tự đặc biệt (`%`, `@`, `-`), tỷ lệ entropy cao.
- **Mạo Danh Thương Hiệu (Brand Impersonation):** Tên thương hiệu lớn (PayPal, Google, Microsoft, v.v.) xuất hiện trong subdomain hoặc đường dẫn path nhưng tên miền đăng ký (Registered Domain) thuộc về kẻ tấn công (ví dụ: `paypal.com.verify-account.attacker.com`).
- **Tấn Công Homograph / Punycode:** Hostname sử dụng tiền tố quốc tế `xn--` để hiển thị ký tự đồng dạng nhằm đánh lừa mắt người.
- **Lạm Dụng Dịch Vụ Rút Gọn (Shortener Abuse):** URL sử dụng các dịch vụ rút gọn liên kết đã biết để che giấu đích đến cuối cùng.
- **Kỹ Thuật Chuyển Hướng Lén (Redirection Patterns):** Chèn ký tự `//` vào giữa đường dẫn để đánh lừa bộ phân giải đường dẫn của trình duyệt.
- **Tên Miền Cấp Cao Rủi Ro Cao (Suspicious TLDs):** Sử dụng các TLD giá rẻ hoặc miễn phí thường xuyên bị lạm dụng trong các chiến dịch spam (`.xyz`, `.top`, `.icu`, `.buzz`, v.v.).

### 2.2. Mối Đe Dọa Nằm Ngoài Phạm Vi Xử Lý (Out-of-Scope / Non-Goals)
PhishGuard ML là một bộ lọc rủi ro mức Lexical URL, **không có khả năng và không cam kết** bảo vệ trước các mối đe dọa sau:
- **Tên Miền Hợp Lệ Bị Chiếm Quyền (Compromised Legitimate Domains):** Website hợp lệ nhưng bị tin tặc hack và chèn form lừa đảo với URL hoàn toàn tự nhiên.
- **Nội Dung Trang Độc Hại (Malicious Web Content):** PhishGuard không cào mã HTML, CSS hoặc phân tích DOM của website.
- **Mã Độc Thực Thi Phía Client (Client-Side JS Exploits / Drive-by Downloads):** Không phân tích hành vi Javascript hay binary payload.
- **Tấn Công Hạ Tầng Mạng (DNS Poisoning / BGP Hijacking):** Không kiểm tra tính toàn vẹn của DNS records hay chứng chỉ số SSL/TLS.
- **Zero-Day Phishing với Cấu Trúc Hoàn Hảo:** Các trang phishing sử dụng tên miền ngắn, cấu trúc sạch sẽ và không chứa từ khóa thương hiệu trong URL.

---

## 3. Kiến Trúc Quyền Riêng Tư (Privacy Architecture)

1. **In-Memory Prediction:** Toàn bộ chuỗi URL (bao gồm cả query string) chỉ được gửi qua kết nối nội bộ `http://127.0.0.1:5000` và tồn tại trong bộ nhớ RAM trong suốt quá trình trích xuất đặc trưng và suy luận mô hình.
2. **Sanitized History Logging:** Extension chỉ lưu trữ phần `origin + pathname` vào bộ nhớ trình duyệt `chrome.storage.local`. Toàn bộ tham số Query String (tránh làm lộ Access Token, Session ID, Email cá nhân) và Hash Fragment đều bị loại bỏ trước khi ghi log.
3. **Băm SHA-256 Bộ Nhớ Đệm:** Bộ nhớ đệm LRU Cache của API băm URL bằng thuật toán SHA-256 kết hợp phiên bản mô hình, không lưu giữ chuỗi URL gốc dạng plain text trong cache keys.
4. **Không Gọi Mạng Ngoài (Zero External Network Calls):** Quá trình suy luận của mô hình hoàn toàn độc lập, không thực hiện bất kỳ truy vấn DNS, Whois hay Safe Browsing API nào qua Internet.

---

## 4. Bảo Mật Model Artifact

- API chỉ nạp mô hình từ định dạng **Native XGBoost JSON** chính thức (`XGB.json`).
- Hệ thống từ chối hoàn toàn việc giải tuần tự hóa các tệp Python Pickle (`.pkl`, `.joblib`) để phòng ngừa lỗ hổng RCE (Remote Code Execution).
- Khi nạp release, API xác thực SHA-256 của model và các artifact theo `metadata.json` trong release bundle. Nếu checksum sai, API từ chối phục vụ với lỗi `MODEL_INTEGRITY_ERROR`.

---

## 5. Báo Cáo Lỗ Hổng

Nếu bạn phát hiện lỗ hổng bảo mật trong mã nguồn hoặc phương thức trích xuất đặc trưng, xin vui lòng tạo Issue riêng tư hoặc liên hệ trực tiếp với tác giả repo. Xin vui lòng không công khai URL độc hại thực tế hoặc thông tin cá nhân lên Issues công cộng.
