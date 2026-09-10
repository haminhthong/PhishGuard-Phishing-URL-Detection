# Chính sách bảo mật — PhishGuard ML

## Phạm vi

PhishGuard ML là bộ phân loại rủi ro URL local-first. API chỉ nhận URL qua localhost, không gọi dịch vụ mạng bên ngoài và không phân tích nội dung trang. Hệ thống hỗ trợ các tín hiệu lexical như IP, Punycode, brand impersonation, shortener, suspicious TLD và redirection pattern.

Các mối đe dọa ngoài phạm vi gồm HTML/DOM/JavaScript độc hại, file tải xuống, DNS/BGP/TLS hijacking, domain hợp lệ bị chiếm quyền nhưng URL tự nhiên và zero-day không có dấu hiệu trong URL.

## Quy tắc xử lý dữ liệu

1. `raw_url` tồn tại trong bộ nhớ trong thời gian trích xuất feature; không gửi URL tới cloud, DNS, Whois hoặc reputation API.
2. Extension chỉ ghi `origin + pathname` vào `chrome.storage.local`; query và fragment bị loại khỏi lịch sử cục bộ.
3. Log API chỉ ghi hostname đã làm sạch, không ghi path, query, token hoặc session.
4. Dữ liệu CSV huấn luyện, URL hoạt động và secret không được commit vào repository.

## Artifact và runtime

- API chỉ nạp XGBoost Native JSON từ `artifacts/model.json`; không nạp pickle/joblib.
- `metadata.json` kiểm tra model checksum, feature contract/hash và resource hashes.
- `calibration.json` phải cùng `model_version` với model.
- `thresholds.json` phải có đúng `caution_threshold`, `block_threshold` và mapping `allow/caution/block`.
- Thiếu artifact, sai checksum, sai contract hoặc sai resource version đều làm API fail-closed.

## Báo cáo lỗ hổng

Không đăng URL phishing đang hoạt động, token, dữ liệu cá nhân hoặc khai thác chưa vá trong issue công khai. Hãy tạo báo cáo riêng tư cho chủ repo, mô tả phiên bản, bước tái hiện an toàn và tác động; có thể thay URL thật bằng domain kiểm thử đã kiểm soát.
