# Nguồn dữ liệu và quy tắc quản trị

## Dữ liệu hiện có

- `Data/legit_url.csv`: URL hợp lệ dùng trong thử nghiệm ban đầu. Cần bổ sung URL nguồn, giấy phép, ngày tải và checksum trước khi công bố benchmark.
- `Data/verified_online.csv`: dữ liệu phishing từ PhishTank. Trường thời gian phải được giữ lại để hỗ trợ đánh giá theo thời gian.

## Quy tắc bắt buộc

1. Không dùng tập test để chọn thuật toán, hyperparameter hoặc threshold.
2. Chuẩn hóa URL và loại bản ghi trùng trước khi chia tập.
3. Loại domain có nhãn mâu thuẫn và báo cáo số lượng bị loại.
4. Chia train/validation/test theo registered domain; giao của ba tập phải rỗng.
5. Lưu seed, checksum đầu vào, phiên bản hợp đồng đặc trưng và thời điểm chạy.
6. Chỉ công bố metric trên tập test sau khi đã khóa model bằng validation.

Notebook cũ là tài liệu khám phá, không còn là nguồn pipeline production. Code huấn luyện mới phải import `phishguard.features` để preprocessing giống hoàn toàn với API.
