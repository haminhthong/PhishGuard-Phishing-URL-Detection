# Nguồn dữ liệu và Quy tắc Quản trị

## Dữ liệu Hiện có

- `Data/legit_url.csv`: URL hợp lệ dùng trong huấn luyện và thử nghiệm (Top Sites toàn cầu Tranco List / Alexa).
- `Data/verified_online.csv`: Dữ liệu phishing từ cộng đồng [PhishTank](https://phishtank.org/). Trường thời gian (`submission_time`) được bảo toàn để phục vụ đánh giá độ suy giảm hiệu năng theo thời gian (Protocol B — Temporal Drift).

## Quy tắc Bắt buộc Quản trị Dữ liệu (Data Governance Rules)

1. **Bảo tồn URL gốc (Raw URL Preservation):** Giữ nguyên chuỗi `raw_url` gốc từ nguồn để phục vụ trích xuất đặc trưng bảo mật (Lexical & Heuristic Features). Tuyệt đối không can thiệp làm mất các biểu diễn quan trọng trước khi trích xuất.
2. **Chuẩn hóa Canonical (Canonical URL):** Chỉ dùng `canonical_url` cho mục đích deduplication, conflict audit và cache identity.
3. **Chính sách Xử lý Mâu thuẫn Nhãn (Label-Conflict Policy):**
   - Loại bỏ các trường hợp **exact canonical URL** có nhãn mâu thuẫn (cùng một URL nhưng xuất hiện cả nhãn 0 và 1 từ 2 nguồn).
   - **KHÔNG LOẠI BỎ registered domain** chỉ vì domain chứa cả URL hợp lệ và URL lừa đảo. Trong thực tế, các nền tảng đám mây, shared hosting và dịch vụ dùng chung (`google.com`, `dropbox.com`, `live.com`, `wix.com`, `wordpress.com`, `t.co`, v.v.) hoàn toàn có thể bị kẻ xấu lợi dụng để tạo trang phishing dưới các đường dẫn path khác nhau. Việc giữ lại các trường hợp này là bắt buộc để mô hình học được các hard cases thực tế.
4. **Chia tập theo Registered Domain (Domain-Grouped 5-Way Split):**
   - Sử dụng Registered Domain (Mozilla Public Suffix List) làm khóa nhóm (grouping key).
   - Toàn bộ các URL thuộc cùng một registered domain (kể cả chứa cả legit và phishing) phải nằm trọn vẹn trong **đúng một split duy nhất** (Train, Development/Validation, Calibration, Policy Validation hoặc Locked Test).
   - Giao giữa 5 tập tuyệt đối rỗng: $\text{Domain Overlap} = 0$ và $\text{Canonical URL Overlap} = 0$.
   - Tỷ lệ chuẩn của lifecycle là Train 60%, Development 15%, Calibration 10%, Policy Validation 5% và Locked Test 10%; số dòng thực tế lấy từ `artifacts/split_manifest.json`.
5. **Độc lập Vòng đời Hiệu chuẩn (Calibration & Test Independence):**
   - Tập Validation chỉ dùng để lựa chọn kiến trúc mô hình (Model Architecture Selection).
   - Tập Calibration chỉ chịu trách nhiệm hiệu chuẩn xác suất (Isotonic/Sigmoid).
   - Tập Policy Validation độc lập chịu trách nhiệm chọn `caution_threshold` và `block_threshold` với FPR/recall gate.
   - Tập Locked Test là tập nguyên bản chưa từng gặp, được đánh giá duy nhất một lần và chỉ dùng cho mục đích báo cáo (Report Only).
6. **Truy xuất Nguồn gốc & Toàn vẹn (Provenance & Checksums):**
   - Mọi lần chuẩn bị dữ liệu đều phải tạo `DatasetManifest` và `SplitManifest` lưu trữ mã băm SHA-256, số lượng bản ghi, số lượng tên miền độc lập, overlap và tỷ lệ phân bố nhãn trên từng split.

## Snapshot hiện tại

Theo báo cáo audit được sinh từ snapshot đang có trong máy, dữ liệu thô gồm
395.356 dòng; sau khi loại exact conflict và canonical duplicate còn 395.034
dòng sạch trên 111.874 registered domains. Đây là thông tin provenance, không phải
metric hiệu năng; hãy đọc `artifacts/data_quality_report.json` và
`artifacts/dataset_manifest.json` sau mỗi lần chạy lại pipeline.
