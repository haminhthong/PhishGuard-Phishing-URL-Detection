# PhishGuard ML

PhishGuard ML là hệ thống phát hiện và cảnh báo sớm URL lừa đảo (Phishing) dựa trên 12 đặc trưng lexical của URL. Hệ thống bao gồm một **Chrome Extension (Manifest V3)** và **REST API (FastAPI)** chạy cục bộ, sử dụng mô hình Machine Learning **XGBoost Native JSON v3.0.0**.

> ⚠️ **Phạm vi sản phẩm:** PhishGuard ML chỉ cung cấp tín hiệu cảnh báo hỗ trợ dựa trên đặc trưng cấu trúc URL, không khẳng định website an toàn hoặc độc hại tuyệt đối. Hệ thống không thay thế cho Google Safe Browsing hoặc phần mềm chống mã độc chuyên dụng.

---

## Mục lục

1. [Tính năng Nổi bật](#tính-năng-nổi- bật)
2. [Kiến trúc Hệ thống](#kiến-trúc-hệ-thống)
3. [Cấu trúc Repository](#cấu-trúc-repository)
4. [Cài đặt & Khởi chạy](#cài-đặt--khởi-chạy)
5. [ML Production Pipeline](#ml-production-pipeline)
6. [Hợp đồng API & Cấu trúc Lỗi](#hợp-đồng-api--cấu-trúc-lỗi)
7. [Kết quả Đánh giá Benchmark](#kết-quả-đánh-giá-benchmark)
8. [Cài đặt Chrome Extension](#cài-đặt-chrome-extension)
9. [Bảo mật & Quyền riêng tư](#bảo-mật--quyền-riêng-tư)
10. [Kiểm thử Tải & Unit Tests](#kiểm-thử-tải--unit-tests)
11. [Trình bày trong CV](#trình-bày-trong-cv)

---

## Tính năng Nổi bật

- **Kiểm tra URL Real-time:** Tự động trích xuất 12 đặc trưng lexical khi Chrome Tab chuyển sang trạng thái `loading`.
- **Hợp đồng Đầu ra Minh bạch:** Trả về nhãn dự đoán (`label`), `model_score` (điểm mô hình [0.0 - 1.0]), `risk_level` ("high", "medium", "low"), `model_version` ("3.0.0") và `feature_contract` ("lexical-v1").
- **Bảo vệ Quyền riêng tư (Sanitized History):** Lịch sử quét của Extension chỉ lưu `origin + pathname`, tự động loại bỏ Query String (tránh lộ Access Token, Session ID, Email).
- **Xuất Mô hình An toàn:** Bỏ hoàn toàn việc nạp file pickle/joblib runtime, chuyển sang định dạng **XGBoost Native JSON** kèm metadata xác minh checksum SHA-256.
- **LRU Cache Thread-safe:** Bộ nhớ đệm băm SHA-256 URL tối đa 1.024 bản ghi giúp xử lý truy vấn lặp lại tức thì.
- **Cơ chế Vượt Cảnh báo An toàn:** Cho phép quay lại, tiếp tục đúng một lần (Allow Once) hoặc thêm tên miền vào danh sách trắng (Whitelist).
- **Xử lý Race Condition:** Đảm bảo không bị phản hồi API cũ đè tab khi người dùng đổi tab liên tục.

---

## Kiến trúc Hệ thống

```text
Chrome Browser Tab
    │ Navigation (loading status)
    ▼
Extension (background.js)
    │ Sanitized URL (strip query string)
    │ POST /phish-url-prediction
    ▼
FastAPI Backend (API/app.py)
    │ 1. Pydantic validation (http/https, hostname, max 2048 chars)
    │ 2. Feature Extraction (phishguard.features -> 12 lexical features)
    │ 3. XGBoost Native JSON model + Thread-safe LRU Cache
    ▼
JSON Response Contract
{
  "url": "https://example.com/login",
  "label": 1,
  "prediction": "Phishing URL",
  "model_score": 0.91,
  "risk_level": "high",
  "model_version": "3.0.0",
  "feature_contract": "lexical-v1",
  "cached": false
}
    │
    ▼
Extension Warning Page / Toolbar Badge
```

---

## Cấu trúc Repository

```text
Do_an/
├── API/                        # REST API Backend Modular Package
│   ├── app.py                  # Khởi tạo FastAPI App và Handlers
│   ├── config.py               # Quản lý cấu hình từ biến môi trường
│   ├── dependencies.py         # Dependency Injections
│   ├── errors.py               # Chuẩn hóa hợp đồng lỗi JSON
│   ├── main.py                 # Entry point khởi chạy Development Server
│   ├── api.py                  # Module tương thích ngược
│   ├── XGB.json                # Model XGBoost Native JSON v3.0.0
│   └── model_metadata.json     # Metadata và SHA-256 Checksums
├── phishguard/                 # Core Python Package
│   ├── features/               # Hợp đồng 12 đặc trưng lexical URL
│   │   ├── contract.py
│   │   └── extractor.py
│   └── training/               # Data cleaning, Baseline & Metrics
│       ├── baseline.py
│       ├── data.py
│       └── evaluation.py
├── scripts/                    # ML Production Scripts CLI
│   ├── audit_data.py           # Audit chất lượng dữ liệu nguồn
│   ├── prepare_splits.py       # Chia 70/15/15 Domain-Grouped Split (Parquet)
│   ├── train.py                # Train Baselines, RF, XGBoost & Threshold tuning
│   ├── evaluate.py             # Đánh giá độc lập 1 lần trên Test Set
│   ├── export_model.py         # Đóng gói XGB.json và Metadata
│   └── load_test.py            # Kịch bản kiểm thử tải Locust
├── artifacts/                  # Chứa báo cáo, Parquet splits và JSON artifacts
│   ├── data_quality_report.json
│   ├── validation_summary.json
│   ├── test_evaluation_report.json
│   └── splits/                 # train.parquet, validation.parquet, test.parquet
├── demo/                       # Safe Demo Fixtures
│   ├── safe_urls.json
│   ├── suspicious_urls.json
│   └── expected_flow.md
├── Extension/                  # Chrome Extension Manifest V3
├── notebooks/archive/          # Lưu trữ Jupyter Notebook nghiên cứu ban đầu
├── tests/                      # Pytest integration tests & fixtures
│   └── fixtures/feature_contract.json
├── .env.example
├── MODEL_CARD.md
├── SECURITY.md
├── pyproject.toml
└── requirements.txt
```

---

## Cài đặt & Khởi chạy

### 1. Khởi tạo Môi trường Virtualenv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Khởi chạy REST API Server

```powershell
python -m API.main
```

Máy lưu trữ khởi chạy tại: <http://127.0.0.1:5000>
- **Swagger UI Document:** <http://127.0.0.1:5000/docs>
- **Health Check:** <http://127.0.0.1:5000/health>
- **Model Info:** <http://127.0.0.1:5000/model-info>
- **System Stats:** <http://127.0.0.1:5000/stats>

---

## ML Production Pipeline

Hệ thống tách biệt hoàn toàn notebook nghiên cứu khỏi pipeline sản xuất. Tất cả được vận hành tự động qua các script CLI trong thư mục `scripts/`:

```powershell
# 1. Audit chất lượng dữ liệu nguồn & tạo báo cáo checksum SHA-256
python -m scripts.audit_data

# 2. Chia tập Train/Validation/Test theo Domain (70/15/15) độc lập tuyệt đối
python -m scripts.prepare_splits

# 3. Huấn luyện các mô hình Baseline, Random Forest, XGBoost và chọn Threshold
python -m scripts.train

# 4. Đánh giá duy nhất 1 lần trên tập Test độc lập
python -m scripts.evaluate

# 5. Đóng gói và xuất mô hình JSON an toàn kèm metadata cho API backend
python -m scripts.export_model
```

---

## Hợp đồng API & Cấu trúc Lỗi

### Dự đoán một URL (POST `/phish-url-prediction`)

```json
{
  "url": "https://example.com/login"
}
```

Phản hồi thành công (HTTP 200 OK):

```json
{
  "url": "https://example.com/login",
  "label": 0,
  "prediction": "Legitimate URL",
  "model_score": 0.0412,
  "risk_level": "low",
  "model_version": "3.0.0",
  "feature_contract": "lexical-v1",
  "cached": false
}
```

### Dự đoán hàng loạt (POST `/phish-url-prediction/batch`)

Nhận danh sách tối đa 50 URL. Phản hồi bảo toàn đúng thứ tự các URL gửi vào.

### Cấu trúc Phản hồi Lỗi JSON Chuẩn

Khi URL không hợp lệ (ví dụ: thiếu scheme http/https, thiếu hostname, vượt độ dài 2.048 ký tự), API trả về mã HTTP tương ứng kèm cấu trúc lỗi:

```json
{
  "error": {
    "code": "INVALID_URL",
    "message": "URL phải sử dụng giao thức HTTP hoặc HTTPS và có hostname hợp lệ",
    "request_id": "a1b2c3d4"
  }
}
```

Các mã lỗi chuẩn: `INVALID_URL`, `BATCH_LIMIT_EXCEEDED`, `MODEL_NOT_FOUND`, `MODEL_CONTRACT_MISMATCH`, `MODEL_OUTPUT_INVALID`, `PREDICTION_FAILED`, `RATE_LIMITED`.

---

## Kết quả Đánh giá Benchmark

### Bảng So sánh Mô hình trên Tập Validation (72.339 rows)

| Mô hình | PR-AUC | Recall | Precision | FPR | FNR | p95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Most Frequent** | 0.0882 | 0.00% | 0.00% | 0.00% | 100.00% | 0.01 ms |
| **Dummy Stratified** | 0.0875 | 9.05% | 8.10% | 9.92% | 90.95% | 0.08 ms |
| **Rule-based Baseline** | 0.2136 | 1.27% | 78.64% | 0.03% | 98.73% | 0.18 ms |
| **Logistic Regression** | 0.8658 | 77.65% | 94.79% | 0.41% | 22.35% | 0.14 ms |
| **Random Forest** | 0.9379 | 86.57% | 97.89% | 0.18% | 13.43% | 8.11 ms |
| **XGBoost (Selected)** | **0.9383** | **88.71%** | **95.79%** | **0.38%** | **11.29%** | **0.92 ms** |

### Kết quả Độc lập Tập Test (55.384 rows / 16.771 Unseen Registered Domains)

- **Accuracy:** 97.77%
- **Precision:** 96.43%
- **Recall:** 81.66%
- **F1 Score:** 0.8843
- **PR-AUC:** 0.9239
- **ROC-AUC:** 0.9712
- **False Positive Rate (FPR):** 0.35% (Chỉ 175 cảnh báo nhầm / 49.594 URL hợp lệ)
- **Brier Score:** 0.0191
- **Latency p50 / p95:** 0.69 ms / 0.80 ms

---

## Cài đặt Chrome Extension

1. Khởi chạy máy chủ API backend (`python -m API.main`).
2. Mở trình duyệt Google Chrome và truy cập `chrome://extensions`.
3. Bật **Chế độ dành cho nhà phát triển (Developer mode)** ở góc trên bên phải.
4. Nhấn **Tải tiện ích đã giải nén (Load unpacked)** và chọn thư mục `Extension/`.
5. Tham khảo kịch bản thử nghiệm an toàn tại [demo/expected_flow.md](demo/expected_flow.md).

---

## Bảo mật & Quyền riêng tư

- **Sanitized URL Log & History:** API Backend chỉ ghi hostname vào log. Extension chỉ ghi `origin + pathname` vào `scanHistory`, loại bỏ hoàn toàn Query String để tránh lộ Session Token, Passwords hoặc Email.
- **Loại bỏ Pickle Runtime:** API từ chối tải mô hình `.pkl`, chỉ chấp nhận định dạng Native XGBoost JSON (`XGB.json`) kèm `model_metadata.json` đã xác minh checksum SHA-256.
- **Fail-Safe Bind:** Máy chủ API mặc định chỉ lắng nghe trên giao thức loopback `127.0.0.1`.

---

## Kiểm thử Tải & Unit Tests

### Chạy Bộ Kiểm thử Pytest (29 Integration Tests Pass)

```powershell
python -m pytest -v
```

### Chạy Kiểm thử Tải Locust (Simulated Users)

```powershell
locust -f scripts/load_test.py --host=http://127.0.0.1:5000
```

---

## Trình bày trong CV

**PhishGuard ML — Production ML URL Phishing Detection System**

- Thiết kế và triển khai hệ thống phát hiện phishing end-to-end kết hợp XGBoost Native JSON, FastAPI Backend và Chrome Extension Manifest V3.
- Xử lý triệt để hiện tượng Data Leakage bằng quy trình **Registered Domain-Grouped Split (70/15/15)** trên 369.115 bản ghi, chứng minh 100% domain overlap = 0 giữa các tập.
- Thiết lập pipeline benchmark so sánh với 4 baseline models (Dummy, Rule-based, Logistic Regression, Random Forest), đạt **PR-AUC 0.9239** và **FPR 0.35%** trên tập Test độc lập (16.771 unseen domains).
- Chuẩn hóa backend API theo kiến trúc modular, nâng cao tính bảo mật bằng cơ chế **Sanitized URL History** và bộ kiểm thử tự động 29 pytest suite.

---

## Giấy phép

Mã nguồn dự án được phát hành theo giấy phép [MIT License](LICENSE).
