# Model Card — PhishGuard ML XGBoost v3.0.0

## Tổng quan Mô hình

Mô hình phân loại URL hợp lệ (`0`) và URL nghi ngờ phishing (`1`) dựa trên 12 đặc trưng lexical. Mô hình được xuất dưới định dạng Native XGBoost JSON (`API/XGB.json`) đi kèm file metadata (`API/model_metadata.json`) đã được kiểm chứng băm SHA-256.

- **Tên mô hình:** PhishGuard ML XGBoost Classifier
- **Phiên bản:** `3.0.0`
- **Hợp đồng đặc trưng:** `lexical-v1` (12 features)
- **Định dạng:** Native XGBoost JSON (Không sử dụng Python Pickle runtime)
- **Ngày huấn luyện:** 2026-09-02
- **Tạo bởi:** PhishGuard ML Maintainers

## Mục đích Sử dụng

- Cung cấp tín hiệu cảnh báo sớm hỗ trợ người dùng nhận biết URL có dấu hiệu lừa đảo.
- Phục vụ học tập, nghiên cứu và trình diễn portfolio Machine Learning end-to-end.
- **Không dùng làm căn cứ duy nhất** để khẳng định website an toàn tuyệt đối hoặc chặn truy cập hệ thống doanh nghiệp.

## Dữ liệu Huấn luyện & Quy trình Chia tập (No Data Leakage)

- **Dữ liệu nguồn:** 345.741 URL legitimate (`Data/legit_url.csv`) và 49.615 URL phishing (`Data/verified_online.csv`).
- **Làm sạch & Audit:**
  - Loại bỏ 39 URL sai định dạng hoặc vượt 2.048 ký tự.
  - Loại bỏ 282 URL trùng lặp hoàn toàn.
  - Loại bỏ 73 registered domain mâu thuẫn nhãn (25.920 bản ghi xuất hiện ở cả 2 nguồn).
  - Tổng số bản ghi sạch sau audit: 369.115 URL (111.801 registered domains độc lập).
- **Quy tắc chia tập (Domain-Grouped Split):**
  - **Train set (70%):** 241.392 rows (78.260 unique domains).
  - **Validation set (15%):** 72.339 rows (16.770 unique domains).
  - **Test set (15%):** 55.384 rows (16.771 unique domains).
  - **Kiểm chứng Leakage:** Đã xác minh 100% domain overlap = 0 và URL overlap = 0 giữa 3 tập Train, Validation và Test.

## Bảng So sánh Baseline trên Tập Validation

| Mô hình | PR-AUC | Recall | Precision | FPR | FNR | p95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Most Frequent** | 0.0882 | 0.00% | 0.00% | 0.00% | 100.00% | 0.01 ms |
| **Dummy Stratified** | 0.0875 | 9.05% | 8.10% | 9.92% | 90.95% | 0.08 ms |
| **Rule-based Baseline** | 0.2136 | 1.27% | 78.64% | 0.03% | 98.73% | 0.18 ms |
| **Logistic Regression** | 0.8658 | 77.65% | 94.79% | 0.41% | 22.35% | 0.14 ms |
| **Random Forest** | 0.9379 | 86.57% | 97.89% | 0.18% | 13.43% | 8.11 ms |
| **XGBoost (Selected)** | **0.9383** | **88.71%** | **95.79%** | **0.38%** | **11.29%** | **0.92 ms** |

## Kết quả Đánh giá Độc lập trên Tập Test (Final Test Set Benchmark)

Đánh giá đúng **1 lần duy nhất (Evaluate Once)** trên tập Test độc lập (55.384 rows, 16.771 registered domains chưa từng xuất hiện trong tập Train):

- **Accuracy:** 97.77%
- **Precision:** 96.43%
- **Recall:** 81.66%
- **F1 Score:** 0.8843
- **PR-AUC:** 0.9239
- **ROC-AUC:** 0.9712
- **False Positive Rate (FPR):** 0.35% (Chỉ 175 cảnh báo nhầm / 49.594 URL hợp lệ)
- **False Negative Rate (FNR):** 18.34%
- **Brier Score:** 0.0191
- **Inference Latency:** p50 = 0.69 ms, p95 = 0.80 ms
- **Optimal Threshold:** 0.57 (Đã chọn từ tập Validation)

## Mức độ Rủi ro (Risk Levels)

```python
def risk_level(score: float) -> str:
    if score >= 0.80:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"
```

## Giới hạn và Bảo mật

1. Mô hình chỉ sử dụng đặc trưng lexical URL, không kiểm tra nội dung HTML, SSL Certificate hoặc DNS Reputation.
2. Dữ liệu lừa đảo có tính chất concept drift theo thời gian, cần được huấn luyện định kỳ.
3. API từ chối khởi động nếu mô hình không khớp checksum SHA-256 hoặc vi phạm hợp đồng 12 đặc trưng.
