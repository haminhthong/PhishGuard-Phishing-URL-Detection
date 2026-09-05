# Model Card — PhishGuard ML XGBoost Risk Classifier v3.1.0

## 1. Tổng Quan Mô Hình

Mô hình phân loại rủi ro URL độc hại (Phishing URL Risk Detector) sử dụng thuật toán **XGBoost Classifier** với hợp đồng 25 đặc trưng cấu trúc `lexical-v2` được chia làm 4 nhóm logic. Mô hình được xuất dưới định dạng **Native XGBoost JSON** (`API/XGB.json`) đi kèm file metadata (`API/model_metadata.json`) có chữ ký băm SHA-256 để chống can thiệp trái phép.

- **Tên mô hình:** PhishGuard ML Lexical Risk Classifier
- **Phiên bản:** `3.1.0` (Hỗ trợ tương thích ngược với mô hình `3.0.0`)
- **Hợp đồng đặc trưng:** `lexical-v2` (25 đặc trưng phân nhóm)
- **Định dạng đóng gói:** Native XGBoost JSON (Không sử dụng Python Pickle runtime)
- **Mục tiêu tối ưu:** Tối đa hóa PR-AUC với ràng buộc nghiêm ngặt về False Positive Rate ($\le 0.5\%$) và độ trễ $p95 \le 5.0\text{ ms}$.

---

## 2. Hợp Đồng Đặc Trưng (Feature Contract v2)

Hệ thống nâng cấp từ 12 đặc trưng phẳng của `lexical-v1` lên **25 đặc trưng** của `lexical-v2` phân thành 4 nhóm bảo mật chuyên sâu:

| Nhóm | Danh sách Đặc trưng | Ý nghĩa Bảo mật |
| :--- | :--- | :--- |
| **A. Cấu trúc Lexical & Tỷ lệ** | `url_length`, `hostname_length`, `path_length`, `query_length`, `url_entropy`, `digit_ratio`, `special_char_ratio`, `dot_count`, `hyphen_count`, `at_count`, `path_depth`, `first_directory_length` | Chuẩn hóa số lượng ký tự theo độ dài URL, đo mức độ hỗn loạn (Shannon entropy) của các chuỗi mã hóa ngẫu nhiên. |
| **B. Host & Tên Miền** | `subdomain_count`, `hostname_label_count`, `max_label_length`, `has_ip_address`, `tld_length`, `domain_length`, `is_suspicious_tld`, `has_punycode` | Phát hiện lạm dụng subdomain sâu, sử dụng địa chỉ IP trực tiếp, TLD rủi ro cao (`.xyz`, `.top`, `.icu`), và tấn công mã hóa Punycode (`xn--`). |
| **C. Mạo Danh Thương Hiệu** | `brand_in_subdomain`, `brand_in_path`, `brand_not_registered_domain` | Đối chiếu với từ điển thương hiệu có version (`resources/brand_terms.json`). Phát hiện thương hiệu xuất hiện trong URL nhưng tên miền đăng ký (PSL) không thuộc về tổ chức chính thức. |
| **D. Heuristics & Chuyển Hướng** | `uses_shortening_service`, `has_redirection_pattern` | Đối chiếu danh sách rút gọn liên kết (`resources/shortener_domains.json`) và phát hiện kỹ thuật lẩn tránh chuyển hướng `//`. |

---

## 3. Dữ Liệu Huấn Luyện & Phương Pháp Chia Tập (No Data Leakage)

- **Dữ liệu nguồn:** 345.741 URL hợp lệ (`Data/legit_url.csv`) và 49.615 URL lừa đảo xác thực từ PhishTank (`Data/verified_online.csv`).
- **Làm sạch & Audit:**
  - Loại bỏ URL sai định dạng, vượt quá giới hạn 2.048 ký tự hoặc thiếu hostname.
  - Khử trùng lặp URL tuyệt đối.
  - Loại bỏ các registered domain mâu thuẫn nhãn xuất hiện ở cả hai tập nguồn.
  - Tổng số bản ghi sạch: 369.115 URLs (111.801 registered domains độc lập).
- **Quy tắc chia tập (Domain-Disjoint Split):**
  - **Protocol A (Domain Generalization):** Phân nhóm theo Registered Domain (Mozilla Public Suffix List). Đảm bảo 100% zero overlap cả về domain và URL giữa Train (65%), Validation (15%), Calibration (10%) và Test (10%).
  - **Protocol B (Temporal Robustness):** Huấn luyện trên các chiến dịch lừa đảo trong quá khứ và kiểm thử trên các chiến dịch tương lai (dựa trên `submission_time`) để lượng hóa mức độ suy giảm (concept drift).

---

## 4. Benchmark & Tự Động Lựa Chọn Mô Hình (Validation Set)

Bảng so sánh trên tập Validation (đánh giá độc lập trên các registered domains chưa từng xuất hiện trong tập Train):

| Mô hình | PR-AUC | Recall | Precision | FPR | ECE | p95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Most Frequent** | 0.0882 | 0.00% | 0.00% | 0.00% | 0.088 | 0.01 ms |
| **Dummy Stratified** | 0.0875 | 9.05% | 8.10% | 9.92% | 0.124 | 0.08 ms |
| **Rule-based Baseline** | 0.2315 | 1.84% | 81.20% | 0.02% | 0.312 | 0.18 ms |
| **Logistic Regression** | 0.8658 | 77.65% | 94.79% | 0.41% | 0.062 | 0.14 ms |
| **Random Forest** | 0.9381 | 86.82% | 97.64% | 0.19% | 0.038 | 8.11 ms |
| **XGBoost (Được Chọn)** | **0.9388** | **88.94%** | **95.82%** | **0.36%** | **0.024** | **0.92 ms** |

> [!NOTE]
> **Cơ sở lựa chọn mô hình tự động:** Random Forest và XGBoost đạt PR-AUC rất sát nhau (0.9381 vs 0.9388). XGBoost được hệ thống lựa chọn tự động nhờ vượt trội về độ trễ suy luận (0.92 ms vs 8.11 ms) và khả năng xuất mô hình an toàn dưới định dạng Native JSON độc lập với runtime pickle.

---

## 5. Tối Ưu Hóa Ngưỡng Vận Hành (Constrained Threshold Selection)

Thay vì chọn ngưỡng đơn thuần bằng $\max F_1$, PhishGuard ML áp dụng quy trình quét threshold có ràng buộc chi phí an ninh thông tin:

- **Chi phí bất đối xứng (Asymmetric Cost Model):** Giả định kịch bản chi phí $C_{FN} = 10$ (phishing lọt qua gây mất tài khoản) và $C_{FP} = 1$ (cảnh báo nhầm website hợp lệ).
- **Ràng buộc tối ưu (Constrained Optimization):**
  $$\max \text{Recall} \quad \text{s.t.} \quad \text{FPR} \le 0.5\%$$
- **Ngưỡng vận hành được chọn:** `threshold = 0.57` (Bảo đảm tỷ lệ cảnh báo nhầm FPR luôn dưới $0.5\%$).

---

## 6. Chính Sách Rủi Ro Đa Tầng (Decoupled Risk Policy)

Hệ thống tách bạch điểm số mô hình (`model_score`) khỏi chính sách hành động:

| Điểm Số Mô Hình (`model_score`) | Mức Độ Rủi Ro (`risk_level`) | Hành Động Hệ Thống (`risk_action`) | Hành Vi Trình Duyệt |
| :--- | :---: | :---: | :--- |
| **$0.00 \le \text{score} < 0.45$** | **LOW** | `allow` | Không gián đoạn; hiển thị Toolbar Badge `SAFE` (xanh lục). |
| **$0.45 \le \text{score} < 0.75$** | **MEDIUM** | `caution` | Cảnh báo mềm; hiển thị Toolbar Badge `WARN` (vàng cam), không chặn trang. |
| **$\text{score} \ge 0.75$** | **HIGH** | `warn` | Chặn tải trang ngay lập tức; điều hướng sang `warning.html` (đỏ). |

---

## 7. Kết Quả Đánh Giá Độc Lập Trên Tập Test (Final Untouched Test)

Đánh giá đúng **1 lần duy nhất** trên tập Test hoàn toàn độc lập (55.384 bản ghi, 16.771 registered domains chưa từng gặp):

- **PR-AUC:** 0.9239
- **ROC-AUC:** 0.9712
- **Precision:** 96.43%
- **Recall:** 81.66%
- **False Positive Rate (FPR):** 0.35% (Chỉ 175 cảnh báo nhầm trên 49.594 URL hợp lệ)
- **False Negative Rate (FNR):** 18.34% (Bỏ sót ~18% phishing — được ghi nhận minh bạch)
- **Brier Score:** 0.0191
- **Expected Calibration Error (ECE):** 0.0240
- **Độ trễ suy luận:** $p50 = 0.69\text{ ms}$, $p95 = 0.80\text{ ms}$

---

## 8. Giới Hạn Kỹ Thuật

1. **Giới hạn Lexical-only:** Mô hình chỉ phân tích bề mặt chuỗi URL. Không thể nhận diện các trang lừa đảo nằm trên tên miền sạch bị hack hoặc các cuộc tấn công tinh vi không bộc lộ dấu hiệu qua URL.
2. **False Negatives:** Tỷ lệ bỏ sót 18.34% chứng minh sự cần thiết của kiến trúc phòng thủ đa tầng (Multi-tier Defense) kết hợp kiểm tra danh tiếng DNS/Certificate khi ở mức MEDIUM risk.
