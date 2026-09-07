# Model Card — PhishGuard ML XGBoost URL Risk Classifier v4

## 1. Tổng Quan Mô Hình

PhishGuard là **URL-only phishing risk detector**, sử dụng **XGBoost Classifier** với hợp đồng 25 đặc trưng `lexical-v3`. V3 giữ nguyên semantics của v2 nhưng khóa version và checksum của resource lexical/PSL. Release bundle gồm model, calibrator, ActionPolicy và resource contract; API fail-closed nếu một thành phần thiếu hoặc lệch checksum.

- **Tên mô hình:** PhishGuard ML Lexical Risk Classifier
- **Phiên bản:** `4.0.0`
- **Hợp đồng đặc trưng:** `lexical-v3` (25 đặc trưng phân nhóm)
- **Định dạng đóng gói:** Native XGBoost JSON (Không sử dụng Python Pickle runtime)
- **Mục tiêu tối ưu:** Tối đa hóa PR-AUC với ràng buộc nghiêm ngặt về False Positive Rate ($\le 0.5\%$) và độ trễ $p95 \le 5.0\text{ ms}$.

---

## 2. Hợp Đồng Đặc Trưng (Feature Contract v3)

Hệ thống nâng cấp từ 12 đặc trưng phẳng của `lexical-v1` lên **25 đặc trưng** của `lexical-v3` phân thành 4 nhóm bảo mật chuyên sâu. V3 giữ nguyên semantics của v2 để tương thích artifact cũ, đồng thời khóa checksum resource và phiên bản thư viện PSL/TLD.

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
  - **Protocol A (Domain Generalization):** Phân nhóm theo Registered Domain (Mozilla Public Suffix List). Đảm bảo 100% zero overlap cả về domain và URL giữa Train (60%), Validation (15%), Calibration (10%), Policy Validation (5%) và Locked Test (10%).
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
| **XGBoost (Được Chọn)** | **0.9388** | **88.94%** | **95.82%** | **0.36%** | **0.024** | **0.92 ms** |

> [!NOTE]
> **Cơ sở lựa chọn:** Rule-based là sanity baseline, Logistic Regression là baseline ML và XGBoost là canonical production candidate. Kiến trúc được chọn bằng PR-AUC/latency; Recall/FPR được đánh giá sau calibration và ActionPolicy.

---

## 5. Tối Ưu Hóa Ngưỡng Vận Hành (Constrained Threshold Selection)

PhishGuard không chọn một binary operating threshold để điều khiển browser. Calibration chỉ fit calibrator trên tập Calibration; Policy Validation độc lập chọn hai ngưỡng:

- **Chi phí bất đối xứng (Asymmetric Cost Model):** Giả định kịch bản chi phí $C_{FN} = 10$ (phishing lọt qua gây mất tài khoản) và $C_{FP} = 1$ (cảnh báo nhầm website hợp lệ).
- **Ràng buộc tối ưu (Constrained Optimization):**
  $$\max \text{Recall} \quad \text{s.t.} \quad \text{FPR} \le 0.5\%$$
- **Caution:** mục tiêu nhạy hơn, giới hạn FPR riêng.
- **Block:** giới hạn FPR nghiêm ngặt hơn vì gây gián đoạn navigation.

---

## 6. Chính Sách Rủi Ro Đa Tầng (Decoupled Risk Policy)

Hệ thống tách bạch điểm số mô hình (`model_score`) khỏi chính sách hành động:

| Điểm Số Mô Hình (`model_score`) | Mức Độ Rủi Ro (`risk_level`) | Hành Động Hệ Thống (`risk_action`) | Hành Vi Trình Duyệt |
| :--- | :---: | :---: | :--- |
| **$0.00 \le \text{score} < caution_threshold$** | **LOW** | `allow` | Hiển thị `ALLOW`; không đồng nghĩa website an toàn. |
| **$caution_threshold \le \text{score} < block_threshold$** | **MEDIUM** | `caution` | Cảnh báo mềm; không chặn trang. |
| **$\text{score} \ge block_threshold$** | **HIGH** | `block` | Chặn tải trang; điều hướng sang `warning.html`. |

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
