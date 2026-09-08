# Model Card — PhishGuard ML XGBoost URL Risk Classifier v4

## 1. Tổng Quan Mô Hình

PhishGuard là **URL-only phishing risk detector**, sử dụng **XGBoost Classifier** với hợp đồng 25 đặc trưng `lexical-v4`. V4 khóa version/checksum resource và dùng brand matching theo token/hostname label để tránh false positive kiểu `pineapple.example.com`. Release bundle gồm model, calibrator, ActionPolicy và resource contract; API fail-closed nếu một thành phần thiếu hoặc lệch checksum.

- **Tên mô hình:** PhishGuard ML Lexical Risk Classifier
- **Phiên bản:** `4.0.0`
- **Hợp đồng đặc trưng:** `lexical-v4` (25 đặc trưng phân nhóm)
- **Định dạng đóng gói:** Native XGBoost JSON (Không sử dụng Python Pickle runtime)
- **Mục tiêu tối ưu:** Tối đa hóa PR-AUC, sau đó kiểm tra ActionPolicy FPR/recall, calibration ECE và latency theo `configs/train_config.yaml`.

---

## 2. Hợp Đồng Đặc Trưng (Feature Contract v4)

Hệ thống nâng cấp từ 12 đặc trưng phẳng của `lexical-v1` lên **25 đặc trưng** của `lexical-v4` phân thành 4 nhóm bảo mật chuyên sâu. V4 giữ nguyên thứ tự cột của v2/v3 nhưng thay semantics brand matching bằng token/label-aware matching; vì vậy metadata v4 không được dùng chung cho artifact v3.

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
  - Chỉ loại exact canonical URL conflict; registered domain có nhiều nhãn được giữ lại để bảo toàn hard cases shared-hosting.
  - Tổng số bản ghi sạch theo audit snapshot: 395.034 URLs (111.874 registered domains độc lập).
- **Quy tắc chia tập (Domain-Disjoint Split):**
  - **Protocol A (Domain Generalization):** Phân nhóm theo Registered Domain (Mozilla Public Suffix List). Đảm bảo 100% zero overlap cả về domain và canonical URL giữa Train (60%), Validation (15%), Calibration (10%), Policy Validation (5%) và Locked Test (10%).
  - **Protocol B (Future-Phishing Unseen-Domain Stress Test):** Huấn luyện trên các chiến dịch lừa đảo trong quá khứ và kiểm thử trên các phishing domain tương lai (dựa trên `submission_time`). Đây là stress test trên phishing snapshot, chưa phải full concept-drift benchmark.

---

## 4. Benchmark & Tự Động Lựa Chọn Mô Hình (Validation Set)

Các baseline được so sánh trên Validation (registered domains chưa từng xuất hiện
trong tập Train); bảng và số liệu chi tiết nằm trong `artifacts/validation_summary.json`
của từng lần train.

> [!NOTE]
> **Cơ sở lựa chọn:** Rule-based là sanity baseline, Logistic Regression là baseline ML và XGBoost là canonical production candidate. Kiến trúc được chọn bằng PR-AUC/latency trên Validation; Recall/FPR được đánh giá sau calibration và ActionPolicy. Metric thực tế nằm trong `reports/<model_version>/locked_test_metrics.json` hoặc validation summary của lần chạy tương ứng.

---

## 5. Tối Ưu Hóa Ngưỡng Vận Hành (Constrained Threshold Selection)

PhishGuard không chọn một binary operating threshold để điều khiển browser. Calibration chỉ fit calibrator trên tập Calibration; Policy Validation độc lập chọn hai ngưỡng:

- **Chi phí bất đối xứng (Asymmetric Cost Model):** Giả định kịch bản chi phí $C_{FN} = 10$ (phishing lọt qua gây mất tài khoản) và $C_{FP} = 1$ (cảnh báo nhầm website hợp lệ).
- **Ràng buộc tối ưu (Constrained Optimization):** tối đa hóa recall dưới các
  ngưỡng FPR `caution_max_fpr` và `block_max_fpr` trong `configs/train_config.yaml`.
- **Caution:** mục tiêu nhạy hơn, giới hạn FPR riêng.
- **Block:** giới hạn FPR nghiêm ngặt hơn vì gây gián đoạn navigation.

---

## 6. Chính Sách Rủi Ro Đa Tầng (Decoupled Risk Policy)

Hệ thống tách bạch điểm rủi ro (`risk_score`) khỏi chính sách hành động:

| Điểm rủi ro (`risk_score`) | Mức Độ Rủi Ro (`risk_level`) | Hành Động Hệ Thống (`action`) | Hành Vi Trình Duyệt |
| :--- | :---: | :---: | :--- |
| **$0.00 \le \text{score} < caution_threshold$** | **LOW** | `allow` | Hiển thị `ALLOW`; không đồng nghĩa website an toàn. |
| **$caution_threshold \le \text{score} < block_threshold$** | **MEDIUM** | `caution` | Cảnh báo mềm; không chặn trang. |
| **$\text{score} \ge block_threshold$** | **HIGH** | `block` | Chặn tải trang; điều hướng sang `warning.html`. |

---

## 7. Kết Quả Đánh Giá Độc Lập Trên Tập Test (Final Untouched Test)

Đánh giá đúng **1 lần duy nhất** trên Locked Test. Số dòng, số domain, PR-AUC,
ROC-AUC, FPR/recall, ECE/Brier và latency phải đọc từ
`reports/<model_version>/locked_test_metrics.json`; không ghi metric cố định vào
Model Card vì mỗi snapshot dữ liệu và mỗi release có lineage riêng.

---

## 8. Giới Hạn Kỹ Thuật

1. **Giới hạn Lexical-only:** Mô hình chỉ phân tích bề mặt chuỗi URL. Không thể nhận diện các trang lừa đảo nằm trên tên miền sạch bị hack hoặc các cuộc tấn công tinh vi không bộc lộ dấu hiệu qua URL.
2. **False Negatives:** URL-only model có thể bỏ sót phishing dùng URL tự nhiên; đây là lý do hệ thống chỉ là Tier 1 và nên kết hợp kiểm tra nội dung/danh tiếng ở lớp sau.
