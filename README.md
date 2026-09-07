# PhishGuard — Local-First Phishing URL Risk Intelligence Platform

> **A privacy-aware, local-first URL-only phishing risk detector with domain-disjoint evaluation, a versioned lexical feature contract, calibrated risk scoring, and one browser action policy: ALLOW / CAUTION / BLOCK.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost Native JSON](https://img.shields.io/badge/XGBoost-Native%20JSON%20v3.1.0-orange.svg)](https://xgboost.readthedocs.io/)
[![Chrome Extension](https://img.shields.io/badge/Chrome%20Extension-Manifest%20V3-4285F4.svg)](https://developer.chrome.com/docs/extensions/mv3/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Mục lục

1. [Định Nghĩa Bài Toán & Mô Hình Đe Dọa](#1-định-nghĩa-bài-toán--mô-hình-đe-dọa)
2. [Phạm Vi Hệ Thống & Non-Goals](#2-phạm-vi-hệ-thống--non-goals)
3. [Kiến Trúc Hệ Thống (Online & Offline)](#3-kiến-trúc-hệ-thống-online--offline)
4. [Kiến Trúc Quyền Riêng Tư (Privacy Architecture)](#4-kiến-trúc-quyền-riêng-tư-privacy-architecture)
5. [Dữ Liệu Nguồn & Data Card](#5-dữ-liệu-nguồn--data-card)
6. [Chiến Lược Chia Tập Leakage-Safe (Domain-Disjoint Split)](#6-chiến-lược-chia-tập-leakage-safe-domain-disjoint-split)
7. [Hợp Đồng Đặc Trưng 4 Nhóm (lexical-v3)](#7-hợp-đồng-đặc-trưng-4-nhóm-lexical-v3)
8. [Benchmark Baseline & Lựa Chọn Mô Hình Tự Động](#8-benchmark-baseline--lựa-chọn-mô-hình-tự-động)
9. [Chính Sách Ngưỡng & Tối Ưu Chi Phí Rủi Ro (Cost-Aware Thresholds)](#9-chính-sách-ngưỡng--tối-ưu-chi-phí-rủi-ro-cost-aware-thresholds)
10. [Kết Quả Đánh Giá Độc Lập Tập Test (Final Benchmark)](#10-kết-quả-đánh-giá-độc-lập-tập-test-final-benchmark)
11. [Phân Tích Lỗi Chuyên Sâu (Error Analysis)](#11-phân-tích-lỗi-chuyên-sâu-error-analysis)
12. [Đóng Gói & Bảo Mật Model Artifact](#12-đóng-gói--bảo-mật-model-artifact)
13. [Hợp Đồng API & Liveness/Readiness Probes](#13-hợp-đồng-api--livenessreadiness-probes)
14. [Tích Hợp Chrome Extension (Manifest V3)](#14-tích-hợp-chrome-extension-manifest-v3)
15. [Khả Năng Giám Sát (Observability & Monitoring)](#15-khả-năng-giám-sát-observability--monitoring)
16. [Kiểm Thử Invariants & Adversarial Evasion](#16-kiểm-thử-invariants--adversarial-evasion)
17. [Giới Hạn Kỹ Thuật & Định Vị Sản Phẩm](#17-giới-hạn-kỹ-thuật--định-vị-sản-phẩm)
18. [Lộ Trình Phát Triển (Roadmap P0 - P3)](#18-lộ-trình-phát-triển-roadmap-p0---p3)
19. [Trình Bày Trong CV / Phỏng Vấn](#19-trình-bày-trong-cv--phỏng-vấn)

---

## 1. Định Nghĩa Bài Toán & Mô Hình Đe Dọa

Tấn công giả mạo (Phishing) là một trong những vector tấn công phổ biến nhất hiện nay, gây thất thoát hàng triệu USD thông qua việc đánh cắp thông tin xác thực, tài khoản ngân hàng và dữ liệu định danh người dùng.

Tuy nhiên, phần lớn các giải pháp bảo vệ trình duyệt truyền thống phụ thuộc vào cơ sở dữ liệu danh sách đen (Blacklists như Google Safe Browsing), vốn có độ trễ cập nhật từ vài giờ đến vài ngày đối với các chiến dịch tấn công Zero-day mới nổi.

**PhishGuard** giải quyết bài toán này bằng một **URL-only phishing risk detector**: phân tích chuỗi URL hoàn toàn cục bộ trước khi tải trang, không fetch nội dung trang và không gửi dữ liệu người dùng ra Internet. Độ trễ phải được báo cáo tách thành feature extraction, model-only và end-to-end API.

### Mô Hình Đe Dọa (Threats Considered)
- **Cấu trúc URL bất thường:** Lạm dụng ký tự số, ký tự đặc biệt (`%`, `@`, `-`), tỷ lệ entropy cao.
- **Mạo danh thương hiệu (Brand Impersonation):** Chèn từ khóa thương hiệu (`paypal`, `google`, `apple`, v.v.) vào subdomain hoặc đường dẫn path trong khi Registered Domain thuộc kẻ tấn công.
- **Tấn công đồng dạng Punycode (Homograph Attacks):** Sử dụng tiền tố `xn--` để hiển thị chữ cái trông giống chữ Latin.
- **Dịch vụ rút gọn liên kết:** Lạm dụng `bit.ly`, `tinyurl.com`, `t.co` để che giấu máy chủ đích.
- **Chuyển hướng lén (Redirection Patterns):** Khai thác dấu `//` sau giao thức.
- **TLD rủi ro cao:** Các tên miền cấp cao giá rẻ thường xuyên bị lạm dụng (`.xyz`, `.top`, `.icu`, `.buzz`).

---

## 2. Phạm Vi Hệ Thống & Non-Goals

> [!WARNING]
> **Định vị sản phẩm:** PhishGuard là một **Lexical Phishing Risk Detector**, không phải một công cụ bảo mật thay thế toàn diện cho antivirus hoặc Google Safe Browsing.

### Những mối đe dọa NẰM NGOÀI phạm vi (Non-Goals):
1. **Tên miền hợp lệ bị chiếm quyền (Compromised Legitimate Sites):** Tin tặc chiếm quyền điều khiển một trang web uy tín và đặt trang lừa đảo với URL hoàn toàn tự nhiên.
2. **Nội dung trang độc hại (Malicious Page Content):** Hệ thống không cào HTML/DOM hoặc phân tích mã nguồn Javascript phía client.
3. **Mã độc tải về máy (Drive-by Downloads / Malware):** Không phát hiện file nhị phân độc hại.
4. **Tấn công hạ tầng mạng:** Không phát hiện giả mạo DNS (DNS Spoofing/Poisoning) hay chiếm quyền định tuyến BGP.

---

## 3. Kiến Trúc Hệ Thống (Online & Offline)

### 3.1. Kiến Trúc Bảo Vệ Trực Tuyến (Online Protection Architecture)

```text
                  ┌─────────────────────┐
                  │  Chrome Navigation  │
                  └─────────┬───────────┘
                            │ Full URL (in-memory)
                            ▼
                ┌──────────────────────┐
                │ Extension Controller │
                │  (Service Worker)    │
                └─────────┬────────────┘
                          │ POST /phish-url-prediction (localhost)
                          ▼
                ┌──────────────────────┐
                │  FastAPI Validation  │
                │ (Pydantic & Scheme)  │
                └─────────┬────────────┘
                          │
                    Feature Contract (lexical-v3)
                          │
                          ▼
              ┌────────────────────────┐
              │ Lexical Risk Detector  │
              │ XGBoost Native JSON    │
              └──────────┬─────────────┘
                         │
                    Model Score [0.0 - 1.0]
                         │
               ┌─────────▼─────────┐
               │  Action Policy    │
               │ ALLOW/CAUTION/BLOCK│
               └───────┬───────────┘
                       │
          ┌────────────┼─────────────┐
          │            │             │
          LOW         MEDIUM         HIGH
     (< caution)  (caution-block)  (>= block)
          │            │             │
        Allow       Caution        Warning
      (ALLOW Badge) (CAUTION Badge) Warning Page / Block
                       │             │
                       └──────┬──────┘
                              │
                        Allow Once /
                         Whitelist
```

### 3.2. Quy Trình Machine Learning Ngoại Tuyến (Offline ML Pipeline)

Pipeline 7 giai đoạn chuẩn mực từ dữ liệu thô đến đóng gói artifact:

```text
1. DATA INGESTION & AUDIT
Legitimate URLs + Phishing URLs ──► Normalization ──► Deduplication ──► PSL Domain Extraction ──► Audit Report
        │
2. LEAKAGE-SAFE SPLIT
Group by Registered Domain (PSL) ──► Train 60% / Validation 15% / Calibration 10% / Policy Validation 5% / Locked Test 10%
        │
3. FEATURE CONTRACT (lexical-v3)
URL ──► 25 Features: Structure (12) + Host/Domain (8) + Brand Abuse (3) + Heuristics (2)
        │
4. MODEL DEVELOPMENT & BENCHMARK
Rule-based sanity baseline + Logistic Regression baseline + canonical XGBoost ──► PR-AUC/latency selection
        │
5. CALIBRATION & CONSTRAINED THRESHOLD
Calibration Split ──► Isotonic/Sigmoid Scaling ──► Policy Validation ──► Select caution/block thresholds
        │
6. FINAL UNTOUCHED TEST
Evaluate Once ──► PR-AUC, ROC-AUC, Recall, FPR, FNR, Confusion Matrix ──► Granular Error Analysis
        │
7. PACKAGING & INTEGRITY
XGBoost Native JSON + Metadata Schema v2 + SHA-256 Checksums + Quality Gate Verification
```

---

## 4. Kiến Trúc Quyền Riêng Tư (Privacy Architecture)

Một trong những vấn đề nghiêm trọng nhất trong các hệ thống bảo mật ML là **Training-Serving Skew do can thiệp URL không đúng chỗ**:

> [!IMPORTANT]
> **Nguyên tắc phân tách xử lý URL:**
> - **In-Memory Prediction:** Tiện ích gửi **Full URL** (bao gồm cả query string) qua kết nối nội bộ `127.0.0.1:5000`. Điều này cho phép trích xuất đầy đủ các đặc trưng lexical sống còn (`%`, `=`, ký tự số trong query) vốn có mặt khi huấn luyện mô hình.
> - **Sanitized History Logging:** Khi lưu lịch sử quét tại trình duyệt (`chrome.storage.local`), hệ thống **tự động cắt bỏ hoàn toàn Query String và Hash Fragment**, chỉ lưu `origin + pathname` để tuyệt đối không làm lộ Access Token, Session ID hoặc Email nhạy cảm của người dùng.

```text
Full URL (https://site.com/login?token=xyz123)
  ├──► Localhost API (in-memory) ──► Feature Extraction ──► Prediction (Zero Logging)
  │
  └──► sanitizeUrlForHistory()   ──► origin + pathname  ──► chrome.storage.local
```

---

## 5. Dữ Liệu Nguồn & Data Card

Dữ liệu được thu thập và làm sạch từ hai nguồn công khai uy tín:
- **Legitimate URLs (`Data/legit_url.csv`):** 345.741 URL từ danh sách Top Sites toàn cầu (Tranco List, Alexa).
- **Verified Phishing URLs (`Data/verified_online.csv`):** 49.615 URL lừa đảo được cộng đồng an ninh mạng xác thực từ [PhishTank](https://phishtank.org/) (bao gồm mốc thời gian `submission_time`).

### Báo Cáo Làm Sạch & Kiểm Toán Dữ Liệu (Data Quality Audit)
| Chỉ Số Kiểm Toán | Số Lượng Bản Ghi |
| :--- | :---: |
| Tổng bản ghi nguồn ban đầu | 395.356 |
| URL sai cú pháp hoặc vượt quá 2.048 ký tự bị loại | 39 |
| URL trùng lặp hoàn toàn bị loại | 282 |
| Tên miền mâu thuẫn nhãn (xuất hiện ở cả 2 nguồn) | 73 domains |
| Bản ghi mâu thuẫn nhãn bị loại bỏ | 25.920 rows |
| **Tổng số URL sạch đưa vào huấn luyện/kiểm thử** | **369.115 rows** |
| **Số lượng Registered Domains độc lập** | **111.801 domains** |

Mã băm toàn vẹn SHA-256 được lưu tự động tại `artifacts/data_quality_report.json`.

---

## 6. Chiến Lược Chia Tập Leakage-Safe (Domain-Disjoint Split)

> **"Generalization is measured on unseen registered domains, not merely unseen URL strings."**

Nếu chia tập ngẫu nhiên theo từng dòng (Random Row Split), mô hình sẽ thấy các URL cùng tên miền ở cả Train và Test (ví dụ: `paypal-fake.com/page1` ở Train và `paypal-fake.com/page2` ở Test). Điều này dẫn đến hiện tượng rò rỉ dữ liệu (Data Leakage) nghiêm trọng — mô hình chỉ "học vẹt" tên miền thay vì học đặc trưng cấu trúc URL.

### Hai Giao Thức Đánh Giá (Evaluation Protocols)

#### Protocol A — Domain Generalization (Benchmark Chính)
Sử dụng quy tắc Mozilla Public Suffix List (PSL) để trích xuất tên miền đăng ký (Registered Domain):
$$\text{sub.example.co.uk} \longrightarrow \text{example.co.uk}$$
Dữ liệu được chia theo nhóm domain:
- **Train (60%), Validation (15%), Calibration (10%), Policy Validation (5%), Locked Test (10%):** Tất cả registered domain disjoint và tỷ lệ nhãn được cân bằng gần nhau.
- **Validation Set (15%):** 72.339 rows | 16.770 unique domains (dùng để chọn mô hình)
- **Calibration Set (10%):** Dùng để hiệu chuẩn xác suất và tối ưu ngưỡng vận hành
- **Test Set (10% - 15%):** 55.384 rows | 16.771 unseen domains (đánh giá độc lập duy nhất một lần)

> [!NOTE]
> Hệ thống kiểm chứng tự động: **$\text{Domain Overlap} = 0$ và $\text{URL Overlap} = 0$ tuyệt đối giữa các tập.**

#### Protocol B — Temporal Robustness (Đo Lường Concept Drift)
Phishing thay đổi liên tục theo thời gian. Protocol B huấn luyện trên các chiến dịch trong quá khứ và kiểm thử trên các chiến dịch tương lai (dựa trên trường `submission_time`) thông qua script `scripts/temporal_benchmark.py` để đo lường độ suy giảm hiệu năng theo thời gian.

---

## 7. Hợp Đồng Đặc Trưng 4 Nhóm (lexical-v3)

Thay vì 12 đặc trưng phẳng của `lexical-v1`, hợp đồng `lexical-v3` đóng băng **25 đặc trưng** phân cấp. V3 giữ nguyên thứ tự cột của v2 và khóa thêm version/checksum của các resource lexical và thư viện PSL/TLD.

```text
URL
 ├── A. Lexical Structure (12 đặc trưng)
 │    ├── url_length, hostname_length, path_length, query_length
 │    ├── url_entropy (Shannon entropy)
 │    ├── digit_ratio, special_char_ratio
 │    ├── dot_count, hyphen_count, at_count
 │    └── path_depth, first_directory_length
 │
 ├── B. Host & Domain (8 đặc trưng)
 │    ├── subdomain_count, hostname_label_count, max_label_length
 │    ├── has_ip_address (IPv4 / IPv6)
 │    ├── tld_length, domain_length
 │    ├── is_suspicious_tld (.xyz, .top, .icu, .buzz, ...)
 │    └── has_punycode (xn--)
 │
 ├── C. Brand Abuse (3 đặc trưng)
 │    ├── brand_in_subdomain
 │    ├── brand_in_path
 │    └── brand_not_registered_domain (brand token in URL but domain is unverified)
 │
 └── D. Heuristics & Redirection (2 đặc trưng)
      ├── uses_shortening_service (versioned shortener dictionary)
      └── has_redirection_pattern (// outside scheme)
```

Tất cả từ điển đều được version hóa rõ ràng:
- `resources/brand_terms.json` (`brand-terms-v1`)
- `resources/shortener_domains.json` (`shortener-list-v1`)

---

## 8. Benchmark Baseline & Lựa Chọn Mô Hình Tự Động

Pipeline huấn luyện (`scripts/train.py`) đánh giá tự động các mô hình baseline và ứng viên trên tập Validation:

| Mô hình | PR-AUC | Recall | Precision | FPR | FNR | ECE | p95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Most Frequent** | 0.0882 | 0.00% | 0.00% | 0.00% | 100.00% | 0.088 | 0.01 ms |
| **Dummy Stratified** | 0.0875 | 9.05% | 8.10% | 9.92% | 90.95% | 0.124 | 0.08 ms |
| **Rule-based Baseline** | 0.2315 | 1.84% | 81.20% | 0.02% | 98.16% | 0.312 | 0.18 ms |
| **Logistic Regression** | 0.8658 | 77.65% | 94.79% | 0.41% | 22.35% | 0.062 | 0.14 ms |
| **Random Forest** | Không còn dùng trong lifecycle production | — | — | — | — | — | — |
| **XGBoost (Được Chọn)** | **0.9388** | **88.94%** | **95.82%** | **0.36%** | **11.06%** | **0.024** | **0.92 ms** |

### Thuật Toán Lựa Chọn Mô Hình Tự Động
Mã nguồn không hardcode tên mô hình. Thuật toán `select_best_candidate_model` áp dụng:
1. **Guardrails an ninh:** Loại bỏ các mô hình có $\text{Recall} < 80\%$, $\text{FPR} > 1.0\%$, hoặc độ trễ $p95 > 5.0\text{ ms}$.
2. **Tiêu chí xếp hạng:** Tối đa hóa PR-AUC trong các ứng viên đủ điều kiện.
3. **Production candidate:** XGBoost là model canonical vì đạt PR-AUC tốt, latency phù hợp và xuất được Native JSON. Recall/FPR chỉ được đánh giá sau calibration và ActionPolicy, không dùng threshold 0.5 để chọn model.

---

## 9. Chính Sách Ngưỡng & Tối Ưu Chi Phí Rủi Ro (Cost-Aware Thresholds)

Trong bài toán an ninh mạng, chi phí của các loại lỗi là bất đối xứng:
- **False Negative (FN):** Website lừa đảo lọt qua dẫn đến người dùng bị mất tiền hoặc tài khoản cá nhân.
- **False Positive (FP):** Chặn nhầm website hợp lệ gây khó chịu và gián đoạn công việc của người dùng.

Calibration chỉ dùng để fit calibrator. Hai ngưỡng `caution_threshold < block_threshold` được chọn trên **Policy Validation** độc lập; Caution có thể nhạy hơn, còn Block giữ FPR nghiêm ngặt hơn. Không hard-code `0.45`, `0.75` hoặc diễn giải score là prevalence phishing ngoài đời.

---

## 10. Kết Quả Đánh Giá Độc Lập Tập Test (Final Benchmark)

Đánh giá **đúng 1 lần duy nhất** trên tập Test nguyên bản (55.384 bản ghi, 16.771 registered domains chưa từng gặp):

| Metric | Giá Trị | Đánh Giá Ý Nghĩa Kỹ Thuật |
| :--- | :---: | :--- |
| **PR-AUC (Headline Metric)** | **0.9239** | Khả năng phân biệt cực mạnh trên dữ liệu mất cân bằng |
| **ROC-AUC** | **0.9712** | Khả năng xếp hạng tổng quát cao |
| **Precision** | **96.43%** | Khi mô hình báo độc hại, độ tin cậy đạt 96.43% |
| **Recall** | **81.66%** | Phát hiện thành công hơn 81.6% các chiến dịch lừa đảo mới |
| **False Positive Rate (FPR)** | **0.35%** | Chỉ 175 cảnh báo nhầm trên tổng số 49.594 URL hợp lệ |
| **False Negative Rate (FNR)** | **18.34%** | Bỏ sót ~18% — được công bố minh bạch trong Model Card |
| **Brier Score** | **0.0191** | Độ chuẩn xác của điểm số xác suất |
| **Expected Calibration Error (ECE)** | **0.0240** | Điểm số phản ánh đúng tần suất xuất hiện thực tế |
| **Độ trễ suy luận p50 / p95** | **0.69 ms / 0.80 ms** | Đạt chuẩn real-time cho tiện ích mở rộng trình duyệt |

---

## 11. Phân Tích Lỗi Chuyên Sâu (Error Analysis)

Được tự động xuất ra file `artifacts/error_analysis_report.json`:
- **False Positives (Cảnh báo nhầm):** Chiếm ưu thế ở nhóm URL tiếp thị có tham số theo dõi rất dài (Google Adwords, Affiliate tracking) và các đường link SSO chuyển hướng phức tạp.
- **False Negatives (Bỏ sót):** Phần lớn rơi vào các chiến dịch phishing có URL rất ngắn, đường dẫn sạch sẽ và không sử dụng từ khóa thương hiệu phổ biến trong URL — đây là căn cứ xác thực để phát triển phòng thủ đa tầng (Tier 2/Tier 3) ở giai đoạn tiếp theo.

---

## 12. Đóng Gói & Bảo Mật Model Artifact

1. **Native XGBoost JSON:** Xuất trực tiếp mô hình thành file JSON thuần (`model.json`), từ chối `pickle`/`joblib` trong production.
2. **Release bundle v4:** Lưu trữ tại `artifacts/models/phishguard-<version>/` gồm model, feature contract, calibration, `action_policy.json`, resource hashes, split manifest và evaluation report.
   - `model_version`: version của bundle
   - `feature_contract`: `lexical-v3` (25 đặc trưng)
   - `feature_contract_hash`: Băm SHA-256 thứ tự các cột đặc trưng
   - `model_sha256`: Chữ ký băm SHA-256 của file `XGB.json`
   - `action_policy_sha256`, `calibration_sha256`, `resource_hashes`: checksum bắt buộc
   - `quality_gate`: Trạng thái kiểm toán chất lượng trước khi phát hành

---

## 13. Hợp Đồng API & Liveness/Readiness Probes

REST API được xây dựng bằng FastAPI, hỗ trợ đầy đủ các endpoint chuẩn cloud-native:

### Các Endpoint Sức Khỏe & Thống Kê
- `GET /health/live`: Liveness Probe xác nhận tiến trình web đang chạy.
- `GET /health/ready`: Readiness Probe kiểm tra mô hình đã nạp thành công, khớp checksum SHA-256 và hợp đồng đặc trưng sẵn sàng.
- `GET /health`: Thông tin tổng quan về phiên bản mô hình, threshold và cache.
- `GET /model-info`: Chi tiết 25 đặc trưng `lexical-v3` và metrics kiểm thử.
- `GET /stats`: Thống kê Uptime và tỷ lệ Cache Hit Rate (%).
- `DELETE /cache`: Xóa bộ nhớ đệm phục vụ kiểm thử.

### Endpoint Dự Đoán: `POST /phish-url-prediction`
**Request:**
```json
{
  "url": "https://paypal.com.verify-user.attacker.com/login"
}
```

**Response (một action policy canonical):**
```json
{
  "url": "https://paypal.com.verify-user.attacker.com/login",
  "model": {
    "score": 0.9842,
    "threshold": 0.75,
    "version": "3.1.0"
  },
  "phishing_risk_score": 0.9842,
  "action": "block",
  "policy_version": "browser-risk-v1",
  "risk": {
    "level": "high",
    "action": "block"
  },
  "feature_contract": "lexical-v3",
  "cached": false,
  "label": 1,
  "prediction": "Phishing URL",
  "model_score": 0.9842,
  "risk_level": "high",
  "model_version": "3.1.0"
}
```

---

## 14. Tích Hợp Chrome Extension (Manifest V3)

### Cài Đặt Tiện Ích
1. Khởi chạy máy chủ API:
   ```powershell
   python -m API.main
   ```
2. Mở trình duyệt Chrome và truy cập `chrome://extensions`.
3. Bật **Chế độ dành cho nhà phát triển (Developer mode)** ở góc trên bên phải.
4. Nhấn **Tải tiện ích đã giải nén (Load unpacked)** và chọn thư mục `Extension/`.

### Quy Trình Xử Lý Tab An Toàn & Fail-Safe
- **Lọc scheme:** Tự động bỏ qua các URL hệ thống (`chrome://`, `file://`, `chrome-extension://`).
- **Xử lý Race Condition:** Sử dụng `checkToken` gán theo tabId để đảm bảo không bị phản hồi API cũ đè khi người dùng chuyển đổi tab nhanh.
- **Fail-Safe Messaging:** Khi máy chủ API tắt hoặc mất kết nối, tiện ích hiển thị badge `?` (Protection unavailable), **không tự ý chặn web** và **không tuyên bố web an toàn giả tạo**.
- **Cơ chế vượt cảnh báo an toàn:** Cung cấp các nút: "Quay lại an toàn", "Bỏ qua 1 lần (Allow Once)" và "Thêm vào Whitelist".

---

## 15. Khả Năng Giám Sát (Observability & Monitoring)

- **LRU Cache Định Danh Mô Hình:** Cache key được băm theo công thức `SHA256(model_version:feature_contract:url)`. Khi cập nhật mô hình, các cache key cũ tự động hết hiệu lực mà không làm sai lệch dự đoán mới.
- **Logging Bảo Vệ Riêng Tư:** API chỉ ghi hostname. Extension chỉ lưu action, score bucket, policy version và URL đã bỏ query/fragment; không ghi raw URL vào telemetry.

---

## 16. Kiểm Thử Invariants & Adversarial Evasion

Hệ thống sở hữu bộ kiểm thử tự động toàn diện (48 unit & integration tests) kiểm chứng các bất biến bảo mật:

```powershell
python -m unittest discover tests
```

### Các nhóm kiểm thử chính:
1. **Kiểm thử bất biến dữ liệu (`test_training_data.py`):** Kiểm chứng 100% zero domain overlap cho split 5-way, xác minh ngữ nghĩa PSL (`sub.example.co.uk` $\to$ `example.co.uk`).
2. **Kiểm thử đặc trưng 4 nhóm (`test_feature_extraction.py`):** Kiểm tra 23 golden fixtures v1 và các đặc trưng v2 (ratios, entropy, subdomain depth).
3. **Kiểm thử mạo danh thương hiệu (`test_feature_extraction.py`):** Kiểm chứng phát hiện brand trong subdomain/path khi domain không chính thức.
4. **Kiểm thử Hard Negatives (`test_hard_benchmarks.py`):** Đảm bảo các URL hợp lệ phức tạp (Google OAuth, Microsoft SSO, AWS S3 signed link, CloudFront CDN) không bị gán nhãn rủi ro mạo danh.
5. **Kiểm thử Adversarial Evasion (`test_adversarial_evasion.py`):** Kiểm tra URL chứa thông tin xác thực (`user@domain`), punycode (`xn--`), mixed casing, percent encoding, đường dẫn dài 1.800 ký tự.
6. **Kiểm thử API & Integrity (`test_api.py`):** Kiểm tra liveness, readiness, composite cache isolation và từ chối khởi động khi file mô hình bị sai checksum.

---

## 17. Giới Hạn Kỹ Thuật & Định Vị Sản Phẩm

1. **Giới hạn Lexical-only:** Mô hình chỉ nhận diện tín hiệu bất thường trên URL. Score thấp chỉ là `low lexical phishing risk`, không phải cam kết website an toàn; hệ thống không phát hiện website uy tín bị chiếm quyền hoặc shared-hosting phishing có URL tự nhiên.
2. **Khuyến nghị triển khai đa tầng:** PhishGuard nên được sử dụng làm **Tier 1 (Lớp lọc cục bộ đầu tiên)** trong kiến trúc phòng thủ đa tầng:
   - *Tier 1 (Local ML):* Xử lý tức thì các URL rõ ràng an toàn hoặc rõ ràng độc hại.
   - *Tier 2 (Deep Check):* Chỉ truy vấn kiểm tra DNS, tuổi domain hoặc Safe Browsing đối với các URL rơi vào nhóm **MEDIUM Risk** để tối ưu quyền riêng tư và băng thông.

---

## 18. Lộ Trình Phát Triển (Roadmap P0 - P3)

| Mức Độ | Trạng Thái | Nhiệm Vụ Kỹ Thuật |
| :---: | :---: | :--- |
| **🔴 P0** | **ĐÃ HOÀN THÀNH** | Sửa triệt để training-serving skew (Full URL in-memory, Sanitized log history). |
| **🔴 P0** | **ĐÃ HOÀN THÀNH** | Tự động hóa Model Selection với guardrails PR-AUC, Recall $\ge 80\%$, FPR $\le 1.0\%$. |
| **🔴 P0** | **ĐÃ HOÀN THÀNH** | Chọn `caution_threshold` và `block_threshold` trên Policy Validation, tách khỏi Calibration. |
| **🔴 P0** | **ĐÃ HOÀN THÀNH** | Một ActionPolicy canonical trả `ALLOW / CAUTION / BLOCK`, không còn mâu thuẫn threshold. |
| **🔴 P0** | **ĐÃ HOÀN THÀNH** | Xác minh Registered Domain Extraction tuân thủ chuẩn Mozilla PSL. |
| **🟠 P1** | **ĐÃ HOÀN THÀNH** | Hợp đồng 25 đặc trưng `lexical-v3` khóa thứ tự cột và checksum resource. |
| **🟠 P1** | **ĐÃ HOÀN THÀNH** | Tín hiệu Punycode (`xn--`) và mạo danh thương hiệu (`brand_terms.json`). |
| **🟠 P1** | **ĐÃ HOÀN THÀNH** | Báo cáo hiệu chuẩn xác suất (ECE, Brier Score). |
| **🟠 P1** | **ĐÃ HOÀN THÀNH** | Bộ benchmark Hard Negatives và Adversarial Evasion Tests. |
| **🟠 P1** | **ĐÃ HOÀN THÀNH** | Cache key cô lập theo phiên bản mô hình và hợp đồng đặc trưng. |
| **🟡 P2** | **ĐÃ HOÀN THÀNH** | Thực nghiệm Protocol B — Temporal Robustness Benchmark đo concept drift. |
| **🟡 P2** | **ĐÃ HOÀN THÀNH** | Cặp endpoint `/health/live` và `/health/ready`. |
| **🟡 P2** | **ĐÃ HOÀN THÀNH** | Viết lại toàn diện README, Model Card và Security Threat Model. |
| **🟢 P3** | *Kế hoạch tương lai* | Bộ phân tích nội dung trang HTML (Page-Content Classifier) bổ trợ. |
| **🟢 P3** | *Kế hoạch tương lai* | Cơ chế Hybrid: Tự động kích hoạt kiểm tra danh tiếng DNS/Domain Age khi điểm số ở mức MEDIUM. |

---

## 19. Trình Bày Trong CV / Phỏng Vấn

**PhishGuard — Local-First Phishing URL Risk Intelligence Platform**

- Thiết kế và phát triển nền tảng phát hiện URL lừa đảo cục bộ kết hợp **FastAPI Backend**, **XGBoost Native JSON** và **Chrome Extension Manifest V3**, đạt độ trễ suy luận **$p95 = 0.80\text{ ms}$**.
- Giải quyết triệt để vấn đề rò rỉ dữ liệu bằng quy trình **Domain-Disjoint Split (Mozilla PSL)** trên 369.115 bản ghi, bảo đảm 100% zero overlap giữa các tập và đo lường tính tổng quát hóa trên **16.771 unseen registered domains**.
- Nâng cấp hợp đồng đặc trưng lên **`lexical-v3` (25 features)** gồm 4 nhóm logic và resource contract tái lập được.
- Thay thế tối ưu $F_1$ truyền thống bằng **Constrained Cost-Aware Threshold Optimization** ($\max \text{Recall} \text{ s.t. } \text{FPR} \le 0.5\%$), đạt **PR-AUC 0.9239** và **FPR 0.35%** trên tập Test độc lập.
- Xây dựng kiến trúc bảo vệ quyền riêng tư nghiêm ngặt: toàn bộ suy luận thực hiện in-memory qua localhost, loại bỏ Query String trước khi ghi log lịch sử, đóng gói mô hình an toàn không phụ thuộc pickle runtime.

---

## Giấy Phép
Dự án được phát hành mã nguồn mở theo giấy phép [MIT License](LICENSE).
