# PhishGuard — Local-First Phishing URL Risk Intelligence Platform

> **A privacy-aware, local-first URL-only phishing risk system with domain-disjoint evaluation, release-bound lexical-v4 features, calibrated risk scoring, and one browser action policy: ALLOW / CAUTION / BLOCK.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![XGBoost Native JSON](https://img.shields.io/badge/XGBoost-Native%20JSON-orange.svg)](https://xgboost.readthedocs.io/)
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
7. [Hợp Đồng Đặc Trưng 4 Nhóm (lexical-v4)](#7-hợp-đồng-đặc-trưng-4-nhóm-lexical-v4)
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

**PhishGuard** là hệ thống **URL-only phishing risk**: phân tích full URL qua localhost trước/sớm trong navigation, không fetch HTML, DNS hay reputation service trong v1. Model chỉ tạo risk signal; `ActionPolicy` mới quyết định browser `ALLOW`, `CAUTION` hay `BLOCK`.

### Mô Hình Đe Dọa (Threats Considered)
- **Cấu trúc URL bất thường:** Lạm dụng ký tự số, ký tự đặc biệt (`%`, `@`, `-`), tỷ lệ entropy cao.
- **Mạo danh thương hiệu (Brand Impersonation):** Chèn từ khóa thương hiệu (`paypal`, `google`, `apple`, v.v.) vào subdomain hoặc đường dẫn path trong khi Registered Domain thuộc kẻ tấn công.
- **Tấn công đồng dạng Punycode (Homograph Attacks):** Sử dụng tiền tố `xn--` để hiển thị chữ cái trông giống chữ Latin.
- **Dịch vụ rút gọn liên kết:** Lạm dụng `bit.ly`, `tinyurl.com`, `t.co` để che giấu máy chủ đích.
- **Chuyển hướng lén (Redirection Patterns):** Khai thác dấu `//` sau giao thức.
- **TLD signal yếu:** Một số TLD bị overrepresented trong snapshot phishing và chỉ được dùng như một feature yếu, không phải luật `TLD → phishing`.

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
                    Release-bound Feature Contract (lexical-v4)
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

Pipeline 9 giai đoạn từ dữ liệu thô đến promote release:

```text
1. DATA INGESTION & AUDIT
Legitimate URLs + Phishing URLs ──► Normalization ──► Deduplication ──► PSL Domain Extraction ──► Audit Report
        │
2. LEAKAGE-SAFE SPLIT
Group by Registered Domain (PSL) ──► Train 60% / Validation 15% / Calibration 10% / Policy Validation 5% / Locked Test 10%
        │
3. FEATURE CONTRACT (lexical-v4)
URL ──► 25 Features: Structure (12) + Host/Domain (8) + Brand Abuse (3) + Heuristics (2)
        │
4. MODEL DEVELOPMENT & BENCHMARK
Rule-based sanity baseline + Logistic Regression baseline + canonical XGBoost ──► PR-AUC/latency selection
        │
5. CALIBRATION & CONSTRAINED THRESHOLD
Calibration Split ──► Isotonic/Sigmoid Scaling ──► Policy Validation ──► Select caution/block thresholds
        │
6. LOCKED TEST
Evaluate Once ──► PR-AUC, ROC-AUC, Recall, FPR, FNR, Confusion Matrix ──► Granular Error Analysis
        │
7. CANDIDATE FREEZE
Candidate bundle + calibration + ActionPolicy + frozen resources
        │
8. LOCKED TEST + SECURITY STRESS
Reports có lineage và checksum, không tự promote
        │
9. EXPLICIT PROMOTION
promote_release.py → atomic releases/current_release.json
```

### 3.3. Lệnh chạy đúng thứ tự

```powershell
python -m phishguard.pipeline run       # audit -> split -> train -> locked test -> stress
python -m phishguard.pipeline promote   # chỉ promote candidate đã qua toàn bộ gate
python -m API.main                      # API chỉ ready khi current_release.json hợp lệ
```

`train`, `evaluate` và `stress` chỉ tạo hoặc cập nhật artifact của candidate. Chỉ
`promote_release.py` được phép thay đổi `releases/current_release.json`; nếu chưa
promote release, API chủ động trả trạng thái chưa sẵn sàng thay vì nạp model legacy.

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
| Exact canonical URL conflict bị loại | 2 rows |
| Canonical duplicate bị loại | 281 rows |
| Multi-label registered domains được giữ lại | 73 domains / 25.920 rows |
| **Tổng số URL sạch** | **395.034 rows** |
| **Số lượng Registered Domains độc lập** | **111.874 domains** |

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
- **Development/Validation (15%), Calibration (10%), Policy Validation (5%) và Locked Test (10%):** counts và positive rate luôn lấy từ `artifacts/split_manifest.json` của lần chạy, không hard-code trong README.

> [!NOTE]
> Hệ thống kiểm chứng tự động: **$\text{Domain Overlap} = 0$ và $\text{Canonical URL Overlap} = 0$ tuyệt đối giữa các tập.**

#### Protocol B — Future-Phishing Unseen-Domain Stress Test
Phishing thay đổi liên tục theo thời gian. Benchmark này dùng các chiến dịch trong
quá khứ để kiểm thử trên phishing domain tương lai chưa từng gặp (dựa trên
`submission_time`) thông qua `scripts/temporal_benchmark.py`. Đây là stress test
đo drift trên phishing snapshot, không được gọi là full concept-drift benchmark vì
legitimate holdout không có timestamp tương ứng.

---

## 7. Hợp Đồng Đặc Trưng 4 Nhóm (lexical-v4)

Hợp đồng `lexical-v4` có **25 đặc trưng**, giữ nguyên thứ tự cột của v2/v3 nhưng khóa resource và thay brand matching bằng token/hostname-label aware matching. Vì semantics đã thay đổi, v4 không dùng chung metadata với v3.

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
 │    └── brand_not_registered_domain (brand token/label ngoài registered domain chính thức)
 │
 └── D. Heuristics & Redirection (2 đặc trưng)
      ├── uses_shortening_service (versioned shortener dictionary)
      └── has_redirection_pattern (// outside scheme)
```

Tất cả từ điển đều được version hóa và copy vào chính release bundle:
- `resources/brand_terms.json` (`brand-terms-v1`)
- `resources/shortener_domains.json` (`shortener-list-v1`)
- `resources/suspicious_tlds.json` (`suspicious-tlds-v1`)

---

## 8. Benchmark Baseline & Lựa Chọn Mô Hình Tự Động

Pipeline huấn luyện (`scripts/train.py`) chỉ tạo candidate. Rule-based và Logistic Regression là sanity baselines; XGBoost là production candidate duy nhất nếu vượt ranking/latency gate và các policy gate sau calibration. Không có model zoo hoặc Random Forest trong lifecycle.

### Nguyên tắc chọn candidate
PR-AUC dùng cho ranking offline trên Development/Validation. Rule-based và Logistic
Regression chỉ là sanity baseline; XGBoost Native JSON là model family duy nhất được
đóng gói cho production. Browser gates dùng metric của `ActionPolicy`: Block
FPR/Recall, Caution FPR/Recall, calibration ECE và latency. Threshold không được
chọn trên Locked Test. Số liệu thực tế phải đọc từ report của candidate tương ứng.

---

## 9. Chính Sách Ngưỡng & Tối Ưu Chi Phí Rủi Ro (Cost-Aware Thresholds)

Trong bài toán an ninh mạng, chi phí của các loại lỗi là bất đối xứng:
- **False Negative (FN):** Website lừa đảo lọt qua dẫn đến người dùng bị mất tiền hoặc tài khoản cá nhân.
- **False Positive (FP):** Chặn nhầm website hợp lệ gây khó chịu và gián đoạn công việc của người dùng.

Calibration chỉ lưu method, params và ECE/Brier. Hai ngưỡng `caution_threshold < block_threshold` nằm duy nhất trong `action_policy.json`, được chọn trên Policy Validation với minimum recall. Score là `calibrated lexical phishing risk under benchmark distribution`, không phải xác suất phishing ngoài đời.

---

## 10. Kết Quả Đánh Giá Độc Lập Tập Test (Final Benchmark)

Không ghi metric thủ công vào README. Locked Test được sinh bởi `scripts/evaluate.py` và lưu tại `reports/<model_version>/locked_test_metrics.json`, kèm evaluation run id, split hash, model/calibration/policy/resource hashes. README chỉ mô tả protocol; số liệu phải đọc từ report của release tương ứng.

Report nằm tại `reports/<model_version>/locked_test_metrics.json` và chỉ được tạo
sau khi candidate đã freeze. File report ghi kèm `evaluation_run_id`, split hash,
artifact hashes, hard slices, latency và metrics của hai ActionPolicy. Không copy
metrics thủ công vào README vì sẽ dễ lệch khỏi artifact đang chạy.

---

## 11. Phân Tích Lỗi Chuyên Sâu (Error Analysis)

Được tự động xuất ra `reports/<model_version>/error_analysis.json`:
- **False Positives (Cảnh báo nhầm):** Chiếm ưu thế ở nhóm URL tiếp thị có tham số theo dõi rất dài (Google Adwords, Affiliate tracking) và các đường link SSO chuyển hướng phức tạp.
- **False Negatives (Bỏ sót):** Phần lớn rơi vào các chiến dịch phishing có URL rất ngắn, đường dẫn sạch sẽ và không sử dụng từ khóa thương hiệu phổ biến trong URL — đây là căn cứ xác thực để phát triển phòng thủ đa tầng (Tier 2/Tier 3) ở giai đoạn tiếp theo.

---

## 12. Đóng Gói & Bảo Mật Model Artifact

1. **Native XGBoost JSON:** Xuất trực tiếp mô hình thành file JSON thuần (`model.json`), từ chối `pickle`/`joblib` trong production.
2. **Candidate/release bundle:** Lưu tại `releases/candidates/phishguard-<version>/` gồm model, feature contract, calibration, `action_policy.json`, resources, manifests và candidate report. Chỉ `releases/current_release.json` được dùng làm production pointer sau promote.
   - `model_version`: version của bundle
   - `feature_contract`: `lexical-v4` (25 đặc trưng)
   - `feature_contract_hash`: Băm SHA-256 thứ tự các cột đặc trưng
    - `model_sha256`: Chữ ký băm SHA-256 của file `model.json`
   - `action_policy_sha256`, `calibration_sha256`, `resource_hashes`: checksum bắt buộc
   - `candidate_eligible`: gate offline trước Locked Test; chưa có nghĩa đã production

---

## 13. Hợp Đồng API & Liveness/Readiness Probes

REST API được xây dựng bằng FastAPI, hỗ trợ đầy đủ các endpoint chuẩn cloud-native:

### Các Endpoint Sức Khỏe & Thống Kê
- `GET /health/live`: Liveness Probe xác nhận tiến trình web đang chạy.
- `GET /health/ready`: Readiness Probe kiểm tra mô hình đã nạp thành công, khớp checksum SHA-256 và hợp đồng đặc trưng sẵn sàng.
- `GET /health`: Thông tin tổng quan về release, ActionPolicy và cache.
- `GET /model-info`: Chi tiết contract, release và artifact hashes.
- `GET /stats`: Thống kê Uptime và tỷ lệ Cache Hit Rate (%).
- `DELETE /cache`: Xóa bộ nhớ đệm phục vụ kiểm thử.

### Endpoint Dự Đoán: `POST /v1/score` (route cũ vẫn alias)
**Request:**
```json
{
  "url": "https://paypal.com.verify-user.attacker.com/login"
}
```

**Response canonical:**
```json
{
  "url": "https://paypal.com.verify-user.attacker.com/login",
  "request_id": "req-uuid",
  "risk_score": 0.9842,
  "decision": {
    "action": "BLOCK",
    "risk_level": "HIGH",
    "reason": "LEXICAL_RISK_ABOVE_BLOCK_THRESHOLD"
  },
  "signals": {"punycode": true, "brand_mismatch": true, "shortener": false, "suspicious_tld": false},
   "versions": {"release": "phishguard-4.0.0", "model": "4.0.0", "feature_contract": "lexical-v4", "feature_contract_hash": "sha256...", "policy": "browser-risk-v2"},
  "cached": false
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
- **Fail-Safe Messaging:** Khi máy chủ API tắt hoặc mất kết nối, tiện ích hiển thị badge `?` (Protection unavailable), **không tự ý cho phép hoặc tuyên bố web an toàn**.
- **Navigation semantics:** Đây là early-navigation screening; navigation đã bắt đầu, sau đó mới stop/điều hướng sang warning interstitial khi action là BLOCK.
- **User Trust Override:** Allow Once và whitelist là bypass do người dùng yêu cầu, không phải verdict website an toàn.

---

## 15. Khả Năng Giám Sát (Observability & Monitoring)

- **LRU Cache Định Danh Release:** Cache key được băm theo công thức `SHA256(model_version:feature_contract:policy_version:url)`. Khi cập nhật model, feature contract hoặc ActionPolicy, các cache key cũ tự động hết hiệu lực.
- **Logging Bảo Vệ Riêng Tư:** API chỉ ghi hostname. Extension chỉ lưu action, score bucket, policy version và URL đã bỏ query/fragment; không ghi raw URL vào telemetry.

---

## 16. Kiểm Thử Invariants & Adversarial Evasion

Hệ thống có bộ kiểm thử unit/integration trong `tests/` để kiểm chứng các bất biến bảo mật:

```powershell
python -m unittest discover tests
```

### Các nhóm kiểm thử chính:
1. **Kiểm thử bất biến dữ liệu (`test_training_data.py`):** Kiểm chứng zero domain/canonical URL overlap cho split 5-way, giữ multi-label domain và xác minh PSL.
2. **Kiểm thử contract (`test_feature_extraction.py`):** Kiểm tra v1/v2 legacy và lexical-v4 token-aware brand matching.
3. **Kiểm thử mạo danh thương hiệu (`test_feature_extraction.py`):** Kiểm chứng phát hiện brand trong subdomain/path khi domain không chính thức.
4. **Kiểm thử Hard Negatives (`test_hard_benchmarks.py`):** Đảm bảo các URL hợp lệ phức tạp (Google OAuth, Microsoft SSO, AWS S3 signed link, CloudFront CDN) không bị gán nhãn rủi ro mạo danh.
5. **Kiểm thử Adversarial Evasion (`test_adversarial_evasion.py`):** Kiểm tra URL chứa thông tin xác thực (`user@domain`), punycode (`xn--`), mixed casing, percent encoding, đường dẫn dài 1.800 ký tự.
6. **Kiểm thử API & Integrity (`test_api.py`):** Kiểm tra canonical response, liveness, readiness, cache isolation và fail-closed checksum/release pointer.

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
| **🟠 P1** | **ĐÃ HOÀN THÀNH** | Hợp đồng 25 đặc trưng `lexical-v4` khóa thứ tự cột, token-aware brand semantics và checksum resource. |
| **🟠 P1** | **ĐÃ HOÀN THÀNH** | Tín hiệu Punycode (`xn--`) và mạo danh thương hiệu (`brand_terms.json`). |
| **🟠 P1** | **ĐÃ HOÀN THÀNH** | Báo cáo hiệu chuẩn xác suất (ECE, Brier Score). |
| **🟠 P1** | **ĐÃ HOÀN THÀNH** | Bộ benchmark Hard Negatives và Adversarial Evasion Tests. |
| **🟠 P1** | **ĐÃ HOÀN THÀNH** | Cache key cô lập theo phiên bản mô hình và hợp đồng đặc trưng. |
| **🟡 P2** | **ĐÃ HOÀN THÀNH** | Future-Phishing Unseen-Domain Stress Test; chưa gọi là full concept drift vì legitimate snapshot không có timestamp tương ứng. |
| **🟡 P2** | **ĐÃ HOÀN THÀNH** | Cặp endpoint `/health/live` và `/health/ready`. |
| **🟡 P2** | **ĐÃ HOÀN THÀNH** | Viết lại toàn diện README, Model Card và Security Threat Model. |
| **🟢 P3** | *Kế hoạch tương lai* | Bộ phân tích nội dung trang HTML (Page-Content Classifier) bổ trợ. |
| **🟢 P3** | *Kế hoạch tương lai* | Cơ chế Hybrid: Tự động kích hoạt kiểm tra danh tiếng DNS/Domain Age khi điểm số ở mức MEDIUM. |

---

## 19. Trình Bày Trong CV / Phỏng Vấn

**PhishGuard — Local-First Phishing URL Risk Intelligence Platform**

- Thiết kế local-first URL risk platform với **PSL domain-disjoint 5-way evaluation**, **lexical-v4 release contract**, calibrated XGBoost và constrained ALLOW/CAUTION/BLOCK browser policy.
- Tách candidate → Locked Test → curated security stress → atomic promotion; failed gate không được thay đổi `current_release.json`.
- Giữ full URL trong memory để suy luận, chỉ lưu origin/path trong extension history và chỉ log hostname ở API.

---

## Giấy Phép
Dự án được phát hành mã nguồn mở theo giấy phép [MIT License](LICENSE).
