/**
 * Xử lý tương tác và hành vi người dùng trên giao diện Cảnh báo Nguy Hiểm (warning.html).
 * 
 * Nhiệm vụ:
 * 1. Trích xuất URL độc hại và % độ tin cậy AI từ URL parameters (query string).
 * 2. Xử lý nút "Quay lại an toàn" (Tự động quay lại trang trước đó hoặc đóng tab).
 * 3. Xử lý nút "Thêm vào Whitelist" (Gửi thông điệp ADD_WHITELIST tới background.js).
 * 4. Xử lý nút "Tiếp tục 1 lần" (Gửi thông điệp ALLOW_URL_ONCE để bỏ qua bảo vệ 1 lần).
 */
document.addEventListener("DOMContentLoaded", () => {
    // Đọc tham số truyền tới từ query string
    const params = new URLSearchParams(window.location.search);
    const suspiciousUrl = params.get("url");
    const rawScore = params.get("score") || params.get("confidence");
    const confidence = Number(rawScore);

    // DOM Elements
    const urlElement = document.getElementById("badUrl");
    const confidenceElement = document.getElementById("confidence");
    const backBtn = document.getElementById("backBtn");
    const whitelistBtn = document.getElementById("whitelistBtn");
    const continueBtn = document.getElementById("continueBtn");

    // Hiển thị chuỗi URL nghi vấn
    if (suspiciousUrl) {
        urlElement.textContent = suspiciousUrl;
        urlElement.title = suspiciousUrl;
    }

    // Hiển thị Điểm đánh giá rủi ro của mô hình Machine Learning
    if (Number.isFinite(confidence) && confidence >= 0 && confidence <= 1) {
        confidenceElement.textContent = `Điểm rủi ro mô hình AI: ${(confidence * 100).toFixed(1)}%`;
        confidenceElement.hidden = false;
    }

    // 1. Nút "Quay lại an toàn" (Khuyến nghị)
    backBtn.addEventListener("click", () => {
        if (window.history.length > 1) {
            window.history.back();
        } else {
            window.close();
        }
    });

    // 2. Nút "Thêm vào Whitelist" (Đánh dấu tên miền này là an toàn vĩnh viễn)
    whitelistBtn.addEventListener("click", () => {
        if (!suspiciousUrl) return;
        try {
            const hostname = new URL(suspiciousUrl).hostname;
            chrome.runtime.sendMessage({ type: "ADD_WHITELIST", domain: hostname }, () => {
                window.location.assign(suspiciousUrl);
            });
        } catch {
            window.location.assign(suspiciousUrl);
        }
    });

    // 3. Nút "Tiếp tục 1 lần" (Bỏ qua cảnh báo đúng 1 lần duy nhất)
    continueBtn.addEventListener("click", () => {
        if (!suspiciousUrl) return;
        chrome.runtime.sendMessage({ type: "ALLOW_URL_ONCE", url: suspiciousUrl }, response => {
            if (chrome.runtime.lastError || !response?.allowed) return;
            window.location.assign(suspiciousUrl);
        });
    });
});

