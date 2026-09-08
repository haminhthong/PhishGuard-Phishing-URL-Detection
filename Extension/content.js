/**
 * Content Script chạy tại thời điểm bắt đầu tải tài liệu (document_start).
 * 
 * Nhiệm vụ:
 * Lắng nghe thông điệp phát hiện website lừa đảo ("PHISHING_DETECTED") từ background.js.
 * Khi phát hiện trang nguy hiểm:
 * 1. Ngừng ngay lập tức quá trình tải trang web độc hại (window.stop()).
 * 2. Điều hướng trình duyệt sang trang cảnh báo an toàn (warning.html) đính kèm % độ tin cậy AI.
 */
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message || message.type !== "PHISHING_DETECTED") return false;
    
    // Phản hồi nhận tín hiệu thành công
    sendResponse({ received: true });

    try {
        // Tạo URL dẫn tới trang cảnh báo warning.html tích hợp sẵn trong Chrome Extension
        const warningUrl = new URL(chrome.runtime.getURL("warning.html"));
        warningUrl.searchParams.set("url", message.url || window.location.href);

        const scoreVal = message.risk_score;
        if (typeof scoreVal === "number") {
            warningUrl.searchParams.set("score", String(scoreVal));
            warningUrl.searchParams.set("confidence", String(scoreVal));
        }

        // Chặn tải trang web nguy hiểm và điều hướng sang warning.html
        window.stop();
        window.location.replace(warningUrl.toString());
    } catch (err) {
        console.error("Lỗi khi điều hướng sang trang cảnh báo PhishGuard ML:", err);
    }

    return false;
});
