/**
 * Tập tin cấu hình hệ thống tập trung cho PhishGuard ML Chrome Extension.
 * Quản lý endpoint API backend, timeout giới hạn và danh sách key lưu trữ local.
 */
const PHISH_GUARD_CONFIG = Object.freeze({
    // Địa chỉ REST API backend FastAPI chạy tại máy cục bộ (localhost)
    apiBaseUrl: "http://127.0.0.1:5000",
    
    // Thời gian chờ tối đa khi gọi API (đơn vị: millisecond)
    requestTimeoutMs: 3500,

    // Các key lưu trữ trong chrome.storage.local
    storageKeys: {
        whitelist: "phishguard_whitelist",
        scanHistory: "phishguard_scan_history",
        shieldEnabled: "phishguard_shield_enabled"
    },

    // Danh sách tên miền an toàn mặc định được tự động bỏ qua kiểm tra (Local Whitelist)
    defaultWhitelist: ["localhost", "127.0.0.1"]
});
