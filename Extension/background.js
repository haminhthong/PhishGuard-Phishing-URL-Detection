/**
 * Service Worker chính của PhishGuard ML Chrome Extension (Manifest V3).
 * 
 * Nhiệm vụ chính:
 * 1. Lắng nghe sự kiện điều hướng tab (chrome.tabs.onUpdated).
 * 2. Kiểm tra danh sách trắng cục bộ (Local Whitelist) và công tắc bảo vệ.
 * 3. Gửi yêu cầu kiểm tra URL tới REST API FastAPI (127.0.0.1:5000/phish-url-prediction).
 * 4. Quản lý Tab Token bất đồng bộ nhằm ngăn chặn hiện tượng Race Condition khi đổi tab nhanh.
 * 5. Cập nhật Dynamic Badge Icon trên Chrome Toolbar (SAFE, WARN, OFF, ?).
 * 6. Lưu trữ lịch sử quét sanitized real-time (không lưu query string nhạy cảm).
 */
importScripts("config.js");

const API_URL = `${PHISH_GUARD_CONFIG.apiBaseUrl}/phish-url-prediction`;
const REQUEST_TIMEOUT_MS = PHISH_GUARD_CONFIG.requestTimeoutMs;
const KEYS = PHISH_GUARD_CONFIG.storageKeys;

const allowedOnce = new Set();
const pendingChecks = new Map();

/** Chuẩn hóa hostname do người dùng nhập. */
function normalizeDomain(value) {
    if (typeof value !== "string") return null;
    const candidate = value.trim().toLowerCase();
    if (!candidate || candidate.includes("/") || candidate.includes(" ")) return null;
    try {
        return new URL(`http://${candidate}`).hostname || null;
    } catch {
        return null;
    }
}

/**
 * Loại bỏ Query String và Hash Fragment khỏi URL trước khi lưu lịch sử
 * để bảo vệ quyền riêng tư người dùng (không làm lộ access token, email, session ID).
 */
function sanitizeUrlForHistory(urlString) {
    try {
        const parsed = new URL(urlString);
        return `${parsed.origin}${parsed.pathname}`;
    } catch {
        return urlString;
    }
}

/**
 * Khởi tạo dữ liệu mặc định trong chrome.storage.local.
 */
async function initializeStorage() {
    try {
        const data = await chrome.storage.local.get([KEYS.whitelist, KEYS.shieldEnabled, KEYS.scanHistory]);
        if (!data[KEYS.whitelist]) {
            await chrome.storage.local.set({ [KEYS.whitelist]: PHISH_GUARD_CONFIG.defaultWhitelist });
        }
        if (data[KEYS.shieldEnabled] === undefined) {
            await chrome.storage.local.set({ [KEYS.shieldEnabled]: true });
        }
        if (!data[KEYS.scanHistory]) {
            await chrome.storage.local.set({ [KEYS.scanHistory]: [] });
        }
    } catch (err) {
        console.error("Lỗi khi khởi tạo bộ nhớ cục bộ extension:", err);
    }
}
initializeStorage();

/**
 * Kiểm tra tên miền (hostname) của URL có thuộc Whitelist an toàn hay không.
 */
async function isDomainWhitelisted(urlStr) {
    try {
        const hostname = new URL(urlStr).hostname.toLowerCase();
        const data = await chrome.storage.local.get(KEYS.whitelist);
        const whitelist = data[KEYS.whitelist] || PHISH_GUARD_CONFIG.defaultWhitelist;
        return whitelist.some(domain => {
            const cleanDomain = domain.toLowerCase().trim();
            return hostname === cleanDomain || hostname.endsWith("." + cleanDomain);
        });
    } catch {
        return false;
    }
}

/**
 * Cập nhật Badge trên Icon tiện ích ở thanh công cụ.
 */
function updateToolbarBadge(tabId, statusText, colorHex) {
    if (!chrome.action) return;
    chrome.action.setBadgeText({ tabId, text: statusText });
    chrome.action.setBadgeBackgroundColor({ tabId, color: colorHex });
}

/**
 * Lưu vết kết quả quét URL đã làm sạch vào lịch sử (Tối đa 10 mục mới nhất).
 */
async function recordScanHistory(url, label, modelScore, riskLevel) {
    try {
        const data = await chrome.storage.local.get(KEYS.scanHistory);
        let history = data[KEYS.scanHistory] || [];
        const sanitizedUrl = sanitizeUrlForHistory(url);
        const newEntry = {
            url: sanitizedUrl,
            label,
            model_score: modelScore,
            risk_level: riskLevel,
            timestamp: new Date().toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" })
        };
        history = history.filter(item => item.url !== sanitizedUrl);
        history.unshift(newEntry);
        if (history.length > 10) history.pop();
        await chrome.storage.local.set({ [KEYS.scanHistory]: history });
    } catch (err) {
        console.error("Lỗi khi ghi lịch sử quét an toàn:", err);
    }
}

/**
 * Gửi HTTP POST request tới máy chủ FastAPI để trích xuất đặc trưng và dự đoán URL.
 */
async function fetchPhishingPrediction(url) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    try {
        const response = await fetch(API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url }),
            signal: controller.signal
        });
        if (!response.ok) throw new Error(`Máy chủ API trả về mã lỗi HTTP ${response.status}`);
        return await response.json();
    } finally {
        clearTimeout(timeoutId);
    }
}

/**
 * Bộ lắng nghe thông điệp từ Popup UI hoặc Content Script.
 */
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (!message || typeof message.type !== "string") return false;

    switch (message.type) {
        case "ALLOW_URL_ONCE":
            if (typeof message.url === "string") {
                allowedOnce.add(message.url);
                sendResponse({ allowed: true });
            }
            break;

        case "GET_STATE":
            chrome.storage.local.get([KEYS.whitelist, KEYS.shieldEnabled, KEYS.scanHistory]).then(data => {
                sendResponse({
                    whitelist: data[KEYS.whitelist] || [],
                    shieldEnabled: data[KEYS.shieldEnabled] !== false,
                    scanHistory: data[KEYS.scanHistory] || []
                });
            });
            return true;

        case "TOGGLE_SHIELD":
            chrome.storage.local.get(KEYS.shieldEnabled).then(data => {
                const newState = !(data[KEYS.shieldEnabled] !== false);
                chrome.storage.local.set({ [KEYS.shieldEnabled]: newState }).then(() => {
                    sendResponse({ shieldEnabled: newState });
                });
            });
            return true;

        case "ADD_WHITELIST":
            {
                const domainToAdd = normalizeDomain(message.domain);
                if (!domainToAdd) {
                    sendResponse({ success: false, error: "Tên miền không hợp lệ" });
                    break;
                }
                chrome.storage.local.get(KEYS.whitelist).then(data => {
                    const currentList = [...(data[KEYS.whitelist] || [])];
                    if (!currentList.includes(domainToAdd)) {
                        currentList.push(domainToAdd);
                        chrome.storage.local.set({ [KEYS.whitelist]: currentList }).then(() => {
                            sendResponse({ success: true, whitelist: currentList });
                        });
                    } else {
                        sendResponse({ success: true, whitelist: currentList });
                    }
                });
                return true;
            }

        case "REMOVE_WHITELIST":
            {
                const domainToRemove = normalizeDomain(message.domain);
                if (!domainToRemove) {
                    sendResponse({ success: false, error: "Tên miền không hợp lệ" });
                    break;
                }
                chrome.storage.local.get(KEYS.whitelist).then(data => {
                    const updatedList = (data[KEYS.whitelist] || []).filter(
                        domain => domain !== domainToRemove
                    );
                    chrome.storage.local.set({ [KEYS.whitelist]: updatedList }).then(() => {
                        sendResponse({ success: true, whitelist: updatedList });
                    });
                });
                return true;
            }

        case "CLEAR_HISTORY":
            chrome.storage.local.set({ [KEYS.scanHistory]: [] }).then(() => {
                sendResponse({ success: true });
            });
            return true;
    }
    return false;
});

/**
 * Lắng nghe sự kiện cập nhật URL của các Tab trong Chrome.
 */
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
    if (changeInfo.status !== "loading" || !tab.url?.startsWith("http")) return;

    const storageData = await chrome.storage.local.get(KEYS.shieldEnabled);
    if (storageData[KEYS.shieldEnabled] === false) {
        updateToolbarBadge(tabId, "OFF", "#64748b");
        return;
    }

    if (allowedOnce.delete(tab.url) || (await isDomainWhitelisted(tab.url))) {
        updateToolbarBadge(tabId, "SAFE", "#22c55e");
        return;
    }

    const checkToken = Symbol(tab.url);
    pendingChecks.set(tabId, checkToken);

    try {
        const result = await fetchPhishingPrediction(tab.url);

        if (pendingChecks.get(tabId) !== checkToken) return;

        const score = result.model_score !== undefined ? result.model_score : result.confidence;
        await recordScanHistory(tab.url, result.label, score, result.risk_level);

        if (result.label === 1) {
            updateToolbarBadge(tabId, "WARN", "#ef4444");
            chrome.tabs.sendMessage(tabId, {
                type: "PHISHING_DETECTED",
                url: tab.url,
                model_score: score,
                risk_level: result.risk_level
            }).catch(() => {});
        } else {
            updateToolbarBadge(tabId, "SAFE", "#22c55e");
        }
    } catch (error) {
        console.warn("Lỗi kết nối API phân tích URL (Hãy khởi chạy server FastAPI tại http://127.0.0.1:5000):", error);
        updateToolbarBadge(tabId, "?", "#f59e0b");
    } finally {
        if (pendingChecks.get(tabId) === checkToken) {
            pendingChecks.delete(tabId);
        }
    }
});

chrome.tabs.onRemoved.addListener(tabId => pendingChecks.delete(tabId));
