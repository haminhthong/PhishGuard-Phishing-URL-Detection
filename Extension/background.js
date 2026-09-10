/** Service worker: score navigations, apply local policy, and update the badge. */
importScripts("config.js");

const API_URL = `${PHISH_GUARD_CONFIG.apiBaseUrl}/v1/score`;
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

/** Bỏ query/hash khỏi lịch sử để không lưu token, email hoặc session ID. */
function sanitizeUrlForHistory(urlString) {
    try {
        const parsed = new URL(urlString);
        return `${parsed.origin}${parsed.pathname}`;
    } catch {
        return urlString;
    }
}

/** Khởi tạo storage local khi service worker được cài hoặc đánh thức. */
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

/** Kiểm tra hostname có thuộc whitelist local hay không. */
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

/** Cập nhật badge trên toolbar. */
function updateToolbarBadge(tabId, statusText, colorHex) {
    if (!chrome.action) return;
    chrome.action.setBadgeText({ tabId, text: statusText });
    chrome.action.setBadgeBackgroundColor({ tabId, color: colorHex });
}

/** Lưu tối đa 10 kết quả gần nhất sau khi đã làm sạch URL. */
async function recordScanHistory(url, action, riskScore, riskLevel, modelVersion) {
    try {
        const data = await chrome.storage.local.get(KEYS.scanHistory);
        let history = data[KEYS.scanHistory] || [];
        const sanitizedUrl = sanitizeUrlForHistory(url);
        const newEntry = {
            url: sanitizedUrl,
            action,
            score_bucket: Math.min(10, Math.floor(Number(riskScore || 0) * 10)),
            risk_level: riskLevel,
            model_version: modelVersion || "unknown",
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

/** Gửi URL đầy đủ tới API local để trích xuất feature và chấm điểm. */
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

/** Nhận thao tác từ popup và content script. */
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

/** Chấm URL mới khi tab bắt đầu điều hướng. */
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
    // URL nội bộ của Chrome không đi qua API scoring.
    if (changeInfo.status !== "loading" || !tab.url) return;
    try {
        const parsedUrl = new URL(tab.url);
        if (!["http:", "https:"].includes(parsedUrl.protocol)) return;
    } catch {
        return;
    }

    const storageData = await chrome.storage.local.get(KEYS.shieldEnabled);
    if (storageData[KEYS.shieldEnabled] === false) {
        updateToolbarBadge(tabId, "OFF", "#64748b");
        return;
    }

    if (allowedOnce.delete(tab.url) || (await isDomainWhitelisted(tab.url))) {
        updateToolbarBadge(tabId, "ALLOW", "#22c55e");
        return;
    }

    const checkToken = Symbol(tab.url);
    pendingChecks.set(tabId, checkToken);

    try {
        // Giữ query trong request vì lexical features dùng cả query length/content.
        const result = await fetchPhishingPrediction(tab.url);

        if (pendingChecks.get(tabId) !== checkToken) return;

        const score = Number(result.risk_score);
        const riskLevel = String(result.risk_level ?? "").toLowerCase();
        const riskAction = String(result.action ?? "").toLowerCase();
        const modelVersion = result.model_version;
        if (!Number.isFinite(score) || !["allow", "caution", "block"].includes(riskAction)) {
            throw new Error("API thiếu risk score hoặc action policy hợp lệ");
        }

        await recordScanHistory(tab.url, riskAction, score, riskLevel, modelVersion);

        if (riskAction === "block") {
            updateToolbarBadge(tabId, "BLOCK", "#ef4444");
            chrome.tabs.sendMessage(tabId, {
                type: "PHISHING_DETECTED",
                url: tab.url,
                risk_score: score,
                risk_level: riskLevel,
                action: "block",
                model_version: modelVersion || "unknown"
            }).catch(() => {});
        } else if (riskAction === "caution") {
            updateToolbarBadge(tabId, "CAUTION", "#f59e0b");
        } else {
            updateToolbarBadge(tabId, "ALLOW", "#22c55e");
        }
    } catch (error) {
        // API lỗi thì hiển thị trạng thái chưa có verdict, không giả lập ALLOW.
        console.warn("Dịch vụ PhishGuard ML API offline hoặc không phản hồi:", error);
        updateToolbarBadge(tabId, "?", "#f59e0b");
    } finally {
        if (pendingChecks.get(tabId) === checkToken) {
            pendingChecks.delete(tabId);
        }
    }
});

chrome.tabs.onRemoved.addListener(tabId => pendingChecks.delete(tabId));
