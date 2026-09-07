/**
 * Quản lý tương tác UI và logic đồng bộ cho Popup của PhishGuard ML Chrome Extension.
 */
document.addEventListener("DOMContentLoaded", async () => {
    const shieldToggle = document.getElementById("shieldToggle");
    const apiStatusBox = document.getElementById("apiStatusBox");
    const apiStatusTitle = document.getElementById("apiStatusTitle");
    const currentTabUrl = document.getElementById("currentTabUrl");
    const retryApiBtn = document.getElementById("retryApiBtn");
    const domainInput = document.getElementById("domainInput");
    const addDomainBtn = document.getElementById("addDomainBtn");
    const whitelistContainer = document.getElementById("whitelistContainer");
    const historyContainer = document.getElementById("historyContainer");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");

    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");

    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            tabContents.forEach(c => c.classList.remove("active"));
            btn.classList.add("active");
            document.getElementById(btn.dataset.tab).classList.add("active");
        });
    });

    try {
        const [activeTab] = await chrome.tabs.query({ active: true, currentWindow: true });
        if (activeTab && activeTab.url) {
            currentTabUrl.textContent = activeTab.url;
            currentTabUrl.title = activeTab.url;
        } else {
            currentTabUrl.textContent = "Không tìm thấy URL tab hiện tại";
        }
    } catch {
        currentTabUrl.textContent = "Không thể lấy thông tin tab trình duyệt";
    }

    async function checkApiHealth() {
        apiStatusBox.className = "status-box";
        apiStatusTitle.textContent = "Đang kết nối tới máy chủ API…";

        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), PHISH_GUARD_CONFIG.requestTimeoutMs);

        try {
            const response = await fetch(`${PHISH_GUARD_CONFIG.apiBaseUrl}/health`, {
                signal: controller.signal
            });
            if (!response.ok) throw new Error("API trả về lỗi");
            const data = await response.json();
            apiStatusBox.classList.add("online");
            apiStatusTitle.textContent = `Hệ thống sẵn sàng (${data.model_version || "v3.0.0"})`;
        } catch {
            apiStatusBox.classList.add("offline");
            apiStatusTitle.textContent = "API chưa hoạt động (Offline - Hãy bật FastAPI server)";
        } finally {
            clearTimeout(timeoutId);
        }
    }

    function loadExtensionState() {
        chrome.runtime.sendMessage({ type: "GET_STATE" }, response => {
            if (!response) return;
            shieldToggle.checked = response.shieldEnabled;
            renderWhitelist(response.whitelist || []);
            renderHistory(response.scanHistory || []);
        });
    }

    function renderWhitelist(whitelist) {
        whitelistContainer.replaceChildren();
        if (whitelist.length === 0) {
            const emptyMessage = document.createElement("div");
            emptyMessage.className = "info-desc empty-state";
            emptyMessage.textContent = "Chưa có tên miền nào trong Whitelist";
            whitelistContainer.appendChild(emptyMessage);
            return;
        }
        whitelist.forEach(domain => {
            const item = document.createElement("div");
            item.className = "whitelist-item";
            const domainText = document.createElement("span");
            domainText.textContent = domain;
            const removeButton = document.createElement("button");
            removeButton.className = "btn-danger-sm remove-domain-btn";
            removeButton.dataset.domain = domain;
            removeButton.textContent = "Xóa";
            item.append(domainText, removeButton);
            whitelistContainer.appendChild(item);
        });

        document.querySelectorAll(".remove-domain-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const domainToRemove = btn.dataset.domain;
                chrome.runtime.sendMessage({ type: "REMOVE_WHITELIST", domain: domainToRemove }, res => {
                    if (res && res.whitelist) renderWhitelist(res.whitelist);
                });
            });
        });
    }

    function renderHistory(history) {
        historyContainer.replaceChildren();
        if (history.length === 0) {
            const emptyMessage = document.createElement("div");
            emptyMessage.className = "info-desc empty-state";
            emptyMessage.textContent = "Chưa có lịch sử quét nào";
            historyContainer.appendChild(emptyMessage);
            return;
        }
        history.forEach(item => {
            const action = String(item.action || "allow").toLowerCase();
            const scoreBucket = Number.isFinite(Number(item.score_bucket)) ? Number(item.score_bucket) * 10 : null;
            const badgeClass = action === "block" ? "badge-phish" : action === "caution" ? "badge-warn" : "badge-safe";
            const scoreText = scoreBucket === null ? "" : ` (${scoreBucket}%)`;
            const badgeText = `${action.toUpperCase()}${scoreText}`;

            const historyItem = document.createElement("div");
            historyItem.className = "history-item";
            const urlText = document.createElement("div");
            urlText.className = "history-url";
            urlText.title = item.url;
            urlText.textContent = item.url;
            const metadata = document.createElement("div");
            metadata.className = "history-meta";
            const badge = document.createElement("span");
            badge.className = badgeClass;
            badge.textContent = badgeText;
            const timestamp = document.createElement("span");
            timestamp.className = "history-time";
            timestamp.textContent = item.timestamp || "";
            metadata.append(badge, timestamp);
            historyItem.append(urlText, metadata);
            historyContainer.appendChild(historyItem);
        });
    }

    addDomainBtn.addEventListener("click", () => {
        const val = domainInput.value.trim();
        if (!val) return;
        domainInput.setCustomValidity("");
        chrome.runtime.sendMessage({ type: "ADD_WHITELIST", domain: val }, res => {
            if (res && res.whitelist) {
                renderWhitelist(res.whitelist);
                domainInput.value = "";
            } else if (res?.error) {
                domainInput.setCustomValidity(res.error);
                domainInput.reportValidity();
            }
        });
    });

    shieldToggle.addEventListener("change", () => {
        chrome.runtime.sendMessage({ type: "TOGGLE_SHIELD" });
    });

    clearHistoryBtn.addEventListener("click", () => {
        chrome.runtime.sendMessage({ type: "CLEAR_HISTORY" }, () => {
            renderHistory([]);
        });
    });

    retryApiBtn.addEventListener("click", checkApiHealth);

    checkApiHealth();
    loadExtensionState();
});
