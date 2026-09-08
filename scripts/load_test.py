"""
Kịch bản Locust kiểm thử tải (Load Test) API PhishGuard ML.
Sử dụng danh sách URL đa dạng để kiểm tra latency p95, error rate và cache memory stability.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

SAMPLE_URLS = [
    "https://www.google.com/search?q=test",
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.facebook.com/login",
    "https://github.com/microsoft/vscode",
    "https://stackoverflow.com/questions/123456",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://amazon.com/dp/B08N5WRWNW",
    "https://bit.ly/3xYz12",
    "http://tinyurl.com/secure-login-2026",
    "http://192.168.1.1/admin/login",
    "http://allegrolokalnie.pl-oferta-38752659027.icu/",
    "https://suivre-mes-livraison.com/auth",
    "https://paypal.com-verify-account.security-update.info/login.php",
    "https://bank-of-america.secure-session-id-998822.com/auth/login",
    "https://user:pass@example.com/login",
    "https://example.com/search%20results%21?q=test",
    "https://a.b.c.d.e.subdomain.example.co.uk/file.aspx",
    "http://cutt.ly/campaign2026",
]


class PhishGuardUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task(5)
    def predict_single_url(self) -> None:
        url = random.choice(SAMPLE_URLS)
        self.client.post(
            "/v1/score",
            json={"url": url},
            name="/v1/score [Single]",
        )

    @task(2)
    def predict_batch_urls(self) -> None:
        batch = random.sample(SAMPLE_URLS, k=random.randint(2, 5))
        self.client.post(
            "/v1/score/batch",
            json={"urls": batch},
            name="/v1/score/batch [Batch]",
        )

    @task(1)
    def check_health(self) -> None:
        self.client.get("/health", name="/health")
