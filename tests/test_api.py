"""
Bộ kiểm thử tích hợp (Integration Tests) mở rộng cho PhishGuard ML FastAPI endpoints.
Kiểm tra toàn bộ REST API contract, validation schema, error code contract, LRU Cache, concurrency và Batch Prediction.
"""

from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from API.app import app
from API.dependencies import get_cache
from API.errors import PhishGuardAPIException
from API.services.model_loader import load_phishguard_model

SCORE_ROUTE = "/v1/score"
BATCH_SCORE_ROUTE = "/v1/score/batch"
LEGACY_SCORE_ROUTE = "/phish-url-prediction"


class RouteRegistrationTests(unittest.TestCase):
    """Kiểm tra contract route không cần nạp model hoặc active release."""

    def test_canonical_and_legacy_prediction_routes_are_registered(self) -> None:
        route_paths = {route.path for route in app.routes}
        self.assertIn(SCORE_ROUTE, route_paths)
        self.assertIn(BATCH_SCORE_ROUTE, route_paths)
        self.assertIn(LEGACY_SCORE_ROUTE, route_paths)
        self.assertIn("/phish-url-prediction/batch", route_paths)


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Khởi tạo TestClient cho ứng dụng FastAPI."""
        registry = Path(__file__).resolve().parents[1] / "releases" / "current_release.json"
        if not registry.is_file():
            raise unittest.SkipTest("Chưa có release được promote; API readiness phải fail-closed")
        cls.client = TestClient(app)

    def setUp(self) -> None:
        """Xóa sạch cache trước mỗi bài test."""
        get_cache().clear()

    def test_health_check_endpoint(self) -> None:
        """Kiểm tra endpoint /health trả về trạng thái 200 OK và metadata mô hình hợp lệ."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "ok")
        self.assertIn("model_version", json_data)
        self.assertIn(json_data["feature_count"], {12, 25})
        self.assertIn(
            json_data["feature_contract"], {"lexical-v1", "lexical-v2", "lexical-v3", "lexical-v4"}
        )
        self.assertIn("action_policy", json_data)

    def test_model_info_endpoint(self) -> None:
        """Kiểm tra endpoint /model-info công bố hợp đồng đặc trưng và version."""
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertIn(json_data["feature_count"], {12, 25})
        self.assertEqual(len(json_data["features"]), json_data["feature_count"])
        self.assertIn(
            json_data["feature_contract"], {"lexical-v1", "lexical-v2", "lexical-v3", "lexical-v4"}
        )

    def test_system_stats_endpoint(self) -> None:
        """Kiểm tra endpoint /stats báo cáo chỉ số thống kê hệ thống và hit rate."""
        response = self.client.get("/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_requests", data)
        self.assertIn("cache_hit_rate_percent", data)
        self.assertIn("cache_capacity", data)

    def test_prediction_response_contract_for_valid_url(self) -> None:
        """Kiểm tra schema phản hồi canonical: risk_score, decision và versions."""
        response = self.client.post(
            SCORE_ROUTE,
            json={"url": "https://google.com"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["url"], "https://google.com")
        self.assertIn("risk_score", data)
        self.assertIn(data["decision"]["risk_level"], {"HIGH", "MEDIUM", "LOW"})
        self.assertIn(data["decision"]["action"], {"ALLOW", "CAUTION", "BLOCK"})
        self.assertIn("release", data["versions"])
        self.assertIn(
            data["versions"]["feature_contract"],
            {"lexical-v1", "lexical-v2", "lexical-v3", "lexical-v4"},
        )
        self.assertFalse(data["cached"])

    def test_legacy_prediction_alias_remains_compatible(self) -> None:
        """Alias cũ vẫn trả cùng schema canonical để hỗ trợ client hiện hữu."""
        response = self.client.post(
            LEGACY_SCORE_ROUTE,
            json={"url": "https://legacy-alias.example"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("risk_score", response.json())

    def test_prediction_lru_cache_hit(self) -> None:
        """Kiểm tra cơ chế LRU Cache Hit khi gửi trùng URL nhiều lần."""
        test_url = "https://example.com/login"

        # Lần 1: Cache Miss
        res1 = self.client.post(SCORE_ROUTE, json={"url": test_url})
        self.assertEqual(res1.status_code, 200)
        self.assertFalse(res1.json()["cached"])

        # Lần 2: Cache Hit
        res2 = self.client.post(SCORE_ROUTE, json={"url": test_url})
        self.assertEqual(res2.status_code, 200)
        self.assertTrue(res2.json()["cached"])

    def test_batch_prediction_endpoint(self) -> None:
        """Kiểm tra endpoint canonical POST /v1/score/batch."""
        urls = [
            "https://google.com",
            "https://github.com",
            "http://192.168.1.1/admin/login",
        ]
        response = self.client.post(BATCH_SCORE_ROUTE, json={"urls": urls})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 3)
        self.assertEqual(len(data["results"]), 3)
        self.assertEqual(data["results"][0]["url"], "https://google.com")
        self.assertEqual(data["results"][2]["url"], "http://192.168.1.1/admin/login")

    def test_batch_rejects_invalid_url(self) -> None:
        """Batch phải từ chối URL sai bằng HTTP 422 thay vì trả nhãn an toàn giả."""
        response = self.client.post(
            BATCH_SCORE_ROUTE,
            json={"urls": ["https://example.com", "not-a-valid-url"]},
        )
        self.assertEqual(response.status_code, 422)
        json_data = response.json()
        self.assertIn("error", json_data)
        self.assertEqual(json_data["error"]["code"], "INVALID_URL")

    def test_rejects_unsupported_url_scheme(self) -> None:
        """API từ chối các URL không thuộc giao thức http/https (mã 422)."""
        response = self.client.post(SCORE_ROUTE, json={"url": "file:///etc/passwd"})
        self.assertEqual(response.status_code, 422)

    def test_rejects_url_without_hostname(self) -> None:
        """API từ chối URL thiếu hostname hợp lệ."""
        response = self.client.post(SCORE_ROUTE, json={"url": "https:///missing-host"})
        self.assertEqual(response.status_code, 422)

    def test_max_url_length_boundary_2048(self) -> None:
        """URL dài đúng 2.048 ký tự hợp lệ phải được xử lý thành công."""
        path = "a" * (2048 - len("https://example.com/"))
        valid_long_url = f"https://example.com/{path}"
        self.assertEqual(len(valid_long_url), 2048)

        response = self.client.post(SCORE_ROUTE, json={"url": valid_long_url})
        self.assertEqual(response.status_code, 200)

    def test_rejects_over_max_url_length_2049(self) -> None:
        """URL dài 2.049 ký tự phải bị từ chối với mã 422."""
        path = "a" * (2049 - len("https://example.com/"))
        invalid_long_url = f"https://example.com/{path}"
        self.assertEqual(len(invalid_long_url), 2049)

        response = self.client.post(SCORE_ROUTE, json={"url": invalid_long_url})
        self.assertEqual(response.status_code, 422)

    def test_batch_limit_exceeded_51_urls(self) -> None:
        """Batch quá 50 URL (51 URLs) phải bị từ chối mã 422."""
        urls = [f"https://example{i}.com" for i in range(51)]
        response = self.client.post(BATCH_SCORE_ROUTE, json={"urls": urls})
        self.assertEqual(response.status_code, 422)

    def test_punycode_domain_support(self) -> None:
        """URL có tên miền Punycode phải được xử lý thành công."""
        response = self.client.post(
            SCORE_ROUTE,
            json={"url": "https://xn--e1afmkfd.xn--p1ai/path"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("risk_score", response.json())

    def test_clear_cache_endpoint(self) -> None:
        """Kiểm tra endpoint DELETE /cache xóa sạch bộ nhớ đệm."""
        self.client.post(SCORE_ROUTE, json={"url": "https://test.org"})
        del_res = self.client.delete("/cache")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["status"], "ok")

    def test_missing_model_file_throws_error(self) -> None:
        """Model loader từ chối mở file mô hình không tồn tại."""
        with self.assertRaises(FileNotFoundError):
            load_phishguard_model(Path("non_existent_model.json"), Path("non_existent_meta.json"))

    def test_reject_pickle_model_format(self) -> None:
        """Model loader từ chối định dạng pickle (.pkl) để tránh rủi ro bảo mật."""
        fake_pkl = Path(__file__).parent / "fake.pkl"
        fake_pkl.touch()
        try:
            with self.assertRaises(PhishGuardAPIException):
                load_phishguard_model(fake_pkl, Path("meta.json"))
        finally:
            if fake_pkl.exists():
                fake_pkl.unlink()

    def test_concurrent_requests_to_same_url(self) -> None:
        """Kiểm tra xử lý đồng thời (Concurrent Threads) đối với cùng một URL mà không gây deadlock hoặc lỗi state."""
        url = "https://concurrent-test-example.com/login"

        def make_request() -> int:
            res = self.client.post(SCORE_ROUTE, json={"url": url})
            return res.status_code

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(20)]
            results = [f.result() for f in futures]

        self.assertTrue(all(code == 200 for code in results))

    def test_batch_preserves_input_order(self) -> None:
        """Batch prediction bảo toàn đúng thứ tự các URL gửi vào."""
        urls = [
            "https://test-order-1.com",
            "https://test-order-2.com",
            "https://test-order-3.com",
        ]
        response = self.client.post(BATCH_SCORE_ROUTE, json={"urls": urls})
        self.assertEqual(response.status_code, 200)

    def test_liveness_and_readiness_endpoints(self) -> None:
        """Kiểm tra các endpoint liveness (/health/live) và readiness (/health/ready)."""
        live_res = self.client.get("/health/live")
        self.assertEqual(live_res.status_code, 200)
        self.assertEqual(live_res.json()["status"], "live")

        ready_res = self.client.get("/health/ready")
        self.assertEqual(ready_res.status_code, 200)
        ready_data = ready_res.json()
        self.assertEqual(ready_data["status"], "ready")
        self.assertIn("model_version", ready_data)
        self.assertIn("action_policy", ready_data)

    def test_decoupled_model_and_risk_policy_response_structure(self) -> None:
        """Kiểm tra API chỉ trả canonical risk score và browser decision."""
        res = self.client.post(SCORE_ROUTE, json={"url": "https://paypal.com"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("risk_score", data)
        self.assertEqual(set(data["decision"]), {"action", "risk_level", "reason"})
        self.assertEqual(
            set(data["signals"]), {"punycode", "brand_mismatch", "shortener", "suspicious_tld"}
        )
        self.assertEqual(
            set(data["versions"]),
            {"release", "model", "feature_contract", "feature_contract_hash", "policy"},
        )

    def test_cache_key_isolated_by_model_version(self) -> None:
        """Kiểm tra cache key thay đổi khi đổi model_version hoặc feature_contract."""
        from API.services.cache import PredictionCache

        url = "https://example.com/check"
        key_v1 = PredictionCache.hash_key(url, model_version="3.0.0", feature_contract="lexical-v1")
        key_v2 = PredictionCache.hash_key(url, model_version="3.1.0", feature_contract="lexical-v2")
        self.assertNotEqual(key_v1, key_v2)
        key_policy = PredictionCache.hash_key(
            url,
            model_version="3.1.0",
            feature_contract="lexical-v2",
            policy_version="browser-risk-v2",
        )
        self.assertNotEqual(key_v2, key_policy)

    def test_checksum_mismatch_raises_integrity_error(self) -> None:
        """Kiểm tra loader từ chối nạp mô hình khi checksum SHA-256 không khớp."""
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            model_file = tmp_path / "model.json"
            meta_file = tmp_path / "meta.json"

            # Sao chép model hiện tại nhưng ghi checksum sai vào meta
            root = Path(__file__).resolve().parents[1]
            registry = __import__("json").loads(
                (root / "releases" / "current_release.json").read_text(encoding="utf-8")
            )
            real_model = root / registry["release_dir"] / "model.json"
            model_file.write_bytes(real_model.read_bytes())
            meta_file.write_text(
                json.dumps(
                    {
                        "model_version": "3.0.0",
                        "feature_contract": "lexical-v1",
                        "threshold": 0.5,
                        "model_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(PhishGuardAPIException) as ctx:
                load_phishguard_model(model_file, meta_file)
            self.assertEqual(ctx.exception.code, "MODEL_INTEGRITY_ERROR")


if __name__ == "__main__":
    unittest.main()
