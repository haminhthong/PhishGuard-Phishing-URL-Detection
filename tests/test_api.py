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


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Khởi tạo TestClient cho ứng dụng FastAPI."""
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
        self.assertEqual(json_data["feature_count"], 12)
        self.assertEqual(json_data["feature_contract"], "lexical-v1")
        self.assertIn("threshold", json_data)

    def test_model_info_endpoint(self) -> None:
        """Kiểm tra endpoint /model-info công bố hợp đồng 12 đặc trưng và version."""
        response = self.client.get("/model-info")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["feature_count"], 12)
        self.assertEqual(len(json_data["features"]), 12)
        self.assertEqual(json_data["feature_contract"], "lexical-v1")

    def test_system_stats_endpoint(self) -> None:
        """Kiểm tra endpoint /stats báo cáo chỉ số thống kê hệ thống và hit rate."""
        response = self.client.get("/stats")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("total_requests", data)
        self.assertIn("cache_hit_rate_percent", data)
        self.assertIn("cache_capacity", data)

    def test_prediction_response_contract_for_valid_url(self) -> None:
        """Kiểm tra schema phản hồi chuẩn: model_score, risk_level, model_version."""
        response = self.client.post(
            "/phish-url-prediction",
            json={"url": "https://google.com"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["url"], "https://google.com")
        self.assertIn("label", data)
        self.assertIn("prediction", data)
        self.assertIn("model_score", data)
        self.assertIn(data["risk_level"], {"high", "medium", "low"})
        self.assertEqual(data["model_version"], "3.0.0")
        self.assertEqual(data["feature_contract"], "lexical-v1")
        self.assertFalse(data["cached"])

    def test_prediction_lru_cache_hit(self) -> None:
        """Kiểm tra cơ chế LRU Cache Hit khi gửi trùng URL nhiều lần."""
        test_url = "https://example.com/login"

        # Lần 1: Cache Miss
        res1 = self.client.post("/phish-url-prediction", json={"url": test_url})
        self.assertEqual(res1.status_code, 200)
        self.assertFalse(res1.json()["cached"])

        # Lần 2: Cache Hit
        res2 = self.client.post("/phish-url-prediction", json={"url": test_url})
        self.assertEqual(res2.status_code, 200)
        self.assertTrue(res2.json()["cached"])

    def test_batch_prediction_endpoint(self) -> None:
        """Kiểm tra endpoint dự đoán hàng loạt POST /phish-url-prediction/batch."""
        urls = [
            "https://google.com",
            "https://github.com",
            "http://192.168.1.1/admin/login",
        ]
        response = self.client.post("/phish-url-prediction/batch", json={"urls": urls})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total"], 3)
        self.assertEqual(len(data["results"]), 3)
        self.assertEqual(data["results"][0]["url"], "https://google.com")
        self.assertEqual(data["results"][2]["url"], "http://192.168.1.1/admin/login")

    def test_batch_rejects_invalid_url(self) -> None:
        """Batch phải từ chối URL sai bằng HTTP 422 thay vì trả nhãn an toàn giả."""
        response = self.client.post(
            "/phish-url-prediction/batch",
            json={"urls": ["https://example.com", "not-a-valid-url"]},
        )
        self.assertEqual(response.status_code, 422)
        json_data = response.json()
        self.assertIn("error", json_data)
        self.assertEqual(json_data["error"]["code"], "INVALID_URL")

    def test_rejects_unsupported_url_scheme(self) -> None:
        """API từ chối các URL không thuộc giao thức http/https (mã 422)."""
        response = self.client.post("/phish-url-prediction", json={"url": "file:///etc/passwd"})
        self.assertEqual(response.status_code, 422)

    def test_rejects_url_without_hostname(self) -> None:
        """API từ chối URL thiếu hostname hợp lệ."""
        response = self.client.post("/phish-url-prediction", json={"url": "https:///missing-host"})
        self.assertEqual(response.status_code, 422)

    def test_max_url_length_boundary_2048(self) -> None:
        """URL dài đúng 2.048 ký tự hợp lệ phải được xử lý thành công."""
        path = "a" * (2048 - len("https://example.com/"))
        valid_long_url = f"https://example.com/{path}"
        self.assertEqual(len(valid_long_url), 2048)

        response = self.client.post("/phish-url-prediction", json={"url": valid_long_url})
        self.assertEqual(response.status_code, 200)

    def test_rejects_over_max_url_length_2049(self) -> None:
        """URL dài 2.049 ký tự phải bị từ chối với mã 422."""
        path = "a" * (2049 - len("https://example.com/"))
        invalid_long_url = f"https://example.com/{path}"
        self.assertEqual(len(invalid_long_url), 2049)

        response = self.client.post("/phish-url-prediction", json={"url": invalid_long_url})
        self.assertEqual(response.status_code, 422)

    def test_batch_limit_exceeded_51_urls(self) -> None:
        """Batch quá 50 URL (51 URLs) phải bị từ chối mã 422."""
        urls = [f"https://example{i}.com" for i in range(51)]
        response = self.client.post("/phish-url-prediction/batch", json={"urls": urls})
        self.assertEqual(response.status_code, 422)

    def test_punycode_domain_support(self) -> None:
        """URL có tên miền Punycode phải được xử lý thành công."""
        response = self.client.post(
            "/phish-url-prediction",
            json={"url": "https://xn--e1afmkfd.xn--p1ai/path"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("model_score", response.json())

    def test_clear_cache_endpoint(self) -> None:
        """Kiểm tra endpoint DELETE /cache xóa sạch bộ nhớ đệm."""
        self.client.post("/phish-url-prediction", json={"url": "https://test.org"})
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
            res = self.client.post("/phish-url-prediction", json={"url": url})
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
        response = self.client.post("/phish-url-prediction/batch", json={"urls": urls})
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        for i, url in enumerate(urls):
            self.assertEqual(results[i]["url"], url)


if __name__ == "__main__":
    unittest.main()
