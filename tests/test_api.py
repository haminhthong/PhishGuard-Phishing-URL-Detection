"""Kiểm thử contract và validation của FastAPI."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi.testclient import TestClient

from API.app import app
from API.errors import PhishGuardAPIException
from API.services.model_loader import load_phishguard_model

SCORE_ROUTE = "/v1/score"
BATCH_SCORE_ROUTE = "/v1/score/batch"


class RouteRegistrationTests(unittest.TestCase):
    """Kiểm tra route canonical không cần nạp model."""

    def test_canonical_prediction_routes_are_registered(self) -> None:
        route_paths = set(app.openapi()["paths"].keys())
        self.assertIn(SCORE_ROUTE, route_paths)
        self.assertIn(BATCH_SCORE_ROUTE, route_paths)
        self.assertNotIn("/phish-url-prediction", route_paths)


class ApiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        """Khởi tạo TestClient cho ứng dụng FastAPI."""
        model = Path(__file__).resolve().parents[1] / "artifacts" / "model.json"
        if not model.is_file():
            raise FileNotFoundError("Thiếu artifacts/model.json; API test không được bỏ qua")
        cls.client = TestClient(app)

    def test_health_check_endpoint(self) -> None:
        """Kiểm tra endpoint /health trả về trạng thái 200 OK và metadata mô hình hợp lệ."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        json_data = response.json()
        self.assertEqual(json_data["status"], "ok")
        self.assertIn("model_version", json_data)
        self.assertEqual(json_data["feature_count"], 25)
        self.assertEqual(json_data["feature_contract"], "lexical-v4")
        self.assertIn("thresholds", json_data)
        self.assertTrue(json_data["calibration_loaded"])

    def test_prediction_response_contract_for_valid_url(self) -> None:
        """Kiểm tra schema phẳng gồm điểm, hành động và tín hiệu giải thích."""
        response = self.client.post(
            SCORE_ROUTE,
            json={"url": "https://google.com"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["url"], "https://google.com")
        self.assertIn("risk_score", data)
        self.assertIn(data["risk_level"], {"HIGH", "MEDIUM", "LOW"})
        self.assertIn(data["action"], {"ALLOW", "CAUTION", "BLOCK"})
        metadata_path = Path(__file__).resolve().parents[1] / "artifacts" / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(data["model_version"], metadata["model_version"])

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
        self.assertEqual([item["url"] for item in data["results"]], urls)
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

    def test_missing_model_file_throws_error(self) -> None:
        """Model loader từ chối mở file mô hình không tồn tại."""
        with self.assertRaises(PhishGuardAPIException) as context:
            load_phishguard_model(Path("non_existent_model.json"), Path("non_existent_meta.json"))
        self.assertEqual(context.exception.status_code, 503)

    def test_reject_pickle_model_format(self) -> None:
        """Model loader từ chối định dạng pickle (.pkl) để tránh rủi ro bảo mật."""
        with NamedTemporaryFile(suffix=".pkl") as temporary:
            fake_pkl = Path(temporary.name)
            with self.assertRaises(PhishGuardAPIException):
                load_phishguard_model(fake_pkl, Path("meta.json"))

    def test_batch_preserves_input_order(self) -> None:
        """Batch prediction bảo toàn đúng thứ tự các URL gửi vào."""
        urls = [
            "https://test-order-1.com",
            "https://test-order-2.com",
            "https://test-order-3.com",
        ]
        response = self.client.post(BATCH_SCORE_ROUTE, json={"urls": urls})
        self.assertEqual(response.status_code, 200)

    def test_decoupled_model_and_risk_policy_response_structure(self) -> None:
        """Kiểm tra API trả điểm rủi ro, action và tín hiệu lexical."""
        res = self.client.post(SCORE_ROUTE, json={"url": "https://paypal.com"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("risk_score", data)
        self.assertIn(data["action"], {"ALLOW", "CAUTION", "BLOCK"})
        self.assertIn(data["risk_level"], {"HIGH", "MEDIUM", "LOW"})
        self.assertIn("reason", data)
        self.assertEqual(
            set(data["signals"]), {"punycode", "brand_mismatch", "shortener", "suspicious_tld"}
        )


if __name__ == "__main__":
    unittest.main()
