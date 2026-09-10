"""Kiểm tra đầu vào dùng chung, chạy độc lập với model production."""

import unittest

from pydantic import ValidationError

from API.routes.prediction import BatchURLRequest, URLRequest


class URLValidationTests(unittest.TestCase):
    def test_invalid_ports_and_control_characters_are_rejected(self):
        for url in (
            "https://example.com:wrong/",
            "https://example.com:65536/",
            "https://exa mple.com/",
            "https://example.com/ab\ncd",
            "https://example.com/ab\tcd",
        ):
            with self.subTest(url=url):
                for model, payload in (
                    (URLRequest, {"url": url}),
                    (BatchURLRequest, {"urls": [url]}),
                ):
                    with self.assertRaises(ValidationError):
                        model(**payload)

    def test_valid_ports_ipv6_and_encoded_spaces_are_preserved(self):
        urls = ["https://example.com:443/a%20b", "http://[::1]:8080/"]
        self.assertEqual(BatchURLRequest(urls=urls).urls, urls)
        for url in urls:
            self.assertEqual(URLRequest(url=url).url, url)
