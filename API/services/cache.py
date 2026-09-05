"""Lớp quản lý LRU Cache an toàn cho môi trường đa luồng với khóa định danh mô hình."""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from threading import RLock
from typing import Any


class PredictionCache:
    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self._cache: OrderedDict[str, Any] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def hash_key(url: str, model_version: str = "default", feature_contract: str = "default") -> str:
        """
        Băm SHA-256 kết hợp model_version + feature_contract + URL.
        Bảo đảm khi mô hình được nâng cấp, các bản ghi cũ sẽ không gây sai lệch verdict.
        """
        raw_key = f"{model_version}:{feature_contract}:{url}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()

    def get(self, url: str, model_version: str = "default", feature_contract: str = "default") -> Any | None:
        key = self.hash_key(url, model_version, feature_contract)
        with self._lock:
            if key in self._cache:
                self.hits += 1
                self._cache.move_to_end(key)
                return self._cache[key]
            self.misses += 1
            return None

    def put(self, url: str, value: Any, model_version: str = "default", feature_contract: str = "default") -> None:
        key = self.hash_key(url, model_version, feature_contract)
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)

    def clear(self) -> int:
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            return count

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            hit_rate = round(self.hits / total * 100.0, 2) if total > 0 else 0.0
            return {
                "hits": self.hits,
                "misses": self.misses,
                "total_requests": total,
                "hit_rate_percent": hit_rate,
                "capacity": self.capacity,
                "used": len(self._cache),
            }
