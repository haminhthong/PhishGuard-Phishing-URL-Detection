"""CLI điều phối duy nhất: audit -> split -> train -> evaluate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def run_audit() -> None:
    """Kiểm tra chất lượng dữ liệu nguồn."""
    from scripts.audit_data import main

    main()


def run_split() -> None:
    """Làm sạch, khử trùng lặp và chia dữ liệu theo registered domain."""
    from scripts.prepare_splits import main

    main()


def run_train() -> None:
    """Chọn model trên validation, refit rồi hiệu chuẩn và chọn thresholds."""
    from scripts.train import main

    main()


def run_evaluate() -> None:
    """Đánh giá test độc lập và báo cáo các edge case."""
    from scripts.evaluate import main

    main()


def run_all() -> None:
    """Chạy toàn bộ quy trình huấn luyện và đánh giá theo đúng thứ tự."""
    run_audit()
    run_split()
    run_train()
    run_evaluate()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pipeline PhishGuard: audit, split, train và evaluate."
    )
    parser.add_argument(
        "command",
        choices=["run", "audit", "split", "train", "evaluate"],
        help="Chọn một bước hoặc run để chạy toàn bộ quy trình.",
    )
    args = parser.parse_args()
    commands = {
        "audit": run_audit,
        "split": run_split,
        "train": run_train,
        "evaluate": run_evaluate,
        "run": run_all,
    }
    commands[args.command]()


if __name__ == "__main__":
    main()
