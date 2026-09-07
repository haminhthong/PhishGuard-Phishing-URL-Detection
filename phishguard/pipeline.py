"""
Canonical Master Pipeline Runner cho PhishGuard ML.
Quản lý toàn diện vòng đời: Ingestion -> Split -> Train -> Select -> Refit -> Calibrate -> Freeze -> Evaluate -> Package.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def run_audit() -> None:
    """Thực thi kiểm toán chất lượng dữ liệu nguồn."""
    from scripts.audit_data import main as audit_main
    print("\n" + "=" * 70)
    print(" [STAGE 1/5] DATA AUDIT & QUALITY VERIFICATION")
    print("=" * 70)
    audit_main()


def run_split() -> None:
    """Thực thi chia tập theo Registered Domain (Zero Leakage)."""
    from scripts.prepare_splits import main as split_main
    print("\n" + "=" * 70)
    print(" [STAGE 2/5] STRATIFIED DOMAIN-GROUPED 5-WAY SPLIT")
    print("=" * 70)
    split_main()


def run_train() -> None:
    """Thực thi huấn luyện, so sánh benchmark, refit champion và hiệu chuẩn xác suất."""
    from scripts.train import main as train_main
    print("\n" + "=" * 70)
    print(" [STAGE 3/5] MODEL SELECTION, CALIBRATION & ACTION POLICY")
    print("=" * 70)
    train_main()


def run_evaluate() -> None:
    """Thực thi đánh giá độc lập duy nhất 1 lần trên tập Test (Report Only)."""
    from scripts.evaluate import main as eval_main
    print("\n" + "=" * 70)
    print(" [STAGE 4/5] FINAL UNTOUCHED TEST EVALUATION & HARD SLICES")
    print("=" * 70)
    eval_main()


def run_package() -> None:
    """Đóng gói model artifact và metadata."""
    from scripts.export_model import main as export_main
    print("\n" + "=" * 70)
    print(" [STAGE 5/5] MODEL EXPORT & INTEGRITY PACKAGING")
    print("=" * 70)
    export_main()


def run_all() -> None:
    """Chạy toàn bộ lifecycle từ đầu đến cuối một cách tuần tự."""
    print("=" * 70)
    print(" 🌟 KHỞI CHẠY TOÀN BỘ PHISHGUARD ML END-TO-END CANONICAL PIPELINE")
    print("=" * 70)
    run_audit()
    run_split()
    run_train()
    run_evaluate()
    print("\n" + "=" * 70)
    print(" 🎉 TOÀN BỘ PIPELINE ĐÃ HOÀN THÀNH THÀNH CÔNG!")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhishGuard ML Canonical Pipeline CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["run", "audit", "split", "train", "evaluate", "package"],
        help=(
            "Lựa chọn giai đoạn thực thi:\n"
            "  run      - Chạy toàn bộ lifecycle (audit -> split -> train -> evaluate)\n"
            "  audit    - Kiểm toán dữ liệu nguồn và xuất DatasetManifest\n"
            "  split    - Chia 5 tập domain-disjoint có stratification\n"
            "  train    - Train LogReg/XGBoost, calibrate và chọn ActionPolicy\n"
            "  evaluate - Đánh giá tập Test và Hard Slices (Report Only)\n"
            "  package  - Đóng gói artifact và kiểm tra Quality Gate\n"
        ),
    )

    args = parser.parse_args()

    if args.command == "run":
        run_all()
    elif args.command == "audit":
        run_audit()
    elif args.command == "split":
        run_split()
    elif args.command == "train":
        run_train()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "package":
        run_package()


if __name__ == "__main__":
    main()
