"""
Canonical Master Pipeline Runner cho PhishGuard ML.
Quản lý vòng đời: Audit -> Split -> Train Candidate -> Locked Test -> Stress Test.
Promote là bước riêng để không đưa model chưa qua gate lên production.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def candidate_dir() -> Path:
    """Lấy candidate version từ config, không đọc production pointer."""
    with (PROJECT_ROOT / "configs" / "train_config.yaml").open(encoding="utf-8") as file:
        version = str((yaml.safe_load(file) or {}).get("model_version", "4.0.0"))
    return PROJECT_ROOT / "releases" / "candidates" / f"phishguard-{version}"


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
    print(" [STAGE 4/5] FINAL LOCKED TEST EVALUATION & HARD SLICES")
    print("=" * 70)
    eval_main(candidate_dir())


def run_stress() -> None:
    """Chạy curated security regression benchmark cho candidate."""
    from scripts.security_stress import main as stress_main

    print("\n" + "=" * 70)
    print(" [STAGE 5/5] CURATED SECURITY STRESS GATES")
    print("=" * 70)
    stress_main(candidate_dir())


def run_promote() -> None:
    """Promote candidate đã qua Locked Test và security gates."""
    from scripts.promote_release import main as promote_main

    print("\n" + "=" * 70)
    print(" [STAGE 1/1] EXPLICIT RELEASE PROMOTION")
    print("=" * 70)
    promote_main(candidate_dir())


def run_all() -> None:
    """Chạy toàn bộ lifecycle từ đầu đến cuối một cách tuần tự."""
    print("=" * 70)
    print(" 🌟 KHỞI CHẠY TOÀN BỘ PHISHGUARD ML END-TO-END CANONICAL PIPELINE")
    print("=" * 70)
    run_audit()
    run_split()
    run_train()
    run_evaluate()
    run_stress()
    print("\n" + "=" * 70)
    print(" ✅ CANDIDATE PIPELINE ĐÃ HOÀN THÀNH; PRODUCTION CHƯA ĐƯỢC THAY ĐỔI")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PhishGuard ML Canonical Pipeline CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "command",
        choices=["run", "audit", "split", "train", "evaluate", "stress", "promote", "package"],
        help=(
            "Lựa chọn giai đoạn thực thi:\n"
            "  run      - Chạy candidate lifecycle (audit -> split -> train -> evaluate -> stress)\n"
            "  audit    - Kiểm toán dữ liệu nguồn và xuất DatasetManifest\n"
            "  split    - Chia 5 tập domain-disjoint có stratification\n"
            "  train    - Train LogReg/XGBoost, calibrate và chọn ActionPolicy\n"
            "  evaluate - Đánh giá Locked Test cho candidate\n"
            "  stress   - Chạy curated security regression benchmark\n"
            "  promote  - Cập nhật current_release.json sau mọi release gate\n"
            "  package  - Alias tương thích cho promote\n"
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
    elif args.command == "stress":
        run_stress()
    elif args.command in {"promote", "package"}:
        run_promote()


if __name__ == "__main__":
    main()
