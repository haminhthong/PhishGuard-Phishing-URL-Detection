"""Kiểm tra gate cuối trước khi vận hành artifact; không ghi pointer tự động."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from API.services.model_loader import load_phishguard_model

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(release_dir: Path | None = None) -> None:
    parser = argparse.ArgumentParser(description="Xác minh artifact trước khi đưa vào vận hành")
    parser.add_argument("--release-dir", type=Path)
    if release_dir is None:
        release_dir = parser.parse_args().release_dir
    artifact_dir = (release_dir or PROJECT_ROOT / "artifacts").resolve()
    load_phishguard_model(artifact_dir / "model.json", artifact_dir / "metadata.json")
    stress_report = artifact_dir / "security_stress_metrics.json"
    if stress_report.exists():
        report = json.loads(stress_report.read_text(encoding="utf-8"))
        if report.get("passed") is not True:
            raise RuntimeError("Artifact chưa đạt security stress gate")
    print(f"[OK] Artifact đã qua kiểm tra integrity: {artifact_dir}")


if __name__ == "__main__":
    main()
