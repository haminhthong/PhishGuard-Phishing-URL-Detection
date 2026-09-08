"""Promote candidate sau Locked Test và security stress gates.

Script này là nơi duy nhất được phép cập nhật ``releases/current_release.json``.
Training và evaluation chỉ tạo artifact/report, tuyệt đối không đổi production.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RELEASES_DIR = PROJECT_ROOT / "releases"
CANDIDATES_DIR = RELEASES_DIR / "candidates"
CURRENT_RELEASE = RELEASES_DIR / "current_release.json"
CONFIG_PATH = PROJECT_ROOT / "configs" / "train_config.yaml"


class ReleaseRejected(RuntimeError):
    """Candidate không đạt một hoặc nhiều release gate."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ReleaseRejected(f"Thiếu artifact bắt buộc: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseRejected(f"Không đọc được {path}: {error}") from error
    if not isinstance(value, dict):
        raise ReleaseRejected(f"Artifact phải là JSON object: {path}")
    return value


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleaseRejected(message)


def verify_hash(path: Path, expected: str | None, name: str) -> None:
    require(path.is_file(), f"Thiếu {name}: {path}")
    require(
        expected and sha256_file(path).lower() == str(expected).lower(), f"Checksum sai: {name}"
    )


def validate_candidate(candidate_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = read_json(candidate_dir / "metadata.json")
    manifest = read_json(candidate_dir / "candidate_manifest.json")
    evaluation = read_json(candidate_dir / "evaluation.json")
    stress = read_json(candidate_dir / "security_stress_metrics.json")

    require(manifest.get("candidate_eligible") is True, "Candidate bị train gate từ chối")
    require(metadata.get("release_id") == manifest.get("release_id"), "Release lineage không khớp")
    require(
        manifest.get("feature_contract_hash") == metadata.get("feature_contract_hash"),
        "Candidate manifest sai feature contract hash",
    )

    verify_hash(candidate_dir / "model.json", metadata.get("model_sha256"), "model")
    verify_hash(
        candidate_dir / "calibration.json", metadata.get("calibration_sha256"), "calibration"
    )
    verify_hash(
        candidate_dir / "action_policy.json", metadata.get("action_policy_sha256"), "action_policy"
    )
    require(
        manifest.get("model_sha256") == metadata.get("model_sha256")
        and manifest.get("calibration_sha256") == metadata.get("calibration_sha256")
        and manifest.get("action_policy_sha256") == metadata.get("action_policy_sha256"),
        "Candidate manifest sai checksum artifact",
    )

    evaluation_hashes = evaluation.get("artifact_hashes", {})
    for name in ("model", "calibration", "action_policy", "feature_contract"):
        require(
            evaluation_hashes.get(name)
            == metadata.get(f"{name}_sha256", metadata.get("feature_contract_hash")),
            f"Evaluation không cùng {name} với serving artifact",
        )
    require(
        evaluation_hashes.get("resources") == metadata.get("resource_hashes"),
        "Evaluation không cùng resources với serving artifact",
    )
    require(evaluation.get("release") == metadata.get("release_id"), "Evaluation sai release")
    require(evaluation.get("locked_test") is True, "Thiếu cờ locked_test")
    require(stress.get("release") == metadata.get("release_id"), "Stress report sai release")
    require(stress.get("passed") is True, "Candidate không đạt security stress gate")
    require(stress.get("model_sha256") == metadata.get("model_sha256"), "Stress sai model checksum")
    require(
        stress.get("calibration_sha256") == metadata.get("calibration_sha256"),
        "Stress sai calibration checksum",
    )
    require(
        stress.get("action_policy_sha256") == metadata.get("action_policy_sha256"),
        "Stress sai action policy checksum",
    )

    split_manifest = candidate_dir / "split_manifest.json"
    if split_manifest.is_file():
        split_data = read_json(split_manifest)
        require(
            split_data.get("domain_overlap") == 0 and split_data.get("canonical_url_overlap") == 0,
            "Candidate split manifest có leakage domain/canonical URL",
        )
        require(
            evaluation.get("split_manifest_sha256") == sha256_file(split_manifest),
            "Evaluation không cùng split manifest với candidate",
        )
    else:
        raise ReleaseRejected("Candidate thiếu split_manifest.json")

    with CONFIG_PATH.open(encoding="utf-8") as file:
        gates = (yaml.safe_load(file) or {}).get("release_gates", {})
    block = evaluation.get("policy_metrics", {}).get("block", {})
    caution = evaluation.get("policy_metrics", {}).get("caution", {})
    block_gate = gates.get("block", {})
    caution_gate = gates.get("caution", {})
    require(
        block.get("false_positive_rate", 1.0) <= block_gate.get("max_fpr", 0.005)
        and block.get("recall", 0.0) >= block_gate.get("min_recall", 0.80),
        "Locked Test không đạt Block FPR/Recall gate",
    )
    require(
        caution.get("false_positive_rate", 1.0) <= caution_gate.get("max_fpr", 0.02)
        and caution.get("recall", 0.0) >= caution_gate.get("min_recall", 0.90),
        "Locked Test không đạt Caution FPR/Recall gate",
    )
    return metadata, evaluation


def atomic_update_pointer(release_id: str, candidate_dir: Path, evaluation: dict[str, Any]) -> None:
    RELEASES_DIR.mkdir(parents=True, exist_ok=True)
    pointer = {
        "release_id": release_id,
        "release_dir": str(candidate_dir.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "evaluation_run_id": evaluation.get("evaluation_run_id"),
        "promoted_at": datetime.now(UTC).isoformat(),
    }
    temp_path = CURRENT_RELEASE.with_suffix(".tmp")
    temp_path.write_text(json.dumps(pointer, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temp_path, CURRENT_RELEASE)


def main(release_dir: Path | None = None) -> None:
    if release_dir is None:
        parser = argparse.ArgumentParser(
            description="Promote một release candidate đã qua test và security gates."
        )
        parser.add_argument("--release-dir", type=Path, required=True)
        release_dir = parser.parse_args().release_dir
    candidate_dir = release_dir.resolve()
    require(
        candidate_dir.is_relative_to(CANDIDATES_DIR.resolve()),
        "Candidate phải nằm trong releases/candidates",
    )
    metadata, evaluation = validate_candidate(candidate_dir)
    atomic_update_pointer(str(metadata["release_id"]), candidate_dir, evaluation)
    print(f"[OK] Đã promote {metadata['release_id']} -> {CURRENT_RELEASE}")


if __name__ == "__main__":
    main()
