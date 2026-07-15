from __future__ import annotations

import csv
import hashlib
import importlib.util
import subprocess
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_public_release_archive.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_public_release_archive", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_plan(root: Path, relative_paths: list[str]) -> Path:
    plan = root / "release" / "package_plan.csv"
    plan.parent.mkdir(parents=True)
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "relative_path",
                "sha256",
                "bytes",
                "artifact_roles",
                "models",
                "referencing_runs",
                "included_referencing_runs",
                "copied_by_builder",
                "package_action",
            ],
        )
        writer.writeheader()
        for relative_path in relative_paths:
            payload = (root / relative_path).read_bytes()
            writer.writerow(
                {
                    "relative_path": relative_path,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "bytes": len(payload),
                    "artifact_roles": "test",
                    "models": "shared",
                    "referencing_runs": 0,
                    "included_referencing_runs": 0,
                    "copied_by_builder": False,
                    "package_action": "include",
                }
            )
    return plan


def test_deterministic_allowlist_archive(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("safe\n", encoding="utf-8")
    script = tmp_path / "scripts" / "replay.sh"
    script.parent.mkdir()
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    plan = _write_plan(tmp_path, ["README.md", "scripts/replay.sh"])
    archive = tmp_path / "release" / "bundle.tar.gz"
    staging = tmp_path / "release" / "staging"

    command = [
        "python3",
        str(SCRIPT),
        "--root",
        str(tmp_path),
        "--package-plan",
        str(plan),
        "--staging-dir",
        str(staging),
        "--archive",
        str(archive),
        "--prefix",
        "example-v1",
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    first = archive.read_bytes()
    subprocess.run(command, check=True, capture_output=True, text=True)
    assert archive.read_bytes() == first

    digest = hashlib.sha256(first).hexdigest()
    assert archive.with_name("bundle.tar.gz.sha256").read_text(encoding="utf-8") == (
        f"{digest}  bundle.tar.gz\n"
    )
    with tarfile.open(archive, "r:gz") as handle:
        file_members = [member for member in handle.getmembers() if member.isfile()]
        assert [member.name for member in file_members] == [
            "example-v1/README.md",
            "example-v1/scripts/replay.sh",
        ]
        assert all(member.uid == member.gid == member.mtime == 0 for member in handle.getmembers())
        assert handle.getmember("example-v1/README.md").mode == 0o644
        assert handle.getmember("example-v1/scripts/replay.sh").mode == 0o755


def test_rejects_unsafe_and_hash_mismatch(tmp_path: Path) -> None:
    module = _load_module()
    for unsafe in ("", "../secret", "/absolute", "safe/._sidecar"):
        try:
            module.safe_relative_path(unsafe)
        except ValueError:
            pass
        else:
            raise AssertionError(f"unsafe path accepted: {unsafe!r}")

    (tmp_path / "README.md").write_text("initial\n", encoding="utf-8")
    plan = _write_plan(tmp_path, ["README.md"])
    (tmp_path / "README.md").write_text("changed\n", encoding="utf-8")
    result = subprocess.run(
        [
            "python3",
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--package-plan",
            str(plan),
            "--staging-dir",
            str(tmp_path / "staging"),
            "--archive",
            str(tmp_path / "bundle.tar.gz"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "mismatch" in result.stderr


def test_removes_only_appledouble_sidecars_from_private_staging(
    tmp_path: Path,
) -> None:
    module = _load_module()
    staging = tmp_path / "staging"
    nested = staging / "metadata"
    nested.mkdir(parents=True)
    keep = nested / ".zenodo.json"
    keep.write_text("{}\n", encoding="utf-8")
    (staging / "._.zenodo.json").write_bytes(b"metadata")
    (nested / "._artifact.csv").write_bytes(b"metadata")

    removed = module.remove_appledouble_sidecars(staging)

    assert removed == 2
    assert keep.is_file()
    assert not list(staging.rglob("._*"))
