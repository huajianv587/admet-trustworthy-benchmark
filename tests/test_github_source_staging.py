from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "release" / "build_github_source_staging.py"


def _module():
    spec = importlib.util.spec_from_file_location("github_source_staging", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(root: Path, relative: str, role: str) -> dict[str, object]:
    payload = (root / relative).read_bytes()
    return {
        "relative_path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "artifact_roles": role,
        "models": "shared",
        "referencing_runs": 0,
        "included_referencing_runs": 0,
        "copied_by_builder": False,
        "package_action": "include",
    }


def test_builds_exact_code_only_staging_and_rejects_stale_hash(tmp_path: Path) -> None:
    module = _module()
    (tmp_path / "src").mkdir()
    (tmp_path / "public").mkdir()
    (tmp_path / "src/tool.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "public/prediction.csv").write_text("prediction\n0.5\n", encoding="utf-8")
    plan = tmp_path / "package_plan.csv"
    rows = [
        _row(tmp_path, "src/tool.py", "software_or_code"),
        _row(tmp_path, "public/prediction.csv", "public_prediction_view"),
    ]
    with plan.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    destination = tmp_path / "github"
    manifest = tmp_path / "github.manifest.json"
    result = module.build(
        root=tmp_path, plan=plan, destination=destination, manifest=manifest
    )
    assert result["file_count"] == 1
    assert (destination / "src/tool.py").is_file()
    assert not (destination / "public").exists()
    assert not list(destination.rglob("._*"))

    (tmp_path / "src/tool.py").write_text("changed\n", encoding="utf-8")
    try:
        module.build(
            root=tmp_path, plan=plan, destination=destination, manifest=manifest
        )
    except ValueError as exc:
        assert "byte/hash mismatch" in str(exc)
    else:
        raise AssertionError("stale package-plan hash was accepted")
