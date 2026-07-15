#!/usr/bin/env python3
"""Build a hash-verified, code-only GitHub staging tree.

The complete public reproducibility archive remains the authoritative release
asset.  This helper selects only allowlisted ``software_or_code`` rows from the
production package plan so the browsable Git repository never receives the
large result/prediction payload or any non-allowlisted project file.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath


ROLE = "software_or_code"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or any(part in {"", "."} or part.startswith("._") for part in path.parts)
    ):
        raise ValueError(f"unsafe package path: {value!r}")
    return path


def selected_rows(plan: Path) -> list[dict[str, str]]:
    with plan.open(newline="", encoding="utf-8") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("package_action") == "include"
            and ROLE in row.get("artifact_roles", "").split(";")
        ]
    if not rows:
        raise ValueError("package plan contains no allowlisted software/code rows")
    seen: set[str] = set()
    for row in rows:
        key = safe_relative(row.get("relative_path", "")).as_posix()
        if key in seen:
            raise ValueError(f"duplicate selected path: {key}")
        seen.add(key)
        digest = row.get("sha256", "")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError(f"invalid SHA-256 for {key}")
        if int(row.get("bytes", "-1")) < 0:
            raise ValueError(f"invalid byte count for {key}")
    return sorted(rows, key=lambda row: row["relative_path"])


def remove_appledouble(root: Path) -> int:
    paths = sorted(
        (path for path in root.rglob("*") if path.name.startswith("._")),
        key=lambda path: (len(path.parts), path.as_posix()),
        reverse=True,
    )
    for path in paths:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    return len(paths)


def build(*, root: Path, plan: Path, destination: Path, manifest: Path) -> dict:
    root = root.resolve()
    rows = selected_rows(plan)
    destination.parent.mkdir(parents=True, exist_ok=True)
    removed = 0
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}.", dir=destination.parent
    ) as tmp:
        staging = Path(tmp) / destination.name
        staging.mkdir()
        for row in rows:
            relative = safe_relative(row["relative_path"])
            source = root.joinpath(*relative.parts)
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"selected source is not a regular file: {relative}")
            size = source.stat().st_size
            digest = sha256_file(source)
            if size != int(row["bytes"]) or digest != row["sha256"]:
                raise ValueError(f"package-plan byte/hash mismatch: {relative}")
            target = staging.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            target.chmod(0o755 if target.suffix == ".sh" else 0o644)
        removed = remove_appledouble(staging)

        actual = {
            path.relative_to(staging).as_posix(): path
            for path in staging.rglob("*")
            if path.is_file()
        }
        expected = {row["relative_path"]: row for row in rows}
        if set(actual) != set(expected):
            raise ValueError("GitHub source staging inventory is not exact")
        for relative, path in actual.items():
            row = expected[relative]
            if path.is_symlink() or path.stat().st_size != int(row["bytes"]):
                raise ValueError(f"staging file mismatch: {relative}")
            if sha256_file(path) != row["sha256"]:
                raise ValueError(f"staging hash mismatch: {relative}")
            if path.stat().st_size >= 100_000_000:
                raise ValueError(f"ordinary GitHub size limit exceeded: {relative}")

        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise ValueError(f"refusing to replace destination: {destination}")
            shutil.rmtree(destination)
        os.replace(staging, destination)

    payload = {
        "schema_version": 1,
        "selection": "package_action=include and artifact_roles contains software_or_code",
        "source_package_plan": plan.relative_to(root).as_posix()
        if plan.is_relative_to(root)
        else str(plan),
        "source_package_plan_sha256": sha256_file(plan),
        "file_count": len(rows),
        "total_bytes": sum(int(row["bytes"]) for row in rows),
        "removed_appledouble_sidecars": removed,
        "files": [
            {
                "relative_path": row["relative_path"],
                "bytes": int(row["bytes"]),
                "sha256": row["sha256"],
            }
            for row in rows
        ],
    }
    manifest.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--package-plan", type=Path, default=Path("release/package_plan.csv"))
    parser.add_argument(
        "--destination", type=Path, default=Path("release/github_source_staging_v1.0.0")
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("release/github_source_staging_v1.0.0.manifest.json"),
    )
    args = parser.parse_args()
    root = args.root.resolve()

    def rooted(path: Path) -> Path:
        return path.resolve() if path.is_absolute() else (root / path).resolve()

    payload = build(
        root=root,
        plan=rooted(args.package_plan),
        destination=rooted(args.destination),
        manifest=rooted(args.manifest),
    )
    print(f"github_source_files={payload['file_count']}")
    print(f"github_source_bytes={payload['total_bytes']}")
    print(f"removed_appledouble_sidecars={payload['removed_appledouble_sidecars']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
