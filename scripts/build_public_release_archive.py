#!/usr/bin/env python3
"""Build a deterministic public archive from release/package_plan.csv.

Only rows explicitly marked ``include`` are copied.  Every source byte count and
SHA-256 digest is checked before staging, and the resulting tarball uses fixed
ownership, permissions, ordering, and timestamps so that repeated builds are
byte reproducible.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Iterable


DEFAULT_PREFIX = "admet-trustworthy-benchmark-v1.0.0"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def safe_relative_path(value: str) -> PurePosixPath:
    rel = PurePosixPath(value)
    if (
        not value
        or rel.is_absolute()
        or ".." in rel.parts
        or any(part in {"", "."} for part in rel.parts)
        or any(part.startswith("._") for part in rel.parts)
    ):
        raise ValueError(f"unsafe public-package path: {value!r}")
    return rel


def included_rows(package_plan: Path) -> list[dict[str, str]]:
    with package_plan.open(newline="", encoding="utf-8") as handle:
        rows = [
            dict(row)
            for row in csv.DictReader(handle)
            if row.get("package_action") == "include"
        ]
    if not rows:
        raise ValueError(f"no include rows found in {package_plan}")

    seen: set[str] = set()
    for row in rows:
        rel = safe_relative_path(row.get("relative_path", ""))
        key = rel.as_posix()
        if key in seen:
            raise ValueError(f"duplicate include path: {key}")
        seen.add(key)
        expected_hash = row.get("sha256", "")
        if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
            raise ValueError(f"invalid SHA-256 for {key}: {expected_hash!r}")
        try:
            expected_bytes = int(row.get("bytes", ""))
        except ValueError as exc:
            raise ValueError(f"invalid byte count for {key}: {row.get('bytes')!r}") from exc
        if expected_bytes < 0:
            raise ValueError(f"negative byte count for {key}: {expected_bytes}")
    return sorted(rows, key=lambda row: row["relative_path"])


def stage_files(*, root: Path, rows: Iterable[dict[str, str]], staging: Path) -> None:
    for row in rows:
        rel = safe_relative_path(row["relative_path"])
        source = root.joinpath(*rel.parts)
        if source.is_symlink() or not source.is_file():
            raise ValueError(f"included source is not a regular non-symlink file: {source}")
        actual_bytes = source.stat().st_size
        expected_bytes = int(row["bytes"])
        if actual_bytes != expected_bytes:
            raise ValueError(
                f"byte-count mismatch for {rel}: expected {expected_bytes}, found {actual_bytes}"
            )
        actual_hash = sha256_file(source)
        if actual_hash != row["sha256"]:
            raise ValueError(
                f"SHA-256 mismatch for {rel}: expected {row['sha256']}, found {actual_hash}"
            )
        destination = staging.joinpath(*rel.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        destination.chmod(0o755 if destination.suffix == ".sh" else 0o644)


def remove_appledouble_sidecars(staging: Path) -> int:
    """Remove filesystem-generated AppleDouble entries from a private staging tree.

    Some external macOS volumes materialize metadata operations (notably for a
    hidden file such as ``.zenodo.json``) as sibling ``._*`` files.  The staging
    tree is freshly created and contains only allowlisted copies, so removing
    these non-allowlisted metadata sidecars before the exact-inventory check is
    safe and deterministic.  Any other unexpected path still fails closed.
    """

    candidates = sorted(
        (path for path in staging.rglob("*") if path.name.startswith("._")),
        key=lambda path: (len(path.parts), path.as_posix()),
        reverse=True,
    )
    for path in candidates:
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            shutil.rmtree(path)
    return len(candidates)


def verify_staging(*, staging: Path, rows: Iterable[dict[str, str]]) -> None:
    expected = {safe_relative_path(row["relative_path"]).as_posix(): row for row in rows}
    actual: dict[str, Path] = {}
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"staging contains a symlink: {path}")
        if path.is_file():
            rel = path.relative_to(staging).as_posix()
            safe_relative_path(rel)
            actual[rel] = path
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ValueError(f"staging inventory mismatch; missing={missing}, extra={extra}")
    for rel, path in actual.items():
        row = expected[rel]
        if path.stat().st_size != int(row["bytes"]) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"staging byte/hash mismatch for {rel}")


def _directory_names(rows: Iterable[dict[str, str]], prefix: str) -> list[str]:
    names = {prefix}
    for row in rows:
        rel = safe_relative_path(row["relative_path"])
        current = PurePosixPath(prefix)
        for part in rel.parts[:-1]:
            current /= part
            names.add(current.as_posix())
    return sorted(names, key=lambda name: (name.count("/"), name))


def write_deterministic_archive(
    *, staging: Path, rows: Iterable[dict[str, str]], archive: Path, prefix: str
) -> None:
    rows = list(rows)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=9) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                for name in _directory_names(rows, prefix):
                    info = tarfile.TarInfo(name)
                    info.type = tarfile.DIRTYPE
                    info.mode = 0o755
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = 0
                    tar.addfile(info)
                for row in rows:
                    rel = safe_relative_path(row["relative_path"])
                    path = staging.joinpath(*rel.parts)
                    data = path.read_bytes()
                    info = tarfile.TarInfo((PurePosixPath(prefix) / rel).as_posix())
                    info.size = len(data)
                    info.mode = 0o755 if path.suffix == ".sh" else 0o644
                    info.uid = info.gid = 0
                    info.uname = info.gname = "root"
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(data))


def verify_archive(
    *, archive: Path, rows: Iterable[dict[str, str]], prefix: str
) -> None:
    rows = list(rows)
    expected = {
        (PurePosixPath(prefix) / safe_relative_path(row["relative_path"])).as_posix(): row
        for row in rows
    }
    with tarfile.open(archive, mode="r:gz") as tar:
        members = tar.getmembers()
        files = {member.name: member for member in members if member.isfile()}
        if set(files) != set(expected):
            missing = sorted(set(expected) - set(files))
            extra = sorted(set(files) - set(expected))
            raise ValueError(f"archive inventory mismatch; missing={missing}, extra={extra}")
        for member in members:
            safe_name = PurePosixPath(member.name)
            if safe_name.is_absolute() or ".." in safe_name.parts or member.issym() or member.islnk():
                raise ValueError(f"unsafe archive member: {member.name}")
            if member.uid != 0 or member.gid != 0 or member.mtime != 0:
                raise ValueError(f"non-normalized archive metadata: {member.name}")
        for name, member in files.items():
            handle = tar.extractfile(member)
            if handle is None:
                raise ValueError(f"could not read archive member: {name}")
            payload = handle.read()
            row = expected[name]
            if len(payload) != int(row["bytes"]):
                raise ValueError(f"archive byte-count mismatch: {name}")
            if hashlib.sha256(payload).hexdigest() != row["sha256"]:
                raise ValueError(f"archive SHA-256 mismatch: {name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--package-plan", type=Path, default=Path("release/package_plan.csv"))
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("release/publication_staging_v1.0.0"),
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=Path("release/admet-trustworthy-benchmark-v1.0.0-public.tar.gz"),
    )
    parser.add_argument("--prefix", default=DEFAULT_PREFIX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    package_plan = args.package_plan if args.package_plan.is_absolute() else root / args.package_plan
    staging = args.staging_dir if args.staging_dir.is_absolute() else root / args.staging_dir
    archive = args.archive if args.archive.is_absolute() else root / args.archive
    safe_relative_path(args.prefix)
    rows = included_rows(package_plan)
    removed_appledouble = 0

    staging.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{staging.name}.", dir=staging.parent) as tmp:
        temporary_staging = Path(tmp) / staging.name
        temporary_staging.mkdir()
        stage_files(root=root, rows=rows, staging=temporary_staging)
        removed_appledouble = remove_appledouble_sidecars(temporary_staging)
        verify_staging(staging=temporary_staging, rows=rows)
        if staging.exists():
            if staging.is_symlink() or not staging.is_dir():
                raise ValueError(f"refusing to replace non-directory staging path: {staging}")
            shutil.rmtree(staging)
        os.replace(temporary_staging, staging)

    temporary_archive = archive.with_name(f".{archive.name}.tmp")
    if temporary_archive.exists():
        temporary_archive.unlink()
    write_deterministic_archive(
        staging=staging,
        rows=rows,
        archive=temporary_archive,
        prefix=args.prefix,
    )
    verify_archive(archive=temporary_archive, rows=rows, prefix=args.prefix)
    os.replace(temporary_archive, archive)
    digest = sha256_file(archive)
    sidecar = archive.with_name(f"{archive.name}.sha256")
    sidecar.write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"included_files={len(rows)}")
    print(f"removed_appledouble_sidecars={removed_appledouble}")
    print(f"archive={archive}")
    print(f"sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
