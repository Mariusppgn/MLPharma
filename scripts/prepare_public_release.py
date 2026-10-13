"""Export reviewed source paths, never the workspace's data or Git history."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = (
    "README.md", "LICENSE", "THIRD_PARTY_NOTICES.md", ".gitignore",
    ".python-version", "pyproject.toml", "uv.lock",
)
SOURCE_DIRECTORIES = ("src", "configs", "scripts", "tests", "docs")
SUFFIXES = {".py", ".ps1", ".yaml", ".yml", ".md", ".json", ".js", ".mjs", ".html", ".css"}
FORBIDDEN_PARTS = {"__pycache__", "node_modules", ".git", ".venv", "data", "artifacts"}
SECRET_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}\b"),
    re.compile(r"\bAKIA[A-Z0-9]{16}\b"),
    re.compile(r"https?://[^\s/@:]+:[^\s/@]+@"),
    re.compile(r'["\'](?:api_key|access_token|password)["\']\s*:\s*["\'][A-Za-z0-9_+/=-]{20,}["\']'),
)


def source_files() -> list[Path]:
    paths = [ROOT / name for name in ROOT_FILES]
    for directory in SOURCE_DIRECTORIES:
        for path in (ROOT / directory).rglob("*"):
            relative = path.relative_to(ROOT)
            if any(part in FORBIDDEN_PARTS for part in relative.parts):
                continue
            if relative.as_posix() == "docs/history_pipeline.md":
                continue  # Locally preserved old README, not current release documentation.
            if path.is_file() and (path.suffix in SUFFIXES or path.name == "CYTOSCAPE_LICENSE"):
                paths.append(path)
    return sorted(set(paths))


def prepare_release(destination: Path) -> dict[str, object]:
    destination = destination.resolve()
    publication_root = (ROOT / "work" / "publication").resolve()
    if not destination.is_relative_to(publication_root) or destination == publication_root:
        raise ValueError("Destination must be a new subdirectory of work/publication/.")
    archive = destination.with_suffix(".zip")
    if destination.exists() or archive.exists():
        raise FileExistsError("Release directory/archive already exists; nothing overwritten.")
    paths = source_files()
    records = []
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Symlink not allowed: {path.relative_to(ROOT)}")
        content = path.read_bytes()
        text = content.decode("utf-8-sig")
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Potential credential in {path.relative_to(ROOT)}; export cancelled.")
        records.append({
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        })
    # All checks precede copying. No workspace file is removed or modified.
    destination.mkdir(parents=True, exist_ok=False)
    for source in paths:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    manifest = {
        "files": records,
        "file_count": len(records),
        "data_and_git_history_included": False,
    }
    # Generated provenance only; no authored source is written by this script.
    destination.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for record in records:
            bundle.write(destination / record["path"], arcname=record["path"])
    return {"directory": str(destination), "archive": str(archive), "file_count": len(records)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare_release(args.destination), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
