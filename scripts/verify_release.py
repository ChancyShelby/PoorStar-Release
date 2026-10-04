"""Verify the exact files and hashes of the release payload."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".ruff_cache", "build", "dist"}


def ignored(path: Path) -> bool:
    return any(part in IGNORED or part.endswith(".egg-info") for part in path.parts)


def main() -> int:
    manifest = json.loads((ROOT / "RELEASE_MANIFEST.json").read_text(encoding="utf-8"))
    names = set()
    errors = []
    for entry in manifest["files"]:
        name = entry["path"]
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts or name in names:
            errors.append("invalid or duplicate path: " + name)
            continue
        names.add(name)
        path = ROOT / relative
        if path.is_symlink() or not path.is_file():
            errors.append("missing or nonregular file: " + name)
            continue
        content = path.read_bytes()
        if len(content) != entry["bytes"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
            errors.append("content changed: " + name)
    if manifest["files_count"] != len(names):
        errors.append("manifest count mismatch")
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if ignored(relative) or relative.as_posix() == "RELEASE_MANIFEST.json":
            continue
        if path.is_file() and relative.as_posix() not in names:
            errors.append("unregistered file: " + relative.as_posix())
    print(
        json.dumps(
            {"status": "FAIL" if errors else "PASS", "files": len(names), "errors": errors},
            indent=2,
        )
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
