"""Freeze local review sources and small evidence records without including datasets."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parent.parent


def sha256(path: Path) -> str:
    """Identify the supplied review file with SHA-256."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    """Package source/config/tests/docs and small records into a new exclusive ZIP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.archive.exists() or args.manifest.exists():
        parser.error("archive or manifest exists; choose new paths")
    paths = {ROOT/"README.md", ROOT/"AGENTS.md"}
    for directory, extensions in [("src", {".py"}), ("tests", {".py"}), ("scripts", {".py"}),
                                  ("configs", {".yaml"}), ("docs/recovery", {".md"}), (".agents", {".md"}),
                                  ("results/recovery", {".json", ".npz"})]:
        paths.update(path for path in (ROOT/directory).rglob("*") if path.is_file() and path.suffix in extensions)
    entries = {str(path.relative_to(ROOT)).replace("\\", "/"): {"sha256": sha256(path), "bytes": path.stat().st_size}
               for path in sorted(paths)}
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(paths):
            archive.write(path, str(path.relative_to(ROOT)))
    with zipfile.ZipFile(args.archive, "r") as archive:
        if archive.testzip() is not None:
            raise AssertionError("ZIP integrity check failed")
        for filename, expected in entries.items():
            if hashlib.sha256(archive.read(filename)).hexdigest() != expected["sha256"]:
                raise AssertionError("archive content differs from the source manifest")
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "audited_baseline_commit": "cba90a1522c3f4a8b3f839b1750a370713f61a50",
              "working_branch": "codex/scientific-validation-recovery", "git_changes": "local working changes; no remote push",
              "archive": str(args.archive.resolve()), "archive_sha256": sha256(args.archive), "files": entries,
              "scope": "review source/config/test/document snapshot and small evidence records",
              "excluded": ["RadioML or other raw datasets", "virtual environment", "large model binaries; referenced local checkpoint paths remain in records"]}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"files": len(entries), "archive_bytes": args.archive.stat().st_size,
                      "archive_sha256": result["archive_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
