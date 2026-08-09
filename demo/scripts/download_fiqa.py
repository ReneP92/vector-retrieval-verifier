import argparse
import hashlib
import shutil
import zipfile
from pathlib import Path

import httpx

from app.config import Settings

FIQA_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"
FIQA_MD5 = "17918ed23cd04fb15047f73e6c3bd9d9"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official BEIR FiQA dataset")
    parser.add_argument("--force", action="store_true", help="replace an existing dataset")
    args = parser.parse_args()
    settings = Settings()
    target = settings.data_dir

    if settings.corpus_path.is_file() and not args.force:
        raise SystemExit(f"FiQA already exists at {target}. Use --force to replace it.")
    if target.exists() and args.force:
        shutil.rmtree(target)

    archive = target.parent / "fiqa.zip"
    archive.parent.mkdir(parents=True, exist_ok=True)
    partial = archive.with_suffix(".zip.partial")
    digest = hashlib.md5(usedforsecurity=False)

    print(f"Downloading {FIQA_URL}")
    with httpx.stream("GET", FIQA_URL, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as destination:
            for chunk in response.iter_bytes():
                destination.write(chunk)
                digest.update(chunk)

    actual_md5 = digest.hexdigest()
    if actual_md5 != FIQA_MD5:
        partial.unlink(missing_ok=True)
        raise SystemExit(f"FiQA checksum mismatch: expected {FIQA_MD5}, got {actual_md5}")
    partial.replace(archive)

    print(f"Extracting verified archive to {target.parent}")
    _safe_extract(archive, target.parent)
    archive.unlink()

    required = [settings.corpus_path, settings.queries_path, settings.qrels_path]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Downloaded archive is missing required BEIR files: {', '.join(missing)}")
    print(f"FiQA is ready at {target}")


def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            output = (destination / member.filename).resolve()
            if not output.is_relative_to(destination):
                raise ValueError(f"Unsafe path in FiQA archive: {member.filename}")
            if member.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            with source.open(member) as compressed, output.open("wb") as extracted:
                shutil.copyfileobj(compressed, extracted)


if __name__ == "__main__":
    main()
