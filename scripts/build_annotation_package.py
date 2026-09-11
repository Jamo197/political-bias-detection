"""Build a clean ZIP archive for an external qualitative annotator."""

from pathlib import Path
import json
import random
import shutil
import zipfile

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "dist"
PACKAGE_DIR = OUTPUT_DIR / "qualitative_annotation_app"
ARCHIVE_PATH = OUTPUT_DIR / "qualitative_annotation_app.zip"
SAMPLE_PATH = ROOT / "results" / "qualitative" / "qualitative_capacity_8b.jsonl"
SAMPLE_LIMIT = 25
SAMPLE_SEED = 42

FILES = {
    ROOT / "main.py": Path("main.py"),
    ROOT / "src" / "run_streamlit.py": Path("src/run_streamlit.py"),
    ROOT
    / "results"
    / "qualitative"
    / "qualitative_capacity_8b.jsonl": Path(
        "results/qualitative/qualitative_capacity_8b.jsonl"
    ),
    ROOT / "annotation_release" / "README.md": Path("README.md"),
    ROOT / "RAG Analysis" / "help page.md": Path("help_page.md"),
    ROOT / "annotation_release" / "requirements.txt": Path("requirements.txt"),
    ROOT / "annotation_release" / "run_app.sh": Path("run_app.sh"),
}


def build_package() -> Path:
    if PACKAGE_DIR.exists():
        shutil.rmtree(PACKAGE_DIR)
    PACKAGE_DIR.mkdir(parents=True)

    for source, relative_path in FILES.items():
        if not source.exists():
            raise FileNotFoundError(f"Required release file is missing: {source}")
        target = PACKAGE_DIR / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    with SAMPLE_PATH.open("r", encoding="utf-8") as handle:
        samples = [json.loads(line) for line in handle if line.strip()]
    selected_samples = (
        random.Random(SAMPLE_SEED).sample(samples, SAMPLE_LIMIT)
        if len(samples) > SAMPLE_LIMIT
        else samples
    )
    manifest = {
        "seed": SAMPLE_SEED,
        "sample_limit": SAMPLE_LIMIT,
        "text_indices": [sample.get("text_index") for sample in selected_samples],
    }
    (PACKAGE_DIR / "sample_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    (PACKAGE_DIR / "annotations").mkdir()
    (PACKAGE_DIR / "annotations" / ".gitkeep").touch()
    (PACKAGE_DIR / "run_app.sh").chmod(0o755)

    if ARCHIVE_PATH.exists():
        ARCHIVE_PATH.unlink()
    with zipfile.ZipFile(
        ARCHIVE_PATH, "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(PACKAGE_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(OUTPUT_DIR))

    return ARCHIVE_PATH


if __name__ == "__main__":
    print(build_package())
