import os
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TESTS_COMPLETED = BASE_DIR / "tests_completed.md"

EMBEDDING_NAMES = {"e5", "bge", "jina", "qwen3", "none"}


def parse_logs_info(path):
    content = path.read_text(encoding="utf-8")

    m = re.search(r"\*\*Run ID:\*\*\s*(\S+)", content)
    run_id = m.group(1) if m else None

    m = re.search(r"\*\*Embedding Model:\*\*\s*(\S+)", content)
    embedding = m.group(1) if m else None

    m = re.search(r"\*\*Strategies:\*\*\s*(.+)", content)
    strategies_raw = m.group(1) if m else ""

    m = re.search(r"\*\*RAG Mode:\*\*\s*(.+)", content)
    rag_mode_raw = m.group(1) if m else None
    is_rag = rag_mode_raw and "True" in rag_mode_raw

    # parse model table
    models = []
    for line in content.splitlines():
        line = line.strip()
        if (
            line.startswith("|")
            and "Key" not in line
            and "-----" not in line
            and line.count("|") >= 3
        ):
            parts = [p.strip() for p in line.split("|")[1:-1]]
            if len(parts) >= 3:
                key, model_id, region = parts[0], parts[1], parts[2]
                models.append({"key": key, "model_id": model_id, "region": region})

    strategies = [
        s.strip().replace("(baseline)", "").strip()
        for s in strategies_raw.split(",")
        if s.strip()
    ]

    return {
        "run_id": run_id,
        "path": path.parent,
        "embedding": embedding,
        "is_rag": is_rag,
        "strategies": strategies,
        "models": models,
    }


def find_result_files(run_dir):
    results = []
    if not run_dir.is_dir():
        return results

    embedding_dirs = [
        d for d in run_dir.iterdir() if d.is_dir() and d.name in EMBEDDING_NAMES
    ]
    for emb_dir in embedding_dirs:
        embedding = emb_dir.name
        for strategy_dir in emb_dir.iterdir():
            if not strategy_dir.is_dir():
                continue
            strategy = strategy_dir.name
            for f in strategy_dir.iterdir():
                if f.suffix != ".jsonl" or not f.name.endswith(
                    "_evaluation_logs.jsonl"
                ):
                    continue
                # filename format: {llm_key}_{strategy}_evaluation_logs.jsonl
                parts = f.stem.split("_")
                if len(parts) >= 3:
                    llm_key = parts[0]
                else:
                    llm_key = None
                lines = sum(1 for _ in f.open(encoding="utf-8"))
                results.append(
                    {
                        "path": f,
                        "embedding": embedding,
                        "strategy": strategy,
                        "llm_key": llm_key,
                        "questions": lines,
                        "dirname": run_dir.name,
                    }
                )
    return results


def build_models_section(completed, info_map):
    header = "| Model ID | Region | RAG | Embedding Model | Retrieval Strategy | Questions | Status | Folder | Ground Truth |\n"
    sep = "|----------|--------|--------|--------|--------|--------|--------|--------|--------|\n"
    rows = []

    for res in completed:
        dirname = res["dirname"]
        info = info_map.get(dirname, {})
        models = info.get("models", [])

        model_id = ""
        region = ""
        for m in models:
            if res["llm_key"] == m["key"]:
                model_id = m["model_id"]
                region = m["region"]
                break
        if not model_id and models:
            model_id = models[0]["model_id"]
            region = models[0]["region"]

        rag = "Yes" if info.get("is_rag", False) else "No"
        embedding = res["embedding"] if res["embedding"] != "none" else "None"
        strategy = res["strategy"]
        questions = res["questions"]
        total = 775
        if questions >= total:
            status = "Complete"
        elif questions > 0:
            status = f"Partial (stopped at {questions}/{total})"
        else:
            status = "Empty"

        rows.append(
            f"| {model_id} | {region} | {rag} | {embedding} | {strategy} | {questions} | {status} | {dirname} | final_label_ideology |"
        )

    rows.sort()
    return "## Models\n" + header + sep + "\n".join(rows) + "\n"


def build_completed_set(all_results, info_map, total=775):
    """Return set of (model_id, embedding, strategy) tuples that are fully complete."""
    completed = set()
    for res in all_results:
        if res["questions"] < total:
            continue
        dirname = res["dirname"]
        info = info_map.get(dirname, {})
        model_id = ""
        for m in info.get("models", []):
            if res["llm_key"] == m["key"]:
                model_id = m["model_id"]
                break
        if not model_id and info.get("models"):
            model_id = info["models"][0]["model_id"]
        if not model_id:
            continue
        embedding = res["embedding"] if res["embedding"] != "none" else "None"
        completed.add((model_id.strip().lower(), embedding.strip().lower(), res["strategy"].strip().lower()))
    return completed


def filter_remaining(remaining_section, completed_set):
    """Remove rows from remaining tests table whose strategies are all completed."""
    lines = remaining_section.splitlines()
    header_idx = next(
        (i for i, l in enumerate(lines) if l.strip().startswith("|") and "Model ID" in l),
        None,
    )
    if header_idx is None:
        return remaining_section

    before = lines[:header_idx]
    header = lines[header_idx]
    sep = lines[header_idx + 1]
    data_rows = []
    rest = []
    in_table = True
    for line in lines[header_idx + 2:]:
        if in_table and line.strip().startswith("|"):
            data_rows.append(line)
        else:
            in_table = False
            rest.append(line)

    filtered = [header, sep]
    for row in data_rows:
        parts = [p.strip() for p in row.split("|")[1:-1]]
        if len(parts) < 5:
            filtered.append(row)
            continue

        model_id = parts[0]
        embedding = parts[3]
        strategies_str = parts[4]

        strategies = [s.strip() for s in strategies_str.split(",")]

        all_completed = all(
            (model_id.strip().lower(), embedding.strip().lower(), s.strip().lower()) in completed_set
            for s in strategies
        )

        if not all_completed:
            filtered.append(row)

    return "\n".join(before + filtered + rest)


def main():
    run_dirs = sorted(
        [d for d in BASE_DIR.iterdir() if d.is_dir() and d.name.startswith("2026-")]
    )

    info_map = {}
    all_results = []

    for d in run_dirs:
        logs_info = d / "logs_info.md"
        if logs_info.exists():
            info = parse_logs_info(logs_info)
            if info:
                info_map[d.name] = info

        results = find_result_files(d)
        all_results.extend(results)

    existing = (
        TESTS_COMPLETED.read_text(encoding="utf-8") if TESTS_COMPLETED.exists() else ""
    )

    completed_set = build_completed_set(all_results, info_map)
    print(f"  Fully completed test combos: {len(completed_set)}")

    remaining_marker = "## Remaining Tests"
    remaining_section = ""
    if remaining_marker in existing:
        _, rest = existing.split(remaining_marker, 1)
        remaining_section = filter_remaining(remaining_marker + rest, completed_set)

    models_section = build_models_section(all_results, info_map)

    content = "# Tested Models\n\n" + models_section + "\n\n" + remaining_section

    TESTS_COMPLETED.write_text(content, encoding="utf-8")
    print(f"Updated {TESTS_COMPLETED}")
    print(f"  Completed tests found: {len(all_results)}")
    print(f"  Batch dirs scanned: {len(run_dirs)}")


if __name__ == "__main__":
    main()
