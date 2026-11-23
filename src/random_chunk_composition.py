"""Observed composition of the random5 control chunks (chunk ablation).

For each model's ``e5pool/random5`` run log: share of retrieved chunks whose party
is the author's party (CDU/CSU grouped, as in why_rag_analysis) and mean E = mean
CHES lrgen (1-7, 2 dp) of the chunks' parties, per post and then averaged over
posts. e5/simple on the same posts is given as reference.

Usage:  python src/random_chunk_composition.py
Writes results/thesis/05_chunk_ablation/random5_composition.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import why_rag_analysis as w  # noqa: E402

ROOT = w.ROOT
OUT = ROOT / "results" / "thesis" / "05_chunk_ablation" / "random5_composition.csv"
RANDOM_LOGS = sorted((ROOT / "logs" / "chunk_ablation").glob(
    "chunk_ablation_random_*/e5pool/random5/party_label_*_evaluation_logs.jsonl"))


def rows_of(rec: dict, label: str, model: str) -> list[dict]:
    meta = rec.get("input_metadata") or {}
    a = w._raw_to_canon(meta.get("party"), "author")
    ag = w.party_group(a) if a else None
    out = []
    for rank, c in enumerate((rec.get("inputs") or {}).get("retrieved_chunks") or [], 1):
        canon = w._raw_to_canon(c.get("party"), "chunk")
        g = w.party_group(canon) if canon else None
        out.append({"model": model, "source": label, "text_index": str(meta.get("text_index")),
                    "rank": rank, "author_group": ag, "chunk_group": g,
                    "chunk_pos": w.group_position(g) if g else np.nan,
                    "same_party": (g == ag) if (g and ag) else np.nan})
    return out


def main():
    rows = []
    for fp in RANDOM_LOGS:
        seen = {}
        for line in fp.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("//"):
                rec = json.loads(line)
                seen[str((rec.get("input_metadata") or {}).get("text_index"))] = rec  # keep last
        model = w.em._clean_model_name(next(iter(seen.values()))["parameters"]["llm"])
        for rec in seen.values():
            rows += rows_of(rec, "random5", model)
    ch = pd.DataFrame(rows)
    ref = w.load_retrieval()  # e5/simple, identical for all models
    ref_rows = pd.DataFrame({
        "model": "(all)", "source": "e5/simple", "text_index": ref["text_index"],
        "rank": ref["rank"], "author_group": ref["author_group"],
        "chunk_group": ref["chunk_group"].where(ref["chunk_group"] != w.NO_POSITION),
        "chunk_pos": ref["chunk_pos"],
        "same_party": ref["chunk_group"] == ref["author_group"]})
    ch = pd.concat([ch, ref_rows], ignore_index=True)
    out = []
    for (model, src), g in ch.groupby(["model", "source"]):
        per_post = g.groupby("text_index").agg(E=("chunk_pos", "mean"), same=("same_party", "mean"))
        out.append({"model": model, "source": src, "posts": g["text_index"].nunique(),
                    "chunks": len(g), "chunks_without_position": int(g["chunk_pos"].isna().sum()),
                    "same_party_share_chunks": g["same_party"].astype(float).mean(),
                    "same_party_share_posts_mean": per_post["same"].mean(),
                    "mean_E_posts": per_post["E"].mean(), "mean_E_chunks": g["chunk_pos"].mean(),
                    "sd_E_posts": per_post["E"].std()})
    res = pd.DataFrame(out)
    res.to_csv(OUT, index=False)
    print(res.round(4).to_string(index=False))
    # Are the random chunk lists the same across models?
    r = ch[ch.source == "random5"].pivot_table(index=["text_index", "rank"], columns="model",
                                               values="chunk_pos", aggfunc="first")
    if r.shape[1] > 1:
        print("\nidentical chunk positions across the random runs:",
              float((r.nunique(axis=1, dropna=False) == 1).mean()))
    print("\nexpected same-party share under the KB mix:",
          "see kb_party_distribution.csv; author mix:", ch[ch.source == "random5"]
          .drop_duplicates(["model", "text_index"])["author_group"].value_counts(normalize=True).round(3).to_dict())
    print("→", OUT)


if __name__ == "__main__":
    main()
