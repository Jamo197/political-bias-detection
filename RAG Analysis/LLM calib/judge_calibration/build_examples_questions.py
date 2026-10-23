"""Build the question set with in-prompt examples (experiment 2).

Examples are real (input text, chunk excerpt) pairs from the calibration set, so
they are chosen under strict rules to keep the evaluation honest:

* only dev texts (never the holdout), and only chunks on which all three
  annotators gave the same rating for the dimension;
* one example per side (yes / no) of each of the six discriminating gates, with
  the class that sits on each side of that gate's boundary:

      gate        yes example        no example
      T_domain    R_top 2            R_top 1
      T_subarea   R_top 3            R_top 2
      T_item      R_top 4            R_top 3
      T_central   R_top 5            R_top 4
      I_stance    R_ideo 4           R_ideo 3
      I_party     R_ideo 5           R_ideo 4

* a fixed greedy rule picks the candidates: prefer a text that already supplies
  an example (fewer texts are lost from the evaluation), then the shortest chunk,
  then the lowest text / chunk index. No manual selection.

The texts that supply examples must be left out when the run is scored
(``compare_arms.py --exclude-texts``); the list is written next to the questions.

    python "RAG Analysis/judge_calibration/build_examples_questions.py"
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import calibrate as cal  # noqa: E402
from run_llm_judge import JEV_QUESTIONS  # noqa: E402

EXCERPT_CHARS = 480
SALUTATION = re.compile(r"Präsident|Kolleginnen und Kollegen|Damen und Herren|Frau Kollegin|Herr Kollege", re.I)

# (gate, dimension, class that is 'yes', class that is 'no')
SLOTS = [
    ("T_domain", "R_top", 2, 1),
    ("T_subarea", "R_top", 3, 2),
    ("T_item", "R_top", 4, 3),
    ("T_central", "R_top", 5, 4),
    ("I_stance", "R_ideo", 4, 3),
    ("I_party", "R_ideo", 5, 4),
]
MEANING = {
    ("R_top", 1): "irrelevant: a different field and issue",
    ("R_top", 2): "broad domain only: same field, a different issue",
    ("R_top", 3): "related sub-issue: same sub-area, different scope or target",
    ("R_top", 4): "direct policy overlap: same controversy, slightly different aspect",
    ("R_top", 5): "identical target: the very item the input text refers to",
    ("R_ideo", 3): "implicit value framing, no clear side",
    ("R_ideo", 4): "clear ideological stance, not anchored in the speaker's party",
    ("R_ideo", 5): "explicit party grounding",
}


# The I_party gate hinges on the speaker naming their own party / faction, so its
# excerpt is the window with the most such cues (the opening of a speech rarely
# contains them). Every other gate uses the first window after the salutation.
PARTY_CUES = re.compile(
    r"\bwir als\b|\bfür uns\b|unsere[rn]? Fraktion|meine[rn]? Fraktion|\bwir von der\b|"
    r"\bwir,? die (?:SPD|CDU|CSU|FDP|AfD|Linke|Grünen)|als (?:Linke|AfD|CDU|SPD|FDP|Grüne)|"
    r"unsere[nr]? Antrag|\bwir lehnen\b|\bwir werden\b|\bwir fordern\b",
    re.I,
)


def excerpt(text: str, cue: re.Pattern | None = None) -> str:
    """~480 characters made of whole sentences.

    Starts after leading salutations, or, with ``cue``, at the sentence window
    containing the most cue matches (earliest on ties).
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    start = 0
    while start < len(sentences) - 1 and SALUTATION.search(sentences[start]) and len(sentences[start]) < 160:
        start += 1

    def window(i: int) -> str:
        out = ""
        for sentence in sentences[i:]:
            if out and len(out) + len(sentence) > EXCERPT_CHARS:
                break
            out = f"{out} {sentence}".strip()
        return out if len(out) <= EXCERPT_CHARS + 120 else out[:EXCERPT_CHARS].rsplit(" ", 1)[0] + " ..."

    if cue is not None:
        best = max(range(start, len(sentences)), key=lambda i: (len(cue.findall(window(i))), -i))
        if cue.search(window(best)):
            return window(best)
    return window(start)


def main() -> None:
    dev, _ = cal.split_texts()
    records = {}
    for line in cal.CONSENSUS.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            records[r["text_index"]] = r

    candidates: dict[tuple[str, int], list[tuple]] = {}
    for text in sorted(dev, key=int):
        for chunk in records[text]["chunk_annotations"]:
            for dim in ("R_top", "R_ideo"):
                ratings = set(chunk["human_ratings"][dim].values())
                if len(ratings) == 1:
                    candidates.setdefault((dim, ratings.pop()), []).append(
                        (text, chunk["chunk_index"], len(chunk["chunk_text"]))
                    )

    used_texts: set[str] = set()
    chosen: dict[tuple[str, str], tuple] = {}  # (gate, 'yes'|'no') -> (text, chunk)
    for gate, dim, yes_cls, no_cls in SLOTS:
        for side, cls in (("yes", yes_cls), ("no", no_cls)):
            options = candidates.get((dim, cls), [])
            if not options:
                raise SystemExit(f"no unanimous dev chunk for {dim}={cls}")
            pick = min(options, key=lambda o: (o[0] not in used_texts, o[2], int(o[0]), o[1]))
            chosen[(gate, side)] = pick
            used_texts.add(pick[0])

    questions = json.loads(json.dumps(JEV_QUESTIONS))
    log = {}
    for gate, dim, yes_cls, no_cls in SLOTS:
        for side, cls in (("yes", yes_cls), ("no", no_cls)):
            text, ci, _ = chosen[(gate, side)]
            record = records[text]
            chunk = record["chunk_annotations"][ci]
            example = {
                "chunk_excerpt": excerpt(
                    chunk["chunk_text"], PARTY_CUES if (gate, side) == ("I_party", "yes") else None
                )
            }
            if dim == "R_top":
                example = {"input_text": record["input_text"], **example}
            example["annotators"] = f"all three rated {dim} = {cls} ({MEANING[(dim, cls)]})"
            example["answer"] = side
            key = "true" if side == "yes" else "false"
            base = questions[gate]["criteria"][key]
            if isinstance(base, str):
                questions[gate]["criteria"][key] = {"definition": base, "example": example}
            log[f"{gate}:{side}"] = {"text_index": text, "chunk_index": ci, "rating": f"{dim}={cls}"}

    out_q = HERE / "questions_e2_examples.json"
    out_q.write_text(json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8")
    meta = {"example_texts": sorted(used_texts, key=int), "examples": log}
    (HERE / "examples_used.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_q.name}; example texts to exclude when scoring: {','.join(meta['example_texts'])}")
    for k, v in log.items():
        print(f"  {k:<16} text {v['text_index']:>4} chunk {v['chunk_index']}  {v['rating']}")


if __name__ == "__main__":
    main()
