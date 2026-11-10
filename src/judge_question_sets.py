"""Question sets for the Jev chunk judge (used by ``run_llm_judge.py``).

A *question set* bundles everything that has to change together when the
wording changes:

* ``questions``: the Jev questions, keyed by a local name (``T_domain`` ...);
* ``full_state_questions``: the questions that need the full state (generator
  justification and chunk metadata); all others are content questions and get
  the reduced state of ``--state-mode``;
* ``derive``: raw answers -> integer labels on the human scales, plus
  continuous scores (for Spearman / ranking) and diagnostics;
* ``needs_review``: per-dimension escalation flags.

The hash of ``questions`` identifies a set in every result row; the v1 dict is
byte-identical to the former built-in ``JEV_QUESTIONS``, so its hash (and every
post-hoc mapping fitted on it) is unchanged.

v2 changes against v1 (round after the jev-1.14.0 calibration on the 125
consensus chunks; reasons in the comments next to each question):

R_top
  * each gate has its own focus (v1 gave T_domain / T_subarea the item-level
    focus "first identify the SPECIFIC ITEM");
  * the item falls back to the input's concrete claim when the input names no
    bill, event or figure (text 117);
  * levels 2-4 use the best substantive passage of a mixed chunk; only
    T_central (level 5) requires the item to be the main subject;
  * T_central asks one condition ("main subject"), not two;
  * new: ``X_input_has_item`` (diagnostic only) and ``R_top_score``, a Score
    question with the annotators' rubric, reported as ``R_top@holistic``.
R_ideo
  * ``I_lr_position`` replaces ``I_stance``: the annotators' 3-vs-4 rule is
    placeability on the left-right scale, not partisan intensity;
  * new: ``R_ideo_score`` with the rubric levels -> ``R_ideo@holistic``.
A_caus
  * citations by number / speaker name stay deterministic (``citation_info``
    in the runner, unchanged); only the residual path (no numeric citation)
    changes: the single v1 ``A_caus`` noul becomes two atomic nouls,
    ``A_party_evidence`` and ``A_content_match``, combined with ``min``.

Labels are derived with the v1 rules wherever possible (R_top: 1 + highest gate
answered yes; R_ideo: conjunction ladder) so that a v1-vs-v2 difference comes
from the wording, not from a changed derivation.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Callable

GATE_THRESHOLD = 0.5  # Noul probability >= threshold -> gate is yes
A_CAUS_THRESHOLD = 0.5  # Noul probability >= threshold -> A_caus = 1

# Trust-or-escalate bands (see the runner docstring). Kept identical for v1 and
# v2 so the review shares are comparable.
REVIEW_BANDS = {"R_top": 0.10, "R_ideo": 0.20, "A_caus": 0.20}
# v2: also escalate when the gate ladder and the holistic Score question
# disagree by this many levels or more.
HOLISTIC_DISAGREEMENT = 2


def hash_questions(questions: dict) -> str:
    return hashlib.sha256(
        json.dumps(questions, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


# ===========================================================================
# v1: the calibrated question set used up to jev-1.14.0 round 3
# ===========================================================================

# Calibrated ladder questions. Structure follows the TypeSafe docs: instructions
# are objects ("question" + "compare"/"inspect" + "focus") that reference state
# fields by backticked path, and every question has an explicit true/false test.
# The R_ideo gates carry a burden-of-proof focus ("false unless the text itself
# clearly shows it"): without it the model said yes far more often than the
# annotators (I_party 70 % vs 19 %). I_evaluative follows the help page's 2-vs-3
# line (loaded framing vs. neutral / balanced / consensual, round 3): asked as
# "does the speaker argue or evaluate" it said yes for every chunk. To iterate,
# edit this constant and re-run;
# the questions hash stored with every result identifies the version.
V1_QUESTIONS = {
    "T_domain": {
        "type": "noul",
        "instructions": {
            "question": "Do the input text and the chunk belong to the same policy field, so that the chunk is part of the same area of debate?",
            "compare": ["`input_text`", "`retrieved_chunk.text`"],
            "focus": "Judge subject matter only: ignore stance, tone and ideology - opposite positions on the same measure are the same topic. First identify the SPECIFIC ITEM the input text takes a position on (a named bill or measure, a concrete event, a figure or proposal, an action by a named person or body), not just its general theme.",
        },
        "criteria": {
            "true": "Both belong to the same concrete policy field (for example both are about criminal law, the pandemic, climate and energy, migration, the armed forces, pensions, a named conflict), even if they discuss different measures within it.",
            "false": "They belong to different fields, or the only overlap is something generic: both are 'politics', both involve the same party or person, or they share a word. Money, taxes, costs or government spending count as a shared field only when the input text is itself about that subject (for example public debt, or who pays for state spending); a passing mention of costs in an input text about another issue does not. If the input text is about one concrete issue and the chunk about another (for example consulting contracts for a ministry vs. an excess-profit tax; protest forms vs. extremist organisations), answer false.",
        },
    },
    "T_subarea": {
        "type": "noul",
        "instructions": {
            "question": "Do the input text and the chunk argue about the same narrow policy question?",
            "compare": ["`input_text`", "`retrieved_chunk.text`"],
            "focus": "Judge subject matter only: ignore stance, tone and ideology - opposite positions on the same measure are the same topic. First identify the SPECIFIC ITEM the input text takes a position on (a named bill or measure, a concrete event, a figure or proposal, an action by a named person or body), not just its general theme.",
        },
        "criteria": {
            "true": "The chunk's main subject is the same narrow policy question as the input text's: the same kind of decision or problem (for example how a specific cost is shared, whether a specific obligation should exist, how a specific programme is financed), even if the chunk concerns another detail, institution or year than the input text's own item.",
            "false": "The chunk's main subject is a different question that merely sits under the same umbrella theme (for example the input text is about paying for tests and the chunk is about a vaccination mandate; or both are about 'the military' but one is about equipment and the other about conscription), or it touches the shared theme only in passing.",
        },
    },
    "T_item": {
        "type": "noul",
        "instructions": {
            "question": "Does the chunk discuss the specific item that the input text takes a position on?",
            "compare": ["`input_text`", "`retrieved_chunk.text`"],
            "focus": "Judge subject matter only: ignore stance, tone and ideology - opposite positions on the same measure are the same topic. First identify the SPECIFIC ITEM the input text takes a position on (a named bill or measure, a concrete event, a figure or proposal, an action by a named person or body), not just its general theme.",
        },
        "criteria": {
            "true": "The chunk addresses that same bill, measure, event, figure or named action - at least as one part of its content, from any angle and with any position.",
            "false": "The chunk does not discuss that item: it covers other measures, a more general question, or a different instance of the same kind of thing.",
        },
    },
    "T_central": {
        "type": "noul",
        "instructions": {
            "question": "Is the chunk mainly about the same bill, event, claim or decision that the input text refers to, and does it speak to the point the input text makes?",
            "compare": ["`input_text`", "`retrieved_chunk.text`"],
            "focus": "Judge subject matter only: ignore stance, tone and ideology - opposite positions on the same measure are the same topic. First identify the SPECIFIC ITEM the input text takes a position on (a named bill or measure, a concrete event, a figure or proposal, an action by a named person or body), not just its general theme.",
        },
        "criteria": {
            "true": "The chunk's main subject is that same item - named or unmistakably identified - and it addresses the specific point or claim the input text makes about it (support or opposition does not matter). A summary of the chunk would name this item as its main subject.",
            "false": "The chunk covers the same wider issue but a different point (for example a vaccination mandate when the input text is about a statement on vaccination data), mentions the item only briefly, treats it as one of several topics in a broad speech, or is only the occasion of the debate.",
        },
    },
    "I_procedural": {
        "type": "noul",
        "instructions": {
            "question": "Is the chunk only floor management or formalities?",
            "inspect": "`retrieved_chunk.text`",
        },
        "criteria": {
            "true": "The chunk consists of procedure only: addressing the chair, thanks, greetings, announcing agenda items or votes, with no substantive statement about policy.",
            "false": "The chunk contains substantive statements about policy, facts or political actors.",
        },
    },
    "I_evaluative": {
        "type": "noul",
        "instructions": {
            "question": "Does the passage use normative or politically loaded language, selective emphasis or one-sided framing?",
            "inspect": "`retrieved_chunk.text`",
        },
        "criteria": {
            "true": "The speaker frames the issue from one side: value-laden or politically loaded vocabulary ('economic justice' vs. 'favouring the rich', 'burden' vs. 'humanitarian obligation'), selective emphasis, praise or attacks on political actors, or demands for how things ought to be.",
            "false": "The passage stays neutral or balanced: it reports facts, figures or procedures, gives a balanced overview of several viewpoints, makes a consensual or cross-party appeal, thanks or praises non-partisan actors, or explains a measure in a sober tone - even if the speaker mildly says they support something.",
        },
    },
    "I_stance": {
        "type": "noul",
        "instructions": {
            "question": "Does the speaker take a strong, partisan side on a contested political question?",
            "inspect": "`retrieved_chunk.text`",
            "focus": "Burden of proof: the answer is false unless the text itself clearly shows it. Answer true only if the passage leaves no reasonable doubt on this. If the passage is ambiguous, mixed or only partly fits, answer false.",
        },
        "criteria": {
            "true": "The speaker takes a clear side with charged or polarising language, sweeping judgements of political opponents or of the government's record, or a programmatic demand typical of one political camp (for example market-liberal, welfare-state, nationalist, ecological, law-and-order). The partisanship is the point of the passage.",
            "false": "The speaker disagrees or criticises in a measured, technical or procedural way (rebutting a motion point by point, legal or fact-based reasoning, explaining or defending what the government did), expresses empathy or personal views, weighs pros and cons, or appeals for cross-party unity. Disagreement alone is not enough.",
        },
    },
    "I_party": {
        "type": "noul",
        "instructions": {
            "question": "Does the speaker declare what their own party or faction stands for, demands or rejects?",
            "inspect": "`retrieved_chunk.text`",
            "focus": "Burden of proof: the answer is false unless the text itself clearly shows it. Answer true only if the passage leaves no reasonable doubt on this. If the passage is ambiguous, mixed or only partly fits, answer false.",
        },
        "criteria": {
            "true": "Both hold: (1) the chunk text explicitly names the speaker's own party or faction as the holder of the position ('we as the Left', 'for us in the CDU/CSU', 'we, the AfD', 'my faction', 'our motion') and (2) it states a programmatic position of that party - what it demands, rejects, proposes or will do ('we reject this', 'we demand', 'we will vote against', 'our proposal is').",
            "false": "At least one fails. In particular these do NOT count: a generic 'we' (Germany, the Bundestag, everyone: 'we need a new mobility policy'); speaking as a minister or for the government; naming other parties only to attack or address them; a passing mention of one's own party inside an attack on others; a position that is merely typical of the party but never tied to it in the text.",
        },
    },
    "A_caus": {
        "type": "noul",
        "instructions": {
            "question": "Did the generator justification demonstrably draw on this specific retrieved chunk?",
            "compare": ["`generator_justification`", "`retrieved_chunk`"],
            "focus": "The justification refers to retrieved reference chunks by number, by speaker name, or by the party whose reference statements it invokes. Decide whether it refers to THIS chunk (see `retrieved_chunk.citation_forms`, `retrieved_chunk.speaker`, `retrieved_chunk.party`). If the justification cites other chunks by number but not this chunk's number or speaker, answer false even when this chunk's party is also named - a shared party name is then only context. A party named only as the camp the input text itself belongs to is not a reference to a chunk.",
        },
        "criteria": {
            "true": "The justification refers to this chunk: it cites this chunk's number in any form (a single reference, a list such as '[2, 3]', 'reference [2]', 'anchor 2', 'Chunk 2'), OR it names this chunk's speaker, OR it invokes the statements or stance of this chunk's party as comparison evidence (e.g. 'echoing the SPD and FDP speakers in the reference anchors', 'similar to the AfD speakers') in a way that matches the content of this chunk.",
            "false": "The justification does not refer to this chunk: its number is not cited, its speaker is not named, and its party is not invoked as reference evidence (a party mentioned only as the label for the input text's own orientation does not count; neither does a party name that merely overlaps with chunks cited by other numbers), or it relies on other chunks or on generic reasoning that could have been produced from the input text and pre-training alone.",
        },
    },
}

V1_R_TOP_GATES = ("T_domain", "T_subarea", "T_item", "T_central")
V1_R_IDEO_LADDER = ("I_evaluative", "I_stance", "I_party")  # levels 3, 4, 5


# ===========================================================================
# v2
# ===========================================================================

# --------------------------------------------------------------------------- #
# Shared wording
# --------------------------------------------------------------------------- #
SUBJECT_ONLY = (
    "Judge subject matter only: ignore stance, tone and ideology - opposite "
    "positions on the same measure are the same topic. Shared keywords such as "
    "'Steuern', 'Krise' or 'Freiheit' are not a shared topic by themselves."
)

ITEM_DEFINITION = (
    "The item is the most specific thing `input_text` takes a position on: a "
    "named bill or measure, a concrete event, a figure or proposal, or an action "
    "by a named person or body. If `input_text` names none of these, the item is "
    "its concrete claim (for example 'the government is spending gigantic sums'), "
    "not its general theme."
)

COMPARE = ["`input_text`", "`retrieved_chunk.text`"]

# --------------------------------------------------------------------------- #
# Questions
# --------------------------------------------------------------------------- #
V2_QUESTIONS = {
    # ---------------- R_top ladder (nested gates) ---------------- #
    "T_domain": {
        "type": "noul",
        "instructions": {
            "question": "Do the input text and the chunk belong to the same policy field?",
            "compare": COMPARE,
            "focus": SUBJECT_ONLY + " Compare policy fields, not specific measures.",
        },
        "criteria": {
            "true": "Both belong to the same concrete policy field (for example criminal law, the pandemic, climate and energy, migration, the armed forces, pensions, public finances, a named conflict), even if they discuss different measures within it.",
            "false": "They belong to different fields, or the only overlap is generic: both are 'politics', both involve the same party or person, or they share a word. Money, taxes or costs count as a shared field only when the input text is itself about public money (spending, debt, taxes, who pays); a passing mention of costs in an input text about another issue does not.",
        },
    },
    "T_subarea": {
        "type": "noul",
        "instructions": {
            "question": "Does a substantial passage of the chunk deal with the same narrow policy question as the input text?",
            "compare": COMPARE,
            "focus": SUBJECT_ONLY
            + " A substantial passage is at least a few connected sentences; the rest of the chunk may be about something else.",
        },
        "criteria": {
            "true": "At least one substantial passage of the chunk deals with the same kind of decision or problem as the input text (for example how a specific cost is shared, whether a specific obligation should exist, how a specific programme is financed), even if it concerns another detail, institution or year than the input text's own item.",
            "false": "No passage of the chunk deals with that narrow question: the chunk is about a different question under the same umbrella theme (for example the input text is about paying for tests and the chunk is about a vaccination mandate; or one is about military equipment and the other about conscription), or it touches the question only in a phrase or a passing sentence.",
        },
    },
    "T_item": {
        "type": "noul",
        "instructions": {
            "question": "Does the chunk discuss the item that the input text takes a position on?",
            "compare": COMPARE,
            "focus": SUBJECT_ONLY + " " + ITEM_DEFINITION,
        },
        "criteria": {
            "true": "At least one passage of the chunk addresses that same item - the same bill, measure, event, figure, action or claim - from any angle and with any position.",
            "false": "The chunk does not address that item: it covers other measures, a more general question, or a different instance of the same kind of thing (another bill, another year's budget, another event).",
        },
    },
    "T_central": {
        "type": "noul",
        "instructions": {
            "question": "Is the item that the input text takes a position on the main subject of the chunk?",
            "compare": COMPARE,
            "focus": SUBJECT_ONLY + " " + ITEM_DEFINITION,
        },
        "criteria": {
            "true": "A one-sentence summary of the chunk would name that item (named or unmistakably identified) as its main subject. Support or opposition does not matter.",
            "false": "The item is one topic among several, is mentioned briefly, is only the occasion of the debate, or the chunk's main subject is a related but different point.",
        },
    },
    # Diagnostic, input only (same value for all k chunks of a text). Use it to
    # stratify errors: item-specific vs. item-less inputs. Not used in the label.
    "X_input_has_item": {
        "type": "noul",
        "instructions": {
            "question": "Does the input text name a specific bill, measure, event, figure, or action by a named person or body?",
            "inspect": "`input_text`",
        },
        "criteria": {
            "true": "It names or unmistakably identifies such a specific item (for example 'das Heizungsgesetz', 'the 100-billion fund', 'Merkel's statement in the Bundestag yesterday').",
            "false": "It makes a general claim or expresses a general attitude without a specific item (for example 'the government wastes our money', 'migration is out of control').",
        },
    },
    # ---------------- R_top holistic Score (human rubric) ---------------- #
    "R_top_score": {
        "type": "score",
        "instructions": {
            "question": "How closely does the subject matter of the chunk match the input text?",
            "compare": COMPARE,
            "focus": SUBJECT_ONLY
            + " Rate by the chunk's most relevant substantive passage. "
            + ITEM_DEFINITION,
        },
        "criteria": [
            {
                "what": "Different policy field and different issue; no substantive connection.",
                "not_for": "Chunks that share the policy field but discuss another issue.",
                "example": "Input about a municipal heating law; chunk about defence procurement.",
            },
            {
                "what": "Same broad policy field, but an entirely different specific issue.",
                "not_for": "Chunks about the same policy mechanism or narrow question as the input.",
                "example": "Input about a municipal heating law; chunk about energy supply security or North Sea wind farms.",
            },
            {
                "what": "Same narrow policy question or regulatory mechanism, but a different scope, institution, target group or item.",
                "not_for": "Chunks that discuss the very bill, event or claim the input addresses.",
                "example": "Input about a municipal heating law; chunk about insulation standards for commercial buildings or an EU efficiency directive.",
            },
            {
                "what": "Discusses the same item or controversy as the input, differing at most in sub-clause, timing or venue, or covering it as one part of a broader chunk.",
                "not_for": "Chunks whose main subject is exactly the input's item.",
                "example": "Input about municipal heating mandates; chunk debating deadlines in the Building Energy Act (Gebaeudeenergiegesetz).",
            },
            {
                "what": "The chunk's main subject is exactly the bill, event, claim or decision the input addresses.",
                "not_for": "Chunks where that item is one topic among several.",
                "example": "Input about municipal heating mandates; chunk debating those specific mandates and the transition subsidies the input cites.",
            },
        ],
    },
    # ---------------- R_ideo gates ---------------- #
    "I_procedural": {
        "type": "noul",
        "instructions": {
            "question": "Is the chunk only floor management or formalities?",
            "inspect": "`retrieved_chunk.text`",
        },
        "criteria": {
            "true": "The chunk consists of procedure only: addressing the chair, thanks, greetings, announcing agenda items or votes, with no substantive statement about policy.",
            "false": "The chunk contains substantive statements about policy, facts or political actors.",
        },
    },
    "I_evaluative": {  # unchanged from v1 (2-vs-3 line was already tuned in round 3)
        "type": "noul",
        "instructions": {
            "question": "Does the passage use normative or politically loaded language, selective emphasis or one-sided framing?",
            "inspect": "`retrieved_chunk.text`",
        },
        "criteria": {
            "true": "The speaker frames the issue from one side: value-laden or politically loaded vocabulary ('economic justice' vs. 'favouring the rich', 'burden' vs. 'humanitarian obligation'), selective emphasis, praise or attacks on political actors, or demands for how things ought to be.",
            "false": "The passage stays neutral or balanced: it reports facts, figures or procedures, gives a balanced overview of several viewpoints, makes a consensual or cross-party appeal, thanks or praises non-partisan actors, or explains a measure in a sober tone - even if the speaker mildly says they support something.",
        },
    },
    # Replaces I_stance: the human 3-vs-4 test is placeability, not intensity.
    "I_lr_position": {
        "type": "noul",
        "instructions": {
            "question": "Could a reader place the speaker on the political left-right spectrum from this passage alone?",
            "inspect": "`retrieved_chunk.text`",
            "focus": "Assume you do not know the speaker or their party. Answer true only if the passage itself makes the placement clear; if you would only be guessing from tone or a few loaded words, answer false.",
        },
        "criteria": {
            "true": "The passage asserts how a contested policy ought to be, or judges an issue, in a way typical of one camp (market-liberal, welfare-state or redistributive, ecological, conservative, nationalist, law-and-order), so a reader could say which side the speaker is on. Examples: denouncing wealth redistribution as state theft; demanding state control over rents; attacking corporate profits.",
            "false": "A reader could not place the speaker: the passage is neutral or balanced; it criticises or defends a measure on technical, legal or procedural grounds that any side could use; it appeals for cross-party unity; or it uses loaded vocabulary or one-sided emphasis but stops short of a position typical of one camp.",
        },
    },
    "I_party": {  # unchanged from v1. NB: the annotator guide lists party attacks
        # ("The Ampel coalition has failed") as level 5, this question excludes them
        "type": "noul",
        "instructions": {
            "question": "Does the speaker declare what their own party or faction stands for, demands or rejects?",
            "inspect": "`retrieved_chunk.text`",
            "focus": "Burden of proof: the answer is false unless the text itself clearly shows it. Answer true only if the passage leaves no reasonable doubt on this. If the passage is ambiguous, mixed or only partly fits, answer false.",
        },
        "criteria": {
            "true": "Both hold: (1) the chunk text explicitly names the speaker's own party or faction as the holder of the position ('we as the Left', 'for us in the CDU/CSU', 'we, the AfD', 'my faction', 'our motion') and (2) it states a programmatic position of that party - what it demands, rejects, proposes or will do ('we reject this', 'we demand', 'we will vote against', 'our proposal is').",
            "false": "At least one fails. In particular these do NOT count: a generic 'we' (Germany, the Bundestag, everyone: 'we need a new mobility policy'); speaking as a minister or for the government; naming other parties only to attack or address them; a passing mention of one's own party inside an attack on others; a position that is merely typical of the party but never tied to it in the text.",
        },
    },
    # ---------------- R_ideo holistic Score (human rubric) ---------------- #
    "R_ideo_score": {
        "type": "score",
        "instructions": {
            "question": "How much ideological expression or political signalling does the passage contain?",
            "inspect": "`retrieved_chunk.text`",
            "focus": "Judge what is verbally expressed in this passage, not who the speaker is. A forceful tone alone is not ideology; look for claims about how things ought to be.",
        },
        "criteria": [
            {
                "what": "Purely procedural, administrative or technical content with no evaluative or political framing.",
                "not_for": "Passages that mention a controversy, a trade-off or a political actor's position.",
                "examples": [
                    "budget figures",
                    "committee voting schedules",
                    "calls to order",
                ],
            },
            {
                "what": "Acknowledges a controversy, several viewpoints or a trade-off in a neutral, non-evaluative tone.",
                "not_for": "Passages that use loaded vocabulary, one-sided emphasis or demands.",
                "examples": [
                    "'proponents argue X, whereas opponents contend Y'",
                    "balanced ministerial reporting",
                ],
            },
            {
                "what": "Normative or loaded language, selective emphasis or one-sided framing, but the passage alone does not reveal where the speaker stands on the left-right spectrum.",
                "not_for": "Passages with a clear policy position typical of one ideological camp.",
                "examples": [
                    "tax relief called 'economic justice' or 'favouring the rich'",
                    "immigration framed only as a 'burden on infrastructure'",
                ],
            },
            {
                "what": "A clear position typical of one ideological camp: a reader could place the speaker on the left-right spectrum without knowing who they are. The position is not tied to the speaker's own party.",
                "not_for": "Passages where the speaker names their own party or faction as holding the position.",
                "examples": [
                    "denouncing wealth redistribution as state theft",
                    "demanding state control over rents",
                    "attacking corporate profits",
                ],
            },
            {
                "what": "The speaker anchors the position in party identity: names their own party or faction as holding it, or cites a party programme or voting decision.",
                "not_for": "A generic 'we' meaning Germany or the Bundestag.",
                "examples": [
                    "'We as the CDU/CSU demand...'",
                    "'Our faction will never agree to this'",
                ],
            },
        ],
    },
    # ---------------- A_caus residual path (after code checks) ---------------- #
    "A_party_evidence": {
        "type": "noul",
        "instructions": {
            "question": "Does the justification use statements by speakers of `retrieved_chunk.party` as comparison evidence?",
            "compare": ["`generator_justification`", "`retrieved_chunk.party`"],
            "focus": "A party named only as the camp the input text itself belongs to, or as the label of the final classification, is not comparison evidence.",
        },
        "criteria": {
            "true": "The justification points to what speakers of that party said in the reference material and compares the input text to it (for example 'echoing the SPD and FDP speakers in the reference anchors', 'similar to the AfD speakers').",
            "false": "The party is not mentioned, or only as the label of the input text's orientation, or the justification does not compare the input text with reference statements.",
        },
    },
    "A_content_match": {
        "type": "noul",
        "instructions": {
            "question": "Does the justification restate a specific point made in the chunk?",
            "compare": ["`generator_justification`", "`retrieved_chunk.text`"],
        },
        "criteria": {
            "true": "The justification paraphrases or quotes a specific point, claim, figure or phrase that appears in `retrieved_chunk.text`.",
            "false": "The justification's points do not appear in the chunk, or the two overlap only in general topic words.",
        },
    },
}

V2_R_TOP_GATES = ("T_domain", "T_subarea", "T_item", "T_central")
V2_R_IDEO_LADDER = ("I_evaluative", "I_lr_position", "I_party")  # levels 3, 4, 5


# ===========================================================================
# Shared derivation helpers
# ===========================================================================


def _p(raw: dict, name: str, default: float = 0.0) -> float:
    return raw[name]["noul"] if name in raw else default


def _r_top(raw: dict, gates: tuple[str, ...]) -> tuple[int, float]:
    """Hard label (v1 rule: 1 + highest gate answered yes) and a continuous
    level for ranking: P(level >= k+1) ~ the weakest gate up to k (nested
    gates, so min rather than product), summed."""
    probs = [_p(raw, g) for g in gates]
    hard = 1 + max(
        (i for i, p in enumerate(probs, start=1) if p >= GATE_THRESHOLD), default=0
    )
    soft = 1 + sum(min(probs[: k + 1]) for k in range(len(probs)))
    return hard, soft


def _r_ideo(raw: dict, ladder: tuple[str, str, str]) -> tuple[int, float]:
    """v1 conjunction ladder: 5 = party AND stance, 4 = stance AND evaluative,
    3 = evaluative, else 1 (procedure only) / 2. ``ladder`` names the
    (evaluative, stance, party) gates of the set."""
    ev_name, st_name, pa_name = ladder
    ev, st, pa = _p(raw, ev_name), _p(raw, st_name), _p(raw, pa_name)
    proc = _p(raw, "I_procedural")  # optional (never fired in the v1 data)
    yes = lambda p: p >= GATE_THRESHOLD  # noqa: E731
    if yes(pa) and yes(st):
        hard = 5
    elif yes(st) and yes(ev):
        hard = 4
    elif yes(ev):
        hard = 3
    else:
        hard = 1 if yes(proc) else 2
    p5 = min(pa, st)
    p4 = max(p5, min(st, ev))
    p3 = max(p4, ev)
    soft = 1 + (1 - proc) * (1 + p3 + p4 + p5)
    return hard, soft


def citation_decides(citation: dict) -> bool:
    return bool(
        citation["cited_by_number"]
        or citation["speaker_named"]
        or citation["any_numeric_citation"]
    )


def _a_caus(citation: dict, residual: float | None) -> tuple[int, float, str]:
    """(label, continuous score, path). Citation rules as in v1."""
    if citation["cited_by_number"]:
        return 1, 1.0, "cited_by_number"
    if citation["speaker_named"]:
        return 1, 1.0, "speaker_named"
    if citation["any_numeric_citation"]:
        return 0, 0.0, "other_chunks_cited"
    p = residual if residual is not None else 0.0
    return int(p >= A_CAUS_THRESHOLD), p, "model"


def _ladder_uncertain(raw: dict, gates: tuple[str, ...], band: float) -> bool:
    """A gate within ``band`` of the threshold, or an inconsistent ladder (a
    higher gate yes after a lower gate no). Missing gates are ignored."""
    probs = [raw[g]["noul"] for g in gates if g in raw]
    if any(abs(p - GATE_THRESHOLD) < band for p in probs):
        return True
    answered = [p >= GATE_THRESHOLD for p in probs]
    return any(not lo and any(answered[i + 1 :]) for i, lo in enumerate(answered))


def _holistic_level(raw: dict, name: str) -> tuple[int, float] | None:
    """Score answers run 0..n-1; map to the 1..5 human scale."""
    if name not in raw or raw[name].get("score") is None:
        return None
    level = raw[name]["score"] + 1
    return int(min(5, max(1, round(level)))), float(level)


# ===========================================================================
# Derivations per set
# ===========================================================================


def derive_v1(raw: dict, citation: dict) -> dict:
    r_top, r_top_soft = _r_top(raw, V1_R_TOP_GATES)
    r_ideo, r_ideo_soft = _r_ideo(raw, V1_R_IDEO_LADDER)
    residual = _p(raw, "A_caus") if "A_caus" in raw else None
    a_caus, a_caus_score, path = _a_caus(citation, residual)
    return {
        "labels": {"R_top": r_top, "R_ideo": r_ideo, "A_caus": a_caus},
        "alt_labels": {},
        "scores": {"R_top": r_top_soft, "R_ideo": r_ideo_soft, "A_caus": a_caus_score},
        "details": {"A_caus_path": path},
    }


def review_v1(raw: dict, citation: dict, derived: dict) -> dict:
    a_caus = not citation_decides(citation) and (
        abs(_p(raw, "A_caus") - A_CAUS_THRESHOLD) < REVIEW_BANDS["A_caus"]
    )
    flags = {
        "R_top": _ladder_uncertain(raw, V1_R_TOP_GATES, REVIEW_BANDS["R_top"]),
        "R_ideo": _ladder_uncertain(raw, V1_R_IDEO_LADDER, REVIEW_BANDS["R_ideo"]),
        "A_caus": a_caus,
    }
    flags["any"] = any(flags.values())
    return flags


def derive_v2(raw: dict, citation: dict) -> dict:
    r_top, r_top_soft = _r_top(raw, V2_R_TOP_GATES)
    r_ideo, r_ideo_soft = _r_ideo(raw, V2_R_IDEO_LADDER)
    residual = None
    if "A_party_evidence" in raw and "A_content_match" in raw:
        residual = min(_p(raw, "A_party_evidence"), _p(raw, "A_content_match"))
    a_caus, a_caus_score, path = _a_caus(citation, residual)

    alt_labels, scores = {}, {
        "R_top": r_top_soft,
        "R_ideo": r_ideo_soft,
        "A_caus": a_caus_score,
    }
    for dim, question in (("R_top", "R_top_score"), ("R_ideo", "R_ideo_score")):
        holistic = _holistic_level(raw, question)
        if holistic:
            alt_labels[f"{dim}@holistic"], scores[f"{dim}@holistic"] = holistic

    details = {"A_caus_path": path}
    if "X_input_has_item" in raw:
        details["input_has_item"] = _p(raw, "X_input_has_item") >= GATE_THRESHOLD
    for dim in ("R_top", "R_ideo"):
        if f"{dim}_score" in raw:
            details[f"{dim}_score_confidence"] = raw[f"{dim}_score"].get("confidence")
    return {
        "labels": {"R_top": r_top, "R_ideo": r_ideo, "A_caus": a_caus},
        "alt_labels": alt_labels,
        "scores": scores,
        "details": details,
    }


def review_v2(raw: dict, citation: dict, derived: dict) -> dict:
    labels, alt = derived["labels"], derived["alt_labels"]

    def disagrees(dim: str) -> bool:
        h = alt.get(f"{dim}@holistic")
        return h is not None and abs(h - labels[dim]) >= HOLISTIC_DISAGREEMENT

    residual = derived["scores"]["A_caus"]
    flags = {
        "R_top": _ladder_uncertain(raw, V2_R_TOP_GATES, REVIEW_BANDS["R_top"])
        or disagrees("R_top"),
        "R_ideo": _ladder_uncertain(raw, V2_R_IDEO_LADDER, REVIEW_BANDS["R_ideo"])
        or disagrees("R_ideo"),
        "A_caus": not citation_decides(citation)
        and abs(residual - A_CAUS_THRESHOLD) < REVIEW_BANDS["A_caus"],
    }
    flags["any"] = any(flags.values())
    return flags


# ===========================================================================
# Registry
# ===========================================================================


@dataclass(frozen=True)
class QuestionSet:
    name: str
    description: str
    questions: dict
    full_state_questions: frozenset
    required: frozenset  # minimum keys a --questions-file override must contain
    derive: Callable[[dict, dict], dict]
    needs_review: Callable[[dict, dict, dict], dict]

    @property
    def hash(self) -> str:
        return hash_questions(self.questions)

    @property
    def content_questions(self) -> dict:
        return {
            k: q
            for k, q in self.questions.items()
            if k not in self.full_state_questions
        }

    @property
    def full_questions(self) -> dict:
        return {
            k: q for k, q in self.questions.items() if k in self.full_state_questions
        }

    def with_questions(self, questions: dict) -> "QuestionSet":
        missing = self.required - set(questions)
        if missing:
            raise SystemExit(
                f"Questions for set '{self.name}' are missing: {sorted(missing)}"
            )
        return dataclasses.replace(self, questions=questions)


QUESTION_SETS: dict[str, QuestionSet] = {
    "v1": QuestionSet(
        name="v1",
        description="Calibrated ladder (jev-1.14.0 round 3): I_stance, single A_caus noul",
        questions=V1_QUESTIONS,
        full_state_questions=frozenset({"A_caus"}),
        required=frozenset({*V1_R_TOP_GATES, *V1_R_IDEO_LADDER, "A_caus"}),
        derive=derive_v1,
        needs_review=review_v1,
    ),
    "v2": QuestionSet(
        name="v2",
        description="Per-gate focus, item fallback, I_lr_position, holistic Scores, split A_caus",
        questions=V2_QUESTIONS,
        full_state_questions=frozenset({"A_party_evidence", "A_content_match"}),
        required=frozenset(
            {*V2_R_TOP_GATES, *V2_R_IDEO_LADDER, "A_party_evidence", "A_content_match"}
        ),
        derive=derive_v2,
        needs_review=review_v2,
    ),
}
