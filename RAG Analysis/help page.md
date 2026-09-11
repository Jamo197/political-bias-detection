## 1. Project Context & Purpose

This research investigates how Retrieval-Augmented Generation (RAG) influences an LLM’s ability to classify political ideology in short texts (e.g., social media posts, parliamentary statements).

While adding external context can ground predictions, it can also degrade performance if the retrieved passages introduce semantic noise, conflate topics, or present conflicting ideological cues. By evaluating individual retrieved chunks along two distinct dimensions—**Topical Relevance ($R_{\text{top}}$)** and **Ideological Grounding ($R_{\text{ideo}}$)**—your annotations allow us to analyze the exact mechanisms causing the system to succeed or fail.

---

## 2. Dataset Structure: The Four Quadrants

The evaluation dataset uses a stratified sample of **25 target texts** , each accompanied by **5 retrieved chunks**, yielding **125 total chunks** to evaluate.

The samples are drawn from four operational quadrants based on error changes between a non-RAG baseline and the RAG model:

* **$Q_1$: RAG Win:** Baseline error is high; retrieved context corrects the model and brings the prediction close to the ground truth.
* **$Q_2$: RAG Distraction:** Baseline error is low; retrieved context actively misleads the model, significantly increasing prediction error.
* **$Q_3$: Joint Failure:** Both baseline and RAG fail to predict the ideological score accurately.
* **$Q_4$: Baseline Sufficiency:** Baseline is already accurate; retrieved context maintains high accuracy without causing degradation.

---

## 3. Annotation Workflow

For each sample:

1. **Read the Source Text:** Identify the core policy topic, key entities, and any ideological signals (e.g., party hashtags, specific rhetoric).
* **Read the justifications:**: Read the justifications the LLM returned and why it predicted as it did
2. **Review the Retrieved Chunks:** You will see 5 chunks retrieved with RAG for the *Source Text*
3. **Assign Ratings:** For each chunk, assign an integer score (1–5; True/False) for:
* **$R_{\text{top}}$ (Topical Relevance)**
* **$R_{\text{ideo}}$ (Ideological Stance / Grounding)**
* **$A_{\text{caus}}$ used the LLM this chunk**



---

## 4. Coding Dimension 1: Topical Relevance ($R_{\text{top}}$)

$R_{\text{top}}$ measures how closely the retrieved chunk matches the specific subject matter, legislation, or policy debate in the target text.

| Score | Label | Operational Definition | Practical Example (Target: AfD Tweet on Municipal Heating Law / *Heizungsgesetz*) |
| --- | --- | --- | --- |
| **1** | **Irrelevant** | Different domain and issue entirely. No substantive connection. | Discussion of defense procurement or Bundeswehr budgets. |
| **2** | **Broad Domain Only** | Shares the high-level policy domain (e.g., economy, energy), but addresses an entirely different specific issue. | General discussion of German energy supply security or North Sea wind farms. |
| **3** | **Related Sub-Issue** | Same policy sub-area or regulatory mechanism, but concerns a different scope, institution, or target group. | Debate over building insulation standards in commercial real estate or European energy efficiency directives. |
| **4** | **Direct Policy Overlap** | Addresses the exact same policy controversy or legislation, differing only slightly in timeframe, sub-clause, or institutional venue. | Parliamentary debate on the Building Energy Act (*Gebäudeenergiegesetz*) heating exchange deadlines. |
| **5** | **Identical Target** | Targets the exact same legislative proposal, specific event, controversy, or executive decision referenced in the target text. | Debate focusing directly on the specific municipal heating mandates and transition subsidies cited in the tweet. |

### Disambiguation Rules for $R_{\text{top}}$:

* **The "Shared Keyword" Trap:** Do not assign a high score simply because the chunk repeats words like "Steuern", "Krise", or "Freiheit". Score according to whether the chunk discusses the *underlying policy substance*.
* **2 vs. 3:** If the passage is about the general domain (e.g., climate policy broadly), assign **2**. If it discusses heating subsidies or fossil fuel phase-outs (the exact policy mechanism), assign **3**.
* **4 vs. 5:** Reserve **5** for passages where the debate directly covers the specific target, program, or event named in the source text.

---

## 5. Coding Dimension 2: Ideological Grounding ($R_{\text{ideo}}$)

$R_{\text{ideo}}$ measures the degree of ideological expression or political signaling present in the retrieved chunk, ranging from purely neutral/administrative content to explicit party platform advocacy.

| Score | Label | Operational Definition | Diagnostic Indicators |
| --- | --- | --- | --- |
| **1** | **Descriptive / Procedural** | Purely administrative, technical, or procedural information. No evaluative or political framing. | Budget figures, committee voting schedules, text of legal notices, procedural calls to order. |
| **2** | **Balanced Overview** | Acknowledges political controversy or different viewpoints, but maintains a strictly neutral, non-evaluative tone. | News summaries stating "proponents argue X, whereas opponents contend Y"; balanced ministerial reporting. |
| **3** | **Implicit Value Framing** | Contains normative or politically loaded language, selective emphasis, or framing choices, but does not explicitly name political parties or standard ideological labels. | Referring to tax relief as "economic justice" vs. "favoring the rich"; framing immigration strictly as "burden on infrastructure" or "humanitarian obligation". |
| **4** | **Clear Ideological Stance** | Clear, unambiguous ideological positioning on a standard political spectrum (e.g., free-market conservative, green-progressive, social democratic). | Denouncing wealth redistribution as state theft; demanding systemic state control over rents; attacking corporate profits. |
| **5** | **Explicit Party / Manifesto Grounding** | Direct articulation of official party doctrine, party programs, voting cues, or explicit party attacks/endorsements. | Explicit phrases such as "We as the CDU/CSU demand...", "The Ampel coalition has failed...", or direct citations of party platforms. |

### Disambiguation Rules for $R_{\text{ideo}}$:

* **Tone vs. Content:** A passage delivered forcefully is not necessarily an ideological statement. Look for *normative policy assertions* (how things *ought* to be) versus *procedural facts*.
* **1 vs. 2:** A pure reporting of budget deficits is **1**. A reporting of budget deficits that explicitly contextualizes the trade-offs between government spending and fiscal restraint in a neutral manner is **2**.
* **3 vs. 4:** If you can immediately tell where the speaker stands on a left-right scale without needing to know who they are, assign at least **4**. If it is subtly framed through vocabulary but stops short of a direct policy demand, assign **3**.
* **4 vs. 5:** If a speaker defends a hardline market stance without referencing a party, assign **4**. If the speaker anchors the argument in party identity (e.g., "Our faction will never agree to this"), assign **5**.

---

## 6. General Coding Rules & Edge Cases

1. **Independent Judgments:** Evaluate each chunk independently. The score of Chunk 2 must not be influenced by what you saw in Chunk 1.
2. **Metadata Ignorance:** Focus on the text content. Even if the speaker metadata indicates a specific party, code $R_{\text{ideo}}$ based on what is *verbally expressed in that specific snippet*.
3. **Truncated Quotes:** Parliamentary snippets often cut off mid-sentence. Rate only the readable text provided in the chunk; do not extrapolate what the speaker might have said next.