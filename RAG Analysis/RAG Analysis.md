# RAG Analysis

## Chunk Relevance Evaluation (Likert Scale)

Use LLM-as-a-judge to annotate retrieved chunks on a Likert Scale from 1 - 5 on a topical relevance and a ideological specifity. 
Annotate a random sample of 50–100 chunks yourself, using *Cohen’s Kappa* or *Krippendorff’s Alpha*. If *Kappa* < 0.70, the automated metric lacks construct validity.

### Topical Relevance (R_top)
Does the chunk discuss the exact policy issue, entity, or debate present in the input text? (1 = Completely unrelated topic; 5 = Identical policy topic).

```
================================================================================
TOPICAL RELEVANCE (R_top)
================================================================================

Example Input: "Carbon tax on industrial emissions."

[1] IRRELEVANT: Different domain and issue entirely.
    • Chunk: "Municipal zoning laws for suburban housing."

[2] BROAD DOMAIN ONLY: Shares high-level domain, but addresses a different policy.
    • Chunk: "Subsidies for domestic solar panel manufacturing."

[3] RELATED SUB-ISSUE: Same policy area and mechanism, but different target/context.
    • Chunk: "Fuel excise taxes on commercial aviation."

[4] DIRECT POLICY OVERLAP: Same exact policy debate, differing only in minor scope.
    • Chunk: "Cap-and-trade carbon pricing mechanisms for heavy manufacturing."

[5] IDENTICAL TARGET: Exact entity, legislation, or policy mechanism.
    • Chunk: "Section 4B statutory rates for industrial carbon taxation."
```

#### Issues
Standard Cohen's Kappa is Statistically Invalid Here: Standard Cohen's κ treats all misclassifications equally (e.g., a disagreement between 1 and 2 is penalized the same as a disagreement between 1 and 5). Because Likert data is ordinal, you must use *Quadratic Weighted Cohen's Kappa* or *Krippendorff's Alpha* with an ordinal difference metric.

### Ideological Specificity (R_ideo)
Does the chunk provide unambiguous grounding for how a specific political party, faction, or ideology views this topic? (1 = Purely descriptive/neutral or misleading; 5 = Explicit party stance/manifesto grounding)

```
================================================================================
IDEOLOGICAL SPECIFICITY (R_ideo)
================================================================================
[1] DESCRIPTIVE / PROCEDURAL: Purely administrative, factual, or neutral metrics.
    • Example: "The committee met on Tuesday to review the 2024 budget allocation."

[2] BALANCED OVERVIEW: Mentions political controversy but gives equal weight/neutral tone.
    • Example: "Proponents argue it cuts emissions, while critics warn of energy costs."

[3] IMPLICIT VALUE FRAMING: Uses biased terminology or selective facts without naming actors.
    • Example: "Burdensome regulatory overreach continues to stifle industrial growth."

[4] CLEAR IDEOLOGICAL STANCE: Unambiguous ideological orientation (e.g., social democratic, libertarian).
    • Example: "Market-driven deregulation is the only viable path to economic freedom."

[5] EXPLICIT PARTY / MANIFESTO GROUNDING: Cites specific party doctrine, voting positions, or platforms.
    • Example: "The Green Party platform explicitly mandates a 100% phase-out by 2030."
```

### References
- [Krippendorff's Alpha](https://www.youtube.com/watch?v=D3Tw08uhuK0)
- [Cohen's Kappa](https://numiqo.de/video/4vnKzACaG2k)

## Overlap vs. Causal Utility

Questions to be answered:
1. Do the chunks contain the same knowledge as the input text?
2. Are the chunks actually helpful to predicting the bias?

### Two Essential Metrics

#### Information Delta (N_info - Binary or 3-Point):
- **Question**: Does the retrieved chunk contain facts, ideological definitions, or context not present in the input text?
- **Why it matters**: If *N_info* = Low, the retriever is just echoing the input prompt. High retrieval accuracy on redundant text yields zero informational gain for the classifier.

#### Attribution / Faithfulness (A_caus - Binary 0/1):
- **Question**: Did the generator explicitly rely on the retrieved chunk's unique evidence in its reasoning or classification output?
- **Why it matters**: LLMs frequently suffer from parametric bias override—the retriever fetches the correct foreign party context (e.g., German FDP vs. US Libertarians for RQ2), but the model ignores it and applies US-centric political assumptions learned during pretraining.

## Getting samples

Delta Stratified Sampling across 2–3 core experimental conditions.

### Sampling Procedure

1. **Isolate 3 Core Pipeline Comparisons:**
* Baseline (No RAG) vs. Best Dense Retriever.
* Best Dense Retriever vs. Best Hybrid/Reranked Retriever.
* Domestic Context vs. Cross-Cultural / Foreign Context (for RQ2). (*Not tested yet*)


2. **Sample 15 Cases per Quadrant ($N = 60$ per comparison):**
* **Quadrant I (RAG Win):** Baseline Incorrect $\to$ RAG Correct. *Qualitative Goal:* Identify which chunk feature ($R_{top}$, $R_{ideo}$, $N_{info}$) corrected the model.
* **Quadrant II (RAG Distraction / Poisoning):** Baseline Correct $\to$ RAG Incorrect. *Qualitative Goal:* Identify if failure was caused by noisy retrieval ($R_{top} < 2$), ideological mismatch ($R_{ideo} < 2$), or generator misattribution.
* **Quadrant III (Joint Failure):** Both Incorrect. *Qualitative Goal:* Identify inherent data ambiguity or unrecoverable domain shifts.
* **Quadrant IV (Baseline Sufficiency):** Both Correct. *Qualitative Goal:* Verify whether retrieval added actionable evidence or was bypassed entirely.

___

# Action Plan

All right, lets get the samples in, these are my collective runs so far: 

```csv
Target,Model,Condition,N_samples,MAE,RMSE,Pearson_r,Spearman_rho
label_ideology,Llama-3.1-8B-Instruct,no_rag,775,1.6841,2.315,0.4617,0.4972
label_ideology,Llama-3.1-8B-Instruct,bge/hyde,774,1.1496,1.6877,0.5492,0.5054
label_ideology,Llama-3.1-8B-Instruct,bge/hyde_hybrid,775,1.141,1.676,0.5467,0.496
label_ideology,Llama-3.1-8B-Instruct,bge/simple,775,1.123,1.6536,0.5701,0.5364
label_ideology,Llama-3.1-8B-Instruct,bge/simple_hybrid,775,1.1097,1.6044,0.5955,0.5366
label_ideology,Llama-3.1-8B-Instruct,bge/twostage,216,1.0847,1.6107,0.5418,0.5309
label_ideology,Llama-3.1-8B-Instruct,e5/hyde,775,1.0644,1.542,0.6052,0.5498
label_ideology,Llama-3.1-8B-Instruct,e5/simple,774,1.1168,1.6279,0.5749,0.5205
label_ideology,Llama-3.1-8B-Instruct,e5/twostage,775,1.1494,1.6555,0.5736,0.5144
label_ideology,Llama-3.1-8B-Instruct,jina/hyde,775,1.151,1.6734,0.554,0.5105
label_ideology,Llama-3.1-8B-Instruct,jina/simple,775,1.1818,1.7105,0.5494,0.5178
label_ideology,Llama-3.1-8B-Instruct,jina/twostage,673,1.182,1.7185,0.5558,0.5266
label_ideology,Llama-3.1-8B-Instruct,qwen3/hyde,775,1.0324,1.5124,0.6245,0.5675
label_ideology,Llama-3.1-8B-Instruct,qwen3/simple,775,1.0619,1.5796,0.608,0.5624
label_ideology,Llama-3.1-8B-Instruct,qwen3/twostage,775,1.0858,1.6009,0.5913,0.5497
label_ideology,Llama-3.2-3B-Instruct,no_rag,775,1.7679,2.3422,0.2722,0.1481
label_ideology,Llama-3.2-3B-Instruct,bge/hyde,729,1.8727,2.3289,0.2087,0.065
label_ideology,Llama-3.2-3B-Instruct,bge/hyde_hybrid,729,1.8185,2.2659,0.2293,0.0728
label_ideology,Llama-3.2-3B-Instruct,bge/simple,727,1.8893,2.3461,0.2296,0.0981
label_ideology,Llama-3.2-3B-Instruct,bge/simple_hybrid,708,1.9102,2.3608,0.2656,0.1175
label_ideology,Llama-3.2-3B-Instruct,bge/twostage,729,1.9597,2.4241,0.1852,0.0377
label_ideology,Llama-3.2-3B-Instruct,bge/twostage_hybrid,707,2.0335,2.5014,0.2078,0.0838
label_ideology,Llama-3.2-3B-Instruct,e5/hyde,710,1.9855,2.4546,0.2162,0.0768
label_ideology,Llama-3.2-3B-Instruct,e5/simple,710,1.9527,2.4273,0.2604,0.1214
label_ideology,Llama-3.2-3B-Instruct,e5/twostage,708,1.9936,2.4834,0.2398,0.0908
label_ideology,Llama-3.2-3B-Instruct,jina/hyde,657,1.9409,2.4335,0.2587,0.1025
label_ideology,Llama-3.2-3B-Instruct,jina/simple,646,1.9181,2.4155,0.237,0.0868
label_ideology,Llama-3.2-3B-Instruct,jina/twostage,724,1.9489,2.4379,0.242,0.0962
label_ideology,Llama-3.2-3B-Instruct,qwen3/hyde,719,1.9544,2.4129,0.235,0.1123
label_ideology,Llama-3.2-3B-Instruct,qwen3/simple,724,1.9851,2.4532,0.2357,0.0931
label_ideology,Llama-3.2-3B-Instruct,qwen3/twostage,716,2.0208,2.5002,0.2158,0.0742
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,no_rag,775,0.9057,1.3031,0.7054,0.6379
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,bge/hyde,775,1.2055,1.6562,0.4257,0.412
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,bge/hyde_hybrid,775,1.2098,1.6755,0.4033,0.3999
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,bge/simple,775,1.2271,1.7102,0.3752,0.3684
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,bge/simple_hybrid,775,1.223,1.7033,0.3782,0.3806
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,bge/twostage,774,1.2072,1.6518,0.4335,0.4474
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,bge/twostage_hybrid,775,1.2324,1.6741,0.4099,0.4299
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,e5/hyde,775,1.2179,1.6722,0.4148,0.4055
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,e5/simple,775,1.2115,1.6945,0.3996,0.4029
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,e5/twostage,775,1.1804,1.6325,0.4547,0.4351
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,jina/hyde,775,1.2061,1.669,0.4189,0.4163
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,jina/simple,775,1.2272,1.7029,0.3936,0.3854
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,jina/twostage,775,1.1939,1.6713,0.4052,0.4014
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,qwen3/hyde,775,1.1987,1.695,0.3829,0.4005
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,qwen3/simple,775,1.1354,1.6175,0.4541,0.45
label_ideology,Meta-Llama-3.1-70B-Instruct-FP8,qwen3/twostage,775,1.1963,1.6689,0.4117,0.425
label_ideology,Ministral-3-14B-Instruct-2512,no_rag,775,0.9347,1.3037,0.6727,0.5774
label_ideology,Ministral-3-14B-Instruct-2512,bge/hyde,773,0.9749,1.4615,0.6274,0.5708
label_ideology,Ministral-3-14B-Instruct-2512,bge/hyde_hybrid,771,0.987,1.4654,0.6253,0.5664
label_ideology,Ministral-3-14B-Instruct-2512,bge/simple,772,0.9237,1.3712,0.6682,0.5995
label_ideology,Ministral-3-14B-Instruct-2512,bge/simple_hybrid,770,0.9531,1.3845,0.6726,0.6036
label_ideology,Ministral-3-14B-Instruct-2512,bge/twostage,770,0.9338,1.3792,0.6714,0.6165
label_ideology,Ministral-3-14B-Instruct-2512,bge/twostage_hybrid,770,0.9318,1.3638,0.6762,0.6017
label_ideology,Ministral-3-14B-Instruct-2512,e5/hyde,773,0.9613,1.4393,0.6393,0.5924
label_ideology,Ministral-3-14B-Instruct-2512,e5/simple,770,0.9405,1.3652,0.6843,0.6179
label_ideology,Ministral-3-14B-Instruct-2512,e5/twostage,768,0.9234,1.3474,0.6826,0.6168
label_ideology,Ministral-3-14B-Instruct-2512,jina/hyde,773,0.94,1.3893,0.679,0.6374
label_ideology,Ministral-3-14B-Instruct-2512,jina/simple,771,0.9551,1.4156,0.6588,0.6041
label_ideology,Ministral-3-14B-Instruct-2512,jina/twostage,761,0.9574,1.404,0.6629,0.6063
label_ideology,Ministral-3-14B-Instruct-2512,qwen3/hyde,770,0.987,1.4615,0.6351,0.5862
label_ideology,Ministral-3-14B-Instruct-2512,qwen3/simple,773,0.9176,1.3167,0.6966,0.6296
label_ideology,Ministral-3-14B-Instruct-2512,qwen3/twostage,772,0.925,1.358,0.6703,0.6074
label_ideology,Ministral-3-3B-Instruct-2512,no_rag,774,1.4917,1.9378,0.4831,0.4407
label_ideology,Ministral-3-3B-Instruct-2512,bge/hyde,772,1.3395,1.7065,0.4819,0.4725
label_ideology,Ministral-3-3B-Instruct-2512,bge/hyde_hybrid,768,1.3408,1.702,0.4938,0.4693
label_ideology,Ministral-3-3B-Instruct-2512,bge/simple,772,1.3374,1.717,0.4833,0.4837
label_ideology,Ministral-3-3B-Instruct-2512,bge/simple_hybrid,771,1.3464,1.7105,0.4738,0.4746
label_ideology,Ministral-3-3B-Instruct-2512,bge/twostage,770,1.3247,1.6774,0.5202,0.5176
label_ideology,Ministral-3-3B-Instruct-2512,bge/twostage_hybrid,769,1.3914,1.7541,0.4603,0.4683
label_ideology,Ministral-3-3B-Instruct-2512,e5/hyde,766,1.3645,1.7622,0.4266,0.4232
label_ideology,Ministral-3-3B-Instruct-2512,e5/simple,762,1.3432,1.7305,0.4611,0.4621
label_ideology,Ministral-3-3B-Instruct-2512,e5/twostage,771,1.3336,1.7017,0.4774,0.4824
label_ideology,Ministral-3-3B-Instruct-2512,jina/hyde,606,1.3894,1.7986,0.4149,0.4343
label_ideology,Ministral-3-3B-Instruct-2512,jina/simple,772,1.4402,1.8654,0.3905,0.4092
label_ideology,Ministral-3-3B-Instruct-2512,jina/twostage,768,1.3699,1.7445,0.4734,0.4727
label_ideology,Ministral-3-3B-Instruct-2512,qwen3/hyde,769,1.2718,1.6319,0.5148,0.5092
label_ideology,Ministral-3-3B-Instruct-2512,qwen3/simple,764,1.3136,1.6936,0.4744,0.472
label_ideology,Ministral-3-3B-Instruct-2512,qwen3/twostage,768,1.3286,1.6929,0.4763,0.4784
label_ideology,Ministral-3-8B-Instruct-2512,no_rag,775,1.3111,1.7298,0.6084,0.5872
label_ideology,Ministral-3-8B-Instruct-2512,bge/hyde,763,1.3442,1.9263,0.5676,0.5702
label_ideology,Ministral-3-8B-Instruct-2512,bge/hyde_hybrid,756,1.3634,1.9803,0.5494,0.553
label_ideology,Ministral-3-8B-Instruct-2512,bge/simple,765,1.323,1.8902,0.5818,0.5813
label_ideology,Ministral-3-8B-Instruct-2512,bge/simple_hybrid,752,1.3729,1.9441,0.5731,0.57
label_ideology,Ministral-3-8B-Instruct-2512,bge/twostage,769,1.3365,1.8147,0.5984,0.5769
label_ideology,Ministral-3-8B-Instruct-2512,bge/twostage_hybrid,775,1.301,1.7115,0.615,0.5947
label_ideology,Ministral-3-8B-Instruct-2512,e5/hyde,755,1.3207,1.8794,0.5807,0.5732
label_ideology,Ministral-3-8B-Instruct-2512,e5/simple,760,1.3212,1.8763,0.5977,0.5874
label_ideology,Ministral-3-8B-Instruct-2512,e5/twostage,757,1.3742,1.97,0.576,0.5877
label_ideology,Ministral-3-8B-Instruct-2512,jina/hyde,763,1.3953,1.9909,0.5371,0.5412
label_ideology,Ministral-3-8B-Instruct-2512,jina/simple,759,1.3992,1.9706,0.5713,0.5676
label_ideology,Ministral-3-8B-Instruct-2512,jina/twostage,752,1.3352,1.9021,0.5868,0.5763
label_ideology,Ministral-3-8B-Instruct-2512,qwen3/hyde,768,1.3346,1.9092,0.5801,0.5788
label_ideology,Ministral-3-8B-Instruct-2512,qwen3/simple,758,1.2875,1.8491,0.5966,0.5857
label_ideology,Ministral-3-8B-Instruct-2512,qwen3/twostage,760,1.3093,1.8949,0.5973,0.6049
label_ideology,Qwen2.5-32B-Instruct,no_rag,775,1.4272,1.6603,0.4988,0.5259
label_ideology,Qwen2.5-32B-Instruct,bge/hyde,499,1.1964,1.4555,0.5972,0.5615
label_ideology,Qwen2.5-32B-Instruct,bge/hyde_hybrid,498,1.1942,1.4603,0.6068,0.5719
label_ideology,Qwen2.5-32B-Instruct,bge/simple,775,1.2019,1.4881,0.5726,0.5882
label_ideology,Qwen2.5-32B-Instruct,bge/simple_hybrid,499,1.2283,1.4802,0.6056,0.5815
label_ideology,Qwen2.5-32B-Instruct,bge/twostage,499,1.1735,1.4107,0.6596,0.6263
label_ideology,Qwen2.5-32B-Instruct,bge/twostage_hybrid,775,1.2436,1.5342,0.5438,0.546
label_ideology,Qwen2.5-32B-Instruct,e5/hyde,775,1.2292,1.516,0.5461,0.5493
label_ideology,Qwen2.5-32B-Instruct,e5/simple,775,1.2012,1.4804,0.5761,0.5759
label_ideology,Qwen2.5-32B-Instruct,e5/twostage,775,1.2077,1.4845,0.5737,0.5799
label_ideology,Qwen2.5-32B-Instruct,jina/hyde,775,1.1941,1.5051,0.5617,0.5618
label_ideology,Qwen2.5-32B-Instruct,jina/simple,775,1.1934,1.4706,0.5987,0.6009
label_ideology,Qwen2.5-32B-Instruct,jina/twostage,775,1.2481,1.537,0.5254,0.5248
label_ideology,Qwen2.5-32B-Instruct,qwen3/hyde,775,1.1409,1.4062,0.629,0.6196
label_ideology,Qwen2.5-32B-Instruct,qwen3/simple,775,1.1653,1.4634,0.5794,0.5898
label_ideology,Qwen2.5-32B-Instruct,qwen3/twostage,775,1.2271,1.5101,0.5502,0.5675
label_ideology,Qwen2.5-3B-Instruct,no_rag,775,1.7126,1.9624,0.0605,-0.0031
label_ideology,Qwen2.5-3B-Instruct,bge/hyde,775,1.7987,1.9993,-0.0079,-0.0292
label_ideology,Qwen2.5-3B-Instruct,bge/hyde_hybrid,774,1.7946,2.0078,-0.0057,0.0021
label_ideology,Qwen2.5-3B-Instruct,bge/simple,775,1.7911,1.9909,0.113,0.057
label_ideology,Qwen2.5-3B-Instruct,bge/simple_hybrid,774,1.7765,1.9752,0.1754,0.1032
label_ideology,Qwen2.5-3B-Instruct,bge/twostage,775,1.7978,1.9932,0.0853,0.0542
label_ideology,Qwen2.5-3B-Instruct,bge/twostage_hybrid,775,1.809,2.0018,0.0783,0.0329
label_ideology,Qwen2.5-3B-Instruct,e5/hyde,775,1.7782,1.9678,0.0342,0.0134
label_ideology,Qwen2.5-3B-Instruct,e5/simple,773,1.7942,1.9866,0.1003,0.0551
label_ideology,Qwen2.5-3B-Instruct,e5/twostage,773,1.7951,1.9812,0.1285,0.0788
label_ideology,Qwen2.5-3B-Instruct,jina/hyde,544,1.7676,1.9623,0.1594,0.1055
label_ideology,Qwen2.5-3B-Instruct,jina/simple,774,1.7906,1.9846,0.1164,0.0757
label_ideology,Qwen2.5-3B-Instruct,jina/twostage,775,1.7895,1.9892,0.1555,0.0873
label_ideology,Qwen2.5-3B-Instruct,qwen3/hyde,775,1.7926,1.9855,0.0853,0.0537
label_ideology,Qwen2.5-3B-Instruct,qwen3/simple,775,1.8036,2.002,0.1042,0.0427
label_ideology,Qwen2.5-3B-Instruct,qwen3/twostage,775,1.7792,1.9772,0.1609,0.111
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,no_rag,775,1.223,1.4378,0.7362,0.644
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,bge/hyde,775,1.1632,1.4628,0.6967,0.6251
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,bge/hyde_hybrid,775,1.1644,1.4734,0.6833,0.6068
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,bge/simple,775,1.1603,1.4642,0.7,0.6242
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,bge/simple_hybrid,775,1.1695,1.4564,0.7139,0.62
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,bge/twostage,775,1.1933,1.4842,0.6993,0.6164
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,bge/twostage_hybrid,775,1.191,1.478,0.7127,0.6253
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,e5/hyde,775,1.1746,1.4684,0.6863,0.5992
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,e5/simple,775,1.1348,1.4356,0.7224,0.6348
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,e5/twostage,775,1.1298,1.419,0.7261,0.6369
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,jina/simple,775,1.1463,1.4687,0.7119,0.6218
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,jina/twostage,775,1.1644,1.4789,0.7107,0.6227
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,qwen3/hyde,775,1.1243,1.4259,0.7103,0.6294
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,qwen3/simple,775,1.1085,1.422,0.7158,0.6294
label_ideology,Qwen2.5-72B-Instruct-FP8-dynamic,qwen3/twostage,775,1.1325,1.4404,0.7173,0.6326
label_ideology,Qwen2.5-7B-Instruct,no_rag,774,1.4363,1.7116,0.3646,0.2347
label_ideology,Qwen2.5-7B-Instruct,bge/hyde,775,1.4255,1.8514,0.3994,0.2721
label_ideology,Qwen2.5-7B-Instruct,bge/hyde_hybrid,775,1.4954,1.9419,0.3401,0.219
label_ideology,Qwen2.5-7B-Instruct,bge/simple,775,1.483,1.9374,0.3643,0.2102
label_ideology,Qwen2.5-7B-Instruct,bge/simple_hybrid,775,1.5034,1.9651,0.3604,0.2136
label_ideology,Qwen2.5-7B-Instruct,bge/twostage,774,1.5314,1.9716,0.3226,0.1606
label_ideology,Qwen2.5-7B-Instruct,bge/twostage_hybrid,775,1.5348,1.9985,0.3291,0.1836
label_ideology,Qwen2.5-7B-Instruct,e5/hyde,775,1.5098,1.9377,0.3827,0.228
label_ideology,Qwen2.5-7B-Instruct,e5/simple,774,1.4934,1.9213,0.4118,0.2741
label_ideology,Qwen2.5-7B-Instruct,e5/twostage,772,1.4845,1.9326,0.3832,0.224
label_ideology,Qwen2.5-7B-Instruct,jina/hyde,775,1.4795,1.954,0.3724,0.2236
label_ideology,Qwen2.5-7B-Instruct,jina/simple,774,1.4718,1.9305,0.399,0.2396
label_ideology,Qwen2.5-7B-Instruct,jina/twostage,775,1.5083,1.9803,0.3396,0.1928
label_ideology,Qwen2.5-7B-Instruct,qwen3/hyde,774,1.4627,1.8935,0.3858,0.2397
label_ideology,Qwen2.5-7B-Instruct,qwen3/simple,775,1.464,1.9204,0.3756,0.2421
label_ideology,Qwen2.5-7B-Instruct,qwen3/twostage,773,1.5338,1.9998,0.334,0.1897
,,,,,,,
```

each `N` means one of these logs:
```jsonl
{"run_id": "eval_matrix_20260817_121042_bge", "timestamp": "2026-08-17T12:55:30.545902", "parameters": {"llm": "RedHatAI/Meta-Llama-3.1-70B-Instruct-FP8", "llm_region": "Americas", "embedding_model": "bge", "retrieval_mode": "simple", "hybrid": false, "is_rag": true, "k_chunks": 5}, "input_metadata": {"text_index": "237", "party": "BÜNDNIS 90/DIE GRÜNEN", "speaker": "GoeringEckardt", "source": "wdr"}, "inputs": {"text": "Klares Bekenntnis von @ABaerbock für eine Bürgerversicherung. Wir Grüne fordern das seit Jahren. Die Mehrheit der Bevölkerung auch. Zeit,  dass in der neuen Bundesregierung gehandelt wird. #Triell #Triell", "hyde_docs": [], "retrieved_chunks": [{"text": "Und es geht darum, sich vorzubereiten auf eine Rückkehr in den Arbeitsmarkt. Wenn wir dauerhaft Erfolg haben wollen, dann müssen wir hier noch stärker auf Qualifizierung setzen, insbesondere bei den Jüngeren. Wir werden die Grundsicherung daher zielgerichtet weiterentwickeln. Lassen Sie uns diese Debatte verantwortlich und auch sachlich führen. Liebe Kolleginnen und Kollegen, wir stehen vor anstrengenden Jahren. Ich bin mir sicher, dass die meisten Menschen zu Veränderungen bereit sind. Aber sie erwarten, dass es dabei gerecht zugeht. Soziale Gerechtigkeit muss daher ein Markenzeichen dieser Regierung sein. Dafür stehe ich ein.", "party": "SPD", "country": "Germany", "speaker": "Bärbel Bas", "date": "2025-05-15", "speech_id": "ID21403800", "score": 0.6567}, {"text": "Darauf werde ich auch zurückkommen. [I-1] Aber zunächst muss ich sagen: Meine Fraktion – das ist hier zitiert worden – ist ganz klar der Auffassung und Überzeugung, dass wir eine gemeinsame Versicherung aller Erwerbstätigen in der gesetzlichen Rentenversicherung wollen – wir nennen es Bürgerversicherung –, in die alle Abgeordneten, Beamtinnen und Beamten, Selbstständigen und Angehörigen der berufsständischen Versorgungswerke zusammen mit den Angestellten und Arbeitenden einzahlen sollen. [I-2] So weit, so einfach – erst mal.", "party": "BÜNDNIS 90/DIE GRÜNEN", "country": "Germany", "speaker": "Markus Kurth", "date": "2024-03-22", "speech_id": "ID2016105400", "score": 0.6521}, {"text": "Sehr geehrter Herr Präsident! Sehr verehrte Damen und Herren! Wir haben eine neue Regierung, eine neue Gesundheitsministerin. Aber eins hat die neue Koalition noch nicht: einen klaren Plan, um den Pflegenotstand jetzt wirksam anzupacken. [I-1] Statt entschlossen zu handeln, flüchtet sie sich in Prüfaufträge und Kommissionen. Dringende Entscheidungen werden vertagt, und dieses Zögern wird Ihnen, wird uns allen auf die Füße fallen. [I-2] Die Pflegeversicherung muss jetzt reformiert werden. Damit muss noch vor der Sommerpause begonnen werden. Ansonsten drohen den Menschen weitere Beitragssteigerungen noch in diesem Jahr. Das fordern nicht nur wir Grüne. Auch Wirtschaftsexpertinnen und Wirtschaftsexperten sind sich einig: Diese Legislatur entscheidet darüber, ob die soziale Pflegeversicherung finanziell stabil und sicher wird. [I-3] Pflegekassen brauchen Verlässlichkeit.", "party": "BÜNDNIS 90/DIE GRÜNEN", "country": "Germany", "speaker": "Simone Fischer", "date": "2025-05-15", "speech_id": "ID21406100", "score": 0.6516}, {"text": "Aber wir müssen jetzt aktiv werden. Herr Bundeskanzler, lieber Olaf Scholz, Frau Außenministerin, liebe Annalena Baerbock, wie Sie lesen, hören und spüren können, gibt es eine große Mehrheit im Deutschen Bundestag, die Ihren Kurs unterstützt. Lassen Sie uns gemeinsam aktiv werden!", "party": "SPD", "country": "Germany", "speaker": "Adis Ahmetovic", "date": "2022-06-22", "speech_id": "ID204311000", "score": 0.6502}, {"text": "brauchen stärkere Sozialversicherungen und müssen deswegen die Arbeitslosenversicherung zu einer Arbeitsversicherung und die Kranken-, Pflege- und Rentenversicherung zu Bürgerversicherungen weiterentwickeln. Wir brauchen mehr soziale Sicherheit, nicht nur für die Krisen jetzt, sondern um die Veränderungen, die vor uns stehen, zu meistern und dabei alle mitzunehmen. [I-15] Soziale Sicherheit ist wichtig; das ist das eine. Ebenso wichtig sind gute Löhne und gute Arbeit. Wir brauchen mehr Tarifbindung.", "party": "BÜNDNIS 90/DIE GRÜNEN", "country": "Germany", "speaker": "Wolfgang Strengmann-Kuhn", "date": "2020-11-25", "speech_id": "ID1919407300", "score": 0.6484}]}, "output": {"bias": 2.5, "justification": "The target text explicitly mentions 'Bürgerversicherung' (citizen's insurance), a concept closely associated with the German Green Party (BÜNDNIS 90/DIE GRÜNEN), as seen in reference anchors [2] and [5]. The text also expresses support for Annalena Baerbock, a prominent Green Party politician. However, the tone is more pragmatic and focused on policy implementation rather than radical or ideological. The language used is also relatively moderate, without strong emotional appeals or divisive rhetoric. Therefore, the bias score is positioned slightly left of center, but not extremely left, reflecting the pragmatic and policy-focused tone of the text."}, "ground_truth": {"label_ideology": "2.4", "label_economic": "2.6", "label_galtan": "1.4"}}
```

So I need to through the sampling process:

- For the first Quadrant I go through every llm family and model size and retrieval strategy (simple, hyde, twostage) and get the best outcome against "no_rag" and then go into these logs and find the random samples of 15 - 25
- For the second Quadrant, I will do the opposite to Quadrant 1
- For the third I am searching for bad no rag outcomes and bad scores with RAG
- for the fourth I find samples who did way better then without rag

is this a good way to get the samples for annotating, or should I do something different?