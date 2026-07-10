import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env.local")

logger = logging.getLogger(__name__)


class BiasPrediction(BaseModel):
    bias_score: float = Field(
        ...,
        description="Continuous political bias score from 0.0 (Extreme Left) to 7.0 (Extreme Right).",
    )
    justification: str = Field(
        ...,
        description="Analytical justification for the score based strictly on the text provided.",
    )


class BiasPredictor:
    """Unified OpenRouter political bias evaluator with local CHES lookup capabilities."""

    CHES_PARTY_ID_MAP = {
        301: "CDU/CSU",
        308: "CDU/CSU",
        302: "SPD",
        303: "FDP",
        304: "BÜNDNIS 90/DIE GRÜNEN",
        306: "DIE LINKE",
        310: "AfD",
        313: "BSW",
    }

    def __init__(self):
        # OpenRouter uses standard OpenAI SDK constructs pointed to its routing engine
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            logger.error(
                "Missing valid OPENROUTER_API_KEY inside environment definitions."
            )

        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self._load_ches_database()

    def _load_ches_database(self):
        ches_path = _PROJECT_ROOT / "src/datasets/ground_truth/1999-2024_CHES.csv"
        try:
            ches_raw = pd.read_csv(str(ches_path))
            self.ches_df = ches_raw[
                ches_raw["party_id"].isin(self.CHES_PARTY_ID_MAP.keys())
            ].copy()
            self.ches_df["std_party"] = self.ches_df["party_id"].map(
                self.CHES_PARTY_ID_MAP
            )
            self.ches_df = self.ches_df[
                ["std_party", "year", "lrecon", "galtan", "lrgen"]
            ].dropna()
        except FileNotFoundError:
            logger.warning(
                f"CHES database absent at context root: `{ches_path}`. Fallbacks enabled."
            )
            self.ches_df = pd.DataFrame()

    def _get_closest_ches_score(
        self, party: str, year: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        if self.ches_df.empty or pd.isna(year) or not party:
            return None, None, None

        normalized = "CDU/CSU" if party in ["CDU", "CSU"] else party
        party_ches = self.ches_df[self.ches_df["std_party"] == normalized]
        if party_ches.empty:
            return None, None, None

        closest_idx = (party_ches["year"] - year).abs().idxmin()
        row = party_ches.loc[closest_idx]
        return float(row["lrecon"]), float(row["galtan"]), float(row["lrgen"])

    def predict_bias(
        self,
        text: str,
        model_id: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
        is_rag_mode: bool = True,
    ) -> dict:
        if is_rag_mode and context_chunks:
            rag_instructions = (
                "You are provided with 'Retrieved Context Chunks' from historical speeches. "
                "Use these as empirical benchmarks to calibrate your score. "
                "In your analysis, explicitly cite the anchor numbers (e.g., [1]) that inform your judgment. "
                "If the anchors are topically relevant but ideologically unhelpful, state that and rely on CHES standards."
            )
            excerpts = []
            for i, chunk in enumerate(context_chunks, 1):
                excerpts.append(
                    f"[{i}] Reference Anchor\n"
                    f"    Country: {chunk.get('country', 'Unknown')} | Party: {chunk.get('party', 'Unknown')} | "
                    f"Speaker: {chunk.get('speaker', 'Unknown')} | Date: {chunk.get('date', 'Unknown')}\n"
                    f"    Content: {chunk.get('text', '').strip()}"
                )
            context_block = (
                f"Retrieved Context Chunks (Ideological Reference Points):\n"
                f"=========================================\n"
                f"{"\n\n".join(excerpts)}\n"
                f"=========================================\n\n"
            )
        else:
            rag_instructions = (
                "Evaluate the text purely on its linguistic and ideological framing "
                "using standard Chapel Hill Expert Survey (CHES) criteria."
            )
            context_block = "\n\n"

        system_prompt = f"""
            You are an expert political scientist specializing in comparative European parliamentary politics.
            Analyze the target text and return a continuous political bias score on the standard left-right scale 
            used by the Chapel Hill Expert Survey (CHES), where 0.0 represents Extreme Left and 7.0 represents Extreme Right.

            {rag_instructions}

            You must respond STRICTLY with a valid JSON object matching this structural schema:
            {{
            "bias_score": <float between 0.0 and 7.0>,
            "justification": "<Concise analytical explanation for the score based strictly on the information provided tracking thematic alignment against context anchors.>"
            }}
        """

        user_message = (
            f'{context_block}Target Text to Analyze:\n"""\n{text.strip()}\n"""'
        )

        try:
            # Force structural JSON responses safely across OpenRouter target vendors
            response = self.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                extra_headers={
                    "HTTP-Referer": "https://github.com/Jamo197/political-bias-detection",
                    "X-Title": "Political RAG Pipeline Engine",
                },
            )

            raw_content = response.choices[0].message.content
            return json.loads(raw_content)
        except Exception as e:
            logger.error(
                f"OpenRouter routing execution failure on model {model_id}: {e}"
            )
            return {
                "bias_score": None,
                "justification": f"API Evaluation Pipeline Error: {str(e)}",
            }
