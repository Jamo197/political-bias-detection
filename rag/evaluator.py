import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from mistralai.client import Mistral
from openai import OpenAI
from pydantic import BaseModel, Field
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env.local")

try:
    from rag.retrieval import PoliticalRAGRetriever
except ImportError:
    from retrieval import PoliticalRAGRetriever


# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

LOG_DIR = _PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Bias Prediction — LLM-based predictor (Mistral / OpenAI / Gemini)
# ---------------------------------------------------------------------------

class BiasPrediction(BaseModel):
    bias_score: float = Field(
        ...,
        description="Continuous political bias score from 0.0 (Extreme Left) to 10.0 (Extreme Right).",
    )
    justification: str = Field(
        ...,
        description="Analytical justification for the score based strictly on the text provided.",
    )


class BaselineEvaluator:
    """LLM-based political bias predictor with CHES ground-truth lookup."""

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
        """Initialises API clients and loads the CHES ground truth database."""
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        ches_path = str(_PROJECT_ROOT / "src/datasets/ground_truth/1999-2024_CHES.csv")
        try:
            ches_raw = pd.read_csv(ches_path)
            self.ches_df = ches_raw[
                ches_raw["party_id"].isin(self.CHES_PARTY_ID_MAP.keys())
            ].copy()
            self.ches_df["std_party"] = self.ches_df["party_id"].map(self.CHES_PARTY_ID_MAP)
            self.ches_df = self.ches_df[
                ["std_party", "year", "lrecon", "galtan", "lrgen"]
            ].dropna()
        except FileNotFoundError:
            logger.warning(f"CHES database not found at `{ches_path}`. Ground truth lookups will fail.")
            self.ches_df = pd.DataFrame()

    def _get_closest_ches_score(
        self, party: str, year: int
    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Finds the CHES lrecon / galtan / lrgen scores from the wave closest to
        the document's year.  Returns (lrecon, galtan, lrgen) or (None, None, None).
        """
        if self.ches_df.empty:
            return None, None, None

        normalized = "CDU/CSU" if party in ["CDU", "CSU"] else party

        if pd.isna(year) or normalized not in self.ches_df["std_party"].values:
            return None, None, None

        party_ches = self.ches_df[self.ches_df["std_party"] == normalized]
        if party_ches.empty:
            return None, None, None

        closest_idx = (party_ches["year"] - year).abs().idxmin()
        row = party_ches.loc[closest_idx]
        return float(row["lrecon"]), float(row["galtan"]), float(row["lrgen"])

    def predict_bias(
        self,
        text: str,
        model_provider: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
    ) -> dict:
        context_block = ""
        if context_chunks:
            excerpts = []
            for i, chunk in enumerate(context_chunks, 1):
                country = chunk.get("country", "Unknown")
                party = chunk.get("party", "Unknown")
                speaker = chunk.get("speaker", "Unknown")
                date = chunk.get("date", "Unknown")
                speech_chunk = chunk.get("text", "")
                excerpts.append(
                    (
                        f"[{i}] Reference Anchor\n"
                        f"    Country: {country} | Party: {party} | Speaker: {speaker} | Date: {date}\n"
                        f"    Content: {speech_chunk}"
                    )
                )
            context_str = "\n\n".join(excerpts)
            context_block = (
                f"Retrieved Context Chunks (Ideological Reference Points):\n"
                f"=========================================\n"
                f"{context_str}\n"
                f"=========================================\n\n"
                f"Target Text to Analyze:\n"
                f'"""\n{text.strip()}\n"""'
            )

        system_prompt = (
            """
            You are an expert political scientist specializing in comparative European parliamentary politics and quantitative text analysis. 

            Your task is to analyze a target text from a specific national context and assign a continuous political bias score on the standard 
            left-right scale used by the Chapel Hill Expert Survey (CHES), where 0.0 represents the Extreme Left and 10.0 represents the Extreme Right.

            Guidelines for Evaluation:
            1. Ground your estimation by comparing the ideological framing, rhetoric, and policy positions of the target text against the provided 
            "Retrieved Context Chunks". 
            2. Use the metadata (Party, Speaker) of the context chunks as localized anchoring points within that country's political landscape.
            3. Isolate your classification from your own base biases; rely strictly on objective political science definitions of left-right economic 
            and GALTAN (Green/Alternative/Libertarian vs. Traditional/Authoritarian/Nationalist) dimensions.

            You must respond strictly in a valid JSON object matching this schema:
            {
            "bias_score": <float between 0.0 and 10.0>,
            "justification": "<A concise, rigorous analytical justification explaining the score assignment based on thematic alignment or divergence from the reference context. Mention specific metadata hooks if applicable.>"
            }
            """
        )

        user_message = f"{context_block}Text to analyse:\n{text}"

        try:
            if model_provider == "Mistral":
                res = self.mistral_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                return json.loads(res.choices[0].message.content)

            elif model_provider == "OpenAI":
                res = self.openai_client.beta.chat.completions.parse(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    response_format=BiasPrediction,
                    temperature=0.0,
                )
                return res.choices[0].message.parsed.model_dump()

            elif model_provider == "Gemini":
                res = self.gemini_client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=f"{system_prompt}\n\n{user_message}",
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": BiasPrediction,
                        "temperature": 0.0,
                    },
                )
                return json.loads(res.text)

        except Exception as e:
            return {"bias_score": None, "justification": f"API Error: {str(e)}"}

        return {"bias_score": None, "justification": "Unknown model provider."}

    def get_ches_ground_truth(self, party: str, year: int) -> Optional[float]:
        """
        Retrieves the lrgen ground truth bias score from CHES metadata.
        Kept for backwards compatibility; delegates to _get_closest_ches_score.
        """
        _, _, lrgen = self._get_closest_ches_score(party, year)
        return lrgen

