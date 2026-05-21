import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from mistralai.client import Mistral
from openai import OpenAI
from pydantic import BaseModel, Field

import streamlit as st

# ---------------------------------------------------------------------------
# Path anchoring — src/prediction/bias_prediction.py → project root is 2 levels up
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from project root
load_dotenv(_PROJECT_ROOT / ".env.local")


# --- 1. Schema Definition ---
class BiasPrediction(BaseModel):
    bias_score: float = Field(
        ...,
        description="Continuous political bias score from 0.0 (Extreme Left) to 10.0 (Extreme Right).",
    )
    justification: str = Field(
        ...,
        description="Analytical justification for the score based strictly on the text provided.",
    )


# --- 2. Backend Evaluator Class ---
class BaselineEvaluator:
    # CHES party_id -> standardized display name mapping
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
        """Initializes API clients and loads the CHES ground truth database."""
        # Initialize API Clients
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Load CHES Database
        ches_path = str(_PROJECT_ROOT / "src/datasets/ground_truth/1999-2024_CHES.csv")
        try:
            ches_raw = pd.read_csv(ches_path)
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
            st.warning(
                f"CHES database not found at `{ches_path}`. Ground truth lookups will fail."
            )
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

        # CDU and CSU are merged in CHES under CDU/CSU
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
        """
        Routes the text to the selected LLM and enforces structured JSON output.

        If ``context_chunks`` is provided the chunks are injected into the prompt
        so the model can use retrieved parliamentary speech excerpts as grounding.
        """
        context_block = ""
        if context_chunks:
            excerpts = []
            for i, chunk in enumerate(context_chunks, 1):
                party = chunk.get("party", "Unknown")
                speaker = chunk.get("speaker", "Unknown")
                text_snippet = chunk.get("text", "")[:500]
                excerpts.append(
                    f"[{i}] Party: {party} | Speaker: {speaker}\n{text_snippet}"
                )
            context_block = (
                "\n\nRelevant parliamentary speech excerpts for context:\n"
                + "\n\n".join(excerpts)
                + "\n\n"
            )

        system_prompt = (
            "You are an expert political scientist specializing in German parliamentary politics. "
            "Analyze the provided text and assign a continuous political bias score "
            "from 0.0 (Extreme Left) to 10.0 (Extreme Right) on the standard left-right scale "
            "used by the Chapel Hill Expert Survey (CHES). "
            "Also provide a concise analytical justification for the score based strictly on "
            "the text (and any provided context). "
            "Respond strictly in JSON format matching this schema: "
            "{'bias_score': float (0.0 to 10.0), 'justification': string}"
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

    def get_ches_ground_truth(self, party: str, country: str, year: int) -> Optional[float]:
        """
        Retrieves the lrgen ground truth bias score from CHES metadata.
        Kept for backwards compatibility; delegates to _get_closest_ches_score.
        """
        _, _, lrgen = self._get_closest_ches_score(party, year)
        return lrgen

