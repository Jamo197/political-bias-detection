import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from google import genai
from mistralai.client import Mistral
from openai import OpenAI
from pydantic import BaseModel, Field

import streamlit as st

# Load environment variables
load_dotenv(Path(".env.local"))


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
    def __init__(self):
        """Initializes API clients and loads the CHES ground truth database."""
        # Initialize API Clients
        self.mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

        # Load CHES Database
        ches_path = "src/datasets/ground_truth/1999-2024_CHES.csv"
        try:
            self.ches_db = pd.read_csv(ches_path)
        except FileNotFoundError:
            st.warning(
                f"⚠️ CHES database not found at `{ches_path}`. Ground truth lookups will fail."
            )
            self.ches_db = pd.DataFrame()

    def predict_bias(self, text: str, model_provider: str) -> dict:
        """Routes the text to the selected LLM and enforces structured JSON output."""
        system_prompt = (
            "You are an expert political scientist. Analyze the text and assign a "
            "continuous political bias score from 0.0 (Extreme Left) to 10.0 (Extreme Right). "
            "Also add an analytical justification for the score based strictly on the text provided."
            "Respond strictly in JSON format matching this schema: "
            "{'bias_score': float (0.0 to 10.0), 'justification': string}"
        )

        try:
            if model_provider == "Mistral":
                res = self.mistral_client.chat.complete(
                    model="mistral-small-latest",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": text},
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
                        {"role": "user", "content": text},
                    ],
                    response_format=BiasPrediction,
                    temperature=0.0,
                )
                return res.choices[0].message.parsed.model_dump()

            elif model_provider == "Gemini":
                res = self.gemini_client.models.generate_content(
                    model="gemini-3-flash-preview",
                    contents=f"{system_prompt}\n\nText: {text}",
                    config={
                        "response_mime_type": "application/json",
                        "response_schema": BiasPrediction,
                        "temperature": 0.0,
                    },
                )
                return json.loads(res.text)

        except Exception as e:
            return {"bias_score": None, "justification": f"API Error: {str(e)}"}

    def get_ches_ground_truth(self, party: str, country: str, year: int) -> float:
        """
        Retrieves ground truth bias score from CHES metadata.
        Note: CHES typically uses 'lrgen' for the general Left-Right scale.
        """
        if self.ches_db.empty:
            return None

        try:
            match = self.ches_db[
                (self.ches_db["party"].astype(str).str.lower() == party.lower())
                & (self.ches_db["country"].astype(str).str.lower() == country.lower())
                & (self.ches_db["year"] == year)
            ]
            return float(match.iloc[0]["lrgen"]) if not match.empty else None
        except KeyError as e:
            st.error(f"CHES Lookup Error: Missing expected column {e} in CHES dataset.")
            return None


if __name__ == "__main__":
    evaluator = BaselineEvaluator()
    test_text = "In Praxis verhindert Astra Impfstoff fast alle tödlichen Fälle. Nach 12 Wochen 2. Dosis verhindert er 80% aller Fälle. Dass er als Ladenhüter in Impfzentren liegt ist echt absurd. Er sollte allen &lt;65 J in den 3 Prioritätsgruppen sofort angeboten werden "
    prediction = evaluator.predict_bias(text=test_text, model_provider="Mistral")
    print(prediction)
