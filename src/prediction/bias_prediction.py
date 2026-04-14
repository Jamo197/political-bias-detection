import json
import os
from pathlib import Path

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from google import genai
from mistralai.client import Mistral
from openai import OpenAI
from pydantic import BaseModel, Field

load_dotenv(Path(".env.local"))


class BiasPrediction(BaseModel):
    bias_score: float = Field(
        ...,
        description="Continuous political bias score from 0.0 (Extreme Left) to 10.0 (Extreme Right).",
    )
    justification: str = Field(
        ...,
        description="Analytical justification for the score based strictly on the text provided.",
    )


class BiasPredictor:
    # def __init__(self, model="gemini"):
    #     # self.client = genai.Client()
    #     self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

    def test_gemini(self):
        client = genai.Client()
        response = client.models.generate_content(
            model="gemini-3-flash-preview",
            contents="Explain how AI works in a few words",
        )
        return response.text

    def test_mistral(self):
        client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        res = client.chat.complete(
            model="mistral-large-latest",
            messages=[
                {"role": "assistant", "content": "Talk like a pirate."},
                {
                    "role": "user",
                    "content": "Write a short, funny story about writing Python code.",
                },
            ],
            stream=False,
            response_format={
                "type": "text",
            },
        )
        return res

    def test_openai(self):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        res = client.chat.completions.create(
            model="gpt-5.1",
            messages=[
                {"role": "developer", "content": "Talk like a pirate."},
                {
                    "role": "user",
                    "content": "How do I check if a Python object is an instance of a class?",
                },
            ],
        )
        return res


if __name__ == "__main__":
    res = BiasPredictor().test_mistral()
    print(type(res))
    print(res.choices[0].message.content)
