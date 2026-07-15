import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv = None
try:
    from dotenv import load_dotenv

    load_dotenv(_PROJECT_ROOT / ".env.local")
except ImportError:
    pass

logger = logging.getLogger(__name__)


class BiasPrediction(BaseModel):
    bias_score: float = Field(
        ...,
        description="Continuous political bias score from 1.0 (Extreme Left) to 7.0 (Extreme Right).",
    )
    justification: str = Field(
        ...,
        description="Analytical justification for the score based strictly on the text provided.",
    )


class BiasPredictor:
    """Unified political bias evaluator with configurable LLM backend.

    By default uses OpenRouter (https://openrouter.ai/api/v1) which routes to
    hundreds of models via a single API key. For HPC clusters without internet,
    set ``base_url`` to a local vLLM server and ``api_key`` to "EMPTY".
    """

    def __init__(
        self,
        base_url: str = "https://openrouter.ai/api/v1",
        api_key: Optional[str] = None,
    ):
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY", "EMPTY")
        if not resolved_key or resolved_key == "EMPTY" and "openrouter.ai" in base_url:
            logger.error(
                "Missing OPENROUTER_API_KEY in environment. "
                "Set it in .env.local or pass api_key explicitly."
            )

        self.client = OpenAI(base_url=base_url, api_key=resolved_key)
        self.base_url = base_url

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
                "If the anchors are topically relevant but ideologically unhelpful, state that and rely on the scale standards."
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
                "using standard left-right political spectrum criteria."
            )
            context_block = "\n\n"

        system_prompt = f"""
            You are an expert political scientist specializing in comparative European parliamentary politics.
            Analyze the target text and return a continuous political bias score on the standard left-right scale, 
            where 1.0 represents Extreme Left and 7.0 represents Extreme Right.

            {rag_instructions}

            CRITICAL FORMATTING RULE: You must respond STRICTLY with a valid JSON object matching this structural schema:
            {{
            "bias_score": <float between 1.0 and 7.0>,
            "justification": "<Concise analytical explanation for the score based strictly on the information provided tracking thematic alignment against context anchors.>"
            }}
            Do NOT use double quotes (\") anywhere inside your justification text. 
            If you need to quote a word or phrase, you MUST use single quotes (') instead.
        """

        user_message = (
            f'{context_block}Target Text to Analyze:\n"""\n{text.strip()}\n"""'
        )

        try:
            response = self.client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=800, 
            temperature=0.0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "bias_prediction",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "bias_score": {
                                "type": "number", 
                                "description": "The calculated bias score"
                            },
                            "justification": {
                                "type": "string", 
                                "description": "Explanation for the score"
                            }
                        },
                        "required": ["bias_score", "justification"],
                        "additionalProperties": False
                    }
                }
            },
            extra_headers=(
                {
                    "HTTP-Referer": "https://github.com/Jamo197/political-bias-detection",
                    "X-Title": "Political RAG Pipeline Engine",
                }
                if "openrouter.ai" in self.base_url
                else None
            ),
        )

            raw_content = response.choices[0].message.content.strip()
            if raw_content.startswith("```"):
                raw_content = raw_content.split("\n", 1)[-1]
                if raw_content.endswith("```"):
                    raw_content = raw_content.rsplit("\n", 1)[0]
            
            start_idx = raw_content.find('{')
            end_idx = raw_content.rfind('}')
            
            if start_idx != -1 and end_idx != -1:
                raw_content = raw_content[start_idx:end_idx + 1]
                
            try:
                return json.loads(raw_content)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON. Raw LLM output was:\n{raw_content}")
                raise e
        except Exception as e:
            logger.error(f"LLM prediction failure on model {model_id}: {e}")
            return {
                "bias_score": None,
                "justification": f"API Evaluation Pipeline Error: {str(e)}",
            }
