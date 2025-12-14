import os
from typing import Dict, List

from onnxruntime.transformers import GenerativeModel

from schemas import LLMResponse

# ==========================
# Config modèle ONNX GenAI
# ==========================
MODEL_PATH = os.path.expanduser("~/llm-models/mistral-onnx-int4/model.onnx")

# Initialisation du modèle avec ONNX Runtime GenAI
gen_model = GenerativeModel(MODEL_PATH)


# ==========================
# Fonction principale
# ==========================
def generate_response(messages: List[Dict[str, str]]) -> LLMResponse:
    """
    Génère une réponse à partir d'une liste de messages de type chat.
    """

    # Concatène les messages dans un format "chat"
    prompt_text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt_text += f"[SYSTEM]: {content}\n"
        elif role == "user":
            prompt_text += f"[USER]: {content}\n"
        elif role == "assistant":
            prompt_text += f"[ASSISTANT]: {content}\n"

    try:
        # Génération de texte avec le modèle GenAI
        output = gen_model.generate(
            prompt_text,
            max_length=512,  # tu peux ajuster selon tes besoins
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

        # Récupère le texte généré
        output_text = output[0]  # output est une liste

        # Essaie d'extraire un JSON depuis le texte
        response_json = try_parse_json(output_text)
        return response_json

    except Exception as e:
        # fallback en cas d'erreur
        return LLMResponse(
            reply="LLM processing error",
            intent="unknown",
            extracted_data={},
            confidence=0.0,
        )


# ==========================
# JSON parsing / fallback
# ==========================
import json


def try_parse_json(text: str) -> LLMResponse:
    try:
        data = json.loads(text)
        return LLMResponse(
            reply=data.get("reply", "…"),
            intent=data.get("intent", "unknown"),
            extracted_data=data.get("extracted_data", {}),
            confidence=float(data.get("confidence", 0.5)),
        )
    except json.JSONDecodeError:
        return LLMResponse(
            reply=text.strip(),
            intent="unknown",
            extracted_data={},
            confidence=0.5,
        )
