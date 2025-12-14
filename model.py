import json
import os
from typing import Dict, List

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from schemas import LLMResponse

# ==========================
# Config modèle ONNX
# ==========================

MODEL_PATH = os.path.expanduser("~/llm-models/mistral‑onnx‑int4/model.onnx")
TOKENIZER_PATH = os.path.expanduser("~/llm-models/mistral‑onnx‑int4")
tokenizer = AutoTokenizer.from_pretrained(
    TOKENIZER_PATH,
    use_fast=True,
    trust_remote_code=True,  # permet d'utiliser le tokenizer custom NVIDIA si nécessaire
)

# Session ONNX
session = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider"])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# ==========================
# Fonction principale
# ==========================


def generate_response(messages: List[Dict[str, str]]) -> LLMResponse:
    """
    Appelle le modèle ONNX INT4 et retourne un LLMResponse valide.
    """

    # Concatène les messages dans un format type "chat"
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

    # Tokenization
    inputs = tokenizer(prompt_text, return_tensors="np")
    input_ids = inputs["input_ids"].astype(np.int64)

    # Génération avec ONNX
    # ⚠️ Ici on fait une génération simple, pas de sampling ni top_p
    outputs = session.run([output_name], {input_name: input_ids})
    generated_ids = outputs[0][0]

    # Décodage
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # ==========================
    # Extraire JSON depuis le texte généré
    # ==========================
    response_json = try_parse_json(output_text)
    return response_json


# ==========================
# JSON parsing / fallback
# ==========================


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
            reply=text.strip(), intent="unknown", extracted_data={}, confidence=0.5
        )
