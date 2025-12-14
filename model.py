import json
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from schemas import LLMResponse

# ==========================
# Config du modèle
# ==========================

MODEL_NAME = "mistral-0.7b-instruct"  # changer selon ton modèle
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# Chargement du modèle
# ==========================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",  # répartit sur GPU automatiquement
    dtype=torch.float16,  # économie mémoire
).to(DEVICE)

# Config de génération par défaut
gen_config = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id,
)

# ==========================
# Fonction principale
# ==========================


def generate_response(messages: List[Dict[str, str]]) -> LLMResponse:
    """
    Appelle le LLM avec une liste de messages construits par builder.py
    et retourne un LLMResponse valide.
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
    inputs = tokenizer(prompt_text, return_tensors="pt").to(DEVICE)

    # Génération
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_config.__dict__)

    # Décodage
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ==========================
    # Extraire JSON depuis le texte généré
    # ==========================

    response_json = try_parse_json(output_text)

    return response_json


# ==========================
# JSON parsing / fallback
# ==========================


def try_parse_json(text: str) -> LLMResponse:
    """
    Essaie d'extraire un JSON strict depuis le texte généré.
    Fallback simple si parse échoue.
    """
    try:
        data = json.loads(text)
        return LLMResponse(
            reply=data.get("reply", "…"),
            intent=data.get("intent", "unknown"),
            extracted_data=data.get("extracted_data", {}),
            confidence=float(data.get("confidence", 0.5)),
        )
    except json.JSONDecodeError:
        # fallback : retourne le texte brut dans reply
        return LLMResponse(
            reply=text.strip(), intent="unknown", extracted_data={}, confidence=0.5
        )
