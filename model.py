import json
import os
from typing import Dict, List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GenerationConfig,
)

from schemas import LLMResponse

# ==========================
# Config modèle
# ==========================

MODEL_NAME = os.path.expanduser("~/llm-models/Mistral-7B-Instruct-v0.3")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE = "cuda"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# ==========================
# Chargement tokenizer
# ==========================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    use_fast=True,
)

# ==========================
# Chargement modèle 4-bit
# ==========================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

model.eval()

# ==========================
# Config génération
# ==========================

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
    Appelle Mistral 7B Instruct 4-bit
    """

    prompt = build_prompt(messages)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            generation_config=gen_config,
        )

    text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    return try_parse_json(text)


# ==========================
# Prompt builder
# ==========================


def build_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Format Instruct Mistral
    """

    prompt = "<s>"
    for msg in messages:
        if msg["role"] == "system":
            prompt += f"[INST] {msg['content']} [/INST]\n"
        elif msg["role"] == "user":
            prompt += f"[INST] {msg['content']} [/INST]\n"
        elif msg["role"] == "assistant":
            prompt += f"{msg['content']}\n"

    return prompt


# ==========================
# JSON parsing / fallback
# ==========================


def try_parse_json(text: str) -> LLMResponse:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        data = json.loads(text[start:end])

        return LLMResponse(
            reply=data.get("reply", "…"),
            intent=data.get("intent", "unknown"),
            extracted_data=data.get("extracted_data", {}),
            confidence=float(data.get("confidence", 0.5)),
        )

    except Exception:
        return LLMResponse(
            reply=text.strip(),
            intent="unknown",
            extracted_data={},
            confidence=0.5,
        )
