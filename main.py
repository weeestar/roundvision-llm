from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from builder import build_messages
from model import generate_response
from schemas import LLMError, LLMRequest, LLMResponse

# ==========================
# App FastAPI
# ==========================

app = FastAPI(
    title="Neo Onboarding LLM Service",
    description="Service LLM pour l'onboarding des joueurs RoundVision",
    version="1.0",
)

# ==========================
# CORS (si nécessaire)
# ==========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adapter selon ton frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# System prompt global
# ==========================
with open("prompt/system.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()


# ==========================
# Endpoint principal
# ==========================


@app.post("/chat", response_model=LLMResponse, responses={400: {"model": LLMError}})
async def chat(request: LLMRequest):
    """
    Reçoit le message du joueur + contexte d'état,
    renvoie la réponse générée par le LLM en JSON strict.
    """

    try:
        # Construire les messages pour le LLM
        messages = build_messages(
            system_prompt=SYSTEM_PROMPT,
            state=request.state,
            state_objective=request.state_objective,
            known_data=request.known_data,
            user_message=request.user_message,
        )

        # Générer la réponse
        llm_response: LLMResponse = generate_response(messages)

        # Validation Pydantic automatique
        return llm_response

    except ValidationError as e:
        # JSON invalide ou problème de schema
        raise HTTPException(status_code=400, detail={"error": str(e)})

    except Exception as e:
        # Fallback général
        return LLMError(error="LLM processing error", raw_output=str(e))
