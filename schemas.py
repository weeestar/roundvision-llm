from typing import Any, Dict, Optional

from pydantic import BaseModel, Field, validator

# =========================
# Request schema (Go -> LLM)
# =========================


class LLMRequest(BaseModel):
    """
    Payload sent by the backend Go to the LLM service.
    """

    state: str = Field(..., description="Current onboarding state")
    state_objective: str = Field(..., description="Objective for the current state")
    known_data: Dict[str, Any] = Field(
        default_factory=dict, description="Already known data (Faceit + onboarding)"
    )
    user_message: str = Field(default="", description="Last message sent by the user")

    @validator("state", "state_objective")
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()


# =========================
# LLM response schema
# =========================


class LLMResponse(BaseModel):
    """
    Strict response expected from the LLM.
    """

    reply: str = Field(
        ..., description="Human-readable reply to display in the chat UI"
    )

    intent: str = Field(
        ..., description="High-level intent of the message (e.g. collect_goals)"
    )

    extracted_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured data extracted from the user message",
    )

    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for the extracted data"
    )

    # ---------- Validators ----------

    @validator("reply")
    def reply_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Reply cannot be empty")
        return v.strip()

    @validator("intent")
    def intent_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Intent cannot be empty")
        return v.strip()

    @validator("extracted_data")
    def extracted_data_is_dict(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("extracted_data must be a dictionary")
        return v


# =========================
# Error schema (optional)
# =========================


class LLMError(BaseModel):
    """
    Returned when the LLM output is invalid or cannot be parsed.
    """

    error: str
    raw_output: Optional[str] = None
