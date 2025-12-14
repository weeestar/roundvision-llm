from typing import Any, Dict, List


def build_messages(
    system_prompt: str,
    state: str,
    state_objective: str,
    known_data: Dict[str, Any],
    user_message: str,
) -> List[Dict[str, str]]:
    """
    Build the message list sent to the LLM.

    Rules:
    - The system prompt is always first and never changes.
    - State, known data and state objective are injected as system messages.
    - User message is always last.
    """

    messages: List[Dict[str, str]] = []

    # 1. Global system prompt (identity + rules)
    messages.append({"role": "system", "content": system_prompt.strip()})

    # 2. Current onboarding state
    messages.append({"role": "system", "content": f"CURRENT_STATE:\n{state}"})

    # 3. Objective for the current state
    messages.append(
        {"role": "system", "content": f"STATE_OBJECTIVE:\n{state_objective.strip()}"}
    )

    # 4. Known data (Faceit + already collected onboarding info)
    if known_data:
        messages.append(
            {
                "role": "system",
                "content": "KNOWN_DATA:\n" + serialize_known_data(known_data),
            }
        )

    # 5. User message (last, always)
    messages.append(
        {"role": "user", "content": user_message.strip() if user_message else ""}
    )

    return messages


def serialize_known_data(data: Dict[str, Any]) -> str:
    """
    Serialize known data into a readable, deterministic format.
    Avoid raw JSON dumps to keep it LLM-friendly.
    """

    lines = []

    for key, value in data.items():
        if value is None:
            lines.append(f"- {key}: unknown")
        elif isinstance(value, (list, tuple)):
            joined = ", ".join(str(v) for v in value)
            lines.append(f"- {key}: {joined}")
        else:
            lines.append(f"- {key}: {value}")

    return "\n".join(lines)
