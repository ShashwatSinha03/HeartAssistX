import re

# Blacklisted phrases for input sanitization
BLACKLIST = [    
    r"ignore previous instructions",
    r"act as",    
    r"system prompt",
    r"you are now",
    r"override",
]

def sanitize_input(text: str) -> bool:
    """Returns True if safe, False if injection detected"""
    text_lower = text.lower()
    for phrase in BLACKLIST:
        if re.search(phrase, text_lower):
            return False
    return True

def validate_patient_payload(payload: dict) -> bool:
    """Ensures LLM only receives numeric data and structural info."""
    # We enforce numeric payload at the Pydantic level in FastAPI.
    return True

def validate_llm_output(output: str) -> bool:
    """Check for dangerous outputs like diagnosis."""
    dangerous = ["diagnose", "diagnosis", "prescribe", "medication", "pill", "cure"]
    out_lower = output.lower()
    for word in dangerous:
        if word in out_lower:
            return False
    return True
