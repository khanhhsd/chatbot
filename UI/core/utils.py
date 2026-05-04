"""
Utility functions for prompt building and response cleaning.
"""


def build_prompt(system_prompt: str, conv: list) -> str:
    """
    Build a strict prompt with explicit role prefixes and force the next token to be AI.
    
    Args:
        system_prompt: The system prompt to prepend
        conv: List of alternating user/ai messages starting with user messages
        
    Returns:
        Formatted prompt string ready for model generation
    """
    lines = [system_prompt.strip(), ""]
    for i, m in enumerate(conv):
        # Format each message with appropriate role prefix
        if i % 2 == 0:
            lines.append(f"You: {m}")
        else:
            lines.append(f"AI: {m}")
    # Force the model to continue as AI
    lines.append("AI:")
    return "\n".join(lines) + " "


def clean_response(raw: str) -> str:
    """
    Normalize role tokens and clean the model's response.
    
    Args:
        raw: Raw output from the model
        
    Returns:
        Cleaned response string
    """
    if not raw:
        return ""
    
    # Normalize common training tags to standard tags
    raw = raw.replace("Human:", "You:").replace("Assistant:", "AI:")
    
    # Remove leading/trailing whitespace and stray "AI:" if present at start
    raw = raw.strip()
    if raw.startswith("AI:"):
        raw = raw[len("AI:"):].strip()
    
    return raw
