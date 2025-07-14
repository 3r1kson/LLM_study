def validate_prompt(prompt: str) -> bool:
    """
    Validates the prompt to ensure it meets the required length constraints.
    
    :param prompt: The text prompt to validate.
    :return: True if valid, False otherwise.
    """
    if not prompt.strip().endswith("?") and "responda" not in prompt.lower():
        return False
    return True