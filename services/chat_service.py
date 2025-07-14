from core.validator import validate_prompt
from llm.llama_service import LLMEngine

class ChatService:
    def __init__(self, llm: LLMEngine):
        self.llm = llm

    def handle_chat(self, prompt: str, system: str) -> str:
        if not validate_prompt(prompt):
            raise ValueError("Prompt precisa ser pergunta ou instrução")
        return self.llm.generate_text(system, prompt)