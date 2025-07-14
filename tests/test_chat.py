from services.chat_service import ChatService

class MockLLM:
    def generate_text(self, prompt: str, system: str) -> str:
        return f"Mock response to system: {system} and prompt: {prompt}"
def test_valid_prompt():
    service = ChatService(MockLLM())
    response = service.handle_chat(prompt="Qual é o seu nome?", system="Você é um assistente útil.")
    assert response.startswith("Mock")

def test_invalid_prompt():
    service = ChatService(MockLLM())
    try:
        service.handle_chat(prompt="Inválido", system="Você é um assistente útil.")
    except ValueError as e:
        assert str(e) == "Prompt precisa ser pergunta ou instrução"
    else:
        assert False, "Expected ValueError to be raised"