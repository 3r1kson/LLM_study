from services.chat_service import ChatService

class MockLLM:
    def generate_text(self, prompt: str) -> str:
        return "Mock response to: " + prompt

def test_valid_prompt():
    service = ChatService(MockLLM())
    response = service.handle_chat("Qual é o seu nome?")
    assert response.startswith("Mock")

def test_invalid_prompt():
    service = ChatService(MockLLM())
    try:
        service.handle_chat("Inválido")
    except ValueError as e:
        assert str(e) == "Prompt precisa ser pergunta ou instrução"
    else:
        assert False, "Expected ValueError to be raised"