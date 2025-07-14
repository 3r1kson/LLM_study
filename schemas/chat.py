from pydantic import BaseModel, constr

class ChatRequest(BaseModel):
    prompt: str
    # system: str = "Você é um assistente útil e direto. Responda sempre em português."
    system: str = "You are a helpful and concise assistant."
class ChatResponse(BaseModel):
    response: str