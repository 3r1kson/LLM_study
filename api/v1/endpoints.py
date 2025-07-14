from fastapi import APIRouter, HTTPException
from schemas.chat import ChatRequest, ChatResponse
from services.chat_service import ChatService
from llm.llama_service import LLMEngine

router = APIRouter()
chat_service = ChatService(LLMEngine(model_path="./models/tigerbot-13b-chat-v5.Q2_K.gguf"))

@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    try:
        print(f"Received chat request: {req}")
        response_text = chat_service.handle_chat(req.prompt, req.system)
        return ChatResponse(response=response_text)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")