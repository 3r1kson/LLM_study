from fastapi import FastAPI
from api.v1.endpoints import router as chat_router

app = FastAPI()

app.include_router(chat_router, prefix="/api/v1")