import os

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from starlette.middleware.cors import CORSMiddleware

from app.services.ai_engine import AIEngine

app = FastAPI(
    title="ChatBotAi",
    description="RAG-powered chatbot API using LangChain, ChromaDB, and Groq/OpenAI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يسمح لأي موقع يتصل بالـ API
    allow_credentials=True,
    allow_methods=["*"],  # GET POST PUT DELETE
    allow_headers=["*"],   # أي headers
)

@app.on_event("startup")
async def startup_event():
    """Validate required environment variables on startup"""
    required_vars = ["GROQ_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}"
        )

# =========================
# SCHEMA
# =========================
class ChatInput(BaseModel):
    client_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    history: List[Dict[str, str]] = []

# =========================
# HEALTH CHECK
# =========================
@app.get("/health")
def health_check():
    return {"status": "ok", "service": "ChatBotAi"}

# =========================
# TRAIN ENDPOINT
# =========================
@app.post("/train")
async def train(client_id: str = Form(...), file: UploadFile = File(...)):

    try:
        content = await file.read()

        if not content:
            raise HTTPException(status_code=400, detail="Empty file")

        text = content.decode("utf-8")

        engine = AIEngine(client_id)
        msg = engine.train_on_data(text)

        return {
            "success": True,
            "message": msg
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
async def chat(data: ChatInput):
    try:
        engine = AIEngine(data.client_id)
        response = engine.get_chat_response(
            question=data.message,
            history=data.history or []
        )
        return {
            "success": True,
            "bot_response": response
        }
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)