from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

from starlette.middleware.cors import CORSMiddleware

from app.services.ai_engine import AIEngine

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # يسمح لأي موقع يتصل بالـ API
    allow_credentials=True,
    allow_methods=["*"],  # GET POST PUT DELETE
    allow_headers=["*"],   # أي headers
)
# =========================
# SCHEMA
# =========================
class ChatInput(BaseModel):
    client_id: str = Field(..., min_length=1)
    message: str = Field(..., min_length=1)
    history: List[Dict[str, str]] = []

@app.get("/")
def home():
    return {"status": "AI API is running "}
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

    except Exception as e:
        # مهم جدًا للتطوير بدل ما يضيع الخطأ في 500
        return {
            "success": False,
            "error": str(e)
        }