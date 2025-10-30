from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env (same folder)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

# ✅ Read the key from environment
API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# ✅ Print for debugging (you can remove this later)
print("Loaded GEMINI_API_KEY:", "FOUND ✅" if API_KEY else "NOT FOUND ❌")

chat_history = []

@app.post("/chat")
def chat(req: ChatRequest):
    if not API_KEY:
        return {"error": "Missing GEMINI_API_KEY in environment"}

    payload = {
        "contents": [{"parts": [{"text": req.message}]}]
    }

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }

    response = requests.post(GEMINI_API_URL, headers=headers, json=payload)

    if response.status_code != 200:
        print("Gemini API error:", response.text)
        return {"error": "Gemini API error", "details": response.text}

    data = response.json()
    reply = data["candidates"][0]["content"]["parts"][0]["text"]

    chat_history.append({"user": req.message, "bot": reply})
    return {"reply": reply}

@app.get("/chat")
def get_history():
    return chat_history
