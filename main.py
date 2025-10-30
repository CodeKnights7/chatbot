from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os
import json
from dotenv import load_dotenv

# ✅ Load .env file (works locally & on Render)
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

app = FastAPI()

class ChatRequest(BaseModel):
    message: str

# ✅ Read Gemini API info
API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

print("Loaded GEMINI_API_KEY:", "FOUND ✅" if API_KEY else "NOT FOUND ❌")

# ✅ Chat history storage file
CHAT_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

# ✅ Helper: Load existing chat history
def load_history():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# ✅ Helper: Save chat history to file
def save_history(history):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

# ✅ Load chat history on startup
chat_history = load_history()

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

    # ✅ Update chat history
    chat_history.append({"user": req.message, "bot": reply})
    save_history(chat_history)  # persist to file

    return {"reply": reply}


@app.get("/chat")
def get_history():
    """✅ Fetch full chat history."""
    return chat_history
