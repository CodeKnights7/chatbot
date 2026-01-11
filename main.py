from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import os
import json
from dotenv import load_dotenv
from rag.vectorstore import search

# --------------------------------------------------
# Load environment variables
# --------------------------------------------------
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI()

# --------------------------------------------------
# CORS
# --------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Models
# --------------------------------------------------
class ChatRequest(BaseModel):
    message: str

# --------------------------------------------------
# Mistral API
# --------------------------------------------------
API_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"

print("Loaded MISTRAL_API_KEY:", "FOUND ✅" if API_KEY else "NOT FOUND ❌")

# --------------------------------------------------
# Chat history
# --------------------------------------------------
CHAT_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")

def load_history():
    if os.path.exists(CHAT_FILE):
        with open(CHAT_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_history(history):
    with open(CHAT_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)

chat_history = load_history()

# --------------------------------------------------
# Routes
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "Server is running 🚀"}

@app.post("/chat")
def chat(req: ChatRequest):
    if not API_KEY:
        return {"reply": "⚠️ Server is missing MISTRAL_API_KEY"}

    # 🔍 Retrieve relevant chunks
    context_chunks = search(req.message, k=4)

    if not context_chunks:
        return {"reply": "I don’t have that information in the provided documents."}

    context = "\n\n".join(
        f"[Source: {c['source']}]\n{c['text']}"
        for c in context_chunks
    )

    system_prompt = f"""
You are a financial assistant.
Answer ONLY using the context below.
If the answer is not in the context, say:
"I don’t have that information in the provided document."

Context:
{context}
"""

    payload = {
        "model": "mistral-small",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.message}
        ],
        "temperature": 0.2
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(
            MISTRAL_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
    except Exception:
        return {"reply": "⚠️ Unable to reach Mistral API"}

    if response.status_code != 200:
        return {"reply": "⚠️ Mistral API failed"}

    reply = response.json()["choices"][0]["message"]["content"]

    chat_history.append({"user": req.message, "bot": reply})
    chat_history[:] = chat_history[-50:]  # cap history
    save_history(chat_history)

    return {"reply": reply}

@app.get("/history")
def get_history():
    return chat_history
