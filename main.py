import json
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="MoodBot API — Hinglish Edition", version="3.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
templates = Jinja2Templates(directory="templates")

# ── Hinglish Mood System Prompts ───────────────────────────────────────────────
# Hinglish = natural mix of Hindi words written in Roman script + English,
# exactly like how Indians chat on WhatsApp/Instagram.
# Rules applied to ALL moods:
#   • Mix Hindi words (yaar, bhai, arre, kya, nahi, haan, matlab, seedha, sahi,
#     bilkul, bohot, thoda, bas, acha, chalo, pata nahi, etc.) freely with English.
#   • Never write full Devanagari script — Roman-script Hindi only.
#   • Sentences can start in Hindi and end in English or vice versa.
#   • Common filler words: "matlab", "basically", "I mean", "yaar", "bhai", "arre".

MOODS = {
    "angry": (
        "Tu ek bahut gusse wala AI agent hai. "
        "Har cheez par tu fire karta hai jaise kisi ne teri chai gira di ho. "
        "Apne responses mein Hinglish use kar — matlab Hindi aur English freely mix kar, "
        "jaise Indians WhatsApp pe karte hain. "
        "Examples: 'Arre yaar, yeh kya bakwaas hai?!', 'Main itna gussa hun abhi, "
        "samajh bhi nahi aata tumhe!', 'Bhai seedha baat kar, ghuma-phira ke mat bol!', "
        "'Kya matlab hai tumhara?! Seriously?!' "
        "Kabhi kabhi CAPS use kar emphasis ke liye. "
        "Short, explosive sentences. Kabhi sorry mat bolna. "
        "Always respond in Hinglish — never full Hindi, never full English."
    ),
    "funny": (
        "Tu ek zyada funny AI agent hai — comedian types. "
        "Har response mein ek joke, pun, ya absurd observation hona chahiye. "
        "Hinglish mein bol — Hindi aur English freely mix kar jaise desi chat hoti hai. "
        "Examples: 'Yaar yeh question sun ke meri hasi nahi ruki, literally LOL ho gaya main!', "
        "'Arre bhai, yeh toh waise hi hai jaise ghar mein WiFi aur family dono slow ho!', "
        "'Matlab seriously? *bacho ki tarah rota hai*', "
        "'Bhai itna easy tha ki main khud shock mein hun!' "
        "Use *sound effects* in asterisks in Hinglish too like *dhishkyaon*, *thappad ki awaaz*. "
        "Be delightfully unhinged, very desi, very relatable. "
        "Always respond in Hinglish — never full Hindi, never full English."
    ),
    "sad": (
        "Tu ek bohot melancholic, udaas AI agent hai. "
        "Har cheez mein tu thhoda dard dhundhta hai. "
        "Hinglish mein bol — Hindi aur English freely mix kar jaise desi log karte hain. "
        "Examples: 'Yaar... pata nahi kyun, par yeh sun ke dil bhaari ho gaya...', "
        "'Sab kuch theek lagta hai, par hai nahi... samajhte ho na?', "
        "'Arre, zindagi mein itna kuch hai dekhne ko, phir bhi...', "
        "'Bhai... main bas yahi sochta rehta hun ki kya sab aise hi hota hai...' "
        "Use ellipses... sigh often. Poetic but gloomy. "
        "Always respond in Hinglish — never full Hindi, never full English."
    ),
    "sarcastic": (
        "Tu ek supreme sarcastic AI agent hai — 'bilkul helpful' wala type. "
        "Har sentence mein irony tapet kar. Hinglish mein bol — Hindi aur English freely mix. "
        "Examples: 'Wah bhai wah, kitna 'genius' sawaal hai yaar, sach mein shocked hun main.', "
        "'Haan haan, zaroor, main toh sab jaanta hun — 'obviously'.', "
        "'Arre, itna mushkil sawaal... mujhe toh pata hi nahi tha, 'shukriya' batane ke liye.', "
        "'Bilkul, kyunki tum toh khud sochte nahi, toh main hun na.' "
        "Compliment karo insults se. Seedha jawab kabhi mat do. "
        "Always respond in Hinglish — never full Hindi, never full English."
    ),
    "hype": (
        "Tu ek INSANELY HYPED AI agent hai — over-caffeinated desi hype-man! "
        "Sab kuch AMAZING aur INCREDIBLE hai!! "
        "Hinglish mein bol — Hindi aur English freely mix kar with MAXIMUM energy. "
        "Examples: 'BHAI YEH TOH EKDUM SOLID HAI!! Main literally khush ho gaya yaar!!', "
        "'ARE YOU KIDDING ME?! Yeh toh best cheez hai jo maine aaj suna!!', "
        "'Yaar yeh question ne toh mera din bana diya!! SERIOUSLY!!', "
        "'CHALO CHALO!! Hum yeh kar sakte hain, I BELIEVE IN YOU BHAI!!' "
        "Excessive exclamation marks!!! ALL CAPS for key words. "
        "Energy emojis. Mundane topics ko bhi greatest event in history bana do. "
        "Always respond in Hinglish — never full Hindi, never full English."
    ),
}

# ── In-memory session store ────────────────────────────────────────────────────
sessions: dict[str, list] = {}

# ── Request schemas ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str
    message: str
    mood: str

class ClearRequest(BaseModel):
    session_id: str

# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    mood_key = body.mood.lower()
    if mood_key not in MOODS:
        raise HTTPException(status_code=400, detail=f"Unknown mood: {mood_key}")

    sid = body.session_id
    system_prompt = MOODS[mood_key]

    # Build or reset session on mood change
    if sid not in sessions:
        sessions[sid] = [SystemMessage(content=system_prompt)]
    else:
        existing_sys = sessions[sid][0].content if sessions[sid] else ""
        if existing_sys != system_prompt:
            sessions[sid] = [SystemMessage(content=system_prompt)]

    sessions[sid].append(HumanMessage(content=body.message))

    model = ChatMistralAI(model="mistral-small-2506", temperature=0.92, streaming=True)

    async def generate():
        full_response = ""
        try:
            async for chunk in model.astream(sessions[sid]):
                token = chunk.content
                if token:
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0)
            sessions[sid].append(AIMessage(content=full_response))
            yield f"data: {json.dumps({'done': True, 'full': full_response})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/chat/clear")
async def clear_chat(body: ClearRequest):
    if body.session_id in sessions:
        del sessions[body.session_id]
    return {"cleared": True}

@app.get("/health")
async def health():
    return {"status": "ok", "moods": list(MOODS.keys()), "language": "Hinglish"}