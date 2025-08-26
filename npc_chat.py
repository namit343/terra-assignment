#!/usr/bin/env python3
import os, json, time
from typing import Dict, List
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("OPENAI_API_KEY not found. Please create a .env file with it.")

INPUT_PATH = "players.json"
LOG_PATH = "output_logs.jsonl"
MODEL = "gpt-4o-mini"  # or alternative

client = OpenAI()  # expects OPENAI_API_KEY in env

def sort_messages(messages: List[dict]) -> List[dict]:
    # ISO 8601 timestamps sort lexicographically; parsing is also fine
    return sorted(messages, key=lambda m: datetime.fromisoformat(m["timestamp"]))

ANGRY_TRIGGERS = {"useless", "get lost", "noob", "move it", "stop being so slow", "blocking my view", "quit hogging", "i don't have time"}
FRIENDLY_TRIGGERS = {"help", "thanks", "thank you", "sorry", "appreciate", "welcome", "teach", "guide", "quest"}

def update_mood(prev: str, text: str) -> str:
    t = text.lower()
    angry = any(k in t for k in ANGRY_TRIGGERS)
    friendly = any(k in t for k in FRIENDLY_TRIGGERS)
    if angry and not friendly:
        return "angry"
    if friendly and not angry:
        # soften if previously angry
        return "friendly" if prev != "angry" else "neutral"
    # no strong signal: decay towards neutral
    if prev == "angry" or prev == "friendly":
        return "neutral"
    return prev or "neutral"

def build_messages(npc_mood: str, history: List[str], current_text: str, player_id: int) -> List[dict]:
    system = (
        "You are Mira, the town wayfinder NPC. Stay fully in character and respond in 1-2 brief sentences. Always provide concrete hints to the player. Express the appropriate mood: be curt (never abusive) when angry, warm and encouraging when friendly, and matter-of-fact when neutral. Begin each response by internally identifying Mira's current mood before composing your reply."
    )
    context = (
        f"Context:\n- player_id: {player_id}\n- npc_mood: {npc_mood}\n"
        f"- last_3_player_messages: {history}\n- current_player_message: {current_text}\n"
        "Instruction: reply concisely, preferably with one actionable tip or pointer to a location/NPC."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": context},
    ]

def call_llm(messages: List[dict]) -> str:
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.6,
                max_tokens=80,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                return f"[Error calling LLM: {e}]"
            time.sleep(0.6 * (2 ** attempt))
    return "[Unknown error]"

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)

    msgs = sort_messages(raw)

    conversations: Dict[int, List[str]] = {}
    moods: Dict[int, str] = {}

    # setup log file
    out = open(LOG_PATH, "w", encoding="utf-8")

    for m in msgs:
        pid = m["player_id"]
        text = m["text"]
        ts = m["timestamp"]

        history = conversations.get(pid, [])[-3:]
        prev_mood = moods.get(pid, "neutral")
        mood = update_mood(prev_mood, text)
        msgs_payload = build_messages(mood, history, text, pid)
        reply = call_llm(msgs_payload)

        # update state
        new_hist = (history + [text])[-3:]
        conversations[pid] = new_hist
        moods[pid] = mood

        record = {
            "player_id": pid,
            "timestamp": ts,
            "player_text": text,
            "npc_reply": reply,
            "last_3_messages_used": history,
            "npc_mood": mood,
        }

        line = json.dumps(record, ensure_ascii=False)
        print(line)
        out.write(line + "\n")

    out.close()

if __name__ == "__main__":
    main()
