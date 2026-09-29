"""
ui/api.py — Backend API Client for the Streamlit UI

Functions:
  - upload_pdf(file)                     → POST /upload
  - ask_question(...)                    → POST /chat
  - get_suggestions(filename)            → POST /suggestions
"""

import os
import time
import json
import requests
import streamlit as st

try:
    BASE_URL = st.secrets.get("BACKEND_URL", "http://127.0.0.1:8000")
except Exception:
    BASE_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")


# ── Upload ─────────────────────────────────────────────────────────────────────

def upload_pdf(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/octet-stream",
        )
    }
    response = requests.post(f"{BASE_URL}/upload", files=files, timeout=120)
    return response


# ── Chat ───────────────────────────────────────────────────────────────────────

def ask_question(
    question: str,
    filename: str,
    history: list,
    session_id: str = "",
) -> dict:
    """Calls the backend /chat endpoint and returns the parsed JSON dict."""
    payload = {
        "question": question,
        "filename": filename,
        "history": history,
        "session_id": session_id,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/chat",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except requests.Timeout:
        return {"answer": "⚠️ The request timed out. Please try again."}
    except requests.ConnectionError:
        return {"answer": "⚠️ Could not connect to the backend. Make sure `uvicorn main:app` is running."}
    except Exception as e:
        return {"answer": f"⚠️ Error: {e}"}


# ── Suggestions ────────────────────────────────────────────────────────────────

def get_suggestions(filename: str):
    response = requests.post(
        f"{BASE_URL}/suggestions",
        json={"filename": filename},
        timeout=30,
    )
    return response


# -- Streaming Chat ------------------------------------------------------------

def stream_chat(
    question: str,
    filename: str,
    history: list,
    session_id: str = "",
):
    """
    Generator that calls POST /chat/stream and yields parsed SSE event dicts.

    Each yielded dict has the shape:
      {"event": "progress", "message": "Querying SQLite database..."}
      {"event": "token",    "text":    "Hello"}
      {"event": "done",     "payload": {"followups": [...], "tools_used": [...], ...}}
      {"event": "error",    "message": "Something went wrong"}

    Usage in Streamlit:
        for evt in stream_chat(...):
            if evt["event"] == "token":
                yield evt["text"]   # pass to st.write_stream()
    """
    payload = {
        "question": question,
        "filename": filename,
        "history": history,
        "session_id": session_id,
    }

    try:
        with requests.post(
            f"{BASE_URL}/chat/stream",
            json=payload,
            stream=True,
            timeout=120,
        ) as response:
            response.raise_for_status()

            current_event = "message"
            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line:
                    # Blank line -- SSE message boundary, reset event type
                    current_event = "message"
                    continue

                if raw_line.startswith("event:"):
                    current_event = raw_line[len("event:"):].strip()
                    continue

                if raw_line.startswith("data:"):
                    data_str = raw_line[len("data:"):].strip()
                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    if current_event == "progress":
                        yield {"event": "progress", "message": data.get("message", "")}
                    elif current_event == "token":
                        yield {"event": "token", "text": data.get("text", "")}
                    elif current_event == "done":
                        yield {"event": "done", "payload": data}
                    elif current_event == "error":
                        yield {"event": "error", "message": data.get("message", "Unknown error")}

    except requests.Timeout:
        yield {"event": "error", "message": "The request timed out. Please try again."}
    except requests.ConnectionError:
        yield {"event": "error", "message": "Could not connect to the backend. Make sure uvicorn is running."}
    except Exception as exc:
        yield {"event": "error", "message": f"Error: {exc}"}