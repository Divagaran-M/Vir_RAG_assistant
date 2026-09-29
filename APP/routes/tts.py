"""
routes/tts.py — Natural AI Voice service powered by Microsoft Edge Neural TTS.
Provides realistic, human-quality text-to-speech without any API keys or billing.
"""

import logging
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import edge_tts

logger = logging.getLogger("vir.tts")

router = APIRouter(tags=["TTS"])

DEFAULT_VOICE = "en-IN-NeerjaNeural"

VOICE_CATALOG = [
    {
        "id": "en-IN-NeerjaNeural",
        "name": "Neerja (Indian English - Female)",
        "gender": "Female",
        "locale": "en-IN",
        "accent": "Indian English",
        "recommended": True
    },
    {
        "id": "en-IN-PrabhatNeural",
        "name": "Prabhat (Indian English - Male)",
        "gender": "Male",
        "locale": "en-IN",
        "accent": "Indian English",
        "recommended": True
    },
    {
        "id": "en-US-JennyNeural",
        "name": "Jenny (US English - Female)",
        "gender": "Female",
        "locale": "en-US",
        "accent": "US English",
        "recommended": False
    },
    {
        "id": "en-US-GuyNeural",
        "name": "Guy (US English - Male)",
        "gender": "Male",
        "locale": "en-US",
        "accent": "US English",
        "recommended": False
    },
    {
        "id": "en-US-AriaNeural",
        "name": "Aria (US English - Female)",
        "gender": "Female",
        "locale": "en-US",
        "accent": "US English",
        "recommended": False
    },
    {
        "id": "en-GB-SoniaNeural",
        "name": "Sonia (UK English - Female)",
        "gender": "Female",
        "locale": "en-GB",
        "accent": "UK English",
        "recommended": False
    },
    {
        "id": "ta-IN-PallaviNeural",
        "name": "Pallavi (Tamil - Female)",
        "gender": "Female",
        "locale": "ta-IN",
        "accent": "Tamil",
        "recommended": False
    }
]


class TTSRequest(BaseModel):
    text: str
    voice: str = DEFAULT_VOICE
    rate: str = "+0%"
    pitch: str = "+0Hz"
    volume: str = "+0%"


@router.get("/api/tts/voices")
@router.get("/tts/voices")
async def list_voices():
    """Return available curated neural voices."""
    return {
        "status": "success",
        "default": DEFAULT_VOICE,
        "voices": VOICE_CATALOG
    }


async def _generate_audio_stream(text: str, voice: str, rate: str = "+0%", pitch: str = "+0Hz", volume: str = "+0%"):
    """Helper to stream audio chunks from edge_tts."""
    selected_voice = voice or DEFAULT_VOICE
    try:
        communicate = edge_tts.Communicate(
            text=text,
            voice=selected_voice,
            rate=rate,
            pitch=pitch,
            volume=volume
        )
    except Exception as exc:
        logger.warning(f"Failed to initialize Communicate with voice {selected_voice}: {exc}, falling back to default voice.")
        communicate = edge_tts.Communicate(text=text, voice=DEFAULT_VOICE)

    async for chunk in communicate.stream():
        if chunk.get("type") == "audio":
            yield chunk["data"]


@router.post("/api/tts")
@router.post("/tts")
async def synthesize_speech_post(req: TTSRequest):
    """
    Synthesize text to speech using Microsoft Edge Neural TTS.
    Accepts text in JSON body to support longer responses without URL length constraints.
    """
    clean_text = (req.text or "").strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Safety truncate if payload is excessively massive (> 10,000 chars)
    if len(clean_text) > 10000:
        clean_text = clean_text[:10000]

    try:
        return StreamingResponse(
            _generate_audio_stream(
                text=clean_text,
                voice=req.voice,
                rate=req.rate,
                pitch=req.pitch,
                volume=req.volume
            ),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": "inline; filename=speech.mp3",
                "X-TTS-Engine": "Microsoft-Edge-Neural"
            }
        )
    except Exception as err:
        logger.error(f"TTS synthesis error: {err}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(err)}")


@router.get("/api/tts")
@router.get("/tts")
async def synthesize_speech_get(
    text: str = Query(..., description="Text to synthesize"),
    voice: str = Query(DEFAULT_VOICE, description="Voice ID"),
    rate: str = Query("+0%", description="Speech rate adjustment"),
    pitch: str = Query("+0Hz", description="Speech pitch adjustment")
):
    """
    Synthesize text to speech via GET query params.
    Useful for direct <audio src="..."> element embedding.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if len(clean_text) > 4000:
        clean_text = clean_text[:4000]

    try:
        return StreamingResponse(
            _generate_audio_stream(
                text=clean_text,
                voice=voice,
                rate=rate,
                pitch=pitch
            ),
            media_type="audio/mpeg",
            headers={
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": "inline; filename=speech.mp3",
                "X-TTS-Engine": "Microsoft-Edge-Neural"
            }
        )
    except Exception as err:
        logger.error(f"TTS synthesis error: {err}")
        raise HTTPException(status_code=500, detail=f"TTS synthesis failed: {str(err)}")
