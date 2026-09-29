"""
tests/test_tts.py — Unit & Integration tests for Microsoft Edge Neural TTS route
"""

import sys
from pathlib import Path

# Add APP directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_tts_voices_list():
    """Verify that available voices list is returned properly."""
    response = client.get("/api/tts/voices")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["default"] == "en-IN-NeerjaNeural"
    assert len(data["voices"]) > 0

    voice_ids = [v["id"] for v in data["voices"]]
    assert "en-IN-NeerjaNeural" in voice_ids
    assert "en-US-JennyNeural" in voice_ids


def test_tts_post_empty_text():
    """Verify that empty text payload returns 400."""
    response = client.post("/api/tts", json={"text": ""})
    assert response.status_code == 400


def test_tts_post_generation():
    """Verify that Edge TTS produces valid MP3 audio stream."""
    response = client.post(
        "/api/tts",
        json={
            "text": "Hello, welcome to P.T. Lee Chengalvaraya Naicker College of Engineering.",
            "voice": "en-IN-NeerjaNeural"
        }
    )
    assert response.status_code == 200
    assert response.headers.get("content-type") == "audio/mpeg"
    assert len(response.content) > 1000  # Valid MP3 audio bytes


def test_tts_get_generation():
    """Verify that GET endpoint produces valid MP3 audio stream."""
    response = client.get(
        "/api/tts",
        params={
            "text": "Quick audio test.",
            "voice": "en-US-GuyNeural"
        }
    )
    assert response.status_code == 200
    assert response.headers.get("content-type") == "audio/mpeg"
    assert len(response.content) > 500


if __name__ == "__main__":
    test_tts_voices_list()
    print("test_tts_voices_list passed!")
    test_tts_post_empty_text()
    print("test_tts_post_empty_text passed!")
    test_tts_post_generation()
    print("test_tts_post_generation passed!")
    test_tts_get_generation()
    print("test_tts_get_generation passed!")
    print("\nAll Edge TTS tests passed successfully!")
