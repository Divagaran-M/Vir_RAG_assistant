"""
services/hum_detector.py — HUM Auto-Greeting (silent, auto-run)

Runs face-api.js invisibly in the browser:
  - Camera feed is hidden from the user
  - Detection fires automatically on page load
  - Stops the camera stream after one detection
  - Gender → greeting:  Male = "How can I help you, Sir?"
                        Female = "How can I help you, Ma'am?"
"""

from __future__ import annotations

BACKEND_URL = "http://127.0.0.1:8000"


def default_greeting() -> str:
    return "Welcome! 👋 I'm **Vir**, your smart campus assistant. How can I help you?"


def hum_component_html(backend_url: str = BACKEND_URL) -> str:
    """
    Returns a tiny invisible HTML/JS component that:
      1. Opens the webcam silently (no UI shown)
      2. Loads face-api.js TinyFaceDetector + AgeGender models
      3. Captures one frame and runs inference
      4. Stops the camera immediately
      5. Posts {gender, age, greeting} back via Streamlit.setComponentValue()
    Height is 0px — completely invisible to the user.
    """
    models_url = f"{backend_url}/hum/models"
    return f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ background: transparent; overflow: hidden; }}
  /* Camera and canvas are completely hidden */
  video, canvas {{ position: absolute; visibility: hidden; width: 1px; height: 1px; }}
</style>
</head>
<body>
<video id="video" autoplay muted playsinline></video>
<canvas id="canvas"></canvas>

<script src="https://cdn.jsdelivr.net/npm/face-api.js@0.22.2/dist/face-api.min.js"></script>
<script>
const MODELS_URL = "{models_url}";

function greet(gender) {{
  if (gender === "Man")   return "How can I help you, Sir? 😊";
  if (gender === "Woman") return "How can I help you, Ma'am? 😊";
  return "How can I help you? 😊";
}}

async function run() {{
  // 1. Load models from local backend
  try {{
    await faceapi.nets.tinyFaceDetector.loadFromUri(MODELS_URL);
    await faceapi.nets.ageGenderNet.loadFromUri(MODELS_URL);
  }} catch(e) {{
    // Models failed — send fallback silently
    Streamlit.setComponentValue({{ success: false, gender: null, age: null,
      greeting: "How can I help you? 😊" }});
    return;
  }}

  // 2. Open camera silently
  let stream;
  try {{
    stream = await navigator.mediaDevices.getUserMedia({{ video: {{ width: 320, height: 240 }} }});
  }} catch(e) {{
    Streamlit.setComponentValue({{ success: false, gender: null, age: null,
      greeting: "How can I help you? 😊" }});
    return;
  }}

  const video = document.getElementById("video");
  const canvas = document.getElementById("canvas");
  video.srcObject = stream;

  // 3. Wait for video to be ready, then detect
  await new Promise(resolve => {{ video.onloadeddata = resolve; }});

  // Give camera a moment to settle (250ms)
  await new Promise(r => setTimeout(r, 250));

  canvas.width  = video.videoWidth  || 320;
  canvas.height = video.videoHeight || 240;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(video, 0, 0);

  const opts = new faceapi.TinyFaceDetectorOptions({{ inputSize: 224, scoreThreshold: 0.3 }});
  let detection = null;
  try {{
    detection = await faceapi.detectSingleFace(canvas, opts).withAgeAndGender();
  }} catch(e) {{ }}

  // 4. Stop camera immediately after capture
  stream.getTracks().forEach(t => t.stop());
  video.srcObject = null;

  // 5. Send result back to Streamlit
  if (detection) {{
    const gender  = detection.gender === "male" ? "Man" : "Woman";
    const age     = Math.round(detection.age);
    Streamlit.setComponentValue({{ success: true, gender, age, greeting: greet(gender) }});
  }} else {{
    Streamlit.setComponentValue({{ success: false, gender: null, age: null,
      greeting: "How can I help you? 😊" }});
  }}
}}

// Auto-run when Streamlit component is ready
if (window.Streamlit) {{
  Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, () => run());
  Streamlit.setComponentReady();
}} else {{
  // Fallback: run after short delay
  setTimeout(run, 500);
}}
</script>
</body>
</html>
"""
