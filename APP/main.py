from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.upload import router as upload_router
from routes.chat import router as chat_router
from routes.suggestions import router as suggestions_router
from routes.hum import router as hum_router


app = FastAPI(
    title="Vir Campus Assistant API",
    version="2.0.0"
)

# Allow the Streamlit frontend (same machine, different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"message": "Vir Campus Assistant Backend Running 🚀"}


app.include_router(upload_router)
app.include_router(chat_router)
app.include_router(suggestions_router)
app.include_router(hum_router)