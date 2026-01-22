import io
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from src.tts import generate_speech

router = APIRouter()

kokoro_pipeline = None  # Will be set from app


@router.get("/tts")
def tts_endpoint(text: str):
    buffer: io.BytesIO = generate_speech(
        text=text, kokoro_pipeline=kokoro_pipeline, voice="af_heart"
    )

    return StreamingResponse(
        buffer,
        media_type="audio/wav",
        headers={"Content-Disposition": 'inline; filename="tts.wav"'},
    )
