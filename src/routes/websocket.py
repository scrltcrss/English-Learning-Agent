import asyncio
import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Annotated
import io

from src.transcription import transcribe
from src.agent import (
    run_streaming_english_learning_agent,
    EnglishLearningAgentDependencies,
)
from src.config import EnvConfig
from src.tts import generate_speech

router = APIRouter()

logger = logging.getLogger(__name__)

whisper_model = None  # Will be set
kokoro_pipeline = None  # Will be set
deps_store: dict[str, EnglishLearningAgentDependencies] = {}


def get_env_config() -> EnvConfig:
    return EnvConfig()  # type: ignore


EnvConfigDependency = Annotated[EnvConfig, Depends(get_env_config)]


async def run_tts(text: str, websocket: WebSocket) -> None:
    buffer: io.BytesIO = await asyncio.to_thread(
        generate_speech, text=text, kokoro_pipeline=kokoro_pipeline, voice="af_heart"
    )
    await websocket.send_bytes(buffer.getvalue())


@router.websocket("/ws/agent")
async def agent_websocket(websocket: WebSocket, env_config: EnvConfigDependency):
    await websocket.accept()
    user_id = websocket.query_params.get("user_id", "default")
    if user_id not in deps_store:
        deps_store[user_id] = EnglishLearningAgentDependencies()
    while True:
        try:
            audio_bytes: bytes = await websocket.receive_bytes()
            await websocket.send_text("Audio received. Processing transcription...")

            transcription: str = await asyncio.to_thread(
                transcribe, audio=audio_bytes, whisper_model=whisper_model
            )
            logger.info(f"Transcription: {transcription}")
            await websocket.send_text(f"Transcription done: {transcription}")

            await run_streaming_english_learning_agent(
                text=transcription,
                chunk_size=10,
                callback=run_tts,
                env_config=env_config,
                deps=deps_store[user_id],
                websocket=websocket,
            )
        except WebSocketDisconnect:
            logger.info("Client disconnected")
            return
        except Exception as e:
            logger.exception(f"Unexpected error: {e}")
            await websocket.send_text(f"Error: {str(e)}")
