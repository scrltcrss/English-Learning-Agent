import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from kokoro import KPipeline
import whisper
import logging

from src.agent import EnglishLearningAgentDependencies
from src.routes.index import router as index_router
from src.routes.tts_routes import router as tts_router
from src.routes.agent_routes import router as agent_router
from src.routes.websocket import router as websocket_router
from src.middlewares.measure_time import MeasureTimeMiddleware

kokoro_pipeline: KPipeline | None = None
whisper_model: whisper.Whisper | None = None
deps_store: dict[str, EnglishLearningAgentDependencies] = {}

logging.basicConfig(level=logging.INFO)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kokoro_pipeline, whisper_model
    kokoro_pipeline, whisper_model = await asyncio.gather(
        asyncio.to_thread(KPipeline, repo_id="hexgrad/Kokoro-82M", lang_code="a"),
        asyncio.to_thread(whisper.load_model, "base"),
    )
    from src.routes import tts_routes as tts, websocket, agent_routes as agent

    tts.kokoro_pipeline = kokoro_pipeline
    websocket.kokoro_pipeline = kokoro_pipeline
    websocket.whisper_model = whisper_model
    agent.deps_store = deps_store
    websocket.deps_store = deps_store
    yield


app: FastAPI = FastAPI(lifespan=lifespan)

app.add_middleware(MeasureTimeMiddleware)

app.include_router(index_router)
app.include_router(tts_router)
app.include_router(agent_router)
app.include_router(websocket_router)
