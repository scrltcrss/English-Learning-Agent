from fastapi import APIRouter, Depends
from typing import Annotated

from src.agent import run_english_learning_agent, EnglishLearningAgentDependencies
from src.config import EnvConfig

router = APIRouter()

deps_store: dict[str, EnglishLearningAgentDependencies] = {}


def get_env_config() -> EnvConfig:
    return EnvConfig()  # type: ignore


EnvConfigDependency = Annotated[EnvConfig, Depends(get_env_config)]


@router.get("/agent")
async def agent_endpoint(
    text: str, user_id: str, env_config: EnvConfigDependency
) -> str:
    if user_id not in deps_store:
        deps_store[user_id] = EnglishLearningAgentDependencies()
    output: str = await run_english_learning_agent(
        text=text, env_config=env_config, deps=deps_store[user_id]
    )
    return output
