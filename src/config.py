from pydantic import ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings


class EnvConfig(BaseSettings):
    model_config: ConfigDict = ConfigDict(env_file=".env", env_file_encoding="utf-8")  # type: ignore

    OPENAI_API_KEY: SecretStr = Field(env="SECRET_KEY")  # type: ignore
    MODEL: str = Field(env="MODEL")  # type: ignore
