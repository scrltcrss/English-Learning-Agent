from typing import Callable
from pydantic import BaseModel, Field
from pydantic_ai import Agent, UsageLimits
from pydantic_ai.models.openai import AsyncOpenAI, OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.tools import RunContext
from pydantic_ai.messages import PartDeltaEvent, TextPartDelta, ModelMessage
from src.config import EnvConfig


class FlashCard(BaseModel):
    word: str = Field(..., description="The word or phrase to learn")
    definition: str = Field(..., description="Meaning of the word")
    example_sentence: str = Field(..., description="Example sentence using the word")


class EnglishLearningAgentDependencies(BaseModel):
    current_flashcard_index: int = 0
    flashcards: list[FlashCard] = Field(
        default_factory=lambda: [
            FlashCard(
                word="scrumptious",
                definition="tasting extremely good",
                example_sentence="a scrumptious breakfast",
            )
        ]
    )
    message_history: list[ModelMessage] = Field(default_factory=list)


class EnglishLearningOutput(BaseModel):
    explanation: str = Field(..., description="Explanation of the word")
    flashcard: FlashCard | None = Field(..., description="Generated flashcard")


def create_english_learning_agent(
    openai_api_key: str, model_name: str
) -> Agent[EnglishLearningAgentDependencies, str]:
    client = AsyncOpenAI(api_key=openai_api_key)
    provider = OpenAIProvider(openai_client=client)
    model = OpenAIModel(model_name, provider=provider)

    system_prompt: str = (
        "You are an English learning assistant. "
        "Given a word, phrase, or sentence, provide clear explanations, "
        "and generate flashcards with word, definition, and example sentence. "
        "Keep output concise and structured."
    )

    agent = Agent(
        model=model,
        deps_type=EnglishLearningAgentDependencies,
        system_prompt=system_prompt,
        # output_type=ToolOutput(EnglishLearningOutput),
    )

    @agent.tool(retries=0)
    def fetch_flashcard(
        context: RunContext[EnglishLearningAgentDependencies],
    ) -> FlashCard:
        return context.deps.flashcards[context.deps.current_flashcard_index]

    @agent.tool(retries=0)
    def add_flashcard(
        context: RunContext[EnglishLearningAgentDependencies],
        word: str,
        definition: str,
        example_sentence: str,
    ) -> str:
        new_flashcard = FlashCard(
            word=word, definition=definition, example_sentence=example_sentence
        )
        context.deps.flashcards.append(new_flashcard)
        return f"Added flashcard: {word}"

    @agent.tool(retries=0)
    def next_flashcard(
        context: RunContext[EnglishLearningAgentDependencies],
    ) -> FlashCard:
        context.deps.current_flashcard_index += 1
        if context.deps.current_flashcard_index >= len(context.deps.flashcards):
            context.deps.current_flashcard_index = 0
        return context.deps.flashcards[context.deps.current_flashcard_index]

    return agent


agent: Agent[EnglishLearningAgentDependencies, str] | None = None


async def run_english_learning_agent(
    text: str, env_config: EnvConfig, deps: EnglishLearningAgentDependencies
) -> str:
    global agent
    if agent is None:
        agent = create_english_learning_agent(
            openai_api_key=env_config.OPENAI_API_KEY.get_secret_value(),
            model_name=env_config.MODEL,
        )
    run_result = await agent.run(
        text,
        deps=deps,
        message_history=deps.message_history,
        usage_limits=UsageLimits(request_limit=10),
    )
    deps.message_history = run_result.all_messages()[-10:]
    return run_result.output


async def run_streaming_english_learning_agent(
    text: str,
    callback: Callable,
    env_config: EnvConfig,
    deps: EnglishLearningAgentDependencies,
    chunk_size: int | None = None,
    *args,
    **kwargs,
) -> str:
    global agent
    if agent is None:
        agent = create_english_learning_agent(
            openai_api_key=env_config.OPENAI_API_KEY.get_secret_value(),
            model_name=env_config.MODEL,
        )
    async with agent.iter(
        text,
        deps=deps,
        message_history=deps.message_history,
        usage_limits=UsageLimits(request_limit=10),
    ) as run:
        collected_response: str = ""
        async for node in run:
            if Agent.is_model_request_node(node):
                async with node.stream(run.ctx) as request_stream:  # type: ignore
                    async for event in request_stream:
                        if isinstance(event, PartDeltaEvent) and isinstance(
                            event.delta, TextPartDelta
                        ):
                            collected_response += event.delta.content_delta
                            if chunk_size is None:
                                await callback(
                                    event.delta.content_delta, *args, **kwargs
                                )
                                continue

                            words: list[str] = collected_response.split(" ")
                            while len(words) > chunk_size:
                                chunk: str = " ".join(words[:chunk_size])
                                words = words[chunk_size:]
                                collected_response = " ".join(words)
                                await callback(chunk, *args, **kwargs)

            elif Agent.is_end_node(node):
                if chunk_size is not None and collected_response:
                    await callback(collected_response, *args, **kwargs)
    deps.message_history = run.all_messages()[-10:]
    return collected_response
