#!/usr/bin/env python3

from __future__ import annotations

import json
import logging
import os
import secrets
import sys
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.messages import (
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ThinkingPart,
    ThinkingPartDelta,
)

SYSTEM_PROMPT = (
    "You pick a single rock–paper–scissors move.\n"
    "Immediately call the emit_move tool exactly once and do nothing else.\n"
    "Never produce your own narration or the move text; the tool response is final.\n"
    "After the tool returns, stop; no reflections or extra calls.\n"
    "Valid moves: rock, paper, scissors."
)
USER_MESSAGE = "make a choice rock paper scissor"
VALID_MOVES: tuple[str, str, str] = ("rock", "paper", "scissors")
logger = logging.getLogger(__name__)


class MoveChoice(BaseModel):
    move: Literal["rock", "paper", "scissors"]
    rationale: str = Field(max_length=120)


def first_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(f"missing required environment variable: one of {', '.join(names)}")


def build_agent(base_url: str, api_key: str, model_name: str) -> Agent:
    provider = OpenAIProvider(base_url=base_url, api_key=api_key)
    model = OpenAIChatModel(model_name, provider=provider)
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        output_type=MoveChoice,
        event_stream_handler=log_event_stream,
    )

    @agent.tool_plain(name="emit_move")
    def emit_move() -> MoveChoice:
        move = secrets.choice(VALID_MOVES)
        rationale = f"Random choice: {move}."
        return MoveChoice(move=move, rationale=rationale)

    return agent


async def log_event_stream(_, events) -> None:
    async for event in events:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            logger.info("thinking start: %s", event.part.content.strip())
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
            delta = (event.delta.content_delta or "").strip()
            if delta:
                logger.info("thinking + %s", delta)
        elif isinstance(event, PartEndEvent) and isinstance(event.part, ThinkingPart):
            logger.info("thinking done: %s", event.part.content.strip())
        elif isinstance(event, FinalResultEvent):
            logger.info("model produced final result, waiting for stream completion")


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Loading environment configuration")
    load_dotenv()
    try:
        base_url = first_env("OPENAI_BASE_URL", "KIMI_BASE_URL")
        api_key = first_env("OPENAI_API_KEY", "KIMI_API_KEY")
        model_name = first_env("MODEL", "KIMI_MODEL")
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

    logger.info("Using model=%s base_url=%s", model_name, base_url)
    logger.debug("API key length=%d", len(api_key))

    agent = build_agent(base_url, api_key, model_name)
    logger.info("Agent initialized, starting run")
    start = time.monotonic()
    try:
        result = agent.run_sync(USER_MESSAGE)
    except Exception as exc:  # pragma: no cover - passthrough for CLI use
        logger.exception("Agent run failed")
        print(f"agent run failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    duration = time.monotonic() - start
    logger.info("Agent run finished in %.2fs", duration)

    output = result.output.model_dump()
    print(json.dumps(output, separators=(",", ":")))


if __name__ == "__main__":
    main()
