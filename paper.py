from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ToolOutput
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
    "Pick exactly one move: rock, paper, or scissors.\n"
    "Call emit_move(move, rationale) once with that choice and a <=120 character justification.\n"
    "Once emit_move returns, stop immediately—no extra text, thoughts, or tool calls."
)
USER_MESSAGE = "make a choice rock paper scissor"
VALID_MOVES: tuple[str, str, str] = ("rock", "paper", "scissors")
logger = logging.getLogger(__name__)


class MoveChoice(BaseModel):
    move: Literal["rock", "paper", "scissors"]
    rationale: str = Field(max_length=120)


def load_configuration() -> tuple[str, str, str]:
    """Load API configuration from environment for reuse in CLI and evals."""
    load_dotenv()
    base_url = first_env("OPENAI_BASE_URL", "KIMI_BASE_URL")
    api_key = first_env("OPENAI_API_KEY", "KIMI_API_KEY")
    model_name = first_env("MODEL", "KIMI_MODEL")
    return base_url, api_key, model_name


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
        output_type=ToolOutput(
            MoveChoice,
            name="emit_move",
            description="Return your single rock-paper-scissors move and a concise rationale.",
        ),
        event_stream_handler=log_event_stream,
    )

    return agent


def run_agent(user_message: str = USER_MESSAGE, *, agent: Agent | None = None) -> MoveChoice:
    """
    Run the paper-scissors agent for a single prompt.

    When no agent is injected we create one from the current environment config,
    which keeps evals and CLI behavior aligned.
    """
    if agent is None:
        base_url, api_key, model_name = load_configuration()
        agent = build_agent(base_url, api_key, model_name)

    result = agent.run_sync(user_message)
    return result.output


async def log_event_stream(_, events) -> None:
    reasoning_chunks: list[str] = []
    current_part: list[str] | None = None
    logged = False

    async for event in events:
        if isinstance(event, PartStartEvent) and isinstance(event.part, ThinkingPart):
            current_part = []
        elif isinstance(event, PartDeltaEvent) and isinstance(event.delta, ThinkingPartDelta):
            if current_part is not None:
                delta = event.delta.content_delta or ""
                if delta:
                    current_part.append(delta)
        elif isinstance(event, PartEndEvent) and isinstance(event.part, ThinkingPart):
            text = _extract_trace_text(current_part, event.part.content)
            if text:
                reasoning_chunks.append(text)
            current_part = None
        elif isinstance(event, FinalResultEvent):
            if current_part:
                text = _extract_trace_text(current_part, "")
                if text:
                    reasoning_chunks.append(text)
                current_part = None
            _maybe_log_reasoning(reasoning_chunks)
            logged = True

    if not logged:
        _maybe_log_reasoning(reasoning_chunks)


def _maybe_log_reasoning(chunks: list[str]) -> None:
    trace = " ".join(chunks).strip()
    if trace:
        logger.info('reasoning trace: "%s"', trace)


def _extract_trace_text(buffer: list[str] | None, fallback: str) -> str:
    text = "".join(buffer) if buffer else (fallback or "")
    return " ".join(text.split())


def main() -> None:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Loading environment configuration")
    try:
        base_url, api_key, model_name = load_configuration()
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        raise SystemExit(1)

    logger.info("Using model=%s base_url=%s", model_name, base_url)
    logger.debug("API key length=%d", len(api_key))

    agent = build_agent(base_url, api_key, model_name)
    logger.info("Agent initialized, starting run")
    start = time.monotonic()
    try:
        move_choice = run_agent(USER_MESSAGE, agent=agent)
    except Exception as exc:  # pragma: no cover - passthrough for CLI use
        logger.exception("Agent run failed")
        print(f"agent run failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
    duration = time.monotonic() - start
    logger.info("Agent run finished in %.2fs", duration)

    output = move_choice.model_dump()
    print(json.dumps(output, separators=(",", ":")))


if __name__ == "__main__":
    main()
