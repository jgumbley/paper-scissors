from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from psycopg.rows import dict_row
from pydantic import BaseModel, Field
from pydantic_ai import Agent, ToolOutput
from pydantic_ai.messages import (
    FinalResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    ThinkingPart,
    ThinkingPartDelta,
)
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from log_config import setup_logging

SDK_SRC = Path(__file__).parent / "third_party" / "absurd" / "sdks" / "python" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

from absurd_sdk import Absurd  # type: ignore

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


def _decide_winner(
    my_move: Literal["rock", "paper", "scissors"],
    opponent_move: str,
) -> Literal["me", "opponent", "draw"]:
    if opponent_move not in VALID_MOVES:
        raise ValueError(f"invalid opponent move: {opponent_move}")
    if my_move == opponent_move:
        return "draw"
    wins_against = {"rock": "scissors", "paper": "rock", "scissors": "paper"}
    return "me" if wins_against[my_move] == opponent_move else "opponent"


def _normalize_move_payload(payload: Any) -> dict[str, str]:
    if not isinstance(payload, dict):
        raise RuntimeError("opponent payload must be a dict")
    move = payload.get("move")
    rationale = payload.get("rationale")
    if move not in VALID_MOVES:
        raise RuntimeError(f"opponent move invalid: {move}")
    if not isinstance(rationale, str):
        raise RuntimeError("opponent rationale missing")
    return {"move": move, "rationale": rationale}


def _load_completed_payload(
    client: Absurd, task_id: str
) -> tuple[dict[str, str] | None, str | None]:
    cursor = client._conn.cursor(row_factory=dict_row)
    cursor.execute(
        "SELECT state, completed_payload FROM absurd.t_default WHERE task_id = %s",
        (task_id,),
    )
    row = cursor.fetchone()
    if not row:
        return None, None
    state = row["state"]
    payload = row["completed_payload"]
    if state == "failed":
        raise RuntimeError(f"rps.llm task {task_id} failed: {payload}")
    if state == "completed" and payload is not None:
        return _normalize_move_payload(payload), state
    return None, state


def _get_move_via_absurd(
    label: str = "opponent", timeout_seconds: float = 30.0, poll_interval: float = 0.25
) -> dict[str, str]:
    client = Absurd()
    client._conn.autocommit = True
    logger.info("Spawning %s via absurd queue 'default'", label)
    spawned = client.spawn("rps.llm", params=None, queue="default")
    task_id = spawned["task_id"]
    logger.info("Spawned %s task id=%s", label, task_id)
    deadline = time.monotonic() + timeout_seconds
    polls = 0
    last_wait_log = time.monotonic()
    last_state: str | None = None
    while time.monotonic() < deadline:
        payload, state = _load_completed_payload(client, task_id)
        if state and state != last_state:
            logger.info("%s task %s state=%s", label, task_id, state)
            last_state = state
        if payload is not None:
            logger.info("%s task %s completed with move=%s", label, task_id, payload["move"])
            return payload
        polls += 1
        now = time.monotonic()
        if now - last_wait_log >= 2:
            remaining = max(deadline - now, 0.0)
            logger.debug(
                "Waiting for %s task_id=%s polls=%d remaining=%.1fs",
                label,
                task_id,
                polls,
                remaining,
            )
            last_wait_log = now
        time.sleep(poll_interval)
    raise RuntimeError(
        f"rps.llm task {task_id} for {label} timed out after {timeout_seconds:.1f}s"
    )


def play_llm_vs_llm_once() -> None:
    setup_logging()
    logger.info("Starting play-llm run")

    start = time.monotonic()
    logger.info("Requesting my move via absurd worker")
    mine = _get_move_via_absurd("me")
    logger.info("My move=%s rationale_len=%d", mine["move"], len(mine["rationale"]))

    logger.info("Requesting opponent move via absurd worker")
    opponent = _get_move_via_absurd("opponent")
    logger.info("Opponent move=%s rationale_len=%d", opponent["move"], len(opponent["rationale"]))

    winner = _decide_winner(mine["move"], opponent["move"])
    logger.info("Winner decided: %s (elapsed=%.2fs)", winner, time.monotonic() - start)

    result = {"me": mine, "opponent": opponent, "winner": winner}
    print(json.dumps(result, separators=(",", ":")))


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
