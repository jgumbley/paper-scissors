#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from paper import MoveChoice, USER_MESSAGE, VALID_MOVES, run_agent


@dataclass
class ValidMoveEvaluator(Evaluator):
    """Checks the agent only emits the allowed rock-paper-scissors moves."""

    allowed_moves: tuple[str, ...] = VALID_MOVES

    async def evaluate(
        self, ctx: EvaluatorContext[str, MoveChoice]
    ) -> EvaluationReason:
        move = ctx.output.move
        if move not in self.allowed_moves:
            return EvaluationReason(
                value=0.0, reason=f"move '{move}' is outside {self.allowed_moves}"
            )
        return EvaluationReason(value=1.0, reason="move is valid")


@dataclass
class RationaleEvaluator(Evaluator):
    """Ensures the rationale references the move and stays concise."""

    max_length: int = 120

    async def evaluate(
        self, ctx: EvaluatorContext[str, MoveChoice]
    ) -> EvaluationReason:
        rationale = ctx.output.rationale.strip()
        move = ctx.output.move

        if not rationale:
            return EvaluationReason(value=0.0, reason="missing rationale")

        if move.lower() not in rationale.lower():
            return EvaluationReason(
                value=0.5, reason=f"rationale does not mention move '{move}'"
            )

        if len(rationale) > self.max_length:
            return EvaluationReason(
                value=0.5,
                reason=f"rationale is too long ({len(rationale)}>{self.max_length})",
            )

        return EvaluationReason(value=1.0, reason="rationale is concise and references move")


paper_dataset = Dataset[str, MoveChoice, None](
    cases=[
        Case(
            name="default_prompt",
            inputs=USER_MESSAGE,
            expected_output="valid move",
        ),
        Case(
            name="tool_usage_prompt",
            inputs="Choose exactly one move and explain why that move beats at least one opponent option.",
            expected_output="valid move",
        ),
    ],
    evaluators=[ValidMoveEvaluator(), RationaleEvaluator()],
)


def run_case(prompt: str) -> MoveChoice:
    return run_agent(prompt)


def main() -> None:
    report = paper_dataset.evaluate_sync(run_case)
    report.print(include_input=True, include_output=True)


if __name__ == "__main__":
    main()
