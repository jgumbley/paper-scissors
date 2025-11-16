"""Long-running worker that completes rps.stub tasks with a static response."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

SDK_SRC = Path(__file__).parent / "third_party" / "absurd" / "sdks" / "python" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

from absurd_sdk import Absurd  # type: ignore
from log_config import setup_logging

logger = logging.getLogger(__name__)


RESULT_PAYLOAD = {"move": "rock", "rationale": "rock beats scissors"}


app = Absurd()
# psycopg disables autocommit by default; enable it so queue setup doesn't hold locks
app._conn.autocommit = True


@app.register_task(name="rps.stub")
def rps_stub_task(_params: Any, _ctx: Any) -> dict[str, str]:
    logger.info("Executing rps.stub task with params: %s", _params)
    logger.debug("Task result: %s", RESULT_PAYLOAD)
    return RESULT_PAYLOAD


def main() -> None:
    setup_logging()
    logger.info("Starting absurd worker for task: rps.stub")

    # Ensure the queue exists before starting the worker
    try:
        logger.info("Creating queue if it doesn't exist")
        app.create_queue()
        logger.info("Queue created successfully")
    except Exception as e:
        logger.info("Queue already exists or failed to create: %s", e)

    logger.info("Starting worker polling loop")
    app.start_worker()


if __name__ == "__main__":
    main()
