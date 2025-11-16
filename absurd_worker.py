"""Long-running worker that completes rps.stub tasks with a static response."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

SDK_SRC = Path(__file__).parent / "third_party" / "absurd" / "sdks" / "python" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

from absurd_sdk import Absurd  # type: ignore


RESULT_PAYLOAD = {"move": "rock", "rationale": "rock beats scissors"}


app = Absurd()
# psycopg disables autocommit by default; enable it so queue setup doesn't hold locks
app._conn.autocommit = True


@app.register_task(name="rps.stub")
def rps_stub_task(_params: Any, _ctx: Any) -> dict[str, str]:
    return RESULT_PAYLOAD


def main() -> None:
    # Ensure the queue exists before starting the worker
    try:
        app.create_queue()
    except Exception:
        pass  # Queue might already exist

    app.start_worker()


if __name__ == "__main__":
    main()
