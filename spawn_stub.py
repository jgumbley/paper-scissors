"""Spawn a single rps.stub task and print the resulting task id."""

from __future__ import annotations

import sys
from pathlib import Path

SDK_SRC = Path(__file__).parent / "third_party" / "absurd" / "sdks" / "python" / "src"
if str(SDK_SRC) not in sys.path:
    sys.path.insert(0, str(SDK_SRC))

from absurd_sdk import Absurd  # type: ignore


def main() -> None:
    client = Absurd()
    # Allow create_queue/spawn changes to commit immediately
    client._conn.autocommit = True

    # Ensure the queue exists before spawning
    try:
        client.create_queue()
    except Exception:
        pass  # Queue might already exist

    spawned = client.spawn("rps.stub", params=None, queue="default")
    print(spawned["task_id"])


if __name__ == "__main__":
    main()
