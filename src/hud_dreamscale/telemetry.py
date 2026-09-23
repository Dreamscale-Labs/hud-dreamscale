"""Small, local timing sidecar; HUD remains canonical for actions, video and grades."""

import json
import time
from contextvars import ContextVar
from pathlib import Path

CURRENT_TASK = ContextVar("hud_dreamscale_task", default=None)


class Evidence:
    def __init__(self, path: Path, *, started=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.started = time.monotonic() if started is None else started
        self._file = path.open("x")
        self.rows = []

    def emit(self, event, **fields):
        if CURRENT_TASK.get() is not None:
            fields.setdefault("task", CURRENT_TASK.get())
        now = time.monotonic()
        row = {
            "event": event,
            "elapsed_s": now - self.started,
            "monotonic_s": now,
            "unix_s": time.time(),
            **fields,
        }
        self._file.write(json.dumps(row, allow_nan=False) + "\n")
        self._file.flush()
        self.rows.append(row)

    def close(self):
        self._file.close()
