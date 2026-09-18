"""Bounded startup-only diagnostics; no observation payloads or request-loop hooks."""

import json
import os
import sys
import time
from contextlib import contextmanager
from uuid import uuid4


class StartupProfile:
    """All durations use this process's monotonic clock; UTC is correlation only."""

    def __init__(self, role, *, episode_id=None, max_events=128, max_bytes=12288):
        self.role = role
        self.process_id = _PROCESS_ID
        self.episode_id = episode_id
        self.started = time.monotonic_ns()
        self.started_unix_ns = time.time_ns()
        self.max_events, self.max_bytes = max_events, max_bytes
        self.events = []
        self.dropped_events = 0
        self._bytes = 0

    @contextmanager
    def span(self, name):
        start = time.monotonic_ns()
        outcome = "ok"
        try:
            yield
        except BaseException:
            outcome = "error"
            raise
        finally:
            event = {
                "name": name,
                "start_offset_ns": start - self.started,
                "duration_ms": (time.monotonic_ns() - start) / 1e6,
                "outcome": outcome,
            }
            size = len(json.dumps(event).encode())
            if len(self.events) >= self.max_events or self._bytes + size > self.max_bytes:
                self.dropped_events += 1
            else:
                self.events.append(event)
                self._bytes += size
                # stdout belongs to HUD's discovery/control protocol.
                print(
                    json.dumps(
                        {
                            "hud_startup": {
                                "schema_version": 1,
                                "process_id": self.process_id,
                                "pid": os.getpid(),
                                "role": self.role,
                                "episode_id": self.episode_id,
                                **event,
                            }
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )

    def snapshot(self):
        return {
            "schema_version": 1,
            "process_id": self.process_id,
            "pid": os.getpid(),
            "role": self.role,
            "episode_id": self.episode_id,
            "started_unix_ns": self.started_unix_ns,
            "clock": "process_monotonic_ns",
            "events": [dict(event) for event in self.events],
            "dropped_events": self.dropped_events,
            "complete": self.dropped_events == 0,
        }


_PROCESS_ID = uuid4().hex
