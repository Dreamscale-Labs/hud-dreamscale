import sys
from pathlib import Path

import pytest
from hud.settings import settings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture(autouse=True)
def no_platform_reporting(monkeypatch):
    monkeypatch.setattr(settings, "telemetry_enabled", False)
    monkeypatch.setattr(settings, "api_key", None)
