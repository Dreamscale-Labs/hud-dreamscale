"""Report the installed client source separately from the simulator build baseline."""

import importlib

import pytest

from hud_dropbear import pooled_cli
from hud_dropbear import provenance as package_metadata

BASE = "0b63b4d3b9acb6d095e0886e18b2c905219e1e5a"
PATCH = "a" * 40


@pytest.mark.parametrize(
    "provenance,expected",
    [
        ({"source_commit": BASE}, BASE),
        ({"source_commit": PATCH}, PATCH),
        ({"version": "0.7.0"}, None),
        ({"local_source_available": False}, None),
        (
            {"source_commit": PATCH, "source_dirty": False, "installed_files_match_source": True},
            PATCH,
        ),
        (
            {"source_commit": PATCH, "source_dirty": True, "installed_files_match_source": True},
            None,
        ),
        (
            {"source_commit": PATCH, "source_dirty": False, "installed_files_match_source": False},
            None,
        ),
        (
            {"source_commit": PATCH, "source_dirty": False, "installed_files_match_source": None},
            None,
        ),
    ],
)
def test_actual_client_revision_never_falls_back_to_environment_baseline(
    monkeypatch, provenance, expected
):
    try:
        with monkeypatch.context() as patch:
            patch.setattr(package_metadata, "package_provenance", lambda name: provenance)
            module = importlib.reload(pooled_cli)
            assert module.HUD_REVISION == expected
            assert module.HUD_BASE_REVISION == BASE
    finally:
        # Restore the real installed metadata for any later tests/imported callers.
        importlib.reload(pooled_cli)
