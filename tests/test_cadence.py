"""Do not silently inherit the SDK sim profile's incompatible 10 Hz default."""

from types import SimpleNamespace

import numpy as np
import pytest

from environments.libero import bridge as module
from hud_dropbear.cli import parser
from hud_dropbear.contract import LEGACY_PROFILE, POOLED_PROFILE, TASK_NAMES


@pytest.mark.parametrize("profile,resolution", [(LEGACY_PROFILE, 256), (POOLED_PROFILE, 360)])
def test_default_cli_and_actual_simulator_use_reference_cadence(
    tmp_path, monkeypatch, profile, resolution
):
    monkeypatch.delenv("LIBERO_CONTROL_HZ", raising=False)
    monkeypatch.setenv("LIBERO_ASSETS_PATH", str(tmp_path))
    monkeypatch.setattr(module, "check_contact_physics", lambda: {"checked": True})
    created = []
    actions = []

    class Sim:
        def __init__(self, **kwargs):
            created.append(kwargs)

        def seed(self, seed):
            pass

        def reset(self):
            pass

        def set_init_state(self, state):
            return {}

        def step(self, action):
            actions.append(action)
            return {}, 0, False, {}

        def close(self):
            pass

    task = SimpleNamespace(
        name=TASK_NAMES[0], language="put bowl on plate", problem_folder="", bddl_file="task"
    )
    suite = SimpleNamespace(
        get_task=lambda _: task, get_task_init_states=lambda _: np.zeros((1, 8))
    )
    modules = {
        "libero.libero": SimpleNamespace(get_libero_path=lambda _: str(tmp_path)),
        "libero.libero.benchmark": SimpleNamespace(
            get_benchmark_dict=lambda: {"libero_spatial": lambda **_: suite}
        ),
        "libero.libero.envs": SimpleNamespace(OffScreenRenderEnv=Sim),
    }
    monkeypatch.setattr(module.importlib, "import_module", modules.__getitem__)
    bridge = module.LiberoBridge(profile=profile)
    try:
        bridge.reset()
        assert created[0]["control_freq"] == 20
        assert created[0]["camera_heights"] == created[0]["camera_widths"] == resolution
        assert actions == [[0.0] * 6 + [-1.0]] * 10
        assert bridge.contract["control_rate"] == 20
        assert parser().parse_args([]).control_hz == 20
        assert parser().parse_args(["--control-hz", "10"]).control_hz == 10
    finally:
        bridge._close_sim()
