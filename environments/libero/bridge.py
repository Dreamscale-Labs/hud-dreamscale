"""CPU LIBERO simulation exposed through HUD's standard robot capability."""

import importlib
import os
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
from hud.environment.robot import RobotBridge

from environments.libero.physics import check_contact_physics
from hud_dropbear.contract import (
    CAMERAS,
    CONTROL_HZ,
    LEGACY_PROFILE,
    MAX_STEPS,
    POOLED_PROFILE,
    POOLED_RESOLUTION,
    TASK_SUITES,
    build_contract,
    finite_array,
    pooled_state_from_raw,
    state_from_raw,
)
from hud_dropbear.startup import StartupProfile


class LiberoBridge(RobotBridge):
    def __init__(self, profile=LEGACY_PROFILE):
        self.startup = StartupProfile("bridge_boot")
        with self.startup.span("bridge_constructor"):
            super().__init__()
        self.profile = profile
        self.control_hz = int(os.environ.get("LIBERO_CONTROL_HZ", CONTROL_HZ))
        self.contract = build_contract(self.control_hz, profile=profile)
        self.resolution = POOLED_RESOLUTION if profile == POOLED_PROFILE else 256
        self.step_timeout = 90.0
        self._env = None
        self._obs = None
        self.steps = 0
        self.reset_seconds = 0.0
        self.first_action_unix_s = None
        self.selection = {}
        self.physics_runtime = None
        self.reset_ordinal = 0
        self.reset_profile = None

    def reset(
        self,
        *,
        suite_name="libero_spatial",
        task_id=0,
        init_state_id=0,
        seed=0,
        max_steps=MAX_STEPS,
        startup_id=None,
    ):
        started = time.monotonic()
        self.reset_ordinal += 1
        self.reset_profile = StartupProfile("bridge_reset", episode_id=startup_id or uuid4().hex)
        with self.reset_profile.span("reset_total"):
            result = self._reset(suite_name, task_id, init_state_id, seed, max_steps)
        self.reset_seconds = time.monotonic() - started
        return result

    def _reset(self, suite_name, task_id, init_state_id, seed, max_steps):
        names = TASK_SUITES.get(suite_name)
        if names is None or type(task_id) is not int or task_id not in range(len(names)):
            raise ValueError("Unsupported LIBERO suite or task ID")
        if type(init_state_id) is not int or init_state_id < 0:
            raise ValueError("init_state_id must be a nonnegative integer")
        if type(max_steps) is not int or not 1 <= max_steps <= MAX_STEPS:
            raise ValueError("max_steps must be between 1 and 600")
        with self.reset_profile.span("previous_sim_close"):
            self._close_sim()
        if self.physics_runtime is None:
            with self.reset_profile.span("physics_preflight"):
                self.physics_runtime = check_contact_physics()
        # Bootstrap is a build/setup command, never a runtime download.
        assets = Path(os.environ.get("LIBERO_ASSETS_PATH", "/opt/libero-assets"))
        if not assets.is_dir():
            raise RuntimeError("LIBERO assets are missing; run the image/setup build first")
        with self.reset_profile.span("libero_imports"):
            raw = importlib.import_module("libero.libero")
            raw._assets_path_cache = str(assets)
            benchmark = importlib.import_module("libero.libero.benchmark")
            envs = importlib.import_module("libero.libero.envs")
        with self.reset_profile.span("task_and_initial_state_load"):
            suite = benchmark.get_benchmark_dict()[suite_name](task_order_index=0)
            task = suite.get_task(task_id)
            if task.name != names[task_id]:
                raise ValueError("Pinned LIBERO task ordering changed")
            states = suite.get_task_init_states(task_id)
            if init_state_id >= len(states):
                raise ValueError("init_state_id is outside the task's available initial states")
        try:
            with self.reset_profile.span("sim_constructor"):
                self._env = envs.OffScreenRenderEnv(
                    bddl_file_name=str(
                        Path(raw.get_libero_path("bddl_files"))
                        / task.problem_folder
                        / task.bddl_file
                    ),
                    camera_heights=self.resolution,
                    camera_widths=self.resolution,
                    control_freq=self.control_hz,
                )
            with self.reset_profile.span("sim_seed_and_reset"):
                self._env.seed(seed)
                self._env.reset()
            with self.reset_profile.span("sim_set_initial_state"):
                self._obs = self._env.set_init_state(states[init_state_id])
            with self.reset_profile.span("settling_ten_steps"):
                for _ in range(10):
                    self._obs, _, _, _ = self._env.step([0.0] * 6 + [-1.0])
        except BaseException:
            self._close_sim()
            raise
        self.steps = 0
        self.success = False
        self.total_reward = 0.0
        self.terminated = False
        self.first_action_unix_s = None
        self.max_steps = max_steps
        self.selection = {
            "suite": suite_name,
            "task_id": task_id,
            "task_name": task.name,
            "init_state_id": init_state_id,
            "seed": seed,
        }
        return str(task.language)

    def step(self, action):
        action = finite_array(action, (1, 7), "batched LIBERO action")
        if self._env is None or self.terminated:
            raise RuntimeError("An active LIBERO episode is required")
        self._obs, reward, _done, _info = self._env.step(action[0].tolist())
        if self.steps == 0:
            self.first_action_unix_s = time.time()
        self.steps += 1
        self.total_reward += float(reward)
        # Termination/time limits are not proof of success; use LIBERO's task predicate.
        self.success = bool(self._env.check_success())
        self.terminated = self.success or self.steps >= self.max_steps

    def get_observation(self):
        if self._obs is None:
            return None
        data = {key: np.asarray(self._obs[key])[None] for key in CAMERAS}
        if self.profile == POOLED_PROFILE:
            data.update(
                {key: value[None] for key, value in pooled_state_from_raw(self._obs).items()}
            )
        else:
            data["state"] = state_from_raw(self._obs)[None]
        return data, np.array([self.terminated], dtype=bool)

    def result(self):
        return {
            **super().result(),
            "info": {
                **self.selection,
                "steps": self.steps,
                "reset_seconds": self.reset_seconds,
                "control_hz": self.control_hz,
                "observation_profile": self.profile,
                "render_resolution": self.resolution,
                "settling_steps": 10,
                "reset_ordinal": self.reset_ordinal,
                "startup_profile": {
                    "bridge_boot": self.startup.snapshot(),
                    "bridge_reset": self.reset_profile.snapshot() if self.reset_profile else None,
                },
                "physics_runtime": self.physics_runtime,
                "first_action_unix_s": self.first_action_unix_s,
                "termination": "success"
                if self.success
                else "action_limit"
                if self.terminated
                else "interrupted",
            },
        }

    def _close_sim(self):
        if self._env is not None:
            self._env.close()
            self._env = None
        self._obs = None

    async def stop(self):
        try:
            await super().stop()
        finally:
            await self._run_on_sim(self._close_sim)
