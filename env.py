"""HUD-deployable LIBERO environment; model inference is supplied by the agent."""

from uuid import uuid4

from hud import Environment
from hud.environment.robot import RobotEndpoint

from environments.libero.bridge import LiberoBridge
from hud_dropbear.contract import (
    ENV_NAME,
    LEGACY_PROFILE,
    MAX_STEPS,
    POOLED_ENV_NAME,
    POOLED_PROFILE,
    TASK_SUITES,
)
from hud_dropbear.startup import StartupProfile


def create_environment(endpoint=None, *, profile=LEGACY_PROFILE, environment=None):
    # HUD deploy discovers names statically, so the declaration must be literal.
    environment = environment or Environment(name="dropbear-libero")
    expected_name = POOLED_ENV_NAME if profile == POOLED_PROFILE else ENV_NAME
    assert environment.name == expected_name
    startup = StartupProfile("environment_boot")
    endpoint = (endpoint or RobotEndpoint(LiberoBridge(profile=profile))).attach(environment)

    @environment.initialize
    async def initialize():
        with startup.span("endpoint_start"):
            await endpoint.start()
        with startup.span("capabilities"):
            for capability in await endpoint.capabilities():
                environment.add_capability(capability)

    @environment.shutdown
    async def shutdown():
        with startup.span("endpoint_stop"):
            await endpoint.stop()

    def register_suite(suite_name):
        @environment.template(id=suite_name)
        async def evaluate_task(
            task_id: int = 0,
            init_state_id: int = 0,
            seed: int = 0,
            max_steps: int = MAX_STEPS,
            startup_id: str = "",
        ):
            startup_id = startup_id or uuid4().hex
            timing = StartupProfile("environment_episode", episode_id=startup_id)
            with timing.span("endpoint_reset"):
                episode = await endpoint.reset(
                    suite_name=suite_name,
                    task_id=task_id,
                    init_state_id=init_state_id,
                    seed=seed,
                    max_steps=max_steps,
                    startup_id=startup_id,
                )
            yield {"prompt": episode["prompt"], "bindings": {"robot": {"token": episode["token"]}}}
            with timing.span("endpoint_result"):
                result = await endpoint.result(token=episode["token"])
            info = result.setdefault("info", {})
            info.setdefault("startup_profile", {}).update(
                {
                    "environment_boot": startup.snapshot(),
                    "environment_episode": timing.snapshot(),
                }
            )
            yield result

    for suite_name in TASK_SUITES:
        register_suite(suite_name)

    return environment


env = create_environment()
