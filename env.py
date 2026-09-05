"""HUD-deployable LIBERO environment; model inference is supplied by the agent."""

from hud import Environment
from hud.environment.robot import RobotEndpoint

from environments.libero.bridge import LiberoBridge
from hud_dropbear.contract import ENV_NAME, MAX_STEPS, TASK_SUITES


def create_environment(endpoint=None):
    # HUD deploy discovers names statically, so the declaration must be literal.
    environment = Environment(name="dropbear-libero")
    assert environment.name == ENV_NAME
    endpoint = (endpoint or RobotEndpoint(LiberoBridge())).attach(environment)

    @environment.initialize
    async def initialize():
        await endpoint.start()
        for capability in await endpoint.capabilities():
            environment.add_capability(capability)

    @environment.shutdown
    async def shutdown():
        await endpoint.stop()

    def register_suite(suite_name):
        @environment.template(id=suite_name)
        async def evaluate_task(
            task_id: int = 0, init_state_id: int = 0, seed: int = 0, max_steps: int = MAX_STEPS
        ):
            episode = await endpoint.reset(
                suite_name=suite_name,
                task_id=task_id,
                init_state_id=init_state_id,
                seed=seed,
                max_steps=max_steps,
            )
            yield {"prompt": episode["prompt"], "bindings": {"robot": {"token": episode["token"]}}}
            yield await endpoint.result(token=episode["token"])

    for suite_name in TASK_SUITES:
        register_suite(suite_name)

    return environment


env = create_environment()
