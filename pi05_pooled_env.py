"""Native pi0.5 camera profile for HUD-hosted LIBERO."""

from hud import Environment
from env import create_environment
from hud_dreamscale.contract import PI05_PROFILE

env = create_environment(
    profile=PI05_PROFILE, environment=Environment(name="dreamscale-libero-pooled-pi05")
)
