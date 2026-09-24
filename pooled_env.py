"""Explicit raw360 pooled profile; the existing scalar environment is unchanged."""

from hud import Environment

from env import create_environment
from hud_dreamscale.contract import POOLED_PROFILE

env = create_environment(
    profile=POOLED_PROFILE, environment=Environment(name="dreamscale-libero-pooled")
)
