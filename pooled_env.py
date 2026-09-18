"""Separate deployment for the native raw360 pooled inference contract."""

from hud import Environment

from env import create_environment
from hud_dropbear.contract import POOLED_PROFILE

env = create_environment(
    profile=POOLED_PROFILE, environment=Environment(name="dropbear-libero-pooled")
)
