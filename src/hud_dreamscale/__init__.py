"""HUD robotics evaluations with managed Dreamscale inference."""

__version__ = "0.1.0"


def __getattr__(name):
    # The CPU simulator imports only the shared contract, without Dreamscale SDK dependencies.
    if name == "DreamscaleRobotAgent":
        from .agent import DreamscaleRobotAgent

        return DreamscaleRobotAgent
    if name == "LiberoAdapter":
        from .adapter import LiberoAdapter

        return LiberoAdapter
    if name == "PooledProvider":
        from .pooled import PooledProvider

        return PooledProvider
    if name == "PooledRobotAgent":
        from .pooled_agent import PooledRobotAgent

        return PooledRobotAgent
    if name == "RuntimePool":
        from .runtime_pool import RuntimePool

        return RuntimePool
    raise AttributeError(name)


__all__ = [
    "DreamscaleRobotAgent",
    "LiberoAdapter",
    "PooledProvider",
    "PooledRobotAgent",
    "RuntimePool",
]
