"""HUD robotics evaluations with managed Dropbear inference."""

__version__ = "0.1.0"


def __getattr__(name):
    if name == "PooledRobotAgent":
        from .pooled_agent import PooledRobotAgent

        return PooledRobotAgent
    if name == "PooledProvider":
        from .pooled import PooledProvider

        return PooledProvider
    if name == "RuntimePool":
        from .runtime_pool import RuntimePool

        return RuntimePool
    # The CPU simulator imports only the shared contract, without Dropbear dependencies.
    if name == "DropbearRobotAgent":
        from .agent import DropbearRobotAgent

        return DropbearRobotAgent
    if name == "LiberoAdapter":
        from .adapter import LiberoAdapter

        return LiberoAdapter
    raise AttributeError(name)


__all__ = [
    "DropbearRobotAgent",
    "LiberoAdapter",
    "PooledRobotAgent",
    "PooledProvider",
    "RuntimePool",
]
