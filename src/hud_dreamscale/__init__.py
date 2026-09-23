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
    raise AttributeError(name)


__all__ = ["DreamscaleRobotAgent", "LiberoAdapter"]
