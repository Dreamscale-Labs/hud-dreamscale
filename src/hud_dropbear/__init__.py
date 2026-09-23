"""HUD robotics evaluations with managed Dreamscale inference."""

__version__ = "0.1.0"


def __getattr__(name):
    # The CPU simulator imports only the shared contract, without Dreamscale SDK dependencies.
    if name == "DropbearRobotAgent":
        from .agent import DropbearRobotAgent

        return DropbearRobotAgent
    if name == "LiberoAdapter":
        from .adapter import LiberoAdapter

        return LiberoAdapter
    raise AttributeError(name)


__all__ = ["DropbearRobotAgent", "LiberoAdapter"]
