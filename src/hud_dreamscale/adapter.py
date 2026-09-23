"""Map HUD's explicit raw LIBERO contract to the published Dreamscale SDK."""

import hashlib
from dataclasses import dataclass
from typing import Any

import numpy as np
from dreamscale.libero import observe
from hud.agents.robot.adapter import Adapter

from .contract import CAMERAS, CHUNK_SIZE, build_contract, finite_array


@dataclass(frozen=True)
class LiberoInput:
    observation: Any
    instruction: str
    input_sha256: str


class LiberoAdapter(Adapter):
    def __init__(self):
        super().__init__(chunk_size=CHUNK_SIZE)

    def bind(self, action_space, observation_space):
        expected = build_contract()["features"]
        for name, actual in [("action", action_space), *observation_space.items()]:
            if name not in expected:
                continue
            for key, value in expected[name].items():
                if key != "role" and actual.get(key) != value:
                    raise ValueError(f"LIBERO contract mismatch: {name}.{key}")
        if not set((*CAMERAS, "state")).issubset(observation_space):
            raise ValueError("LIBERO requires raw agent and wrist cameras plus named state")
        super().bind(action_space, observation_space)

    def adapt_observation(self, obs, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A LIBERO task instruction is required")
        data = obs["data"]
        images = []
        for key in CAMERAS:
            image = np.asarray(data[key])
            if image.dtype != np.uint8 or image.shape != (256, 256, 3):
                raise ValueError(f"{key} must be raw 256x256 RGB uint8")
            images.append(image)
        state = finite_array(data["state"], (8,), "LIBERO state")
        # Do not rotate or normalize: the SDK's observation.to_wire owns preprocessing.
        digest = hashlib.sha256(prompt.encode("utf-8") + b"\0")
        for value in (*images, state):
            digest.update(str(value.dtype).encode() + b"\0")
            digest.update(str(value.shape).encode() + b"\0")
            digest.update(value.tobytes(order="C"))
        return LiberoInput(
            observe(agent_frame=images[0], wrist_frame=images[1], state=state.tolist()),
            prompt,
            digest.hexdigest(),
        )

    def adapt_chunk(self, chunk, obs):
        return finite_array(chunk, (CHUNK_SIZE, 7), "model action chunk")

    def adapt_action(self, action, obs):
        return finite_array(action, (7,), "LIBERO action")
