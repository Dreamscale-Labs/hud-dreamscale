"""Check the installed physics runtime without inference or LIBERO assets."""

import platform
import time

from hud_dropbear.contract import validate_contact_height


def check_contact_physics():
    import mujoco

    started = time.monotonic()
    model = mujoco.MjModel.from_xml_string("""
        <mujoco>
          <option timestep="0.002"/>
          <worldbody>
            <body pos="0 0 0.875">
              <geom type="box" size="0.5 0.6 0.025"/>
            </body>
            <body pos="0.05341 0.20518 0.97">
              <freejoint/>
              <geom type="box" size="0.02 0.02 0.005" mass="0.05"/>
            </body>
          </worldbody>
        </mujoco>
    """)
    data = mujoco.MjData(model)
    for _ in range(500):
        mujoco.mj_step(model, data)
    height = float(data.qpos[2])
    validate_contact_height(height)
    return {
        "system": platform.system(),
        "machine": platform.machine(),
        "mujoco": mujoco.__version__,
        "contact_probe_height_m": height,
        "contact_probe_seconds": time.monotonic() - started,
    }


if __name__ == "__main__":
    import json

    print(json.dumps(check_contact_physics()))
