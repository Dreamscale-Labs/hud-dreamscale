"""Apply the RoboTwin wheel patches after install.

mplib 0.2.1 treats a missing collision flag as a failed screw plan.
sapien 3.0.0b1 opens SRDF files without an encoding and checks the wrong suffix.
"""

import pathlib
import re

root = pathlib.Path("/app/.venv/lib/python3.12/site-packages")
planner = root / "mplib" / "planner.py"
loader = root / "sapien" / "wrapper" / "urdf_loader.py"
if not planner.exists() or not loader.exists():
    raise SystemExit(f"missing sim packages: planner={planner.exists()} loader={loader.exists()}")

planner.write_text(re.sub(r"\bor collide\b", "", planner.read_text(), count=1))
src = loader.read_text()
src = src.replace(
    "with open(srdf_path) as f:",
    'with open(srdf_path, encoding="utf-8") as f:',
).replace('"srdf"', '".srdf"')
loader.write_text(src)
