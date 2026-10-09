"""Build Dockerfile.hud on Modal once and print the image id for --image.

    uv run --project environments/robolab python environments/robolab/build_image.py

Modal caches layers, so rebuilding after a change to env.py / sim.py only redoes
the final COPY layer. Pass the printed ``modal://im-...`` to ``run_waves.py
--image`` (or export it as ``ROBOLAB_MODAL_IMAGE``).
"""

from __future__ import annotations

from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP_NAME = "hud-dreamscale-robolab"


def main() -> None:
    app = modal.App.lookup(APP_NAME, create_if_missing=True)
    image = modal.Image.from_dockerfile(HERE / "Dockerfile.hud", context_dir=HERE)
    with modal.enable_output():
        image.build(app)
    print(f"modal://{image.object_id}", flush=True)


if __name__ == "__main__":
    main()
