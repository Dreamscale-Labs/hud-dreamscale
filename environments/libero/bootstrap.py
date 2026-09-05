"""Prepare immutable simulation assets and noninteractive LIBERO configuration."""

import importlib.util
import os
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


def main():
    spec = importlib.util.find_spec("libero")
    package = Path(next(iter(spec.submodule_search_locations))) / "libero"
    assets = Path(os.environ.get("LIBERO_ASSETS_PATH", "/opt/libero-assets"))
    config = Path(os.environ.get("LIBERO_CONFIG_PATH", "/opt/libero-config"))
    config.mkdir(parents=True, exist_ok=True)
    (config / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "benchmark_root": str(package),
                "bddl_files": str(package / "bddl_files"),
                "init_states": str(package / "init_files"),
                "datasets": "/tmp/libero-datasets",
                "assets": str(assets),
            }
        )
    )
    revision = os.environ["LIBERO_ASSETS_REVISION"]
    snapshot_download(
        "lerobot/libero-assets", repo_type="dataset", revision=revision, local_dir=assets
    )
    (assets / "revision.txt").write_text(revision + "\n")
    print(f"LIBERO assets ready: {revision}")


if __name__ == "__main__":
    main()
