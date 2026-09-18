"""Create a minimal HUD remote-build context with the pooled entrypoint.

Usage: python scripts/prepare_pooled_build.py /private/path/empty-build-context
No dependency changes, credentials, meeting notes, tests, or artifacts are copied.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path


def prepare(destination, *, root=None):
    root = root or Path(__file__).resolve().parents[1]
    destination = Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Build context destination must be empty")
    destination.mkdir(parents=True, exist_ok=True)
    paths = [Path("env.py"), Path("pooled_env.py")]
    for directory in ("src", "environments/libero"):
        paths += [
            path.relative_to(root)
            for path in (root / directory).rglob("*")
            if path.is_file()
            and ".venv" not in path.parts
            and "__pycache__" not in path.parts
            and (path.suffix in (".py", ".json") or path.name in ("pyproject.toml", "uv.lock"))
        ]
    for relative in paths:
        if (root / relative).is_symlink():
            raise ValueError("Build context sources must not be symbolic links")
        (destination / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination / relative)
    dockerfile = (root / "Dockerfile.hud").read_text()
    old = 'CMD ["hud", "serve", "env.py", "--host", "0.0.0.0", "--port", "8765"]'
    if dockerfile.count(old) != 1:
        raise ValueError("Expected one standard HUD environment entrypoint")
    (destination / "Dockerfile.hud").write_text(
        dockerfile.replace(old, old.replace("env.py", "pooled_env.py"))
    )
    # HUD source discovery requires root project metadata. Docker installs only
    # the simulator's independently locked project, never the agent dependencies.
    (destination / "pyproject.toml").write_text(
        '[project]\nname = "hud-dropbear-pooled-environment"\nversion = "0.1.0"\n'
        'requires-python = ">=3.12,<3.13"\nlicense = "MIT"\ndependencies = []\n\n'
        "[tool.uv]\npackage = false\n"
    )
    paths.extend((Path("Dockerfile.hud"), Path("pyproject.toml")))
    manifest = {
        "schema_version": 1,
        "entrypoint": "pooled_env.py",
        "files": {
            str(path): hashlib.sha256((destination / path).read_bytes()).hexdigest()
            for path in sorted(paths)
        },
    }
    (destination / "build-context.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    print(json.dumps(prepare(Path(sys.argv[1])), indent=2))
