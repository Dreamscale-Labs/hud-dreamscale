# HUD recording patches

`hud-video-thread-limit.patch` includes two focused recording fixes. It bounds libx264 to one codec thread per camera by default. HUD already starts an encoder thread for each camera; allowing each codec to choose its own thread count exhausted native threads during a 64-lane, two-camera recording fixture. The patch uses the normal HUD `RobotAgent` recording path, adds an explicit positive `thread_count` option to `SegmentEncoder`, and includes real PyAV/CMAF tests. It also adds `TraceRecorder.aclose()` and makes `RobotAgent` await it: encoder finalization runs off the asyncio event loop, and cancellation is deferred until the owned close operation completes. Concurrent asynchronous callers share the flush; the synchronous `close()` API remains available. A nested `finally` ensures recorder flushing even when wire teardown is cancelled. It does not change model inference or simulator images.

The existing best-effort encoder policy is unchanged: each camera has a 15-second join timeout and callback errors are logged. This does not guarantee complete recording or thread termination if an encoder or exporter is stuck beyond that timeout.

Apply it to HUD base revision `0b63b4d3b9acb6d095e0886e18b2c905219e1e5a`. The internal qualification checkout is `b31a73935084b8e56740f4787462798484e4f77c`; that identifier is not an upstream release or an assumed publicly fetchable commit. The patch and its content hashes are included here so reproduction does not require that checkout. HUD's MIT copyright and permission notice are preserved in [LICENSE.hud](LICENSE.hud).

Use an existing Python 3.12 integration environment with its dependencies already resolved. From this repository's root:

```sh
patch_file="$(pwd)/patches/hud-video-thread-limit.patch"
git clone https://github.com/hud-evals/hud-python.git ../hud-python-video
git -C ../hud-python-video checkout --detach 0b63b4d3b9acb6d095e0886e18b2c905219e1e5a
git -C ../hud-python-video apply --check "$patch_file"
git -C ../hud-python-video apply "$patch_file"
git -C ../hud-python-video add hud/telemetry/robot/video.py hud/telemetry/robot/recorder.py hud/agents/robot/agent.py hud/telemetry/tests/test_robot_video.py hud/telemetry/tests/test_robot_recorder_async.py
test "$(git -C ../hud-python-video write-tree)" = b5d9f588e1dcbe41e6b580cb3073479805293fdb
git -C ../hud-python-video commit -m "Bound HUD camera threads and finalize recording asynchronously"
uv pip install --python .venv/bin/python --no-deps ../hud-python-video
.venv/bin/python -m pytest -q ../hud-python-video/hud/telemetry/tests/test_robot_video.py ../hud-python-video/hud/telemetry/tests/test_robot_recorder_async.py
```

Run this in a new clone: the expected tree check covers the entire patched source tree and rejects unrelated changes. The JSON manifest also contains the patch SHA-256 and individual changed-file hashes. The new local commit can differ from `b31a739…` because its author, timestamp and message differ even when the source tree is identical. Record that actual local revision; never label it with the internal commit ID just because the patch applied.

After the noneditable install, retain the package provenance:

```sh
.venv/bin/python -c 'import json; from hud_dropbear.provenance import package_provenance; print(json.dumps(package_provenance("hud"), indent=2))'
```

Require a clean local checkout, `installed_files_match_source: true`, and the actual source commit alongside installed-file hashes. The evaluation's `hud_revision` describes this agent-side client; `hud_base_revision` describes the original baseline. The hosted simulator still uses the original pinned HUD build. A normal `uv sync` restores the lockfile's baseline, so reinstall this explicit patch afterward when reproducing the wide recording run. Remove this extra installation step once the required fix is available in a verified upstream release and the integration pin has been updated.

This patch is an explicit interim dependency for wide recording. Its focused tests cover codec behavior, responsive peer tasks during finalization, repeated cancellation, preservation of primary model errors, and both camera streams. A separate bounded offline64-lane qualification decoded all256 streams from128 synthetic episodes with ordinary HUD recording; it does not establish hosted-policy quality or performance. Offline integration and real hosted-policy results must be reported separately. It does not establish publication or compatibility of an unreleased Dropbear SDK.
