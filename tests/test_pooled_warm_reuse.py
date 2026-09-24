"""Sequential cohort reuse over real HUD control and the SDK's mock HTTP boundary."""

from types import SimpleNamespace

import numpy as np
import pytest
from hud.eval.job import Job
from hud.settings import settings

from hud_dreamscale.cohort import cohort_manifest, cohort_tasks, run_cohort, summarize_cohort
from hud_dreamscale.pooled_agent import PooledRobotAgent
from hud_dreamscale.telemetry import Evidence
from tests.test_pooled import Server, predict
from tests.test_pooled_agent import local_pool


@pytest.mark.parametrize("provider_width,runtime_width", [(8, 8), (8, 3), (8, 1), (64, 63)])
def test_active_width_can_use_a_subset_of_provider_slots(provider_width, runtime_width):
    provider = SimpleNamespace(concurrency=provider_width)
    runtimes = SimpleNamespace(concurrency=runtime_width)
    agent = PooledRobotAgent(provider=provider, runtimes=runtimes)
    assert agent.provider is provider and agent.runtimes is runtimes


@pytest.mark.parametrize(
    "provider_width,runtime_width",
    [(8, 9), (3, 8), (8, 0), (8, -1), (8, True), (8, 1.5), (8, 65), (0, 1), (True, 1)],
)
def test_active_width_rejects_invalid_or_unexposed_slots(provider_width, runtime_width):
    with pytest.raises(ValueError, match="concurrency"):
        PooledRobotAgent(
            provider=SimpleNamespace(concurrency=provider_width),
            runtimes=SimpleNamespace(concurrency=runtime_width),
        )


async def test_eight_three_one_cohorts_share_one_provider_without_resetting_slots(
    tmp_path, monkeypatch
):
    """No GPU claim: mock results qualify transport, queues, identities and cleanup only."""
    monkeypatch.setattr(settings, "telemetry_local_dir", str(tmp_path / "traces"))
    server = Server()
    jobs, manifests, all_episode_ids = set(), set(), set()
    calls_per_slot = [0] * 8
    async with server.provider(concurrency=8, journal_path=tmp_path / "requests.jsonl") as provider:
        clients = tuple(server.clients)
        assert len(clients) == 9  # One management client plus eight persistent slot clients.
        for width in (8, 3, 1):
            rows = cohort_tasks(width, max_steps=3)
            _manifest, digest = cohort_manifest(rows, concurrency=width, episodes_per_lane=2)
            assert digest not in manifests
            manifests.add(digest)
            job = await Job.start(f"warm-reuse-{width}")
            assert job.id not in jobs
            jobs.add(job.id)
            evidence = Evidence(tmp_path / f"cohort-{width}" / "timings.jsonl")
            start = len(server.posts)
            try:
                async with local_pool(rows, emit=evidence.emit) as (pool, bridges):
                    agent = PooledRobotAgent(
                        provider=provider, runtimes=pool, max_steps=3, emit=evidence.emit
                    )
                    await run_cohort(agent, rows, runtime=pool, job=job, concurrency=width)
                    summary = summarize_cohort(job, rows, evidence.rows)
                    assert summary["successes"] == len(rows)
                    assert summary["integration_errors"] == 0
                    assert summary["completed_episodes"] == len(rows)
                    cohort_posts = server.posts[start:]
                    assert len(cohort_posts) == len(rows)
                    assert {post["robot_slot"] for post in cohort_posts} == set(range(width))
                    episode_ids = {post["episode_id"] for post in cohort_posts}
                    assert len(episode_ids) == len(rows)
                    assert all_episode_ids.isdisjoint(episode_ids)
                    all_episode_ids.update(episode_ids)
                    assert episode_ids == {row["episode_id"] for row in summary["runs"]}
                    for slot, bridge in bridges.items():
                        posts = [post for post in cohort_posts if post["robot_slot"] == slot]
                        assert [post["sequence"] for post in posts] == list(
                            range(calls_per_slot[slot], calls_per_slot[slot] + 2)
                        )
                        # Every episode ends after two of ten actions. Its eight
                        # leftover actions must never reach the next episode.
                        for episode, post in zip(bridge.episodes, posts, strict=True):
                            assert len(episode) == 2
                            assert all(
                                np.all(action == slot + post["sequence"] + 1) for action in episode
                            )
                        calls_per_slot[slot] += 2
                        assert bridge._registry.all_free
                    for run in job.runs:
                        identity = run.trace.extra["dreamscale"]
                        assert identity["active_concurrency"] == width
                        assert identity["provider_concurrency"] == 8
                        assert identity["warm_robots"] == identity["max_robots"] == 8
                assert pool.cleanup_confirmed
                assert len(server.creates) == 1 and server.stops == 0
                assert tuple(server.clients) == clients and not server.closed
                assert provider.capacity == 8 and not provider.fenced_slots
            finally:
                evidence.close()
        assert calls_per_slot == [6, 4, 4, 2, 2, 2, 2, 2]
    assert provider.cleanup_confirmed and server.stops == 1
    assert server.stop_ids == ["deployment-test"]
    assert set(server.closed) == set(clients) and len(server.closed) == len(clients)


async def test_narrower_cohort_preserves_fenced_slot_without_remapping(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "telemetry_local_dir", str(tmp_path / "traces"))
    server = Server()
    async with server.provider(concurrency=8) as provider:
        server.mode = "pending_forever"
        with pytest.raises(TimeoutError):
            await predict(provider, slot=0, episode="uncertain-before-cohort")
        assert provider.fenced_slots == {0}
        server.mode = "success"
        rows = cohort_tasks(3, max_steps=3)
        job = await Job.start("warm-reuse-fenced")
        async with local_pool(rows) as (pool, bridges):
            await run_cohort(
                PooledRobotAgent(provider=provider, runtimes=pool, max_steps=3),
                rows,
                runtime=pool,
                job=job,
                concurrency=3,
            )
            summary = summarize_cohort(job, rows, [])
            assert summary["successes"] == 4 and summary["integration_errors"] == 2
            assert all(not episode for episode in bridges[0].episodes)
            assert all(bridge._registry.all_free for bridge in bridges.values())
            assert all(row["lane_id"] == 0 for row in summary["runs"] if row["integration_error"])
        assert provider.fenced_slots == {0}
        assert [post["robot_slot"] for post in server.posts].count(0) == 1
        assert {post["robot_slot"] for post in server.posts} == {0, 1, 2}
        for slot in (1, 2):
            sequences = [post["sequence"] for post in server.posts if post["robot_slot"] == slot]
            assert sequences == [0, 1]
    assert provider.cleanup_confirmed and server.stops == 1
