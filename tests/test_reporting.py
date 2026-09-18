import pytest

from hud_dropbear.reporting import cohort_gate, concurrency_report, distribution, timing_report


def result():
    return {
        "concurrency": 1,
        "expected_episodes": 6,
        "provenance": {"runtime": "hud", "max_steps": 600},
        "platform_evidence": {"verified": True},
        "provider_cleanup_confirmed": True,
        "simulator_cleanup": {"verified": True},
        "concurrency_evidence": {
            "active_episodes": {"peak_distinct_lanes": 1},
            "provider_ready_robots": 8,
            "errors": [],
        },
        "runs": [
            {
                "case_index": case,
                "lane_id": 0,
                "trace_id": f"{case:032x}",
                "episode_id": str(case),
                "reward": int(case % 2 == 0),
                "args": {"task_id": task, "init_state_id": initial, "seed": 0},
            }
            for case, (task, initial) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)])
        ],
    }


def test_every_attempt_is_in_denominator_and_platform_is_required():
    data = result()
    assert cohort_gate(data, concurrency=1)["passed"]
    data["runs"].pop()
    gate = cohort_gate(data, concurrency=1)
    assert not gate["passed"] and gate["success_rate"] == 0.5
    data = result()
    data["platform_evidence"]["verified"] = False
    assert not cohort_gate(data, concurrency=1)["passed"]


def test_short_single_lane_smoke_run_cannot_pass_fixed_campaign_acceptance():
    data = result()
    data["runs"] = data["runs"][:2]
    data["expected_episodes"] = 2
    gate = cohort_gate(data, concurrency=1)
    assert gate["success_rate"] == 0.5
    assert not gate["passed"]
    assert gate["fixed_task_coverage"]["missing_task_initial_states"] == [
        [1, 0],
        [1, 1],
        [2, 0],
        [2, 1],
    ]


@pytest.mark.parametrize("mutation", ["repeat", "reorder", "missing_args", "wrong_seed"])
def test_six_episodes_require_the_actual_frozen_task_initial_state_assignments(mutation):
    data = result()
    if mutation == "repeat":
        data["runs"][5]["args"] = dict(data["runs"][0]["args"])
    elif mutation == "reorder":
        data["runs"][0]["args"], data["runs"][1]["args"] = (
            data["runs"][1]["args"],
            data["runs"][0]["args"],
        )
    elif mutation == "missing_args":
        del data["runs"][0]["args"]
    else:
        data["runs"][0]["args"]["seed"] = 1
    gate = cohort_gate(data, concurrency=1)
    assert not gate["passed"]
    assert gate["fixed_task_coverage"]["mismatched_case_indices"]


@pytest.mark.parametrize("width", [2, 3, 8, 17, 32, 64])
def test_other_widths_retain_two_episodes_per_lane_acceptance(width):
    data = result()
    data.update(concurrency=width, expected_episodes=2 * width)
    data["runs"] = [
        {
            "case_index": case,
            "lane_id": case % width,
            "trace_id": f"{case:032x}",
            "episode_id": str(case),
            "reward": case % 2,
        }
        for case in range(2 * width)
    ]
    data["concurrency_evidence"]["active_episodes"]["peak_distinct_lanes"] = width
    data["concurrency_evidence"]["provider_ready_robots"] = 8 * ((width + 7) // 8)
    assert cohort_gate(data, concurrency=width)["passed"]


@pytest.mark.parametrize(
    "mutation",
    ["duplicate", "duplicate_trace", "duplicate_episode", "error", "nan", "smoke", "local"],
)
def test_success_cannot_hide_invalid_evaluation(mutation):
    data = result()
    if mutation == "duplicate":
        data["runs"][1]["case_index"] = 0
    elif mutation == "duplicate_trace":
        data["runs"][1]["trace_id"] = data["runs"][0]["trace_id"]
    elif mutation == "duplicate_episode":
        data["runs"][1]["episode_id"] = data["runs"][0]["episode_id"]
    elif mutation == "error":
        data["runs"][1]["integration_error"] = True
    elif mutation == "nan":
        data["runs"][1]["reward"] = float("nan")
    elif mutation == "smoke":
        data["provenance"]["max_steps"] = 20
    else:
        data["provenance"]["runtime"] = "local-container"
    assert not cohort_gate(data, concurrency=1)["passed"]


def test_each_episode_first_inference_is_separate_even_with_same_slug():
    events = []
    for episode in ("a", "b"):
        events.extend(
            [
                {"event": "inference", "episode_id": episode, "task": "same", "duration_s": 2},
                {"event": "inference", "episode_id": episode, "task": "same", "duration_s": 0.2},
                {
                    "event": "provider_inference",
                    "episode_id": episode,
                    "duration_s": 0.19,
                    "timing": {"gpu_ms": 100},
                },
            ]
        )
    report = timing_report(events)
    assert report["episode_first_inference_s"]["n"] == 2
    assert report["steady_inference_s"]["n"] == 2
    assert report["steady_inference_s"]["p50"] == 0.2
    assert report["successful_model_calls_s"]["n"] == 4
    assert report["server_reported_ms"]["gpu_ms"]["p50"] == 100


def test_distribution_preserves_small_sample_size_and_rejects_impossible_intervals():
    assert distribution([])["n"] == 0
    assert distribution([1])["p95"] == 1
    assert distribution([1, 2, 3])["p50"] == 2
    with pytest.raises(ValueError):
        distribution([-1])


def test_failed_slow_attempt_is_retained_in_attempt_distribution():
    events = [
        {"event": "inference_attempt_finished", "duration_s": 10, "outcome": "error"},
        {"event": "inference_post", "duration_s": 5, "response_received": False, "request_id": "a"},
        {"event": "inference_recovery", "request_id": "a"},
        {"event": "slot_fenced", "request_id": "a"},
    ]
    report = timing_report(events)
    assert report["successful_model_calls_s"]["n"] == 0
    assert report["all_model_attempts_s"]["p50"] == 10
    assert report["sdk_post_attempts_s"]["p50"] == 5
    assert report["request_counts"]["unknown_post_outcomes"] == 1
    assert report["request_counts"]["fenced_requests"] == 1


def test_serial_episodes_cannot_pass_as_parallel_lanes():
    data = result()
    data["concurrency"] = 2
    data["runs"][1]["lane_id"] = 1
    gate = cohort_gate(data, concurrency=2)
    assert "Requested simulator concurrency was not observed" in gate["reasons"]


def test_successful_grades_do_not_hide_an_overall_operational_error():
    data = result()
    data["error_type"] = "TimeoutError"
    assert not cohort_gate(data, concurrency=1)["passed"]


def test_journal_phases_are_not_double_counted_across_post_and_result():
    report = timing_report(
        [
            {
                "event": "inference_post",
                "duration_s": 1,
                "response_received": True,
                "journal_before_post_s": 0.1,
            },
            {
                "event": "provider_inference",
                "journal_before_post_s": 0.1,
                "journal_terminal_s": 0.2,
            },
            {
                "event": "inference_cancelled",
                "journal_before_post_s": 0.3,
                "journal_terminal_s": 0.4,
            },
        ]
    )
    assert report["journal_before_post_s"]["n"] == 1
    assert report["journal_before_post_s"]["p50"] == 0.1
    assert report["journal_terminal_s"]["n"] == 2
    assert report["journal_terminal_s"]["p50"] == pytest.approx(0.3)


def test_client_overlap_counts_distinct_lanes_and_retains_duration():
    events = [{"event": "provider_ready", "ready_robots": 8}]
    for lane, begin, end in ((0, 1, 5), (1, 3, 6), (0, 6, 8)):
        identity = f"{lane}-{begin}"
        events.extend(
            [
                {
                    "event": "environment_ready",
                    "lane_id": lane,
                    "episode_id": identity,
                    "elapsed_s": begin,
                },
                {
                    "event": "episode_driven",
                    "lane_id": lane,
                    "episode_id": identity,
                    "elapsed_s": end,
                },
            ]
        )
    # HTTP requests overlap for 0.2 seconds, but are not GPU timing evidence.
    events.extend(
        [
            {"event": "inference_post", "robot_slot": 0, "elapsed_s": 4, "duration_s": 0.5},
            {"event": "inference_post", "robot_slot": 1, "elapsed_s": 4.3, "duration_s": 0.5},
        ]
    )
    report = concurrency_report(events)
    assert report["active_episodes"] == {
        "peak_distinct_lanes": 2,
        "seconds_at_peak": 2,
        "closed_intervals": 3,
    }
    assert report["sdk_posts"]["peak_distinct_lanes"] == 2
    assert report["sdk_posts"]["seconds_at_peak"] == pytest.approx(0.2)
    assert not report["errors"]


def test_touching_and_unclosed_episodes_are_not_parallel_evidence():
    events = []
    for lane, begin, end in ((0, 1, 2), (1, 2, 3)):
        events.extend(
            [
                {
                    "event": "environment_ready",
                    "lane_id": lane,
                    "episode_id": str(lane),
                    "elapsed_s": begin,
                },
                {
                    "event": "episode_driven",
                    "lane_id": lane,
                    "episode_id": str(lane),
                    "elapsed_s": end,
                },
            ]
        )
    events.append(
        {"event": "environment_ready", "lane_id": 2, "episode_id": "unfinished", "elapsed_s": 0.5}
    )
    report = concurrency_report(events)
    assert report["active_episodes"]["peak_distinct_lanes"] == 1
    assert report["errors"] == ["Active episode intervals were not closed"]


def test_checkpoint_phase_distributions_preserve_null_missing_and_durability():
    rows = [
        {
            "event": "journal_checkpoint",
            "phase": "before_post",
            "duration_s": 10,
            "lock_wait_s": 6,
            "dispatch_delay_s": 0.25,
            "executor_queue_s": 2,
            "write_flush_fsync_s": 0.5,
            "resume_delay_s": 1.25,
            "durable": True,
        },
        {
            "event": "journal_checkpoint",
            "phase": "before_post",
            "duration_s": 1,
            "lock_wait_s": 1,
            "dispatch_delay_s": None,
            "executor_queue_s": None,
            "write_flush_fsync_s": None,
            "resume_delay_s": None,
            "durable": False,
        },
        {"event": "journal_checkpoint", "phase": "before_post", "durable": None},
        {
            "event": "journal_checkpoint",
            "phase": "completed",
            "duration_s": 0.2,
            "lock_wait_s": 0,
            "dispatch_delay_s": 0,
            "executor_queue_s": 0,
            "write_flush_fsync_s": 0.2,
            "resume_delay_s": 0,
            "durable": True,
        },
        {"event": "journal_checkpoint", "phase": "stopped"},
    ]
    report = timing_report(rows)
    checkpoints = report["journal_checkpoints"]
    assert checkpoints["count"] == 5
    before = checkpoints["by_phase"]["before_post"]
    assert before["count"] == 3
    assert before["durable"] == {"true": 1, "false": 1, "unknown": 1, "missing": 0, "null": 1}
    assert before["timings_s"]["executor_queue_s"] == {
        "n": 1,
        "min": 2,
        "p50": 2,
        "p95": 2,
        "max": 2,
        "unknown": 2,
        "missing": 1,
        "null": 1,
    }
    assert before["timings_s"]["lock_wait_s"]["p50"] == 3.5
    terminal = checkpoints["by_phase"]["completed"]["timings_s"]
    assert terminal["executor_queue_s"]["n"] == 1 and terminal["executor_queue_s"]["p50"] == 0
    unknown = checkpoints["by_phase"]["stopped"]
    assert unknown["durable"]["unknown"] == unknown["durable"]["missing"] == 1
    assert unknown["timings_s"]["duration_s"]["n"] == 0
    assert unknown["timings_s"]["duration_s"]["p50"] is None
    # Raw measurements cannot fabricate old coarse fields or model calls.
    assert report["journal_before_post_s"]["n"] == report["journal_terminal_s"]["n"] == 0
    assert report["successful_model_calls_s"]["n"] == 0


def test_journal_subspans_and_resubmissions_do_not_add_to_logical_model_duration():
    rows = []
    for index, request in enumerate(("failed-1", "failed-2", "success")):
        rows.extend(
            [
                {
                    "event": "journal_checkpoint",
                    "phase": "before_post",
                    "duration_s": 1,
                    "lock_wait_s": 0.6,
                    "dispatch_delay_s": 0.1,
                    "executor_queue_s": 0.2,
                    "write_flush_fsync_s": 0.05,
                    "resume_delay_s": 0.05,
                    "durable": True,
                },
                {
                    "event": "inference_post",
                    "duration_s": 1,
                    "request_id": request,
                    "logical_call_id": "one-call",
                    "attempt_index": index,
                    "response_received": True,
                    "journal_before_post_s": 1,
                },
            ]
        )
        if index < 2:
            rows.extend(
                [
                    {"event": "inference_failed", "request_id": request, "journal_terminal_s": 0.2},
                    {
                        "event": "inference_resubmission",
                        "request_id": request,
                        "logical_call_id": "one-call",
                        "attempt_index": index,
                    },
                ]
            )
    rows.extend(
        [
            {
                "event": "provider_inference",
                "duration_s": 10,
                "request_id": "success",
                "logical_call_id": "one-call",
                "journal_before_post_s": 1,
                "journal_terminal_s": 0.2,
            },
            {"event": "inference", "duration_s": 10.1, "episode_id": "episode"},
            {"event": "inference_attempt_finished", "duration_s": 10.1, "outcome": "success"},
        ]
    )
    report = timing_report(rows)
    assert report["successful_model_calls_s"] == distribution([10.1])
    assert report["all_model_attempts_s"] == distribution([10.1])
    assert report["sdk_post_attempts_s"] == distribution([1, 1, 1])
    assert report["journal_before_post_s"] == distribution([1, 1, 1])
    assert report["journal_terminal_s"] == distribution([0.2, 0.2, 0.2])
    assert report["journal_checkpoints"]["count"] == 3
    assert report["request_counts"]["terminal_failures"] == 2
    assert report["request_counts"]["resubmission_decisions"] == 2
    assert report["request_counts"]["resubmitted_post_attempts"] == 2
    assert report["request_counts"]["post_attempts_without_known_attempt_index"] == 0
    assert report["model_attempt_outcomes"] == {"success": 1}
    data = result()
    before = cohort_gate(data, concurrency=1)
    data["timing"] = report
    assert cohort_gate(data, concurrency=1) == before
    data["runs"][1]["integration_error"] = True
    assert not cohort_gate(data, concurrency=1)["passed"]


def test_resubmission_decision_does_not_imply_replacement_post_and_legacy_is_unknown():
    report = timing_report(
        [
            {"event": "inference_resubmission", "request_id": "original"},
            {
                "event": "inference_post",
                "request_id": "legacy",
                "duration_s": 2,
                "response_received": False,
            },
        ]
    )
    assert report["request_counts"]["resubmission_decisions"] == 1
    assert report["request_counts"]["resubmitted_post_attempts"] == 0
    assert report["request_counts"]["post_attempts_without_known_attempt_index"] == 1
    assert report["journal_checkpoints"] == {"count": 0, "by_phase": {}}


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "1"])
def test_raw_journal_timing_rejects_invalid_measurements(value):
    with pytest.raises(ValueError):
        timing_report(
            [{"event": "journal_checkpoint", "phase": "before_post", "executor_queue_s": value}]
        )


def test_raw_journal_durability_rejects_non_boolean_and_keeps_unknown_phase():
    with pytest.raises(ValueError):
        timing_report([{"event": "journal_checkpoint", "durable": 1}])
    report = timing_report([{"event": "journal_checkpoint", "duration_s": None}])
    assert (
        report["journal_checkpoints"]["by_phase"]["<unknown>"]["timings_s"]["duration_s"]["null"]
        == 1
    )


async def test_real_sdk_resubmission_events_report_one_complete_model_call(tmp_path):
    from hud_dropbear.pooled_agent import PooledInput, PooledModel
    from tests.test_pooled import observation
    from tests.test_pooled_resubmission import RejectionServer

    server, events = RejectionServer(rejections=2), []
    server.resolution_delay = 0.002
    server.journal = tmp_path / "requests.jsonl"

    def emit(name, **fields):
        events.append({"event": name, **fields})

    async with server.provider(
        concurrency=1,
        max_not_admitted_resubmissions=2,
        journal_path=server.journal,
        emit=emit,
    ) as provider:
        model = PooledModel(
            provider,
            slot=0,
            episode_id="episode",
            trace_id="trace",
            seed=197,
            emit=emit,
            fields={"episode_id": "episode"},
        )
        await model.ainfer(PooledInput(observation(), "pick up the bowl", "0" * 64))
    report = timing_report(events)
    logical = next(row for row in events if row["event"] == "inference")
    assert logical["duration_s"] >= 2 * server.resolution_delay
    assert report["successful_model_calls_s"] == distribution([logical["duration_s"]])
    assert report["model_attempt_outcomes"] == {"completed": 1}
    assert report["request_counts"]["post_attempts"] == 3
    assert report["request_counts"]["terminal_failures"] == 2
    assert report["request_counts"]["resubmission_decisions"] == 2
    assert report["request_counts"]["resubmitted_post_attempts"] == 2
    assert report["journal_checkpoints"]["by_phase"]["before_post"]["count"] == 3
    assert report["journal_checkpoints"]["by_phase"]["failed"]["durable"]["true"] == 2
    assert report["journal_checkpoints"]["by_phase"]["completed"]["durable"]["true"] == 1
