from scripts import analyze_request_concurrency as arc


def test_build_request_intervals_spans_rescheduling_until_final_finish():
    events = [
        {"event_type": "arrival-router", "request_id": "r0", "timestamp": 0.0},
        {"event_type": "arrival", "request_id": "r0", "timestamp": 0.1, "device_id": 0},
        {
            "event_type": "finish",
            "request_id": "r0",
            "timestamp": 0.2,
            "finish_reason": "rejected-arrival-oom",
            "device_id": 0,
        },
        {"event_type": "rescheduling", "request_id": "r0", "timestamp": 0.25},
        {"event_type": "arrival", "request_id": "r0", "timestamp": 0.3, "device_id": 1},
        {
            "event_type": "finish",
            "request_id": "r0",
            "timestamp": 1.0,
            "finish_reason": "length",
            "device_id": 1,
        },
        {"event_type": "arrival-router", "request_id": "r1", "timestamp": 0.4},
        {
            "event_type": "finish",
            "request_id": "r1",
            "timestamp": 0.9,
            "finish_reason": "router_rejection",
            "device_id": -1,
        },
    ]

    intervals, stats = arc.build_request_intervals(events, start_event="arrival-router")

    assert stats["requests_seen"] == 2
    assert stats["requests_analyzed"] == 2
    assert stats["requests_with_retries"] == 1

    r0 = next(interval for interval in intervals if interval.request_id == "r0")
    assert r0.start_time == 0.0
    assert r0.finish_time == 1.0
    assert r0.duration_s == 1.0
    assert r0.finish_reason == "length"
    assert r0.terminal_status == "completed"
    assert r0.finish_events == 2
    assert r0.retry_count == 1
    assert r0.reschedule_events == 1
    assert r0.engine_arrival_events == 2

    r1 = next(interval for interval in intervals if interval.request_id == "r1")
    assert r1.terminal_status == "rejected"
    assert r1.retry_count == 0


def test_build_concurrency_step_rows_processes_finishes_before_starts_at_same_timestamp():
    intervals = [
        arc.RequestInterval(
            request_id="r0",
            start_time=0.0,
            finish_time=1.0,
            duration_s=1.0,
            finish_reason="length",
            terminal_status="completed",
            finish_events=1,
            retry_count=0,
            reschedule_events=0,
            engine_arrival_events=1,
            started_from="arrival-router",
        ),
        arc.RequestInterval(
            request_id="r1",
            start_time=1.0,
            finish_time=2.0,
            duration_s=1.0,
            finish_reason="length",
            terminal_status="completed",
            finish_events=1,
            retry_count=0,
            reschedule_events=0,
            engine_arrival_events=1,
            started_from="arrival-router",
        ),
    ]

    rows, start_time, end_time = arc.build_concurrency_step_rows(intervals)

    assert start_time == 0.0
    assert end_time == 2.0
    assert rows == [
        {
            "timestamp": 0.0,
            "concurrency_before": 0,
            "finishes": 0,
            "starts": 1,
            "concurrency_after": 1,
        },
        {
            "timestamp": 1.0,
            "concurrency_before": 1,
            "finishes": 1,
            "starts": 1,
            "concurrency_after": 1,
        },
        {
            "timestamp": 2.0,
            "concurrency_before": 1,
            "finishes": 1,
            "starts": 0,
            "concurrency_after": 0,
        },
    ]


def test_build_window_rows_integrates_average_and_peak_concurrency():
    intervals = [
        arc.RequestInterval(
            request_id="r0",
            start_time=0.0,
            finish_time=2.0,
            duration_s=2.0,
            finish_reason="length",
            terminal_status="completed",
            finish_events=1,
            retry_count=0,
            reschedule_events=0,
            engine_arrival_events=1,
            started_from="arrival-router",
        ),
        arc.RequestInterval(
            request_id="r1",
            start_time=0.5,
            finish_time=1.5,
            duration_s=1.0,
            finish_reason="length",
            terminal_status="completed",
            finish_events=1,
            retry_count=0,
            reschedule_events=0,
            engine_arrival_events=1,
            started_from="arrival-router",
        ),
    ]

    step_rows, start_time, end_time = arc.build_concurrency_step_rows(intervals)
    window_rows = arc.build_window_rows(
        step_rows,
        start_time=start_time,
        end_time=end_time,
        window_size=1.0,
    )
    summary = arc.summarize_concurrency(
        step_rows,
        start_time=start_time,
        end_time=end_time,
    )

    assert window_rows == [
        {
            "window_start_s": 0.0,
            "window_end_s": 1.0,
            "avg_concurrency": 1.5,
            "max_concurrency": 2,
            "started_requests": 2,
            "finished_requests": 0,
        },
        {
            "window_start_s": 1.0,
            "window_end_s": 2.0,
            "avg_concurrency": 1.5,
            "max_concurrency": 2,
            "started_requests": 0,
            "finished_requests": 2,
        },
    ]
    assert summary["max_concurrency"] == 2
    assert summary["time_weighted_mean"] == 1.5
    assert summary["time_weighted_p50"] == 1.0
    assert summary["time_weighted_p90"] == 2.0
