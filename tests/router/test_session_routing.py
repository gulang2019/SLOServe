from SLOsServe.router import api_server_ray


def _make_request(request_id: str, session_id: str | None = None):
    payload = {
        "model": "test-model",
        "prompt": [1, 2, 3],
        "max_tokens": 4,
        "stream": True,
        "vllm_xargs": {
            "input_length": 12,
            "output_length": 4,
            "slo_tpot": 0.05,
            "prefill_ddl": 100.0,
            "cached_tokens": 5,
        },
    }
    if session_id is not None:
        payload["vllm_xargs"]["session_id"] = session_id
    return api_server_ray.RequestInstance(
        request_id=request_id,
        payload=payload,
        response_queue=None,
        arrival_time=0.0,
    )


def test_round_robin_router_reuses_session_home_when_sticky_enabled():
    router = api_server_ray.RoundRobinRouter(3, {"sticky_sessions": True})

    first = _make_request("req-0", session_id="session-1")
    router.run([first], [])
    router.note_request_state(first, api_server_ray.RequestState.DECODE_FINISHED)

    second = _make_request("req-1", session_id="session-1")
    router.run([second], [])

    assert first.prefill_device_id == second.prefill_device_id
    assert first.decode_device_id == second.decode_device_id


def test_round_robin_router_keeps_round_robin_for_new_sessions():
    router = api_server_ray.RoundRobinRouter(3, {"sticky_sessions": True})

    first = _make_request("req-0", session_id="session-1")
    second = _make_request("req-1", session_id="session-2")
    router.run([first, second], [])

    assert first.prefill_device_id == 0
    assert second.prefill_device_id == 1
