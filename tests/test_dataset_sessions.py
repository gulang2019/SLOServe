from Dataset.download_dataset import _build_sharegpt_chat_requests


def test_build_sharegpt_chat_requests_preserves_session_order_and_ids():
    dataset = [
        {
            "conversation": [
                {"role": "user", "content": "u1"},
                {"role": "assistant", "content": "a1"},
                {"role": "user", "content": "u2"},
                {"role": "assistant", "content": "a2"},
            ]
        },
        {
            "conversation": [
                {"role": "user", "content": "u3"},
                {"role": "assistant", "content": "a3"},
            ]
        },
    ]

    requests = _build_sharegpt_chat_requests(
        dataset,
        count_length_fn=len,
        session_prefix="session",
    )

    assert [req.session_id for req in requests] == [
        "session-0",
        "session-0",
        "session-1",
    ]
    assert requests[0].cached_length == 0
    assert requests[1].cached_length > requests[0].cached_length
    assert requests[0].answer == "a1"
    assert requests[1].answer == "a2"
    assert requests[2].answer == "a3"
