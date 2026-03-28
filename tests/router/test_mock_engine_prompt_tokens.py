from SLOsServe.router.mock_engine import _normalize_mock_prompt_token_ids


def test_normalize_mock_prompt_token_ids_preserves_token_prompt_ids():
    prompt_token_ids = [11, 22, 33, 44]

    normalized = _normalize_mock_prompt_token_ids(
        {"prompt_token_ids": prompt_token_ids},
        input_length=4,
        request_id="req-1",
    )

    assert normalized == prompt_token_ids


def test_normalize_mock_prompt_token_ids_preserves_list_prompt_ids():
    prompt_token_ids = [101, 102, 103]

    normalized = _normalize_mock_prompt_token_ids(
        prompt_token_ids,
        input_length=3,
        request_id="req-2",
    )

    assert normalized == prompt_token_ids


def test_normalize_mock_prompt_token_ids_is_deterministic_for_text_prompts():
    normalized_a = _normalize_mock_prompt_token_ids(
        "hello",
        input_length=8,
        request_id="req-3",
    )
    normalized_b = _normalize_mock_prompt_token_ids(
        "hello",
        input_length=8,
        request_id="req-3",
    )

    assert normalized_a == normalized_b
    assert len(normalized_a) == 8
    assert all(isinstance(token, int) for token in normalized_a)
