import numpy as np
import pytest

from scripts.plot_beam_slo_timeline import evaluate_request, rebin_counts


def request_row(**changes):
    row = {
        "success": True,
        "ttft_s": 0.05,
        "itls_s": [0.01, 0.01],
        "actual_output_tokens": 3,
        "ttft_budget_s": 0.2,
    }
    row.update(changes)
    return row


def evaluate_without_allowance_change(row, tpot_s=0.02):
    return evaluate_request(
        row, saved_allowance_s=0.0, total_allowance_s=0.0, tpot_s=tpot_s,
    )


def test_cumulative_deadlines_can_pass_when_individual_itls_exceed_tpot():
    row = request_row(ttft_s=0.05, itls_s=[0.08, 0.08])
    assert np.percentile(row["itls_s"], 90) > 0.02
    result = evaluate_without_allowance_change(row)
    # Completion times .13/.21 meet cumulative deadlines .22/.24.
    assert result["violation_lower"] is False
    assert result["violation_upper"] is False
    assert result["ttft_violation"] is False
    assert result["missing_token_timestamps"] == 0


def test_cumulative_decode_deadline_failure_is_separate_from_ttft():
    result = evaluate_without_allowance_change(
        request_row(ttft_s=0.05, itls_s=[0.20, 0.20]),
    )
    assert result["ttft_violation"] is False
    assert result["violation_lower"] is True
    assert result["violation_upper"] is True


def test_decode_rule_uses_p90_lateness_not_worst_token_lateness():
    row = request_row(itls_s=[0.01] * 9 + [0.30], actual_output_tokens=11)
    lateness = (row["ttft_s"] + np.cumsum(row["itls_s"])
                - row["ttft_budget_s"] - np.arange(1, 11) * 0.02)
    assert max(lateness) > 0
    assert np.percentile(lateness, 90) < 0
    result = evaluate_without_allowance_change(row)
    assert result["violation_lower"] is False
    assert result["violation_upper"] is False


def test_ttft_failure_always_fails_even_if_later_deadlines_are_met():
    result = evaluate_without_allowance_change(
        request_row(ttft_s=0.21, itls_s=[0.0, 0.0]),
    )
    assert result["ttft_violation"] is True
    assert result["violation_lower"] is True
    assert result["violation_upper"] is True


def test_total_allowance_replaces_saved_allowance_instead_of_adding_to_it():
    result = evaluate_request(
        request_row(ttft_s=0.16, itls_s=[], actual_output_tokens=1,
                    ttft_budget_s=0.12),
        saved_allowance_s=0.02, total_allowance_s=0.05,
    )
    assert result["budget_s"] == pytest.approx(0.15)
    assert result["ttft_violation"] is True
    assert result["violation_lower"] is True
    assert result["violation_upper"] is True


def test_missing_token_timestamps_produce_optimistic_and_pessimistic_bounds():
    result = evaluate_without_allowance_change(
        request_row(ttft_s=0.02, itls_s=[0.12, 0.01],
                    actual_output_tokens=5, ttft_budget_s=0.10),
    )
    # Observed events are .02/.14/.15, but there are five tokens.
    # Missing at first event: .02/.02/.02/.14/.15 passes.
    # Missing at last event: .02/.14/.15/.15/.15 fails.
    assert result["missing_token_timestamps"] == 2
    assert result["ttft_violation"] is False
    assert result["violation_lower"] is False
    assert result["violation_upper"] is True


def test_all_tokens_in_first_event_have_identical_bounds():
    result = evaluate_without_allowance_change(
        request_row(itls_s=[], actual_output_tokens=4),
    )
    assert result["missing_token_timestamps"] == 3
    assert result["violation_lower"] is False
    assert result["violation_upper"] is False


@pytest.mark.parametrize("ttft, fails", [(0.05, False), (0.3, True)])
def test_single_output_token_has_no_decode_deadline(ttft, fails):
    result = evaluate_without_allowance_change(
        request_row(ttft_s=ttft, itls_s=[], actual_output_tokens=1),
    )
    assert result["missing_token_timestamps"] == 0
    assert result["ttft_violation"] is fails
    assert result["violation_lower"] is fails
    assert result["violation_upper"] is fails


def test_exact_deadline_equality_is_satisfied():
    result = evaluate_without_allowance_change(
        request_row(ttft_s=0.125, itls_s=[0.25], actual_output_tokens=2,
                    ttft_budget_s=0.25),
        tpot_s=0.125,
    )
    assert result["violation_lower"] is False
    assert result["violation_upper"] is False


def test_failed_request_is_a_violation_regardless_of_fast_timings():
    result = evaluate_without_allowance_change(request_row(success=False))
    assert result["violation_lower"] is True
    assert result["violation_upper"] is True


def test_failed_request_without_output_or_timings_still_counts_as_violation():
    result = evaluate_without_allowance_change(
        request_row(success=False, ttft_s=None, itls_s=[], actual_output_tokens=0),
    )
    assert result["violation_lower"] is True
    assert result["violation_upper"] is True


@pytest.mark.parametrize("output_tokens", [0, -1, 1.5, np.nan, np.inf])
def test_invalid_output_token_count_rejected(output_tokens):
    with pytest.raises(ValueError):
        evaluate_without_allowance_change(
            request_row(actual_output_tokens=output_tokens, itls_s=[]),
        )


def test_more_observed_events_than_tokens_rejected():
    with pytest.raises(ValueError):
        evaluate_without_allowance_change(request_row(actual_output_tokens=2))


@pytest.mark.parametrize("gap", [-0.01, np.nan, np.inf])
def test_invalid_inter_token_gaps_rejected(gap):
    with pytest.raises(ValueError):
        evaluate_without_allowance_change(request_row(itls_s=[gap, 0.01]))


@pytest.mark.parametrize("field", ["ttft_s", "ttft_budget_s"])
@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_request_timing_rejected(field, value):
    with pytest.raises(ValueError):
        evaluate_without_allowance_change(request_row(**{field: value}))


def test_rebin_counts_sums_bins_and_conserves_total():
    times = np.arange(0, 60, 5)
    counts = np.arange(1, 13)
    edges, rebinned = rebin_counts(
        times, counts, source_window_s=5, target_window_s=30, duration_s=60,
    )
    np.testing.assert_array_equal(edges, [0, 30, 60])
    np.testing.assert_array_equal(rebinned, [21, 57])
    assert rebinned.sum() == counts.sum()


def test_rebin_counts_keeps_zero_bins_for_complete_window():
    edges, counts = rebin_counts(np.arange(0, 600, 5), np.zeros(120))
    np.testing.assert_array_equal(edges, np.arange(0, 601, 30))
    np.testing.assert_array_equal(counts, np.zeros(20))


def test_rebin_counts_can_preserve_original_resolution():
    edges, counts = rebin_counts(
        [0, 5, 10], [2, 0, 7],
        source_window_s=5, target_window_s=5, duration_s=15,
    )
    np.testing.assert_array_equal(edges, [0, 5, 10, 15])
    np.testing.assert_array_equal(counts, [2, 0, 7])


@pytest.mark.parametrize(
    "times, counts",
    [
        ([0, 5, 10], [1, 2, 3]),  # Missing final source bin.
        ([0, 5, 10, 15], [1, 2, 3]),
        ([0, 5, 5, 15], [1, 2, 3, 4]),
        ([0, 10, 5, 15], [1, 2, 3, 4]),
        ([0, 5, 11, 15], [1, 2, 3, 4]),
        ([5, 10, 15, 20], [1, 2, 3, 4]),
        ([0, 5, 10, 15], [1, -1, 3, 4]),
        ([0, 5, 10, 15], [1, np.nan, 3, 4]),
        ([0, 5, 10, 15], [1, np.inf, 3, 4]),
        ([0, 5, np.nan, 15], [1, 2, 3, 4]),
    ],
)
def test_rebin_counts_rejects_incomplete_misaligned_or_invalid_bins(times, counts):
    with pytest.raises(ValueError):
        rebin_counts(times, counts, source_window_s=5, target_window_s=10,
                     duration_s=20)


@pytest.mark.parametrize(
    "source, target, duration",
    [(0, 10, 20), (-5, 10, 20), (5, 0, 20), (5, 7, 20), (5, 10, 25)],
)
def test_rebin_counts_requires_positive_aligned_windows(source, target, duration):
    with pytest.raises(ValueError):
        rebin_counts([0, 5, 10, 15], [1, 2, 3, 4],
                     source_window_s=source, target_window_s=target,
                     duration_s=duration)
