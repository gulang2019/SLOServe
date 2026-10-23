from datetime import datetime, timedelta

import numpy as np
import pytest

from scripts.plot_beam_figure8_mixed import (
    estimate_replay_start,
    interval_mean_power,
    scale_power,
)


def test_power_scaler_matches_pcie_and_sxm_endpoints():
    scaled = scale_power([52.0, 176.0, 300.0])
    assert isinstance(scaled, np.ndarray)
    np.testing.assert_allclose(scaled, [70.0, 235.0, 400.0])


def test_power_scaler_is_linear_without_clipping_and_preserves_missing_data():
    values = np.array([40.0, 320.0, np.nan])
    np.testing.assert_allclose(
        scale_power(values),
        70.0 + (values - 52.0) * 330.0 / 248.0,
        equal_nan=True,
    )


def test_interval_power_uses_energy_differences_not_sample_power_average():
    # The first half-second uses 100 W and the second uses 200 W.
    result = interval_mean_power(
        [0.0, 0.5, 1.0], [1000.0, 1050.0, 1150.0],
        [0.0, 0.25, 0.75, 1.0],
    )
    np.testing.assert_allclose(result, [100.0, 150.0, 200.0])


def test_interval_power_conserves_energy_across_nonuniform_bins():
    times = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    energy = [100.0, 120.0, 145.0, 205.0, 255.0, 325.0]
    edges = np.array([0.0, 0.1, 0.7, 1.25, 2.0])
    power = interval_mean_power(times, energy, edges)
    assert np.isfinite(power).all()
    assert np.dot(power, np.diff(edges)) == pytest.approx(225.0)


def test_interval_power_never_extrapolates_partial_coverage_bins():
    power = interval_mean_power(
        [0.2, 0.5, 1.0, 1.8], [20.0, 50.0, 100.0, 180.0],
        [0.0, 0.5, 1.0, 2.0],
    )
    np.testing.assert_allclose(power, [np.nan, 100.0, np.nan], equal_nan=True)


def test_interval_power_masks_large_gaps_but_not_adjacent_covered_bins():
    power = interval_mean_power(
        [0.0, 0.5, 2.0, 2.5], [0.0, 50.0, 200.0, 250.0],
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        max_gap_s=1.0,
    )
    np.testing.assert_allclose(
        power, [100.0, np.nan, np.nan, np.nan, 100.0], equal_nan=True,
    )


def test_interval_power_masks_a_bin_spanning_an_internal_large_gap():
    power = interval_mean_power(
        [0.0, 0.5, 2.0, 2.5], [0.0, 50.0, 200.0, 250.0],
        [0.0, 2.5], max_gap_s=1.0,
    )
    assert np.isnan(power[0])


def test_interval_power_accepts_gap_equal_to_limit():
    np.testing.assert_allclose(
        interval_mean_power([0.0, 1.0], [0.0, 100.0], [0.0, 1.0]),
        [100.0],
    )


@pytest.mark.parametrize(
    "times, energy, edges",
    [
        ([0.0, 0.0], [0.0, 1.0], [0.0, 1.0]),
        ([1.0, 0.0], [0.0, 1.0], [0.0, 1.0]),
        ([0.0, 1.0], [1.0, 0.0], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, 1.0], [0.0, 0.0]),
        ([0.0, 1.0], [0.0, 1.0], [1.0, 0.0]),
        ([0.0, np.nan], [0.0, 1.0], [0.0, 1.0]),
        ([0.0, np.inf], [0.0, 1.0], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, np.nan], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, np.inf], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, 1.0], [0.0, np.nan]),
        ([0.0, 1.0], [0.0, 1.0], [0.0, np.inf]),
        ([0.0, 1.0], [0.0], [0.0, 1.0]),
        ([], [], [0.0, 1.0]),
        ([0.0], [0.0], [0.0, 1.0]),
        ([0.0, 1.0], [0.0, 1.0], [0.0]),
    ],
)
def test_interval_power_rejects_invalid_samples_or_edges(times, energy, edges):
    with pytest.raises(ValueError):
        interval_mean_power(times, energy, edges)


@pytest.mark.parametrize("gap", [0.0, -1.0, np.nan])
def test_interval_power_rejects_invalid_gap_limit(gap):
    with pytest.raises(ValueError):
        interval_mean_power([0.0, 1.0], [0.0, 100.0], [0.0, 1.0], gap)


def test_replay_start_prefers_final_batch_and_latest_client_completion():
    result = {
        "date": "20261023-100000",
        "duration": 100.0,
        "entry_times": [1.0, 80.0, 40.0],
        "e2el": [4.0, 3.0, 50.0],
    }
    final_batch = datetime(2026, 10, 23, 9, 0, 0, 250000)
    # Latest completion is 40 + 50, not the last arrival or max(e2el).
    assert estimate_replay_start(result, final_batch_time=final_batch) == (
        final_batch - timedelta(seconds=90.0)
    )


def test_replay_start_fallback_uses_half_second_timestamp_midpoint():
    result = {"date": "20261023-100000", "duration": 40.25}
    assert estimate_replay_start(result) == datetime(
        2026, 10, 23, 9, 59, 20, 250000,
    )
