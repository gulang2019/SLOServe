from copy import deepcopy

import pytest

from scripts.plot_beam_coder_mixed_unscaled import select_coder_frontier


def point(n, energy, violation):
    return dict(n_device=n, raw_energy_kj=energy, violation_pct_upper=violation)


def test_filter_omits_only_dominated_points_without_mutating_measurements():
    points = [point(6, 381.76, 4.83), point(7, 402.58, 6.95),
              point(8, 439.61, 6.44), point(10, 523.37, 3.30)]
    original = deepcopy(points)
    kept, omitted = select_coder_frontier(points)
    assert [p['n_device'] for p in kept] == [6, 10]
    assert [p['n_device'] for p in omitted] == [7, 8]
    assert all(p['dominated_by_n_devices'] == [6] for p in omitted)
    assert points == original


def test_identical_points_are_retained_and_equal_coordinate_dominance_is_strict():
    points = [point(1, 100, 2), point(2, 100, 2), point(3, 100, 3), point(4, 120, 2)]
    kept, omitted = select_coder_frontier(points)
    assert [p['n_device'] for p in kept] == [1, 2]
    assert [p['n_device'] for p in omitted] == [3, 4]


@pytest.mark.parametrize('invalid', [float('nan'), float('inf'), -1])
@pytest.mark.parametrize('field', ['raw_energy_kj', 'violation_pct_upper'])
def test_invalid_coordinates_are_rejected(field, invalid):
    entry = point(1, 100, 2)
    entry[field] = invalid
    with pytest.raises(ValueError, match='finite and nonnegative'):
        select_coder_frontier([entry])
