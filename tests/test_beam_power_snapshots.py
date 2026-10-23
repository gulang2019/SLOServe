import csv
import json

import numpy as np
import pytest

from scripts import plot_beam_figure3_paper_style as figure3
from scripts import plot_beam_figure8_paper_style as figure8


@pytest.fixture
def metadata():
    return json.loads(figure3.METADATA.read_text())


def test_portable_snapshots_replay_the_same_measured_shards(metadata):
    edges = np.arange(601, dtype=float)
    paths = [plotter.SNAPSHOT_DIR / name for plotter in (figure3, figure8)
             for name in ('aligned_power.csv', 'provenance.json')]
    before = {path: figure3.file_hash(path) for path in paths}
    cluster = figure3.load_snapshot(figure3.SNAPSHOT_DIR, metadata, edges, [0, 3, 7])
    devices = figure8.load_snapshot(figure8.SNAPSHOT_DIR, metadata, edges)
    for paper_index, beam_index in figure8.PANEL_TO_BEAM.items():
        for column in (0, 1):
            np.testing.assert_array_equal(cluster[beam_index][column], devices[paper_index][column])
        assert devices[paper_index][2]['alignment_is_approximate'] is True
    assert {path: figure3.file_hash(path) for path in paths} == before


def test_snapshot_can_replay_an_explicit_subset_without_fabricating_shards(metadata):
    measurements = figure3.load_snapshot(figure3.SNAPSHOT_DIR, metadata, np.arange(601.), [3])
    assert list(measurements) == [3]
    with pytest.raises(ValueError, match='does not contain BEAM shard'):
        figure3.load_snapshot(figure3.SNAPSHOT_DIR, metadata, np.arange(601.), [4])


def changed_snapshot(plotter, tmp_path, column, value):
    with (plotter.SNAPSHOT_DIR / 'aligned_power.csv').open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields, rows = reader.fieldnames, list(reader)
    rows[0][column] = value
    with (tmp_path / 'aligned_power.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (tmp_path / 'provenance.json').write_text((plotter.SNAPSHOT_DIR / 'provenance.json').read_text())
    return tmp_path


@pytest.mark.parametrize('column,value,error', [
    ('time_s', '1', 'ordered one-second bins'),
    ('vllm_total_power_w', '-1', 'original curves'),
    ('beam_shard_0_raw_power_w', 'nan', 'finite'),
    ('beam_shard_0_scaled_power_w', '-1', 'affine scaling'),
    ('beam_estimated_total_power_w', '-1', 'cluster estimate'),
])
def test_figure3_snapshot_rejects_changed_measurements(metadata, tmp_path, column, value, error):
    path = changed_snapshot(figure3, tmp_path, column, value)
    with pytest.raises(ValueError, match=error):
        figure3.load_snapshot(path, metadata, np.arange(601.), [0, 3, 7])


@pytest.mark.parametrize('column,value,error', [
    ('time_s', '1', 'ordered one-second bins'),
    ('vllm_power_w', '-1', 'original curves'),
    ('beam_raw_pcie_power_w', 'nan', 'finite'),
    ('beam_scaled_power_w', '-1', 'affine scaling'),
    ('beam_source_device_index', '4', 'proxy mapping'),
    ('beam_is_proxy', 'True', 'proxy mapping'),
])
def test_figure8_snapshot_rejects_changed_measurements(metadata, tmp_path, column, value, error):
    path = changed_snapshot(figure8, tmp_path, column, value)
    with pytest.raises(ValueError, match=error):
        figure8.load_snapshot(path, metadata, np.arange(601.))
