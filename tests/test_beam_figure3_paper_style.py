from copy import deepcopy

from matplotlib.figure import Figure
import numpy as np
import pytest

from scripts import plot_beam_figure3_paper_style as plotter
from scripts.plot_beam_figure8_mixed import scale_power


def test_cluster_power_extrapolates_per_bin_mean_not_sum_of_measured_shards():
    shards = {0: [70.0, 100.0], 3: [100.0, 130.0], 7: [130.0, 190.0]}
    original = deepcopy(shards)
    np.testing.assert_allclose(
        plotter.estimate_cluster_power(shards), [800.0, 1120.0],
    )
    assert shards == original


def test_cluster_power_supports_one_shard_and_explicit_replica_count():
    np.testing.assert_allclose(
        plotter.estimate_cluster_power({3: [70.0, 200.0]}, n_device=4),
        [280.0, 800.0],
    )


def test_affine_scaling_precedes_cluster_extrapolation():
    raw_shards = {
        0: np.array([52.0, 176.0, 300.0]),
        3: np.array([52.0, 300.0, 300.0]),
        7: np.array([52.0, 52.0, 300.0]),
    }
    scaled_shards = {shard: scale_power(raw) for shard, raw in raw_shards.items()}
    result = plotter.estimate_cluster_power(scaled_shards)
    # The idle calibration is 70 W per GPU, so the cluster anchor is 8 * 70 W.
    # It is not a hard clipping floor: the scaler deliberately does not clip.
    np.testing.assert_allclose(result, [8 * 70.0, 8 * 235.0, 8 * 400.0])
    incorrect_scale_after_extrapolation = scale_power(
        8 * np.mean(np.stack(list(raw_shards.values())), axis=0),
    )
    assert not np.allclose(result, incorrect_scale_after_extrapolation)


def test_cluster_power_does_not_add_an_idle_floor_or_clip_scaled_values():
    scaled = scale_power([40.0, 320.0])
    result = plotter.estimate_cluster_power({0: scaled})
    np.testing.assert_allclose(result, 8 * scaled)
    assert result[0] < 8 * 70.0
    assert result[1] > 8 * 400.0


@pytest.mark.parametrize("n_device", [0, -1, 1.5, True, False, "8", None])
def test_cluster_power_rejects_invalid_replica_counts(n_device):
    with pytest.raises(ValueError, match="positive integer"):
        plotter.estimate_cluster_power({0: [70.0]}, n_device=n_device)


def test_cluster_power_requires_at_least_one_shard():
    with pytest.raises(ValueError, match="At least one measured shard"):
        plotter.estimate_cluster_power({})


@pytest.mark.parametrize(
    "shards",
    [
        {0: []},
        {0: 70.0},
        {0: [[70.0, 80.0]]},
        {0: [70.0, 80.0], 3: [90.0]},
        {0: [70.0, 80.0], 3: []},
        {0: [70.0, np.nan], 3: [80.0, 90.0]},
        {0: [np.inf]},
        {0: [-np.inf]},
    ],
)
def test_cluster_power_rejects_missing_or_misaligned_bins(shards):
    with pytest.raises(ValueError, match="equal-length, finite, complete"):
        plotter.estimate_cluster_power(shards)


@pytest.fixture
def metadata():
    return {
        "figure": "timeline_comparison",
        "schema_version": 2,
        "traces": [
            {
                "label": label,
                "n_device": 8,
                "power": {
                    "time_s": [0.0, 1.0, 2.0],
                    "values_w": total,
                    "per_device_values_w": {"0": per_device},
                    "window_s": 1.0,
                    "source": "measured",
                },
                "arrivals": {"time_s": [0.0, 1.0, 2.0], "values": arrivals},
            }
            for label, total, per_device, arrivals in [
                ("Baseline", [1000.0, 1200.0, 900.0], [125.0, 150.0, 112.5], [2, 6, 3]),
                ("Ours", [600.0, 700.0, 500.0], [75.0, 87.5, 62.5], [1, 7, 3]),
            ]
        ],
    }


def test_original_power_selects_cluster_totals_and_requested_label(metadata):
    original = deepcopy(metadata)
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(
        plotter.original_total_power(metadata, "Baseline", edges),
        [1000.0, 1200.0, 900.0],
    )
    np.testing.assert_array_equal(
        plotter.original_total_power(metadata, "Ours", edges[1:]),
        [700.0, 500.0],
    )
    assert metadata == original


def test_original_power_requires_eight_device_source(metadata):
    metadata["traces"][0]["n_device"] = 4
    with pytest.raises(ValueError, match="eight-device"):
        plotter.original_total_power(metadata, "Baseline", np.arange(4.0))


@pytest.mark.parametrize(
    "times, values",
    [
        ([0.0, 1.0, 2.0], [1000.0, 1200.0]),
        ([0.0, 1.0], [1000.0, 1200.0]),
        ([0.0, 1.5, 2.0], [1000.0, 1200.0, 900.0]),
        ([], []),
    ],
)
def test_original_power_rejects_unmatched_bins(metadata, times, values):
    metadata["traces"][0]["power"].update(time_s=times, values_w=values)
    with pytest.raises(ValueError, match="matching one-second bins"):
        plotter.original_total_power(metadata, "Baseline", np.arange(4.0))


def test_render_preserves_original_artists_arrivals_and_metadata(
    metadata, monkeypatch, tmp_path,
):
    original_metadata = deepcopy(metadata)
    real_renderer = plotter.renderer.render_timeline_comparison_metadata
    snapshot = {}
    saves = []

    def capture_original(*args, **kwargs):
        fig, axes = real_renderer(*args, **kwargs)
        snapshot["axes"] = axes
        snapshot["lines"] = [
            [(line, line.get_xdata().copy(), line.get_ydata().copy(),
              line.get_color(), line.get_linestyle(), line.get_linewidth(),
              line.get_label()) for line in axis.lines]
            for axis in axes
        ]
        return fig, axes

    monkeypatch.setattr(
        plotter.renderer, "render_timeline_comparison_metadata", capture_original,
    )
    # Exercise the real renderer and overlay without writing PNG/PDF artifacts.
    monkeypatch.setattr(Figure, "savefig", lambda self, path, **kwargs: saves.append(str(path)))
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    cluster_power = np.array([720.0, 800.0, 900.0])
    fig, axes = plotter.render_figure(
        metadata, cluster_power, edges, tmp_path / "figure3",
    )
    try:
        assert axes is snapshot["axes"]
        assert [len(axis.lines) for axis in axes] == [1, 3]
        for axis, original_lines in zip(axes, snapshot["lines"]):
            for index, (line, x, y, color, style, width, label) in enumerate(original_lines):
                assert axis.lines[index] is line
                np.testing.assert_array_equal(line.get_xdata(), x)
                np.testing.assert_array_equal(line.get_ydata(), y)
                assert (line.get_color(), line.get_linestyle(), line.get_linewidth(),
                        line.get_label()) == (color, style, width, label)
        np.testing.assert_array_equal(axes[0].lines[0].get_ydata(), [2, 6, 3])
        np.testing.assert_array_equal(axes[1].lines[0].get_ydata(), [1000.0, 1200.0, 900.0])
        np.testing.assert_array_equal(axes[1].lines[1].get_ydata(), [600.0, 700.0, 500.0])
        np.testing.assert_array_equal(axes[1].lines[2].get_xdata(), edges)
        np.testing.assert_array_equal(axes[1].lines[2].get_ydata(), [720.0, 800.0, 900.0, 900.0])
        assert axes[1].lines[2].get_drawstyle() == "steps-post"
        assert len(fig.legends) == 1
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            "vLLM", "SLO-Packer", "BEAM",
        ]
        assert metadata == original_metadata
        assert saves[-2:] == [str(tmp_path / "figure3.png"), str(tmp_path / "figure3.pdf")]
    finally:
        plotter.plt.close(fig)
