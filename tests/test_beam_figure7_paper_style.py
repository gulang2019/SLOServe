from copy import deepcopy
import csv
import json

import pytest

from plots import common
from scripts import plot_beam_figure7_paper_style as plotter


@pytest.fixture(scope="module")
def merged():
    return plotter.build_merge()


def test_actual_snapshot_has_171_original_38_included_and_two_omitted_beam_points(merged):
    rows = merged["exported_points"]
    original = [row for row in rows if row["method_key"] != plotter.BEAM_KEY]
    beam = [row for row in rows if row["method_key"] == plotter.BEAM_KEY]
    assert len(original) == 171
    assert len(beam) == 40
    assert sum(row["included_in_curve"] for row in beam) == 38
    assert all(row["included_in_curve"] for row in original)
    assert {(row["workload"], row["n_device"]) for row in beam if not row["included_in_curve"]} == {
        ("code_7b", 7), ("code_7b", 8),
    }
    assert sum(len(points) for panel in merged["panels"] for points in panel["series"].values()) == 209
    assert sum(row["included_in_curve"] and row["within_paper_x_range"] for row in beam) == 18


def test_every_original_point_and_method_order_is_unchanged(merged):
    for panel in merged["panels"]:
        info = merged["original_sources"][panel["workload"]]
        series, order, _ = plotter.load_paper_series(plotter.ROOT / info["source"])
        assert [method for method in panel["order"] if method != plotter.BEAM_KEY] == order
        assert {method: points for method, points in panel["series"].items()
                if method != plotter.BEAM_KEY} == series
        exported = [row for row in merged["exported_points"]
                    if row["workload"] == panel["workload"] and row["method_key"] != plotter.BEAM_KEY]
        expected = [(method, point["n_device"], point["slo_violation_rate"], point["energy_consumption"])
                    for method in order for point in series[method]]
        assert [(row["method_key"], row["n_device"], row["violation_pct"], row["energy_kj"])
                for row in exported] == expected


def test_beam_energy_variants_and_slo_values_match_previous_sources(merged):
    prior = {(point["workload"], point["n_device"]): point
             for point in merged["source"]["beam_points"]}
    for row in merged["exported_points"]:
        if row["method_key"] != plotter.BEAM_KEY:
            continue
        point = prior[row["workload"], row["n_device"]]
        raw = row["workload"] in ("code_7b", "mixed_7b")
        assert row["energy_kj"] == point["raw_energy_kj" if raw else "energy_kj"]
        assert row["energy_variant"] == ("unscaled_pcie" if raw else "affine_scaled")
        assert row["violation_pct"] == point["violation_pct_upper"]
        assert row["violation_pct_lower"] == point["violation_pct_lower"]
        assert row["sources"] == ";".join(point["sources"])
        assert row["method"] == "BEAM"
    for panel in merged["panels"]:
        by_n = {row["n_device"]: row for row in merged["exported_points"]
                if row["workload"] == panel["workload"] and row["method_key"] == plotter.BEAM_KEY}
        for point in panel["series"].get(plotter.BEAM_KEY, []):
            assert point["energy_consumption"] == by_n[point["n_device"]]["energy_kj"]
            assert point["slo_violation_rate"] == by_n[point["n_device"]]["violation_pct"]


def test_coder_and_mixed_match_the_prior_unscaled_frontier_csv(merged):
    with (plotter.UNSCALED_PREVIEW.parent / "beam_points.csv").open(newline="") as handle:
        prior = {(row["workload"], int(row["n_device"])): row for row in csv.DictReader(handle)}
    actual = [row for row in merged["exported_points"] if row["method_key"] == plotter.BEAM_KEY
              and row["workload"] in ("code_7b", "mixed_7b")]
    assert len(actual) == len(prior) == 24
    for row in actual:
        old = prior[row["workload"], row["n_device"]]
        assert row["energy_kj"] == float(old["raw_energy_kj"])
        assert row["violation_pct"] == float(old["violation_pct"])
        assert row["included_in_curve"] == (old["included_in_curve"] == "True")
        assert row["omission_reason"] == old["omission_reason"]
    assert [point["n_device"] for point in merged["omitted_coder_points"]] == [7, 8]
    assert all(point["dominated_by_n_devices"] == [6] for point in merged["omitted_coder_points"])


def test_multiround_remains_25ms_evaluation_with_50ms_recorded_targets(merged):
    assert merged["source"]["beam_multiround_evaluation_tpot_ms"] == 25.0
    sessions = [point for point in merged["source"]["beam_points"] if point["workload"] == "multiround_chat"]
    assert [point["n_device"] for point in sessions] == [8, 12, 16]
    for point in sessions:
        assert point["eval_tpot_s"] == .025
        assert point["original_csv_tpot_s"] == .05
        assert all(target["tbt_s"] == target["tpot_s"] == .05 for target in point["control_targets"])
        assert all(run["expected_cache_adjusted"] for run in point["per_shard"])
    assert sessions[1]["counts"]["lower"] == 6
    assert sessions[1]["counts"]["upper"] == 8


def test_panel_order_and_absence_of_fabricated_32b_measurements(merged):
    assert [panel["workload"] for panel in merged["panels"]] == [
        "chat_7b", "code_7b", "mixed_7b", "chat_32b", "multiround_chat", "large_scale_chat",
    ]
    panel = merged["panels"][3]
    assert plotter.BEAM_KEY not in panel["series"]
    assert plotter.BEAM_KEY not in panel["order"]
    assert not any(row["workload"] == "chat_32b" and row["method"] == "BEAM"
                   for row in merged["exported_points"])


def test_build_merge_never_mutates_snapshot_files_or_loaded_source(merged):
    paths = [plotter.SOURCE_PREVIEW, plotter.UNSCALED_PREVIEW,
             plotter.UNSCALED_PREVIEW.parent / "beam_points.csv"]
    paths.extend(plotter.ROOT / info["source"] for info in merged["original_sources"].values())
    before = {path: plotter.file_hash(path) for path in paths}
    source = json.loads(plotter.SOURCE_PREVIEW.read_text())
    result = plotter.build_merge()
    assert result["source"] == source == merged["source"]
    assert {path: plotter.file_hash(path) for path in paths} == before


def test_plain_beam_style_is_temporary_and_restores_existing_registration(monkeypatch):
    monkeypatch.setitem(common.LABEL_MAP, plotter.BEAM_KEY, "previous label")
    monkeypatch.setitem(common.COLOR_MAP, plotter.BEAM_KEY, "#123456")
    monkeypatch.setitem(common.MARKER_MAP, plotter.BEAM_KEY, "o")
    maps = (common.LABEL_MAP, common.COLOR_MAP, common.MARKER_MAP)
    before = [dict(mapping) for mapping in maps]
    with pytest.raises(RuntimeError, match="render failure"):
        with plotter.beam_style():
            assert common.get_method_label(plotter.BEAM_KEY) == "BEAM"
            assert common.get_method_color(plotter.BEAM_KEY) == plotter.BEAM_COLOR
            assert common.get_method_marker(plotter.BEAM_KEY) == "D"
            raise RuntimeError("render failure")
    assert [dict(mapping) for mapping in maps] == before


class ReferenceOnlyAxis:
    """Any attempted dot, extension, or annotation would lack its axis method."""

    def __init__(self):
        self.lines = []

    def axvline(self, target, **kwargs):
        self.lines.append((target, kwargs))


@pytest.mark.parametrize("cap, enabled, expected", [
    (10, True, [1.0, 5.0]), (1, True, [1.0]), (0, True, []), (10, False, []),
])
def test_annotation_hook_draws_only_requested_reference_lines(cap, enabled, expected):
    original = plotter.paper._add_threshold_gap_annotations
    axis = ReferenceOnlyAxis()
    with plotter.reference_lines_only():
        assert plotter.paper._add_threshold_gap_annotations is not original
        result = plotter.paper._add_threshold_gap_annotations(
            axis, max_slo_violation=cap, show_target_reference_lines=enabled,
            series_by_method={}, method_order=[], y_key="energy_consumption",
        )
    assert result == []
    assert [target for target, _ in axis.lines] == expected
    assert all(style == dict(color=plotter.paper.ANNOTATION_COLOR, linestyle=":",
                             linewidth=1.0, zorder=0) for _, style in axis.lines)
    assert plotter.paper._add_threshold_gap_annotations is original


def test_annotation_hook_is_restored_after_render_failure():
    original = plotter.paper._add_threshold_gap_annotations
    with pytest.raises(RuntimeError, match="render failure"):
        with plotter.reference_lines_only():
            raise RuntimeError("render failure")
    assert plotter.paper._add_threshold_gap_annotations is original


def test_render_wrapper_uses_original_renderer_and_no_savings_artifacts(merged, monkeypatch, tmp_path):
    calls, legends = [], []
    original_hook = plotter.paper._add_threshold_gap_annotations
    original_merged = deepcopy(merged)

    def render(**kwargs):
        calls.append(kwargs)
        assert common.get_method_label(plotter.BEAM_KEY) == "BEAM"
        axis = ReferenceOnlyAxis()
        assert plotter.paper._add_threshold_gap_annotations(
            axis, max_slo_violation=10, series_by_method=kwargs["series_by_method"],
            method_order=kwargs["method_order"], y_key="energy_consumption",
        ) == []
        assert [target for target, _ in axis.lines] == [1.0, 5.0]

    monkeypatch.setattr(plotter.paper, "_render_triptych_figure", render)
    monkeypatch.setattr(plotter.paper, "save_shared_legend",
                        lambda *args, **kwargs: legends.append((args, kwargs)))
    options = plotter.render_panels(merged, tmp_path)
    assert len(calls) == 6
    for call, panel in zip(calls, merged["panels"]):
        assert call["series_by_method"] is panel["series"]
        assert call["method_order"] is panel["order"]
        assert call["output_stem"] == tmp_path / panel["stem"]
        assert call["panel_keys"] == ("energy",)
        assert call["show_legend"] is call["show_panel_titles"] is False
        assert call["show_threshold_gap_labels"] is call["show_threshold_gap_connectors"] is False
    assert options["show_slo_reference_lines"] is True
    assert len(legends) == 1
    assert legends[0][0][1][-1] == plotter.BEAM_KEY
    assert legends[0][1]["include_modeled_optimal"] is False
    assert plotter.paper._add_threshold_gap_annotations is original_hook
    assert merged == original_merged


def test_composition_has_six_panels_shared_legend_and_no_caption(merged, tmp_path):
    path = plotter.write_composition_tex(merged["panels"], tmp_path)
    tex = path.read_text()
    assert "shared_legend.pdf" in tex
    assert tex.count(r"\includegraphics") == 7
    assert r"\caption" not in tex
    assert "sensitivity" not in tex.lower()
    positions = [tex.index(panel["stem"] + ".pdf") for panel in merged["panels"]]
    assert positions == sorted(positions)


def test_main_figure_selects_only_chat_code_mixed_without_changing_values(merged):
    before = deepcopy(merged)
    main = plotter.select_main_panels(merged)
    assert [p['workload'] for p in main['panels']] == list(plotter.MAIN_WORKLOADS)
    assert main['panels'] == merged['panels'][:3]
    rows = main['exported_points']
    assert sum(row['method_key'] != plotter.BEAM_KEY for row in rows) == 116
    assert sum(row['method_key'] == plotter.BEAM_KEY and row['included_in_curve']
               for row in rows) == 30
    assert len(rows) == 148  # Also retain the two omitted Coder measurements.
    assert all(row['workload'] in plotter.MAIN_WORKLOADS for row in rows)
    assert merged == before


def test_main_composition_has_three_columns_and_no_extra_panels(merged, tmp_path):
    main = plotter.select_main_panels(merged)
    tex = plotter.write_composition_tex(main['panels'], tmp_path).read_text()
    assert r'\begin{tabular}{@{}ccc@{}}' in tex
    assert tex.count(r'\includegraphics') == 4
    assert r'\caption' not in tex
    for stem in ('32B_chat', 'e2e_mulitiround_chatting', 'chatting_large_scale'):
        assert stem not in tex


def test_main_legend_contains_only_present_methods(merged, monkeypatch, tmp_path):
    legends = []
    monkeypatch.setattr(plotter.paper, '_render_triptych_figure', lambda **kwargs: [])
    monkeypatch.setattr(plotter.paper, 'save_shared_legend',
                        lambda *args, **kwargs: legends.append((args, kwargs)))
    plotter.render_panels(plotter.select_main_panels(merged), tmp_path)
    assert legends[0][0][1] == [
        'vllm / round_robin', 'qlm / llumnix_load', 'sarathi / round_robin',
        'atfc / slosserve_planner', plotter.BEAM_KEY,
    ]
    assert legends[0][1]['ncol'] == 5


def _changed_previews(tmp_path, mutate):
    source = json.loads(plotter.SOURCE_PREVIEW.read_text())
    unscaled = json.loads(plotter.UNSCALED_PREVIEW.read_text())
    mutate(source)
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    unscaled["source_preview_sha256"] = plotter.file_hash(source_path)
    unscaled_path = tmp_path / "unscaled.json"
    unscaled_path.write_text(json.dumps(unscaled))
    return source_path, unscaled_path


def test_merge_rejects_mismatched_snapshot_revision(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(plotter.SOURCE_PREVIEW.read_text() + "\n")
    with pytest.raises(ValueError, match="different source revisions"):
        plotter.build_merge(source, plotter.UNSCALED_PREVIEW)


def test_merge_rejects_duplicate_beam_point(tmp_path):
    paths = _changed_previews(tmp_path, lambda source: source["beam_points"].append(
        deepcopy(source["beam_points"][0]),
    ))
    with pytest.raises(ValueError, match="Duplicate BEAM point"):
        plotter.build_merge(*paths)


def test_merge_rejects_inconsistent_raw_energy(tmp_path):
    def mutate(source):
        source["beam_points"][0]["raw_energy_kj"] += 1

    with pytest.raises(ValueError, match="Inconsistent raw energy"):
        plotter.build_merge(*_changed_previews(tmp_path, mutate))
