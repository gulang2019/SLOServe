import sys

import pytest

if "SLOsServe_C" in sys.modules and getattr(
    sys.modules["SLOsServe_C"],
    "AdmCtrlScheduler",
    None,
) is None:
    sys.modules.pop("SLOsServe_C", None)

SLOsServe_C = pytest.importorskip("SLOsServe_C")


def _make_scheduler(
    *,
    tpots: list[float] | None = None,
    hardware_params: list[float] | None = None,
) -> SLOsServe_C.AdmCtrlScheduler:
    if tpots is None:
        tpots = [0.1]
    if hardware_params is None:
        hardware_params = [1e-5, 0.0, 0.0, 0.0, 0.0]
    scheduler = SLOsServe_C.AdmCtrlScheduler("edf_sim", 16, False, False)
    scheduler.set_ar_planner(tpots, hardware_params, False)
    return scheduler


def test_cpp_admission_reason_reports_memory_reject():
    scheduler = _make_scheduler()
    req = SLOsServe_C.Request(
        id="req-mem",
        is_new_req=True,
        ddl=10.0,
        input_length=32,
        n_computed_tokens=0,
        max_tokens=1,
        profit=1.0,
        mem=2,
        tpot_idx=0,
        prefill_mem=2,
        prefill_device_id=0,
        decode_device_id=0,
        prefill_only=True,
        arrival_time=0.0,
    )

    is_feasible, is_accepteds, reject_reason = scheduler.adm_ctrl_with_reason([req], 1, 0.0)

    assert is_feasible
    assert is_accepteds == [False]
    assert reject_reason == "MEM"


def test_cpp_admission_reason_reports_compute_reject():
    scheduler = _make_scheduler()
    req = SLOsServe_C.Request(
        id="req-cmp",
        is_new_req=True,
        ddl=0.0,
        input_length=1,
        n_computed_tokens=0,
        max_tokens=1,
        profit=1.0,
        mem=1,
        tpot_idx=0,
        prefill_mem=1,
        prefill_device_id=0,
        decode_device_id=0,
        prefill_only=True,
        arrival_time=0.0,
    )

    is_feasible, is_accepteds, reject_reason = scheduler.adm_ctrl_with_reason([req], 1024, 0.0)

    assert is_feasible
    assert is_accepteds == [False]
    assert reject_reason == "CMP"


def test_cpp_admission_reason_rejects_inconsistent_decode_horizon():
    scheduler = _make_scheduler()
    req = SLOsServe_C.Request(
        id="req-decode-horizon",
        is_new_req=False,
        ddl=10.0,
        input_length=10,
        n_computed_tokens=15,
        max_tokens=4,
        profit=1.0,
        mem=1,
        tpot_idx=0,
        prefill_mem=1,
        prefill_device_id=0,
        decode_device_id=0,
        prefill_only=False,
        arrival_time=0.0,
    )

    is_feasible, is_accepteds, reject_reason = scheduler.adm_ctrl_with_reason([req], 1024, 0.0)

    assert not is_feasible
    assert is_accepteds == [False]
    assert reject_reason == "CMP"


def test_cpp_admission_decode_only_fastpath_handles_partial_first_interval(
    monkeypatch,
):
    reqs = [
        SLOsServe_C.Request(
            id="req-partial-first-interval",
            is_new_req=False,
            ddl=0.001,
            input_length=1,
            n_computed_tokens=1,
            max_tokens=2,
            profit=1.0,
            mem=1,
            tpot_idx=0,
            prefill_mem=1,
            prefill_device_id=0,
            decode_device_id=0,
            prefill_only=False,
            arrival_time=0.0,
        ),
    ]
    tpots = [0.025]
    hardware_params = [1e-5, 0.0, 0.0, 0.0, 0.012]

    monkeypatch.delenv("SLOSERVE_ENABLE_DECODE_ONLY_FASTPATH", raising=False)
    scheduler = _make_scheduler(tpots=tpots, hardware_params=hardware_params)
    is_feasible, is_accepteds, reject_reason = scheduler.adm_ctrl_with_reason(
        reqs, 1024, 0.0
    )

    assert is_feasible
    assert is_accepteds == [True]
    assert reject_reason == "NONE"

    monkeypatch.setenv("SLOSERVE_ENABLE_DECODE_ONLY_FASTPATH", "1")
    scheduler = _make_scheduler(tpots=tpots, hardware_params=hardware_params)
    is_feasible, is_accepteds, reject_reason = scheduler.adm_ctrl_with_reason(
        reqs, 1024, 0.0
    )

    assert is_feasible
    assert is_accepteds == [True]
    assert reject_reason == "NONE"


def test_cpp_admission_reason_online_slack_penalizes_compute_budget():
    scheduler = _make_scheduler(
        tpots=[0.025],
        hardware_params=[1e-5, 0.0, 0.0, 0.0, 0.012],
    )
    req = SLOsServe_C.Request(
        id="req-online-slack",
        is_new_req=True,
        ddl=0.001,
        input_length=1,
        n_computed_tokens=1,
        max_tokens=2,
        profit=1.0,
        mem=1,
        tpot_idx=0,
        prefill_mem=1,
        prefill_device_id=0,
        decode_device_id=0,
        prefill_only=False,
        arrival_time=0.0,
    )

    is_feasible, is_accepteds, reject_reason = scheduler.adm_ctrl_with_reason(
        [req],
        1024,
        0.0,
        0.014,
    )

    assert is_feasible
    assert is_accepteds == [False]
    assert reject_reason == "CMP"
