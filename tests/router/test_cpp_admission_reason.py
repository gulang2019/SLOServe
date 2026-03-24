import SLOsServe_C


def _make_scheduler() -> SLOsServe_C.AdmCtrlScheduler:
    scheduler = SLOsServe_C.AdmCtrlScheduler("edf_sim", 16, False, False)
    scheduler.set_ar_planner([0.1], [1e-5, 0.0, 0.0, 0.0, 0.0], False)
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
