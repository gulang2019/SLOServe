import sys
from types import SimpleNamespace

import pytest

sys.modules.setdefault(
    "SLOsServe_C",
    SimpleNamespace(AdmCtrlScheduler=None, Request=None),
)

from Dataset.dataset import Request
from SLOsServe.analysis import headroom_analysis
from SLOsServe.decode_length_predictor import DecodeLengthPredictorPlugin
from SLOsServe.router import adm_ctrl


class _DummyCppRequest:

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _DummyAdmCtrlScheduler:

    def __init__(self, *args, **kwargs):
        pass

    def adm_ctrl(self, reqs, _num_free_blocks, _current_time):
        return True, [True] * len(reqs)

    def adm_ctrl_with_reason(self, reqs, _num_free_blocks, _current_time):
        return True, [True] * len(reqs), None


class _FakePerfModel:

    def get_kv_mem_per_token(self):
        return 1.0

    def get_batch_time(self, batch):
        total_tokens = sum(n_tokens for _n_past, n_tokens in batch)
        return 0.001 * max(total_tokens, 1)

    def get_bs(self,
               t: float,
               num_reqs: int = 1,
               num_past_tokens: int = 0,
               num_decode_steps: int = 1):
        del t, num_reqs, num_past_tokens, num_decode_steps
        return 64

    def get_max_decode_length(self):
        return 64

    def configure_cpp_ar_planner(self, *args, **kwargs):
        pass


@pytest.fixture
def _patched_runtime(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "AdmCtrlScheduler", _DummyAdmCtrlScheduler)
    monkeypatch.setattr(adm_ctrl.SLOsServe_C, "Request", _DummyCppRequest)
    monkeypatch.setattr(
        headroom_analysis.PerfModel,
        "get_perf_model",
        staticmethod(lambda _model_name: _FakePerfModel()),
    )
    headroom_analysis.Instance._printed_kv_cache_info = False


def _make_instance(predictor):
    return headroom_analysis.Instance(
        device_id=0,
        event_queue=headroom_analysis.EventQueue(),
        slo_ttft_scale=1.0,
        slo_tpot=0.1,
        model_name="fake-model",
        block_size=16,
        kv_cache_mem=32,
        max_decode_length=64,
        admission_policy="conservative_arrival_reservation",
        decode_length_predictor=predictor,
    )


def test_conservative_arrival_reservation_counts_false_reject(_patched_runtime):
    instance = _make_instance(DecodeLengthPredictorPlugin.fixed(32))

    accepted = instance.add_request(
        0.0,
        headroom_analysis.RequestInstance(
            Request(input_length=16, output_length=5),
            "req-0",
            mode="normal",
        ),
    )

    metrics = instance.get_metrics()

    assert accepted is False
    assert metrics["rejected_requests"] == 1
    assert metrics["false_rejects"] == 1
    assert metrics["false_reject_rate"] == 1.0


def test_conservative_arrival_reservation_counts_false_admit(_patched_runtime):
    instance = _make_instance(DecodeLengthPredictorPlugin.fixed(1))

    accepted = instance.add_request(
        0.0,
        headroom_analysis.RequestInstance(
            Request(input_length=16, output_length=20),
            "req-0",
            mode="normal",
        ),
    )

    metrics = instance.get_metrics()

    assert accepted is True
    assert metrics["admitted_requests"] == 1
    assert metrics["false_admits"] == 1
    assert metrics["false_admit_rate"] == 1.0
