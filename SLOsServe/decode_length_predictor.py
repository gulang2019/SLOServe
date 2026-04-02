from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


DEFAULT_PROMPT_BUCKET_UPPERS: tuple[int, ...] = (
    128,
    256,
    512,
    1024,
    2048,
    4096,
    8192,
    16384,
)


class DecodeLengthPredictor(Protocol):
    def predict_mean(self, request: Any) -> int:
        ...

    def predict_quantile(self, request: Any, q: float) -> int:
        ...



def _coerce_non_negative_int(value: Any) -> int:
    return max(0, int(value))



def extract_prompt_length(request: Any) -> int:
    if hasattr(request, "input_length"):
        return _coerce_non_negative_int(getattr(request, "input_length"))
    if hasattr(request, "num_prompt_tokens"):
        return _coerce_non_negative_int(getattr(request, "num_prompt_tokens"))
    if hasattr(request, "req"):
        return extract_prompt_length(getattr(request, "req"))
    payload = getattr(request, "payload", None)
    if isinstance(payload, dict):
        extra_args = payload.get("vllm_xargs", {})
        if isinstance(extra_args, dict) and "input_length" in extra_args:
            return _coerce_non_negative_int(extra_args["input_length"])
    raise ValueError(f"Cannot extract prompt length from request of type {type(request)!r}")



def extract_actual_decode_length(request: Any) -> int:
    if hasattr(request, "output_length"):
        output_length = getattr(request, "output_length")
        if output_length is not None:
            return _coerce_non_negative_int(output_length)
    if hasattr(request, "req"):
        return extract_actual_decode_length(getattr(request, "req"))
    payload = getattr(request, "payload", None)
    if isinstance(payload, dict):
        extra_args = payload.get("vllm_xargs", {})
        if isinstance(extra_args, dict) and "output_length" in extra_args:
            return _coerce_non_negative_int(extra_args["output_length"])
        if "max_tokens" in payload:
            return _coerce_non_negative_int(payload["max_tokens"])
    raise ValueError(f"Cannot extract decode length from request of type {type(request)!r}")



def extract_workload_type(request: Any, *, default: str = "default") -> str:
    for attr in ("workload_type", "length_pattern", "service_tier"):
        if hasattr(request, attr):
            value = getattr(request, attr)
            if value is not None:
                return str(value)
    if hasattr(request, "req"):
        nested = getattr(request, "req")
        if nested is not None and nested is not request:
            return extract_workload_type(nested, default=default)
    payload = getattr(request, "payload", None)
    if isinstance(payload, dict):
        extra_args = payload.get("vllm_xargs", {})
        if isinstance(extra_args, dict):
            for key in ("workload_type", "length_pattern"):
                value = extra_args.get(key)
                if value is not None:
                    return str(value)
    return str(default)



def _nearest_rank_quantile(values: np.ndarray, q: float) -> int:
    if values.size == 0:
        raise ValueError("Cannot compute a quantile from an empty bucket.")
    normalized_q = min(max(float(q), 0.0), 1.0)
    sorted_values = np.sort(np.asarray(values, dtype=np.float64))
    rank = int(math.ceil(normalized_q * sorted_values.size))
    index = min(sorted_values.size - 1, max(0, rank - 1))
    return _coerce_non_negative_int(math.ceil(float(sorted_values[index])))


@dataclass(frozen=True)
class BucketedQuantileDecodeLengthPredictor:
    bucket_uppers: tuple[int, ...]
    bucket_samples: dict[tuple[str, int], tuple[int, ...]]
    workload_samples: dict[str, tuple[int, ...]]
    all_samples: tuple[int, ...]
    default_workload_type: str = "default"

    @classmethod
    def fit_from_requests(
        cls,
        requests: list[Any],
        *,
        workload_type: str = "default",
        prompt_bucket_uppers: tuple[int, ...] | list[int] | None = None,
    ) -> "BucketedQuantileDecodeLengthPredictor":
        if not requests:
            raise ValueError("Cannot fit a decode-length predictor on an empty request list.")
        bucket_uppers = tuple(
            sorted(
                {
                    _coerce_non_negative_int(upper)
                    for upper in (
                        DEFAULT_PROMPT_BUCKET_UPPERS
                        if prompt_bucket_uppers is None else prompt_bucket_uppers
                    )
                }
            )
        )
        bucket_samples: dict[tuple[str, int], list[int]] = defaultdict(list)
        workload_samples: dict[str, list[int]] = defaultdict(list)
        all_samples: list[int] = []
        for request in requests:
            actual_decode_length = extract_actual_decode_length(request)
            prompt_length = extract_prompt_length(request)
            request_workload = extract_workload_type(
                request,
                default=workload_type,
            )
            bucket_idx = bisect.bisect_right(bucket_uppers, prompt_length)
            bucket_samples[(request_workload, bucket_idx)].append(actual_decode_length)
            workload_samples[request_workload].append(actual_decode_length)
            all_samples.append(actual_decode_length)

        return cls(
            bucket_uppers=bucket_uppers,
            bucket_samples={
                key: tuple(sorted(values))
                for key, values in bucket_samples.items()
            },
            workload_samples={
                key: tuple(sorted(values))
                for key, values in workload_samples.items()
            },
            all_samples=tuple(sorted(all_samples)),
            default_workload_type=str(workload_type),
        )

    def _lookup_samples(self, request: Any) -> np.ndarray:
        prompt_length = extract_prompt_length(request)
        workload_type = extract_workload_type(
            request,
            default=self.default_workload_type,
        )
        bucket_idx = bisect.bisect_right(self.bucket_uppers, prompt_length)
        bucket_samples = self.bucket_samples.get((workload_type, bucket_idx))
        if bucket_samples:
            return np.asarray(bucket_samples, dtype=np.float64)
        workload_samples = self.workload_samples.get(workload_type)
        if workload_samples:
            return np.asarray(workload_samples, dtype=np.float64)
        return np.asarray(self.all_samples, dtype=np.float64)

    def predict_mean(self, request: Any) -> int:
        samples = self._lookup_samples(request)
        return _coerce_non_negative_int(int(round(float(samples.mean()))))

    def predict_quantile(self, request: Any, q: float) -> int:
        return _nearest_rank_quantile(self._lookup_samples(request), q)


@dataclass(frozen=True)
class OracleDecodeLengthPredictor:
    def predict_mean(self, request: Any) -> int:
        return extract_actual_decode_length(request)

    def predict_quantile(self, request: Any, q: float) -> int:
        del q
        return extract_actual_decode_length(request)


@dataclass(frozen=True)
class DecodeLengthPredictorPlugin:
    mode: str
    predictor: DecodeLengthPredictor | None = None
    fixed_length: int | None = None
    quantile: float | None = None

    @classmethod
    def fixed(cls, fixed_length: int) -> "DecodeLengthPredictorPlugin":
        return cls(mode="fixed", fixed_length=_coerce_non_negative_int(fixed_length))

    @classmethod
    def oracle(cls) -> "DecodeLengthPredictorPlugin":
        return cls(mode="oracle", predictor=OracleDecodeLengthPredictor())

    @classmethod
    def mean(cls, predictor: DecodeLengthPredictor) -> "DecodeLengthPredictorPlugin":
        return cls(mode="mean", predictor=predictor)

    @classmethod
    def quantile_plugin(
        cls,
        predictor: DecodeLengthPredictor,
        q: float,
    ) -> "DecodeLengthPredictorPlugin":
        return cls(mode="quantile", predictor=predictor, quantile=float(q))

    @property
    def label(self) -> str:
        if self.mode == "quantile" and self.quantile is not None:
            return f"q{int(round(self.quantile * 100))}"
        return self.mode

    def predict_length(self, request: Any) -> int:
        if self.mode == "fixed":
            if self.fixed_length is None:
                raise ValueError("fixed_length is required when mode='fixed'.")
            return _coerce_non_negative_int(self.fixed_length)
        if self.mode == "oracle":
            return extract_actual_decode_length(request)
        if self.predictor is None:
            raise ValueError(f"predictor is required when mode={self.mode!r}.")
        if self.mode == "mean":
            return self.predictor.predict_mean(request)
        if self.mode == "quantile":
            if self.quantile is None:
                raise ValueError("quantile is required when mode='quantile'.")
            return self.predictor.predict_quantile(request, self.quantile)
        raise ValueError(f"Unsupported decode-length prediction mode: {self.mode!r}")

    def predict_mean(self, request: Any) -> int:
        if self.mode == "fixed":
            if self.fixed_length is None:
                raise ValueError("fixed_length is required when mode='fixed'.")
            return _coerce_non_negative_int(self.fixed_length)
        if self.mode == "oracle":
            return extract_actual_decode_length(request)
        if self.predictor is None:
            raise ValueError(f"predictor is required when mode={self.mode!r}.")
        return self.predictor.predict_mean(request)

    def predict_quantile(self, request: Any, q: float) -> int:
        if self.mode == "fixed":
            if self.fixed_length is None:
                raise ValueError("fixed_length is required when mode='fixed'.")
            return _coerce_non_negative_int(self.fixed_length)
        if self.mode == "oracle":
            return extract_actual_decode_length(request)
        if self.predictor is None:
            raise ValueError(f"predictor is required when mode={self.mode!r}.")
        return self.predictor.predict_quantile(request, q)
