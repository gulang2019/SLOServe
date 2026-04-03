from Dataset.dataset import Request

from SLOsServe.decode_length_predictor import BucketedQuantileDecodeLengthPredictor


def test_bucketed_quantile_decode_length_predictor_uses_prompt_buckets():
    predictor = BucketedQuantileDecodeLengthPredictor.fit_from_requests(
        [
            Request(input_length=64, output_length=10),
            Request(input_length=64, output_length=20),
            Request(input_length=64, output_length=30),
            Request(input_length=1500, output_length=40),
            Request(input_length=1500, output_length=50),
        ],
        workload_type="chat",
        prompt_bucket_uppers=(128, 1024),
    )

    small_request = Request(input_length=96, output_length=0)
    large_request = Request(input_length=1800, output_length=0)

    assert predictor.predict_mean(small_request) == 20
    assert predictor.predict_quantile(small_request, 0.90) == 30
    assert predictor.predict_quantile(small_request, 0.95) == 30

    assert predictor.predict_mean(large_request) == 45
    assert predictor.predict_quantile(large_request, 0.90) == 50
    assert predictor.predict_quantile(large_request, 0.95) == 50
