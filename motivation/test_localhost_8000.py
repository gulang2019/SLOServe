import json
import sys
import time
from typing import Any, Dict

import requests


def post_completions(url: str, payload: Dict[str, Any], timeout: float = 10.0) -> requests.Response:
    headers = {
        "Content-Type": "application/json",
        # Some OpenAI-compatible servers require an Authorization header, even if unused
        "Authorization": "Bearer EMPTY",
    }
    return requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout)


def main():
    base_url = "http://localhost:8000/v1"
    completions_url = f"{base_url}/completions"

    # Adjust model/prompt as needed for your local server
    payload = {
        "model": "unsloth/Llama-3.2-1B-Instruct",
        "prompt": "Say hello in 3 words.",
        "max_tokens": 32,
        # Optional sampling params that many OpenAI-compatible servers accept
        "temperature": 0.7,
        "top_p": 0.9,
    }

    print(f"POST {completions_url}")
    try:
        start = time.time()
        resp = post_completions(completions_url, payload)
        elapsed = time.time() - start
        print(f"Status: {resp.status_code} ({elapsed:.2f}s)")

        try:
            data = resp.json()
            print("Response JSON:")
            print(json.dumps(data, indent=2)[:2000])
        except Exception:
            print("Raw Response Text:")
            print(resp.text[:2000])

        if resp.status_code != 200:
            sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        sys.exit(2)


if __name__ == "__main__":
    main()



