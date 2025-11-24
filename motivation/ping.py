import requests
import time

url = "http://0.0.0.0:8000/v1/completions"
headers = {
    "Content-Type": "application/json"
}
data = {
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "prompt": [0] * 1000,
    "max_tokens": 110,
    "temperature": 0,
    "stream": True,
    "vllm_xargs": {
        "input_length": 1000,
        "output_length": 110,
        "prefill_ddl": time.time() + 10,
        "slo_ttft": 10,
        "profit": 1
    }
}

import aiohttp
import asyncio

async def main():
    n_lines = 0
    async with aiohttp.ClientSession() as session:
        # print('sending request at {}'.format(time.time()))
        async with session.post(url, headers=headers, json=data) as response:
            async for line in response.content:
                print(line)
                if line.startswith(b'data:'):
                    n_lines += 1     
                print('--------------------------------')
    print(f'n_lines: {n_lines}')
if __name__ == "__main__":
    N = 1  # or any number of requests you want to run in parallel

    async def run_n_requests(n):
        tasks = [main() for _ in range(n)]
        await asyncio.gather(*tasks)

    asyncio.run(run_n_requests(N))

# from motivation.events_analysis import analyze_events, analyze_slo_violation

# filename = 'events/Qwen-7B_azure_code_23_azure_code_23_0:10_1.0_2.jsonl'

# events, reqs = analyze_events(filename)

# for req in reqs.values():
#     print(req)
#     exit(0)

# results = analyze_slo_violation(reqs, events, slo_ttft_fn = lambda x: 2e-4 * x + 0.1, slo_tpot = 0.05)

# print(results)
