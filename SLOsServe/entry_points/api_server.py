"""
Usage example:

python -m distserve.api_server.distserve_api_server \\
    --host 0.0.0.0 \\
    --port {port} \\
    --model {args.model} \\
    --tokenizer {args.model} \\
    \\
    --context-tensor-parallel-size {context_tp} \\
    --context-pipeline-parallel-size {context_pp} \\
    --decoding-tensor-parallel-size {decoding_tp} \\
    --decoding-pipeline-parallel-size {decoding_pp} \\
    \\
    --block-size 16 \\
    --max-num-blocks-per-req 128 \\
    --gpu-memory-utilization 0.95 \\
    --swap-space 16 \\
    \\
    --context-sched-policy fcfs \\
    --context-max-batch-size 128 \\
    --context-max-tokens-per-batch 8192 \\
    \\
    --decoding-sched-policy fcfs \\
    --decoding-max-batch-size 1024 \\
    --decoding-max-tokens-per-batch 65536
"""

import argparse
import json
from typing import AsyncGenerator, List, Tuple, Callable
import asyncio
import traceback
import os
import numpy as np
import logging

from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
import uvicorn

from SLOsServe.args import add_dyserve_args
from SLOsServe.initialize import init
from SLOsServe.programs import RequestInput, RequestOutput, get_program

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()


@app.post("/generate")
async def generate(request: Request) -> Response:
    """Generate completion for the request.

    The request should be a JSON object with the following fields:
    - prompt: the prompt to use for the generation.
    - stream: whether to stream the results or not.
    - other fields: the sampling parameters (See `SamplingParams` for details).
    """
    request_dict = await request.json()
    prompt = request_dict.pop("prompt")
    max_tokens = request_dict.pop("max_tokens")
    ignore_eos = request_dict.pop('ignore_eos', False)
    stream = request_dict.pop("stream", False)
    prompt_len = len(prompt.split())
    logger.info(f"Received a request. #word in prompt: {prompt_len}, max_tokens: {max_tokens}")

    if stream:
        # Streaming case
        async def stream_results() -> AsyncGenerator[bytes, None]:
            async with global_context.context(max_tokens = max_tokens, prompt_len = prompt_len) as ctx:
                async for res in stream_func(ctx, prompt, max_tokens, ignore_eos):
                    ret = {"text": res}
                    yield (json.dumps(ret) + "\0").encode("utf-8")

        background_tasks = BackgroundTasks()
        # Abort the request if the client disconnects.
        # Currently we do not support request abortion, so we comment this line.
        # TODO implement request abortion.
        # background_tasks.add_task(abort_request)
        return StreamingResponse(stream_results(), background=background_tasks)
    else:
        # Non-streaming case
        import time 
        start = time.perf_counter()
        with global_context.context(max_tokens = max_tokens, prompt_len = prompt_len) as ctx:
            output: RequestOutput = await func(ctx, RequestInput(prompt, max_tokens, ignore_eos))
        all_outputs.append(output)
        elapsed = time.perf_counter() - start 
        ret = {
            "text": output.generated_text,
            "elapsed": elapsed 
        }
        return JSONResponse(ret)
    
@app.get('/report')
async def report(request: Request) -> Response:
    ret = {
        'program': args.program, 
        'batch_strategy': args.batch_strategy, 
        '#result': len(all_outputs),
        'spec_step': args.spec_step,
        'drafter_decode_length': 0 if args.draft_decode_length is None else args.draft_decode_length,
    }
    for attr in [
        'n_generated', 
        'n_spec', 
        'acc_rate', 
        'get_time', 
        'n_interpreted', 
        'issue_time', 
        'executor_get_delay', 
        'frontend_get_delay',
        'n_get',
        'n_alg_spec'
    ]:
        values = [getattr(output, attr) for output in all_outputs]
        ret[f'{attr} mean'] = np.mean(values) 
        ret[f'{attr} std'] = np.std(values)
    ret['executor'] = executor.report()
    print('ret', ret)
    return JSONResponse(ret)

@app.get('/reset')
async def reset(request: Request) -> Response:
    global all_outputs
    all_outputs = []
    await executor.reset()
    global_context.reset()

@app.post('/vis')
async def vis(request: Request) -> Response:
    request_dict = await request.json()
    file_path = request_dict.pop('dir', "vis.txt")
    executor.visualize(file_path)

    # return JSONResponse()

@app.post('/set')
async def set(request: Request) -> Response:
    request_dict = await request.json()
    print(f"Set Config: ")
    from pprint import pprint 
    pprint(request_dict)
    window_size = request_dict.pop("window_size", None)
    sch_budget = request_dict.pop('sch_budget', None)
    profile = request_dict.pop('profile', None)
    program = request_dict.pop("program", None)
    batch_strategy = request_dict.pop("strategy", None)
    spec_step = request_dict.pop("spec_step", None)
    draft_decode_length = request_dict.pop('draft_decode_length', None)
    executor.set_config(window_size = window_size, sch_budget=sch_budget, profile = profile)
    global func, stream_func
    if program is not None: 
        args.program = program
    if batch_strategy is not None:
        args.batch_strategy = batch_strategy
    if spec_step is not None:
        args.spec_step = spec_step
    if draft_decode_length is not None: 
        args.draft_decode_length = draft_decode_length
    if program is not None\
    or batch_strategy is not None\
    or spec_step is not None\
    or draft_decode_length is not None:
        func, stream_func = get_program(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    
    add_dyserve_args(parser)
    args = parser.parse_args()
    
    global_context, executor = init(
        model_tags = [args.model, args.draft_model] if args.draft_model is not None else [args.model], 
        dtype = args.dtype, 
        seed = args.seed, 
        backend_tag = args.backend, 
        debug = args.debug,
        num_gpus = args.num_gpus,
        profile = args.profile,
        window_size = args.window_size,
        enable_adaws = args.enable_adaws,
        sch_tot_budget=args.sch_budget,
        use_cuda_graph=args.use_cuda_graph,
        block_size = args.block_size, 
        cache_type = args.cache_type)
    
    func, stream_func = get_program(args)
    
    all_outputs: List[RequestOutput] = []

    uvicorn_config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="warning",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)
    
    async def main_coroutine():
        task2 = asyncio.create_task(uvicorn_server.serve())
        
        async def start_event_loop_wrapper():
            try:
                task = asyncio.create_task(executor.schedule())
                logger.info('SLOsServe Initialized')
                await task
            except Exception as e:
                traceback.print_exc()
                task2.cancel()
                os._exit(1) # Kill myself, or it will print tons of errors. Don't know why.
        
        task1 = asyncio.create_task(start_event_loop_wrapper())
        
        try:
            await task2
        except:
            # This is a workaround
            # When task1 exited for some reason (e.g. error in the engine),
            # task2 will raise many exceptions, which is annoying and I do 
            # not know why
            pass
    
    asyncio.run(main_coroutine())
    