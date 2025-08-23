import asyncio
import networkx as nx
from collections import defaultdict
from aiohttp import web
import argparse


import SLOsServe 
from SLOsServe.executor import Executor
from SLOsServe.initialize import init

global_context = None 

async def handle_request(request):
    data = await request.json()
    prompt = data['prompt']
    model_tag = data['model_tag']
    program_tag = data['program_tag']
    
    # Create an interpreter for this request
    with SLOsServe.RequestContext(global_context) as ctx:
        response = await global_context.get_program(program_tag)(ctx, prompt, model_tag)

    return web.json_response({"status": "interpreted", "data": response})

async def init_app():
    app = web.Application()
    app.router.add_post('/interpret', handle_request)
    return app

async def main():
    args = parse_args()
    global global_context 
    global_context, executor = SLOsServe.init(args.models, args.programs)
    asyncio.create_task(executor.execute())

    app = await init_app()
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, 'localhost', 8080)
    await site.start()

    print("Server started at http://localhost:8080")
    while True:
        await asyncio.sleep(3600)  # Keep the server running


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prorgam', type = str, n_args = '+')
    parser.add_argument('--models', type = str, n_args = '+') 
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    asyncio.run(main())