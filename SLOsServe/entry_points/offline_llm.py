import asyncio 

from SLOsServe.initialize import init
from SLOsServe.programs import forward
from SLOsServe.context import RequestContext 
from SLOsServe.args import parse_args 

async def main(
    args
):
    global_context, executor = init(
        args.models,
    )
    await executor.execute()
    with RequestContext(global_context) as ctx:
        await forward(ctx, 'I am okay', 'gpt2')

if __name__ == '__main__':
    args = parse_args()
    asyncio.run(main(args))