import marimo._code_mode as cm

async def run_training():
    async with cm.get_context() as ctx:
        print("Queueing cell DFbu for execution...")
        ctx.run_cell("DFbu")

await run_training()
print("Cell DFbu execution queued and running!")
