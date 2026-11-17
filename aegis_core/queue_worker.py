"""Bounded asynchronous fan-out over isolated worktrees and a shared WAL ledger."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from . import cli


def run_contracts(paths, *, db, workers=4, router=None):
    if type(workers) is not int or not 1 <= workers <= 32:
        raise ValueError('workers 1..32')
    # Never redirect process-global stdout from worker threads.
    def execute(path):
        import argparse
        try:
            return cli._run(argparse.Namespace(contract=str(path), db=str(db), router=router))
        except Exception as error:
            return {'status': 'error', 'error_type': type(error).__name__, 'detail': str(getattr(error,'stderr',str(error)))[:500]}, 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(execute, paths))


async def run_async(paths, *, db, workers=4):
    return await asyncio.to_thread(run_contracts, paths, db=db, workers=workers)
