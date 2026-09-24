"""One shared worker thread for every local model call (load, encode, predict).

Each sentence-transformers / torch call already spreads across every core.
Run from separate ``asyncio.to_thread`` workers they overlap — a history
rerank while the ingest worker embeds, two users searching at once — and
contend so badly that two overlapping calls take ~10x longer than the same
calls back to back, with all cores pinned throughout. Funnelling them through
a single dedicated thread makes them queue instead.
"""

import asyncio
import concurrent.futures
import functools
from typing import Any, Callable, TypeVar

T = TypeVar('T')

_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='local-model')


async def run_local_model(fn: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """Run a blocking local-model call on the shared model thread."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(_EXECUTOR, functools.partial(fn, *args, **kwargs))
