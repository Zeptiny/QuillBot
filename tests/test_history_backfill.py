"""HistoryRAG backfill: resuming after the startup watermark and the duplicate-append guard."""

import asyncio
from types import SimpleNamespace

from cogs.history_rag import HistoryRAG

from helpers import check


def _history():
    rag = HistoryRAG.__new__(HistoryRAG)
    rag._chunks = {}
    rag._matrices = {}
    rag._mat_bufs = {}
    rag._msg_index = {}
    rag._recent = {}
    rag._locks = {}
    rag._backfill_after = {}
    return rag


def test_watermarks_and_resume():
    """backfill resumes after startup watermark."""
    rag = _history()
    rag._chunks = {1: [
        {'channel_id': '10', 'msg_id': '100'},
        {'channel_id': '10', 'msg_id': '105'},
        {'channel_id': '11', 'msg_id': '50'},
    ]}
    marks = rag._indexed_watermarks()
    check('newest id per channel', marks == {10: 105, 11: 50}, marks)

    seen = {}

    class Channel:
        id = 10

        def history(self, **kwargs):
            seen.update(kwargs)

            async def _empty():
                return
                yield  # pragma: no cover

            return _empty()

    rag._backfill_after = marks
    rag._msg_index = {1: {}}
    asyncio.run(rag._backfill_channel(SimpleNamespace(id=1), Channel()))
    after = seen.get('after')
    check('history walked after the watermark', getattr(after, 'id', None) == 105, seen)
    check('oldest first', seen.get('oldest_first') is True)
    check('watermark consumed', 10 not in rag._backfill_after)

    seen.clear()
    asyncio.run(rag._backfill_channel(SimpleNamespace(id=1), Channel()))
    check('unknown channel walks full history', seen.get('after') is None, seen)


def test_append_skips_duplicates():
    """append guard skips already-indexed messages."""
    rag = _history()
    rag._chunks = {1: []}
    rag._matrices = {1: None}
    rag._msg_index = {1: {}}
    a = {'msg_id': '1'}
    b = {'msg_id': '2'}
    c = {'msg_id': '3'}
    check('first batch appended', rag._append_chunks(1, [a, b], [[1.0, 0.0], [2.0, 0.0]]))
    check('overlapping batch appended', rag._append_chunks(1, [b, c], [[9.0, 0.0], [3.0, 0.0]]))
    ids = [ch['msg_id'] for ch in rag._chunks[1]]
    check('no duplicate chunk', ids == ['1', '2', '3'], ids)
    check('matrix rows match chunks', rag._matrices[1][:, 0].tolist() == [1.0, 2.0, 3.0], rag._matrices[1])
    check('fully duplicate batch rejected', rag._append_chunks(1, [c], [[3.0, 0.0]]) is False)
