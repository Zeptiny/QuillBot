"""Regression tests for the review fixes (no Discord or network access).

Covers: docs reindex keeping untouched/failed sources, wall-clock index
timestamps, tool exceptions surfacing to the model instead of aborting the
turn, history backfill resuming after the startup watermark, and the
duplicate-append guard.

Run: python3 test_review_fixes.py
"""

import asyncio
import time
from types import SimpleNamespace

import numpy as np

from cogs.docs_rag import DocsRAG
from cogs.history_rag import HistoryRAG
from cogs.utils import run_tool_loop
from config import DOC_SOURCES

PASS = 0
FAIL = 0


def check(name, cond, detail=''):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f'  ok  {name}')
    else:
        FAIL += 1
        print(f'FAIL  {name}  {detail}')


# --- DocsRAG reindex -------------------------------------------------------

LABELS = [s['label'] for s in DOC_SOURCES]


class _Resp:
    status = 200

    async def json(self):
        return {'sha': 'per-source-sha'}


class _CM:
    async def __aenter__(self):
        return _Resp()

    async def __aexit__(self, *exc):
        return False


class _Session:
    def get(self, url):
        return _CM()


def _chunk(label, tag):
    return {'content': f'{label}-{tag}', 'path': f'{tag}.md', 'title': tag,
            'source': label, 'doc_url': f'https://x/{tag}', 'tag': tag}


def _make_rag(*, fresh, raise_labels=(), embed_fails=False):
    """DocsRAG with 2 old chunks per source; `fresh` maps label -> new chunk count."""
    rag = DocsRAG.__new__(DocsRAG)
    rag.chunks = []
    rows = []
    for i, label in enumerate(LABELS):
        for j in range(2):
            tag = float(100 * i + j)
            rag.chunks.append(_chunk(label, f'old-{tag}') | {'tag': tag})
            rows.append([tag, 1.0])
    rag._emb_matrix = np.array(rows, dtype=np.float32)
    rag._last_commit_sha = 'composite-old'
    rag._source_shas = {}
    rag._source_last_index = {}
    rag._indexing = False
    rag.session = _Session()
    rag.saved = 0
    tag_by_content: dict[str, float] = {}

    async def index_source(src, semaphore):
        label = src['label']
        if label in raise_labels:
            raise RuntimeError('boom')
        base = 1000 * (LABELS.index(label) + 1)
        chunks = [_chunk(label, f'new-{base + k}') | {'tag': float(base + k)}
                  for k in range(fresh.get(label, 0))]
        tag_by_content.update((c['content'], c['tag']) for c in chunks)
        return chunks

    async def embed(texts):
        if embed_fails:
            raise RuntimeError('embedding API down')
        # First component encodes the chunk's tag so row alignment is checkable.
        return [[tag_by_content[t], 1.0] for t in texts]

    async def composite():
        return 'composite-new'

    def save():
        rag.saved += 1

    rag._index_source = index_source
    rag._embed_batch = embed
    rag._get_composite_sha = composite
    rag._save_vectors = save
    return rag


def _aligned(rag):
    return all(
        float(rag._emb_matrix[i][0]) == c['tag'] for i, c in enumerate(rag.chunks)
    ) and len(rag.chunks) == rag._emb_matrix.shape[0]


def test_partial_reindex_keeps_other_sources():
    print('partial /reindex keeps other sources')
    rag = _make_rag(fresh={'PaperMC': 3})
    asyncio.run(rag.index_docs([s for s in DOC_SOURCES if s['label'] == 'PaperMC']))
    by_source = {}
    for c in rag.chunks:
        by_source.setdefault(c['source'], []).append(c['title'])
    check('other sources untouched', all(len(by_source[l]) == 2 for l in LABELS if l != 'PaperMC'), by_source)
    check('reindexed source replaced', sorted(by_source['PaperMC']) == ['new-2000', 'new-2001', 'new-2002'], by_source['PaperMC'])
    check('matrix rows stay aligned with chunks', _aligned(rag))
    check('composite SHA unchanged on partial', rag._last_commit_sha == 'composite-old')
    ts = rag._source_last_index.get('PaperMC', 0)
    check('index time is wall-clock', abs(ts - time.time()) < 60, ts)
    check('saved once', rag.saved == 1)


def test_full_reindex_keeps_failed_sources():
    print('full reindex keeps failed/empty sources')
    fresh = {label: 1 for label in LABELS}
    fresh['PurpurMC'] = 0  # fetched nothing
    rag = _make_rag(fresh=fresh, raise_labels=('Spark',))
    asyncio.run(rag.index_docs())
    by_source = {}
    for c in rag.chunks:
        by_source.setdefault(c['source'], []).append(c['title'])
    check('failed source kept', len(by_source.get('Spark', [])) == 2, by_source)
    check('empty source kept', len(by_source.get('PurpurMC', [])) == 2, by_source)
    check('healthy source replaced', len(by_source.get('PaperMC', [])) == 1, by_source)
    check('matrix aligned', _aligned(rag))
    check('composite SHA not advanced after failures', rag._last_commit_sha == 'composite-old')
    check('failed sources get no index time', 'Spark' not in rag._source_last_index)


def test_full_reindex_success_and_total_failure():
    print('full reindex success / total failure')
    rag = _make_rag(fresh={label: 1 for label in LABELS})
    asyncio.run(rag.index_docs())
    check('all sources replaced', len(rag.chunks) == len(LABELS) and _aligned(rag))
    check('composite SHA advanced', rag._last_commit_sha == 'composite-new')

    rag = _make_rag(fresh={label: 1 for label in LABELS}, embed_fails=True)
    before = list(rag.chunks)
    asyncio.run(rag.index_docs())
    check('embedding outage leaves index untouched', rag.chunks == before and _aligned(rag))
    check('nothing saved on total failure', rag.saved == 0)
    check('composite SHA unchanged on total failure', rag._last_commit_sha == 'composite-old')


# --- run_tool_loop ---------------------------------------------------------

class _Completions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append([dict(m) if isinstance(m, dict) else m for m in kwargs['messages']])
        return self._responses.pop(0)


def _response(content=None, tool_calls=None):
    tcs = [
        SimpleNamespace(id=f'call_{i}', type='function',
                        function=SimpleNamespace(name=name, arguments=args))
        for i, (name, args) in enumerate(tool_calls or [])
    ] or None
    message = SimpleNamespace(role='assistant', content=content, tool_calls=tcs)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason='stop')], usage=None)


def test_tool_exception_reported_to_model():
    print('tool exception does not abort the turn')
    completions = _Completions([
        _response(tool_calls=[('search_docs', '{"max_results": "cinco"}')]),
        _response('resposta final'),
    ])
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))

    async def exec_tool(name, args):
        int(args['max_results'])  # ValueError, like a malformed model argument
        return 'unreachable', []

    answer, sources, _ = asyncio.run(run_tool_loop(
        client, 'model-x', [{'role': 'user', 'content': 'q'}], [], exec_tool,
    ))
    check('answer still produced', answer == 'resposta final', answer)
    check('no sources', sources == [])
    tool_msgs = [m for m in completions.calls[1] if isinstance(m, dict) and m.get('role') == 'tool']
    check('error surfaced as tool result',
          bool(tool_msgs) and 'Erro ao executar a ferramenta search_docs' in tool_msgs[0]['content'],
          tool_msgs)


# --- HistoryRAG backfill ---------------------------------------------------

def _history():
    rag = HistoryRAG.__new__(HistoryRAG)
    rag._chunks = {}
    rag._matrices = {}
    rag._msg_index = {}
    rag._recent = {}
    rag._locks = {}
    rag._backfill_after = {}
    return rag


def test_watermarks_and_resume():
    print('backfill resumes after startup watermark')
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
    print('append guard skips already-indexed messages')
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


if __name__ == '__main__':
    test_partial_reindex_keeps_other_sources()
    test_full_reindex_keeps_failed_sources()
    test_full_reindex_success_and_total_failure()
    test_tool_exception_reported_to_model()
    test_watermarks_and_resume()
    test_append_skips_duplicates()
    print(f'\n{PASS} passed, {FAIL} failed')
    raise SystemExit(1 if FAIL else 0)
