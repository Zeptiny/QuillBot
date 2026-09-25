"""Regression tests for the history CPU/latency fixes (no Discord, network or models).

Covers: local model calls serialized on one thread, count_mentions skipping
the reranker, the full-scan FTS delete skipped for new rows (and only the
chunks a batch appended being persisted), search scoring running off the
event loop on the live matrix without copying it, and the matrix growth
buffer keeping earlier views intact.

Run: python3 test_history_perf.py
"""

import asyncio
import datetime
import os
import shutil
import sqlite3
import tempfile
import threading
import time
from types import SimpleNamespace

import numpy as np

from cogs.history_rag import HistoryRAG
from cogs.local_inference import run_local_model

GID = 1
GENERAL = "100"
DEV = "200"

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


def chunk(mid, cid, content, vec):
    return {
        "msg_id": str(mid), "guild_id": str(GID), "channel_id": cid, "channel_name": cid,
        "author_id": "7", "author_name": "Ana", "author_full": "Ana (@ana)",
        "content": content, "chunk_text": content, "window_line": content, "window_lines": [],
        "reply_to": None, "ts": f"2025-06-01T10:{mid:02d}:00+00:00",
        "jump_url": f"https://discord.com/channels/{GID}/{cid}/{mid}",
    }, vec


def fresh_rag(tmpdb):
    rag = HistoryRAG(bot=None)
    rag._db_path = lambda: tmpdb  # type: ignore[method-assign]
    rag._ensure_db()
    return rag


class TracedConnect:
    """Record every SQL statement run through sqlite3.connect while active."""

    def __init__(self):
        self.statements: list[str] = []
        self._real = sqlite3.connect

    def __enter__(self):
        def traced(*a, **k):
            con = self._real(*a, **k)
            con.set_trace_callback(self.statements.append)
            return con
        sqlite3.connect = traced
        return self

    def __exit__(self, *exc):
        sqlite3.connect = self._real

    def fts_deletes(self):
        # startswith: _ensure_db re-runs the CREATE TRIGGER whose body has the same text.
        return [s for s in self.statements if s.lstrip().startswith('DELETE FROM chunks_fts')]


def fts_counts(tmpdb):
    con = sqlite3.connect(tmpdb)
    try:
        return dict(con.execute("SELECT msg_id, COUNT(*) FROM chunks_fts GROUP BY msg_id").fetchall())
    finally:
        con.close()


# --- local model thread -----------------------------------------------------

def test_local_model_calls_serialized():
    print('local model calls run one at a time on one thread')
    state = {'active': 0, 'peak': 0, 'threads': set()}
    guard = threading.Lock()

    def work(x):
        with guard:
            state['active'] += 1
            state['peak'] = max(state['peak'], state['active'])
            state['threads'].add(threading.get_ident())
        time.sleep(0.02)
        with guard:
            state['active'] -= 1
        return x * 2

    def boom():
        raise ValueError('model failed')

    async def main():
        results = await asyncio.gather(*(run_local_model(work, i) for i in range(5)))
        try:
            await run_local_model(boom)
            raised = False
        except ValueError:
            raised = True
        return results, raised

    results, raised = asyncio.run(main())
    check('results returned in order', results == [0, 2, 4, 6, 8], results)
    check('never two calls at once', state['peak'] == 1, state['peak'])
    check('single dedicated thread', len(state['threads']) == 1, state['threads'])
    check('exceptions propagate', raised)


# --- search / count_mentions -------------------------------------------------

def _indexed_rag(tmpdb):
    rag = fresh_rag(tmpdb)
    rows = [
        chunk(1, GENERAL, "lag no servidor depois do update", [1.0, 0.0, 0.0]),
        chunk(2, GENERAL, "lag de novo, tps caiu", [0.9, 0.1, 0.0]),
        chunk(3, DEV, "lag no ambiente de dev", [0.8, 0.0, 0.2]),
        chunk(4, GENERAL, "bom dia pessoal", [0.0, 1.0, 0.0]),
        chunk(5, DEV, "deploy feito", [0.0, 0.0, 1.0]),
    ]
    chunks = [c for c, _ in rows]
    embs = [v for _, v in rows]
    rag._chunks[GID] = []
    rag._matrices[GID] = None
    rag._msg_index[GID] = {}
    rag._append_chunks(GID, chunks, embs)
    rag._upsert_chunks_to_db(GID, chunks, embs)

    async def embed(texts):
        return [[1.0, 0.0, 0.0] for _ in texts]
    rag._embed_batch = embed
    return rag


def test_count_mentions_skips_rerank(tmpdb):
    print('count_mentions counts without the reranker; search still reranks')
    rag = _indexed_rag(tmpdb)
    calls = []

    async def rerank(query, candidates, top_n):
        calls.append(len(candidates))
        return candidates[:top_n]
    rag._rerank_history = rerank

    groups = asyncio.run(rag.count_mentions(GID, 'lag'))
    check('count_mentions never calls the reranker', calls == [], calls)
    check('count_mentions still counts matches', groups and groups[0]['count'] >= 3, groups)

    res = asyncio.run(rag.search('lag', GID, limit=2, dedupe=False))
    check('search reranks its whole pool', calls == [5], calls)
    check('search returns limit results', len(res) == 2, res)


def test_search_scores_live_matrix_in_thread(tmpdb):
    print('search scores the live matrix off the event loop')
    rag = _indexed_rag(tmpdb)
    live = rag._matrices[GID]
    seen = {}
    real_score = rag._score_candidates

    def spy(query, guild_id, chunks, mat, *rest):
        seen['thread'] = threading.current_thread() is not threading.main_thread()
        seen['same_matrix'] = mat is live
        return real_score(query, guild_id, chunks, mat, *rest)
    rag._score_candidates = spy

    res = asyncio.run(rag.search('servidor', GID, limit=3, search_mode='semantic', dedupe=False, rerank=False))
    check('scoring ran in a worker thread', seen.get('thread') is True, seen)
    check('live matrix scored without a copy', seen.get('same_matrix') is True, seen)
    check('nearest vector ranks first', [r['msg_id'] for r in res][:1] == ['1'], [r['msg_id'] for r in res])

    res = asyncio.run(rag.search('servidor', GID, limit=5, channel_id=DEV, search_mode='semantic', dedupe=False, rerank=False))
    check('channel filter applied', [r['msg_id'] for r in res] == ['3', '5'], [r['msg_id'] for r in res])

    res = asyncio.run(rag.search('servidor', GID, limit=5, channel_id='999', search_mode='semantic', dedupe=False, rerank=False))
    check('empty filter result short-circuits', res == [], res)

    res = asyncio.run(rag.search('lag', GID, limit=5, search_mode='hybrid', dedupe=False, rerank=False))
    check('hybrid (vector + FTS) still ranks the lag messages first',
          {r['msg_id'] for r in res[:3]} == {'1', '2', '3'}, [r['msg_id'] for r in res])


# --- FTS delete / persistence --------------------------------------------------

def test_fts_delete_only_for_stored_rows(tmpdb):
    print('FTS delete (full scan) only runs for rows that may already exist')
    rag = fresh_rag(tmpdb)
    c, v = chunk(10, GENERAL, "mensagem nova", [1.0, 0.0])
    with TracedConnect() as tr:
        rag._upsert_chunks_to_db(GID, [c], [v], replace_ids=set())
    check('new row: no FTS delete', tr.fts_deletes() == [], tr.fts_deletes())

    edited = dict(c, content="mensagem editada", chunk_text="mensagem editada")
    with TracedConnect() as tr:
        rag._upsert_chunks_to_db(GID, [edited], [v], replace_ids={"10"})
    check('known-stored row: FTS delete runs', len(tr.fts_deletes()) == 1, tr.statements)

    with TracedConnect() as tr:
        rag._upsert_chunks_to_db(GID, [edited], [v])
    check('unknown (None): FTS delete runs as before', len(tr.fts_deletes()) == 1, tr.statements)
    check('still one FTS row per message', fts_counts(tmpdb) == {"10": 1}, fts_counts(tmpdb))

    broken = HistoryRAG(bot=None)
    broken._db_path = lambda: os.path.join(os.path.dirname(tmpdb), 'empty.db')  # type: ignore[method-assign]
    check('failed stored-probe returns None', broken._existing_msg_ids(GID, ["1"]) is None)


def _msg(mid, content):
    return SimpleNamespace(
        id=mid, content=content, attachments=[], embeds=[], reference=None,
        author=SimpleNamespace(id=7, name='ana', display_name='Ana', bot=False),
        created_at=datetime.datetime(2025, 6, 1, 10, mid, tzinfo=datetime.timezone.utc),
    )


def test_index_batch_persists_only_appended(tmpdb):
    print('_index_batch persists only what it appended, without FTS deletes')
    rag = fresh_rag(tmpdb)
    rag._chunks[GID] = []
    rag._matrices[GID] = None
    rag._msg_index[GID] = {}

    async def embed(texts):
        return [[float(len(t)), 1.0] for t in texts]
    rag._embed_batch = embed
    guild = SimpleNamespace(id=GID)
    channel = SimpleNamespace(id=int(GENERAL), name='geral')

    with TracedConnect() as tr:
        asyncio.run(rag._index_batch(guild, channel, [_msg(1, 'um'), _msg(2, 'dois')]))
    check('new messages: no FTS delete', tr.fts_deletes() == [], tr.fts_deletes())
    check('both stored', fts_counts(tmpdb) == {"1": 1, "2": 1}, fts_counts(tmpdb))

    # Message 3 was just appended in memory by the other ingest path (live vs
    # backfill) and is not persisted yet: this batch must leave it to that path.
    other, v = chunk(3, GENERAL, 'tres', [3.0, 1.0])
    rag._append_chunks(GID, [other], [v])
    with TracedConnect() as tr:
        asyncio.run(rag._index_batch(guild, channel, [_msg(2, 'dois'), _msg(3, 'tres'), _msg(4, 'quatro')]))
    counts = fts_counts(tmpdb)
    check('only the freshly appended message persisted', counts == {"1": 1, "2": 1, "4": 1}, counts)
    check('overlapping batch: no FTS delete', tr.fts_deletes() == [], tr.fts_deletes())
    check('in-memory index has no duplicates', [c['msg_id'] for c in rag._chunks[GID]] == ['1', '2', '3', '4'],
          [c['msg_id'] for c in rag._chunks[GID]])


# --- matrix growth buffer ----------------------------------------------------

def test_matrix_growth_buffer():
    print('matrix grows in place and keeps earlier views intact')
    rag = HistoryRAG.__new__(HistoryRAG)
    rag._chunks = {1: []}
    rag._matrices = {1: None}
    rag._mat_bufs = {}
    rag._msg_index = {1: {}}

    rag._append_chunks(1, [{'msg_id': '1'}, {'msg_id': '2'}], [[1.0, 0.0], [2.0, 0.0]])
    early = rag._matrices[1]
    snapshot = early.copy()
    reallocs = 0
    buf = rag._mat_bufs[1]
    for i in range(3, 2503):
        rag._append_chunks(1, [{'msg_id': str(i)}], [[float(i), 0.0]])
        if rag._mat_bufs[1] is not buf:
            reallocs += 1
            buf = rag._mat_bufs[1]
    mat = rag._matrices[1]
    check('all rows present in order', mat[:, 0].tolist() == [float(i) for i in range(1, 2503)], mat[:5])
    check('matrix is a view of the buffer', mat.base is rag._mat_bufs[1])
    check('2500 appends -> at most 2 reallocations', reallocs <= 2, reallocs)
    check('earlier view unchanged by later appends', np.array_equal(early, snapshot), early)

    # Deletes swap in a fresh array (np.delete); the next append must not
    # write into the old buffer a running search may still be reading.
    old_buf = rag._mat_bufs[1]
    rag._mat_bufs.pop(1, None)
    rag._matrices[1] = np.delete(rag._matrices[1], 0, axis=0)
    rag._chunks[1].pop(0)
    rag._msg_index[1] = {int(c['msg_id']): i for i, c in enumerate(rag._chunks[1])}
    before = old_buf[:3].copy()
    rag._append_chunks(1, [{'msg_id': '9999'}], [[9999.0, 0.0]])
    check('append after delete reallocates', rag._mat_bufs[1] is not old_buf)
    check('old buffer untouched', np.array_equal(old_buf[:3], before))
    check('rows after delete + append', rag._matrices[1][0, 0] == 2.0 and rag._matrices[1][-1, 0] == 9999.0)


if __name__ == '__main__':
    tmp = tempfile.mkdtemp(prefix='quillbot_perf_test_')
    try:
        test_local_model_calls_serialized()
        for i, test in enumerate((
            test_count_mentions_skips_rerank,
            test_search_scores_live_matrix_in_thread,
            test_fts_delete_only_for_stored_rows,
            test_index_batch_persists_only_appended,
        )):
            test(os.path.join(tmp, f'history{i}.db'))
        test_matrix_growth_buffer()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f'\n{PASS} passed, {FAIL} failed')
    raise SystemExit(1 if FAIL else 0)
