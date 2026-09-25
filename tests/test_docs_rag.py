"""DocsRAG reindex: partial and full reindex keep untouched or failed sources, wall-clock index times."""

import asyncio
import time

import numpy as np

from cogs.docs_rag import DocsRAG
from config import DOC_SOURCES

from helpers import check


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
    """partial /reindex keeps other sources."""
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
    """full reindex keeps failed/empty sources."""
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
    """full reindex success / total failure."""
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
