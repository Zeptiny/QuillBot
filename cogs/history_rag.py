import asyncio
import collections
import datetime
import json
import logging
import math
import os
import re
import sqlite3
import time

import discord
import numpy as np
try:
    from cachetools import TTLCache
except ImportError:
    TTLCache = dict  # type: ignore
from discord import app_commands
from discord.ext import commands, tasks

from cogs.utils import format_chunk_line, format_message_line, message_content_text, render_search_results

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    HISTORY_BACKFILL_LIMIT,
    HISTORY_DB_PATH,
    HISTORY_DEDUPE_WINDOW_MINUTES,
    HISTORY_ENABLED,
    HISTORY_EXCLUDE_BOTS,
    HISTORY_HYBRID_WEIGHT_KEYWORD,
    HISTORY_HYBRID_WEIGHT_SEMANTIC,
    HISTORY_INGEST_BATCH_SIZE,
    HISTORY_INGEST_FLUSH_SECONDS,
    HISTORY_MAX_MSG_LENGTH,
    HISTORY_QUERY_CACHE_SIZE,
    HISTORY_RERANK_ENABLED,
    HISTORY_RERANK_MODEL,
    HISTORY_RERANK_PROVIDER,
    HISTORY_RRF_K,
    HISTORY_SQL_MAX_ROWS,
    HISTORY_SQL_TIMEOUT_SECONDS,
    HISTORY_TIME_DECAY_LAMBDA,
    HISTORY_WINDOW_OVERLAP,
    HISTORY_WINDOW_SIZE,
    LOCAL_EMBEDDING_DEVICE,
    LOCAL_EMBEDDING_MODEL,
    LOCAL_RERANK_DEVICE,
    LOCAL_RERANK_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    RERANK_PROVIDER,
)

logger = logging.getLogger(__name__)

BR_TZ = None
try:
    from zoneinfo import ZoneInfo
    BR_TZ = ZoneInfo("America/Sao_Paulo")
except Exception:
    pass

def _parse_dt(s: str | None) -> datetime.datetime | None:
    if not s:
        return None
    s = s.strip()
    try:
        dt = datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return dt
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y"):
        try:
            dt = datetime.datetime.strptime(s, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt
        except Exception:
            continue
    for fmt in ("%Y-%m-%d", "%Y/%m/%d"):
        try:
            dt = datetime.datetime.strptime(s[:10], fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)
            return dt
        except Exception:
            continue
    return None

# Explicit column list for full-row reads: positional access avoids
# sqlite3.Row's per-access name scan, and the fixed order is immune to
# reply_to being appended by ALTER TABLE on DBs created before that column.
_CHUNK_COLS = (
    "msg_id", "guild_id", "channel_id", "channel_name", "author_id", "author_name",
    "author_full", "content", "chunk_text", "window_line", "window_lines",
    "reply_to", "ts", "jump_url", "embedding",
)
_EMB_COL = 14


def _jump_url(msg: discord.Message) -> str:
    try:
        return f"https://discord.com/channels/{msg.guild.id}/{msg.channel.id}/{msg.id}"
    except Exception:
        return ""

def _extract_query_terms(query: str) -> list[str]:
    """Extract search terms: quoted phrases kept verbatim, loose words > 2 chars."""
    terms: list[str] = []
    for m in re.finditer(r'"([^"]+)"', query or ''):
        t = ' '.join(m.group(1).split())
        if t and t not in terms:
            terms.append(t)
    remainder = re.sub(r'"[^"]*"', ' ', query or '')
    for t in re.findall(r'\w+', remainder):
        if len(t) > 2 and t not in terms:
            terms.append(t)
    return terms


def _keyword_score(query: str, chunk: dict) -> float:
    terms = _extract_query_terms(query)
    if not terms:
        return 0.0
    text = (chunk.get("chunk_text", "") + " " + chunk.get("content", "")).lower()
    matched = sum(1 for t in terms if t.lower() in text)
    return matched / len(terms)


def _sanitize_fts_query(query: str) -> str | None:
    """Build an FTS5 OR-query; quoted phrases are preserved as exact phrase queries."""
    terms = _extract_query_terms(query)
    if not terms:
        return None
    escaped = []
    for t in terms[:10]:
        escaped.append('"' + t.replace('"', '""') + '"')
    return " OR ".join(escaped)


def _time_decay_factor(ts_str: str, lambda_: float) -> float:
    if lambda_ <= 0:
        return 1.0
    try:
        dt = datetime.datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        age_days = (datetime.datetime.now(datetime.timezone.utc) - dt).total_seconds() / 86400
        if age_days < 0:
            return 1.0
        return math.exp(-lambda_ * age_days)
    except Exception:
        return 1.0


def _rrf_fuse(rank_dicts: list[dict[str, float]], k: int = 60) -> dict[str, float]:
    fused: dict[str, float] = {}
    for rank_map in rank_dicts:
        sorted_ids = sorted(rank_map, key=lambda x: rank_map[x], reverse=True)
        for rank, mid in enumerate(sorted_ids, start=1):
            fused[mid] = fused.get(mid, 0) + 1 / (k + rank)
    return fused


def _escape_like(s: str) -> str:
    return s.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


# --- Read-only SQL tool guards (LLM-written analytical queries) ---

_SQL_LITERAL_RE = re.compile(r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"")
_SQL_COMMENT_RE = re.compile(r'--[^\n]*|/\*.*?\*/', re.S)
_SQL_SELECT_ONLY_RE = re.compile(r'^\s*(?:WITH\b.*?\bSELECT\b|SELECT\b)', re.I | re.S)
_SQL_AUTHOR_ALLOWED = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}
_SQL_MAX_LENGTH = 4000
_SQL_MAX_PATTERN = 512
_SQL_CELL_CHARS = 160
_SQL_OUTPUT_CHARS = 12000


def _validate_history_sql(sql: str, guild_id: int | None) -> None:
    """Reject anything but a single read-only SELECT scoped to one guild."""
    if not sql or not sql.strip():
        raise ValueError("Consulta SQL vazia.")
    if len(sql) > _SQL_MAX_LENGTH:
        raise ValueError(f"Consulta muito longa (máx. {_SQL_MAX_LENGTH} caracteres).")
    if guild_id is None:
        raise ValueError("Consulta requer estar em um servidor (guild_id).")
    if str(guild_id) not in sql:
        raise ValueError(f"A consulta deve filtrar pelo servidor atual: inclua guild_id={guild_id}.")
    stripped = _SQL_COMMENT_RE.sub(' ', _SQL_LITERAL_RE.sub("''", sql)).strip()
    if not _SQL_SELECT_ONLY_RE.match(stripped):
        raise ValueError("Apenas um único SELECT (ou WITH ... SELECT) é permitido.")
    if ';' in stripped.rstrip(';'):
        raise ValueError("Apenas uma instrução por consulta.")


def _sql_cell(v, limit: int = _SQL_CELL_CHARS) -> str:
    if v is None:
        return 'NULL'
    if isinstance(v, bytes):
        return f'<BLOB {len(v)}B>'
    if isinstance(v, float):
        return f'{v:.4f}'.rstrip('0').rstrip('.')
    s = str(v).replace('\n', ' ').replace('\r', ' ').strip()
    return s if len(s) <= limit else s[:limit] + '…'


def _handle_from_label(full_label: str) -> str | None:
    """Extract the @handle from a 'Display (@handle)' author label."""
    if not full_label or ' (@' not in full_label:
        return None
    return full_label.rsplit(' (@', 1)[1].rstrip(')') or None


def _aggregate_authors(rows) -> dict[tuple[int, str], dict]:
    """Aggregate (guild_id, author_id, author_name, author_full, ts) rows into author stats."""
    agg: dict[tuple[int, str], dict] = {}
    for guild_id, aid, name, full, ts in rows:
        if not aid:
            continue
        key = (int(guild_id), str(aid))
        a = agg.setdefault(key, {
            'display_name': name or '',
            'full_label': full or name or str(aid),
            'aliases': set(),
            'first_seen': ts or '',
            'last_seen': ts or '',
            'count': 0,
        })
        if name:
            a['aliases'].add(name)
            a['display_name'] = name or a['display_name']
        if full:
            a['full_label'] = full
        if ts:
            if not a['first_seen'] or ts < a['first_seen']:
                a['first_seen'] = ts
            if not a['last_seen'] or ts > a['last_seen']:
                a['last_seen'] = ts
        a['count'] += 1
    return agg

class HistoryRAG(commands.Cog, name="HistoryRAG"):
    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.client = None
        self._local_model = None
        self._local_dim = None
        self._local_rerank_model = None
        try:
            self._query_cache = TTLCache(maxsize=max(50, HISTORY_QUERY_CACHE_SIZE), ttl=300)  # type: ignore
        except Exception:
            self._query_cache = {}  # type: ignore
        if EMBEDDING_PROVIDER != "local":
            try:
                from openai import AsyncOpenAI
                api_key = OPENAI_API_KEY or "not-needed"
                self.client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=api_key)
            except Exception:
                logger.exception("Failed to init OpenAI client for HistoryRAG")
        self._chunks: dict[int, list[dict]] = {}
        self._matrices: dict[int, np.ndarray | None] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._recent: dict[int, collections.deque] = {}
        self._backfilling: set[int] = set()
        self._msg_index: dict[int, dict[int, int]] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None

    def _get_local_model(self):
        if self._local_model is not None:
            return self._local_model
        from sentence_transformers import SentenceTransformer
        logger.info("Loading local history embedding model %s", LOCAL_EMBEDDING_MODEL)
        self._local_model = SentenceTransformer(LOCAL_EMBEDDING_MODEL, device=LOCAL_EMBEDDING_DEVICE)
        test = self._local_model.encode(["test"], normalize_embeddings=True)
        self._local_dim = test.shape[1] if hasattr(test, "shape") else len(test[0])
        logger.info("History local model dim=%s", self._local_dim)
        return self._local_model

    def _get_local_reranker(self):
        if self._local_rerank_model is not None:
            return self._local_rerank_model
        try:
            from sentence_transformers import CrossEncoder
            logger.info("Loading local rerank model %s (device=%s)", LOCAL_RERANK_MODEL, LOCAL_RERANK_DEVICE)
            self._local_rerank_model = CrossEncoder(LOCAL_RERANK_MODEL, device=LOCAL_RERANK_DEVICE)
            logger.info("Local reranker loaded")
        except Exception:
            logger.exception("Failed to load local reranker %s", LOCAL_RERANK_MODEL)
            raise
        return self._local_rerank_model

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        if EMBEDDING_PROVIDER == "local":
            model = self._get_local_model()
            emb = await asyncio.to_thread(model.encode, texts, normalize_embeddings=True, show_progress_bar=False)
            if hasattr(emb, "tolist"):
                return emb.tolist()
            return [list(e) for e in emb]
        if not self.client:
            raise RuntimeError("OpenAI client not initialized for history embeddings")
        resp = await self.client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
        return [d.embedding for d in resp.data]

    def _db_path(self) -> str:
        return HISTORY_DB_PATH

    def _ensure_db(self):
        path = self._db_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        con = sqlite3.connect(path)
        try:
            try:
                con.execute("PRAGMA journal_mode=WAL;")
                con.execute("PRAGMA synchronous=NORMAL;")
            except Exception:
                pass
            con.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                msg_id TEXT PRIMARY KEY,
                guild_id INTEGER NOT NULL,
                channel_id TEXT NOT NULL,
                channel_name TEXT,
                author_id TEXT NOT NULL,
                author_name TEXT,
                author_full TEXT,
                content TEXT,
                chunk_text TEXT,
                window_line TEXT,
                window_lines TEXT,
                reply_to TEXT,
                ts TEXT,
                jump_url TEXT,
                embedding BLOB NOT NULL
            )""")
            try:
                con.execute("ALTER TABLE chunks ADD COLUMN reply_to TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column name" not in str(e).lower():
                    raise
            con.execute("CREATE INDEX IF NOT EXISTS idx_guild ON chunks(guild_id)")
            # Satisfies the per-guild ORDER BY ts load without a temp-B-tree sort
            # over full rows (embedding blobs included).
            con.execute("CREATE INDEX IF NOT EXISTS idx_guild_ts ON chunks(guild_id, ts)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_author ON chunks(author_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_channel ON chunks(channel_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON chunks(ts)")
            # Covering indexes for the authors top-channels aggregation and the
            # indexed message-context window queries (else both full-scan the guild).
            con.execute("CREATE INDEX IF NOT EXISTS idx_guild_author_chan ON chunks(guild_id, author_id, channel_name)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_guild_chan_ts ON chunks(guild_id, channel_id, ts)")
            try:
                con.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    msg_id UNINDEXED, guild_id UNINDEXED, chunk_text, content, tokenize='porter unicode61'
                )""")
                con.execute("CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN DELETE FROM chunks_fts WHERE msg_id=old.msg_id; END;")
            except Exception:
                logger.warning("FTS5 not available, keyword search will fallback to substring")
            con.execute("""
            CREATE TABLE IF NOT EXISTS authors (
                guild_id INTEGER NOT NULL,
                author_id TEXT NOT NULL,
                display_name TEXT,
                handle TEXT,
                full_label TEXT,
                aliases TEXT NOT NULL DEFAULT '[]',
                first_seen TEXT,
                last_seen TEXT,
                msg_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (guild_id, author_id)
            )""")
            con.execute("CREATE INDEX IF NOT EXISTS idx_authors_guild ON authors(guild_id)")
            con.commit()
        finally:
            con.close()

    def _load_guild_from_db(self, guild_id: int) -> bool:
        path = self._db_path()
        if not os.path.exists(path):
            return False
        try:
            con = sqlite3.connect(path)
            cur = con.execute(
                f"SELECT {', '.join(_CHUNK_COLS)} FROM chunks WHERE guild_id=? ORDER BY ts ASC",
                (guild_id,),
            )
            rows = cur.fetchall()
            con.close()
            if not rows:
                return False
            emb_len = len(rows[0][_EMB_COL])
            if emb_len == 0 or emb_len % 4 or any(len(r[_EMB_COL]) != emb_len for r in rows):
                logger.warning("DB embedding blob mismatch guild %s, skipping DB load", guild_id)
                return False
            dim = emb_len // 4
            if EMBEDDING_PROVIDER == "local" and self._local_dim and dim != self._local_dim:
                logger.warning("DB dim mismatch guild %s, skipping DB load", guild_id)
                return False
            chunks = []
            for (msg_id, gid, channel_id, channel_name, author_id, author_name,
                 author_full, content, chunk_text, window_line, window_lines,
                 reply_to, ts, jump_url, _emb) in rows:
                chunks.append({
                    "msg_id": msg_id,
                    "channel_id": channel_id,
                    "guild_id": str(gid),
                    "channel_name": channel_name or channel_id,
                    "author_id": author_id,
                    "author_name": author_name or "?",
                    "author_full": author_full or author_name or "?",
                    "content": content or "",
                    "chunk_text": chunk_text or "",
                    "window_line": window_line or "",
                    "window_lines": json.loads(window_lines) if window_lines else [],
                    "reply_to": reply_to,
                    "ts": ts or "",
                    "jump_url": jump_url or "",
                })
            # Bulk-decode all embeddings into one contiguous, writable matrix
            # (on_message_edit overwrites rows in place). The matrix is the
            # single in-memory home for embeddings — chunks never hold copies.
            raw = b"".join(r[_EMB_COL] for r in rows)
            matrix = np.frombuffer(raw, dtype=np.float32).reshape(len(rows), dim).copy()
            self._chunks[guild_id] = chunks
            self._matrices[guild_id] = matrix
            self._msg_index[guild_id] = {int(c["msg_id"]): i for i, c in enumerate(chunks)}
            self._rebuild_recent(guild_id)
            logger.info("Loaded history %d chunks for guild %s from DB", len(chunks), guild_id)
            return True
        except Exception:
            logger.exception("Failed to load history DB for guild %s", guild_id)
            return False

    def _upsert_chunks_to_db(self, guild_id: int, chunks: list[dict], embeddings: list | None = None):
        if not chunks:
            return
        path = self._db_path()
        self._ensure_db()
        con = sqlite3.connect(path, timeout=30)
        try:
            rows = []
            for i, c in enumerate(chunks):
                emb = embeddings[i] if embeddings is not None else c.get("embedding")
                if emb is None:
                    continue
                if isinstance(emb, np.ndarray):
                    blob = emb.astype(np.float32).tobytes()
                else:
                    blob = np.array(emb, dtype=np.float32).tobytes()
                rows.append((
                    str(c["msg_id"]), int(guild_id), str(c["channel_id"]), c.get("channel_name", ""), str(c["author_id"]), c.get("author_name", ""), c.get("author_full", ""), c.get("content", ""), c.get("chunk_text", ""), c.get("window_line", ""), json.dumps(c.get("window_lines", []), ensure_ascii=False), c.get("reply_to"), c.get("ts", ""), c.get("jump_url", ""), blob
                ))
            con.executemany("""
            INSERT OR REPLACE INTO chunks (msg_id, guild_id, channel_id, channel_name, author_id, author_name, author_full, content, chunk_text, window_line, window_lines, reply_to, ts, jump_url, embedding)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, rows)
            try:
                # FTS5 has no PK on msg_id: REPLACE on chunks never cleans the old
                # FTS row, so re-upserts (edits, re-backfill) would append duplicates.
                con.executemany(
                    "DELETE FROM chunks_fts WHERE guild_id=? AND msg_id=?",
                    [(r[1], r[0]) for r in rows],
                )
                con.executemany(
                    "INSERT INTO chunks_fts (msg_id, guild_id, chunk_text, content) VALUES (?,?,?,?)",
                    [(r[0], r[1], r[8], r[7]) for r in rows],
                )
            except Exception:
                logger.debug("FTS upsert skipped", exc_info=True)
            con.commit()
        except Exception:
            logger.exception("Failed upserting %d history chunks guild %s", len(chunks), guild_id)
        finally:
            con.close()

    def _delete_chunk_from_db(self, guild_id: int, msg_id: str):
        con = sqlite3.connect(self._db_path(), timeout=30)
        try:
            con.execute("DELETE FROM chunks WHERE guild_id=? AND msg_id=?", (guild_id, msg_id))
            try:
                con.execute("DELETE FROM chunks_fts WHERE guild_id=? AND msg_id=?", (guild_id, msg_id))
            except Exception:
                pass
            con.commit()
        except Exception:
            logger.exception("Failed deleting history chunk guild %s msg %s", guild_id, msg_id)
        finally:
            con.close()

    # --- Authors table maintenance (rename-proof identity + aliases) ---

    @staticmethod
    def _insert_author_rows(con: sqlite3.Connection, agg: dict[tuple[int, str], dict]) -> None:
        rows = []
        for (gid, aid), a in agg.items():
            rows.append((
                gid, aid, a['display_name'], _handle_from_label(a['full_label']),
                a['full_label'], json.dumps(sorted(a['aliases']), ensure_ascii=False),
                a['first_seen'], a['last_seen'], a['count'],
            ))
        con.executemany("""
        INSERT OR REPLACE INTO authors (guild_id, author_id, display_name, handle, full_label, aliases, first_seen, last_seen, msg_count)
        VALUES (?,?,?,?,?,?,?,?,?)
        """, rows)

    def _upsert_authors(self, guild_id: int, chunks: list[dict], count_msg_ids: set[str] | None = None):
        """Merge newly indexed chunks into the authors table.

        ``count_msg_ids`` limits the msg_count increment to messages that were
        NOT already stored (re-indexing an edited or re-backfilled message must
        not inflate counts); names/aliases are merged from every chunk. Pass an
        empty set for alias-only refreshes.
        """
        if not chunks:
            return
        rows = [
            (guild_id, str(c.get('author_id', '')), c.get('author_name'), c.get('author_full'), c.get('ts') or '')
            for c in chunks
        ]
        agg = _aggregate_authors(rows)
        if not agg:
            return
        if count_msg_ids is not None:
            for a in agg.values():
                a['count'] = 0
            for c in chunks:
                if str(c.get('msg_id', '')) in count_msg_ids:
                    aid = str(c.get('author_id', ''))
                    if (guild_id, aid) in agg:
                        agg[(guild_id, aid)]['count'] += 1
        con = sqlite3.connect(self._db_path(), timeout=30)
        try:
            con.execute("BEGIN IMMEDIATE")
            for (gid, aid) in agg:
                r = con.execute(
                    "SELECT aliases, first_seen, last_seen, msg_count FROM authors WHERE guild_id=? AND author_id=?",
                    (gid, aid),
                ).fetchone()
                if r is None:
                    continue
                a = agg[(gid, aid)]
                try:
                    a['aliases'] |= set(json.loads(r[0] or '[]'))
                except Exception:
                    pass
                if r[1] and (not a['first_seen'] or r[1] < a['first_seen']):
                    a['first_seen'] = r[1]
                if r[2] and (not a['last_seen'] or r[2] > a['last_seen']):
                    a['last_seen'] = r[2]
                a['count'] += int(r[3] or 0)
            self._insert_author_rows(con, agg)
            con.commit()
        except Exception:
            con.rollback()
            logger.exception("Failed upserting authors guild %s", guild_id)
        finally:
            con.close()

    def _existing_msg_ids(self, guild_id: int, msg_ids: list[str]) -> set[str]:
        """Return the subset of msg_ids already stored for a guild."""
        msg_ids = [str(m) for m in msg_ids if m]
        if not msg_ids:
            return set()
        con = sqlite3.connect(self._db_path(), timeout=30)
        try:
            placeholders = ','.join('?' * len(msg_ids))
            cur = con.execute(
                f"SELECT msg_id FROM chunks WHERE guild_id=? AND msg_id IN ({placeholders})",
                (guild_id, *msg_ids),
            )
            return {str(r[0]) for r in cur.fetchall()}
        except Exception:
            logger.debug("existing msg_id probe failed", exc_info=True)
            return set()
        finally:
            con.close()

    def _rebuild_authors_from_chunks(self) -> int:
        """Migration: populate authors for every guild that has chunks but no authors rows.

        Per-guild (not global): one guild's existing rows must not block
        another guild's backfill. Rows are read in ts order so the canonical
        name is the most recent one seen.
        """
        con = sqlite3.connect(self._db_path(), timeout=30)
        try:
            try:
                chunk_guilds = [int(r[0]) for r in con.execute("SELECT DISTINCT guild_id FROM chunks")]
                author_guilds = {int(r[0]) for r in con.execute("SELECT DISTINCT guild_id FROM authors")}
            except sqlite3.OperationalError:
                return 0
            missing = [g for g in chunk_guilds if g not in author_guilds]
            if not missing:
                return 0
            total = 0
            for g in missing:
                rows = con.execute(
                    "SELECT guild_id, author_id, author_name, author_full, ts FROM chunks WHERE guild_id=? ORDER BY ts ASC",
                    (g,),
                ).fetchall()
                agg = _aggregate_authors(rows)
                if not agg:
                    continue
                self._insert_author_rows(con, agg)
                total += len(agg)
            con.commit()
            if total:
                logger.info("Rebuilt authors table: %d authors across %d guilds", total, len(missing))
            return total
        except Exception:
            con.rollback()
            logger.exception("Failed rebuilding authors table")
            return 0
        finally:
            con.close()

    def _adjust_author_count(self, guild_id: int, author_id: str | None, delta: int):
        if not author_id:
            return
        try:
            con = sqlite3.connect(self._db_path(), timeout=30)
            try:
                con.execute(
                    "UPDATE authors SET msg_count = MAX(0, msg_count + ?) WHERE guild_id=? AND author_id=?",
                    (delta, guild_id, author_id),
                )
                con.commit()
            finally:
                con.close()
        except Exception:
            logger.debug("Failed adjusting author count", exc_info=True)

    def _load_guild(self, guild_id: int) -> bool:
        return self._load_guild_from_db(guild_id)

    def _rebuild_recent(self, guild_id: int):
        chunks = self._chunks.get(guild_id, [])
        per_channel: dict[int, list[dict]] = {}
        for c in chunks:
            per_channel.setdefault(int(c["channel_id"]), []).append(c)
        for cid, lst in per_channel.items():
            recent = collections.deque(maxlen=HISTORY_WINDOW_SIZE)
            for ch in lst[-HISTORY_WINDOW_SIZE:]:
                recent.append(format_chunk_line(ch))
            self._recent[cid] = recent

    def _lock(self, guild_id: int) -> asyncio.Lock:
        if guild_id not in self._locks:
            self._locks[guild_id] = asyncio.Lock()
        return self._locks[guild_id]

    def _load_all_guilds(self):
        path = self._db_path()
        if os.path.exists(path):
            con = sqlite3.connect(path)
            cur = con.execute("SELECT DISTINCT guild_id FROM chunks")
            for row in cur.fetchall():
                gid = int(row[0])
                if gid not in self._chunks:
                    self._load_guild_from_db(gid)
            con.close()

    async def cog_load(self):
        if not HISTORY_ENABLED:
            logger.info("HistoryRAG disabled via HISTORY_ENABLED=false")
            return
        if EMBEDDING_PROVIDER == "local":
            try:
                await asyncio.to_thread(self._get_local_model)
            except Exception:
                logger.warning("HistoryRAG local model failed, will fallback to remote if possible")
        try:
            await asyncio.to_thread(self._ensure_db)
        except Exception:
            logger.exception("Failed to ensure DB")
        try:
            await asyncio.to_thread(self._load_all_guilds)
        except Exception:
            logger.exception("Failed to scan history store")
        try:
            await asyncio.to_thread(self._rebuild_authors_from_chunks)
        except Exception:
            logger.exception("Failed authors table migration")
        self._worker_task = asyncio.create_task(self._ingest_worker())
        self.bot.loop.create_task(self._background_backfill_all())

    async def cog_unload(self):
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def _ingest_worker(self):
        batch: list[tuple[discord.Guild, discord.abc.Messageable, discord.Message]] = []
        while True:
            try:
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=HISTORY_INGEST_FLUSH_SECONDS)
                    batch.append(item)
                    while len(batch) < HISTORY_INGEST_BATCH_SIZE and not self._queue.empty():
                        try:
                            batch.append(self._queue.get_nowait())
                        except asyncio.QueueEmpty:
                            break
                    if len(batch) >= HISTORY_INGEST_BATCH_SIZE:
                        await self._flush_batch(batch)
                        batch = []
                except asyncio.TimeoutError:
                    if batch:
                        await self._flush_batch(batch)
                        batch = []
            except asyncio.CancelledError:
                if batch:
                    try:
                        await self._flush_batch(batch)
                    except Exception:
                        pass
                break
            except Exception:
                logger.exception("Ingest worker error")
                batch = []
                await asyncio.sleep(1)

    async def _flush_batch(self, batch: list[tuple]):
        grouped: dict[tuple[int, int], list[discord.Message]] = {}
        guild_map: dict[int, discord.Guild] = {}
        channel_map: dict[tuple[int,int], discord.abc.Messageable] = {}
        for guild, channel, msg in batch:
            key = (guild.id, channel.id)
            grouped.setdefault(key, []).append(msg)
            guild_map[guild.id] = guild
            channel_map[key] = channel
        for (gid, cid), msgs in grouped.items():
            guild = guild_map[gid]
            channel = channel_map[(gid, cid)]
            try:
                await self._index_batch(guild, channel, msgs)
                await asyncio.sleep(0.2)
            except Exception:
                logger.exception("Flush batch failed guild %s channel %s", gid, cid)

    async def _background_backfill_all(self):
        await self.bot.wait_until_ready()
        await asyncio.sleep(5)
        for guild in self.bot.guilds:
            if guild.id in self._backfilling:
                continue
            asyncio.create_task(self._backfill_guild(guild))

    async def _backfill_guild(self, guild: discord.Guild):
        if guild.id in self._backfilling:
            return
        self._backfilling.add(guild.id)
        try:
            lock = self._lock(guild.id)
            async with lock:
                if guild.id not in self._chunks:
                    self._chunks[guild.id] = []
                    self._matrices[guild.id] = None
                    self._msg_index[guild.id] = {}
                channels: list[discord.abc.Messageable] = []
                for ch in guild.channels:
                    if isinstance(ch, (discord.TextChannel, discord.Thread, discord.VoiceChannel)):
                        if isinstance(ch, discord.VoiceChannel):
                            continue
                        perms = ch.permissions_for(guild.me)
                        if perms.read_message_history and perms.view_channel:
                            channels.append(ch)
                    elif isinstance(ch, discord.ForumChannel):
                        for thread in ch.threads:
                            if thread.permissions_for(guild.me).read_message_history:
                                channels.append(thread)
                        try:
                            async for thread in ch.archived_threads(limit=None):
                                if thread.permissions_for(guild.me).read_message_history and thread.permissions_for(guild.me).view_channel:
                                    channels.append(thread)
                                await asyncio.sleep(0)
                        except Exception:
                            pass
                for cat in guild.channels:
                    if isinstance(cat, discord.CategoryChannel):
                        for ch in cat.channels:
                            if isinstance(ch, discord.Thread):
                                perms = ch.permissions_for(guild.me)
                                if perms.read_message_history and perms.view_channel and ch not in channels:
                                    channels.append(ch)
                channels = list(dict.fromkeys(channels))
                logger.info("History backfill guild %s (%s) %d channels", guild.name, guild.id, len(channels))
                for channel in channels:
                    await self._backfill_channel(guild, channel)
        except Exception:
            logger.exception("Backfill failed for guild %s", guild.id)
        finally:
            self._backfilling.discard(guild.id)

    async def _backfill_channel(self, guild: discord.Guild, channel: discord.abc.Messageable):
        cid = channel.id
        gid = guild.id
        existing = self._msg_index.get(gid, {})
        recent = self._recent.get(cid)
        if recent is None:
            recent = collections.deque(maxlen=HISTORY_WINDOW_SIZE)
            self._recent[cid] = recent
        to_index: list[discord.Message] = []
        try:
            async for msg in channel.history(limit=HISTORY_BACKFILL_LIMIT, oldest_first=True):
                if msg.id in existing:
                    line = format_message_line(msg)
                    recent.append(line)
                    continue
                if HISTORY_EXCLUDE_BOTS and msg.author.bot:
                    continue
                if not msg.content and not msg.attachments and not msg.embeds:
                    continue
                to_index.append(msg)
                if len(to_index) >= 200:
                    await self._index_batch(guild, channel, to_index)
                    to_index = []
                    await asyncio.sleep(0.3)
            if to_index:
                await self._index_batch(guild, channel, to_index)
        except discord.Forbidden:
            logger.warning("No permission to read history %s/%s", guild.id, cid)
        except Exception:
            logger.exception("Failed backfill channel %s", cid)

    def _append_chunks(self, gid: int, chunks: list[dict], embeddings: list):
        if not chunks or not embeddings:
            return
        stacked = np.asarray(embeddings, dtype=np.float32)
        if stacked.ndim != 2 or stacked.shape[0] != len(chunks):
            logger.warning(
                "Embedding batch mismatch guild %s: %s embeddings for %s chunks",
                gid, stacked.shape, len(chunks),
            )
            return
        mat = self._matrices.get(gid)
        self._matrices[gid] = (
            stacked if mat is None or mat.shape[0] == 0 else np.vstack([mat, stacked])
        )
        for ch in chunks:
            self._chunks[gid].append(ch)
            self._msg_index[gid][int(ch["msg_id"])] = len(self._chunks[gid]) - 1

    async def _index_batch(self, guild: discord.Guild, channel: discord.abc.Messageable, msgs: list[discord.Message]):
        if not msgs:
            return
        gid = guild.id
        cid = channel.id
        recent = self._recent.get(cid)
        if recent is None:
            recent = collections.deque(maxlen=max(1, HISTORY_WINDOW_SIZE))
            self._recent[cid] = recent
        chunks: list[dict] = []
        texts: list[str] = []
        for msg in msgs:
            window_lines = list(recent)
            cur_line = format_message_line(msg)
            if "```" in cur_line:
                cur_line = cur_line.replace("```", "ˋˋˋ")
            if window_lines:
                chunk_text = "\n".join(window_lines + [cur_line])
            else:
                chunk_text = cur_line
            texts.append(chunk_text)
            jump = _jump_url(msg)
            ref = getattr(msg, "reference", None)
            ref_id = getattr(ref, "message_id", None)
            chunk = {
                "msg_id": str(msg.id),
                "channel_id": str(cid),
                "guild_id": str(gid),
                "channel_name": getattr(channel, "name", str(cid)),
                "author_id": str(msg.author.id),
                "author_name": getattr(msg.author, "display_name", str(msg.author)),
                "author_full": f"{getattr(msg.author, 'display_name', str(msg.author))} (@{msg.author.name})",
                "content": message_content_text(msg, max_length=HISTORY_MAX_MSG_LENGTH),
                "chunk_text": chunk_text,
                "window_lines": window_lines.copy(),
                "window_line": cur_line,
                "reply_to": str(ref_id) if ref_id else None,
                "ts": msg.created_at.isoformat(),
                "jump_url": jump,
            }
            chunks.append(chunk)
            recent.append(cur_line)
        try:
            embeddings = await self._embed_batch(texts)
        except Exception:
            logger.exception("Failed to embed history batch guild %s channel %s", gid, cid)
            return
        lock = self._lock(gid)
        if lock.locked():
            self._append_chunks(gid, chunks, embeddings)
        else:
            async with lock:
                self._append_chunks(gid, chunks, embeddings)
        logger.info("Indexed %d msgs guild %s channel %s (total %d)", len(chunks), gid, cid, len(self._chunks[gid]))
        try:
            batch_ids = [str(c['msg_id']) for c in chunks]
            already_stored = await asyncio.to_thread(self._existing_msg_ids, gid, batch_ids)
            await asyncio.to_thread(self._upsert_chunks_to_db, gid, chunks, embeddings)
            await asyncio.to_thread(
                self._upsert_authors, gid, chunks,
                count_msg_ids=set(batch_ids) - already_stored,
            )
        except Exception:
            logger.exception("Failed persisting history batch guild %s", gid)

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not HISTORY_ENABLED or not message.guild:
            return
        if HISTORY_EXCLUDE_BOTS and message.author.bot:
            return
        if not message.content and not message.attachments and not message.embeds:
            return
        gid = message.guild.id
        if gid not in self._chunks:
            self._chunks[gid] = []
            self._matrices[gid] = None
            self._msg_index[gid] = {}
            await asyncio.to_thread(self._load_guild, gid)
        if message.id in self._msg_index.get(gid, {}):
            return
        await self._queue.put((message.guild, message.channel, message))

    @commands.Cog.listener()
    async def on_message_edit(self, before: discord.Message, after: discord.Message):
        if not HISTORY_ENABLED or not after.guild:
            return
        gid = after.guild.id
        idx = self._msg_index.get(gid, {}).get(after.id)
        if idx is None:
            return
        try:
            new_content = message_content_text(after, max_length=HISTORY_MAX_MSG_LENGTH)
            chunk = self._chunks[gid][idx]
            if chunk.get("content") == new_content:
                return
            chunk["content"] = new_content
            chunk["window_line"] = format_message_line(after)
            chunk["chunk_text"] = "\n".join(chunk.get("window_lines", []) + [chunk["window_line"]])
            emb = np.asarray((await self._embed_batch([chunk["chunk_text"]]))[0], dtype=np.float32)
            self._matrices[gid][idx] = emb
            await asyncio.to_thread(self._upsert_chunks_to_db, gid, [chunk], [emb])
            await asyncio.to_thread(self._upsert_authors, gid, [chunk], count_msg_ids=set())
        except Exception:
            logger.exception("Failed to update edited message %s", after.id)

    @commands.Cog.listener()
    async def on_message_delete(self, message: discord.Message):
        if not message.guild:
            return
        gid = message.guild.id
        idx = self._msg_index.get(gid, {}).get(message.id)
        if idx is None:
            return
        async with self._lock(gid):
            chunks = self._chunks[gid]
            chunks.pop(idx)
            self._msg_index[gid] = {int(c["msg_id"]): i for i, c in enumerate(chunks)}
            if chunks:
                self._matrices[gid] = np.delete(self._matrices[gid], idx, axis=0)
                await asyncio.to_thread(self._delete_chunk_from_db, gid, str(message.id))
                await asyncio.to_thread(
                    self._adjust_author_count, gid,
                    str(getattr(message.author, 'id', '') or ''), -1,
                )
            else:
                self._matrices[gid] = None
                try:
                    def _purge_guild():
                        con = sqlite3.connect(self._db_path(), timeout=30)
                        try:
                            con.execute("DELETE FROM chunks WHERE guild_id=?", (gid,))
                            try:
                                con.execute("DELETE FROM chunks_fts WHERE guild_id=?", (gid,))
                            except Exception:
                                pass
                            try:
                                con.execute("DELETE FROM authors WHERE guild_id=?", (gid,))
                            except Exception:
                                pass
                            con.commit()
                        finally:
                            con.close()
                    await asyncio.to_thread(_purge_guild)
                except Exception:
                    pass

    async def _resolve_author(self, guild_id: int, author_id: str | None, author_name: str | None) -> list[str]:
        if author_id:
            return [str(author_id)]
        if not author_name:
            return []
        ids = await asyncio.to_thread(self._resolve_author_db, guild_id, author_name)
        if ids is None:
            ids = []
        if not ids:
            # Empty-but-successful lookup (authors table missing rows for this
            # guild — failed migration, legacy JSON store) is not authoritative:
            # fall back to the in-memory chunk scan.
            name_low = author_name.lower()
            matched = set()
            for c in self._chunks.get(guild_id, []):
                if name_low in c.get("author_name","").lower() or name_low in c.get("author_full","").lower() or name_low in c.get("author_id",""):
                    matched.add(c["author_id"])
            return list(matched)
        return ids

    def _resolve_author_db(self, guild_id: int, author_name: str) -> list[str] | None:
        """Resolve a name to author IDs via the authors table (aliases included).

        Returns None when the table is unusable so callers fall back to the
        in-memory chunk scan.
        """
        try:
            con = sqlite3.connect(self._db_path())
            con.row_factory = sqlite3.Row
            try:
                like = f"%{_escape_like(author_name.lower())}%"
                cur = con.execute(
                    """SELECT author_id FROM authors
                       WHERE guild_id=? AND (
                           lower(display_name) LIKE ? ESCAPE '\\'
                           OR lower(handle) LIKE ? ESCAPE '\\'
                           OR lower(full_label) LIKE ? ESCAPE '\\'
                           OR lower(aliases) LIKE ? ESCAPE '\\')
                       ORDER BY msg_count DESC LIMIT 20""",
                    (guild_id, like, like, like, like),
                )
                return [str(r["author_id"]) for r in cur.fetchall()]
            finally:
                con.close()
        except Exception:
            logger.debug("authors table resolve failed", exc_info=True)
            return None

    def _fts_where(
        self,
        guild_id: int,
        channel_id: str | None = None,
        author_ids: set[str] | None = None,
        dt_after: datetime.datetime | None = None,
        dt_before: datetime.datetime | None = None,
    ) -> tuple[str, list]:
        """Build the JOIN+WHERE fragment pushing channel/author/date filters into SQL."""
        sql = "JOIN chunks c ON c.msg_id = f.msg_id WHERE f.guild_id=?"
        params: list = [guild_id]
        if channel_id:
            sql += " AND c.channel_id=?"
            params.append(str(channel_id))
        if author_ids:
            sql += f" AND c.author_id IN ({','.join('?' * len(author_ids))})"
            params.extend(str(a) for a in author_ids)
        if dt_after:
            sql += " AND datetime(c.ts) >= datetime(?)"
            params.append(dt_after.isoformat())
        if dt_before:
            sql += " AND datetime(c.ts) <= datetime(?)"
            params.append(dt_before.isoformat())
        return sql, params

    def _fts_search(
        self,
        query: str,
        guild_id: int,
        limit: int = 100,
        channel_id: str | None = None,
        author_ids: set[str] | None = None,
        dt_after: datetime.datetime | None = None,
        dt_before: datetime.datetime | None = None,
    ) -> dict[str, float]:
        fts_q = _sanitize_fts_query(query)
        if not fts_q:
            return {}
        where_sql, params = self._fts_where(guild_id, channel_id, author_ids, dt_after, dt_before)
        try:
            con = sqlite3.connect(self._db_path())
            con.row_factory = sqlite3.Row
            try:
                cur = con.execute(
                    f"SELECT f.msg_id AS msg_id, f.rank AS rank FROM chunks_fts f {where_sql} AND f.chunks_fts MATCH ? ORDER BY f.rank LIMIT ?",
                    (*params, fts_q, limit),
                )
                rows = cur.fetchall()
            finally:
                con.close()
            scores: dict[str, float] = {}
            for idx, r in enumerate(rows):
                try:
                    rank_val = float(r["rank"])
                    scores[str(r["msg_id"])] = -rank_val if rank_val < 0 else 1 / (1 + rank_val)
                except Exception:
                    scores[str(r["msg_id"])] = 1 / (1 + idx)
            return scores
        except Exception:
            return {}

    def _fts_search_rows(
        self,
        query: str,
        guild_id: int,
        limit: int,
        channel_id: str | None = None,
        author_ids: set[str] | None = None,
        dt_after: datetime.datetime | None = None,
        dt_before: datetime.datetime | None = None,
    ) -> list[tuple[dict, float]]:
        """FTS search returning full chunk rows (rank order) with their scores."""
        fts_q = _sanitize_fts_query(query)
        if not fts_q:
            return []
        where_sql, params = self._fts_where(guild_id, channel_id, author_ids, dt_after, dt_before)
        try:
            con = sqlite3.connect(self._db_path())
            con.row_factory = sqlite3.Row
            try:
                cur = con.execute(
                    "SELECT c.msg_id, c.guild_id, c.channel_id, c.channel_name, c.author_id, c.author_name, c.author_full, c.content, c.chunk_text, c.window_line, c.window_lines, c.reply_to, c.ts, c.jump_url, f.rank AS fts_rank "
                    f"FROM chunks_fts f {where_sql} AND f.chunks_fts MATCH ? ORDER BY f.rank LIMIT ?",
                    (*params, fts_q, limit),
                )
                rows = cur.fetchall()
            finally:
                con.close()
        except Exception:
            return []
        out: list[tuple[dict, float]] = []
        for idx, r in enumerate(rows):
            try:
                rank_val = float(r["fts_rank"])
                score = -rank_val if rank_val < 0 else 1 / (1 + rank_val)
            except Exception:
                score = 1 / (1 + idx)
            d = dict(r)
            d.pop("fts_rank", None)
            try:
                d["window_lines"] = json.loads(d.get("window_lines") or "[]")
            except Exception:
                d["window_lines"] = []
            out.append((d, score))
        return out

    async def _rerank_history(self, query: str, candidates: list[dict], top_n: int) -> list[dict]:
        if not candidates or not HISTORY_RERANK_ENABLED:
            return candidates[:top_n]
        provider = HISTORY_RERANK_PROVIDER if HISTORY_RERANK_PROVIDER != "auto" else RERANK_PROVIDER
        use_local = provider in ("local", "auto") and (provider == "local" or "openrouter.ai" not in OPENAI_BASE_URL)
        if use_local:
            try:
                reranker = self._get_local_reranker()
                docs = [c.get("chunk_text", c.get("content", ""))[:800] for c in candidates]
                pairs = [[query, d] for d in docs]
                scores = await asyncio.to_thread(reranker.predict, pairs)
                scored = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
                return [c for _, c in scored[:top_n]]
            except Exception:
                logger.exception("Local rerank failed, fallback to vector order")
                return candidates[:top_n]
        if not provider == "remote" and "openrouter.ai" not in OPENAI_BASE_URL:
            return candidates[:top_n]
        try:
            import aiohttp as _aiohttp
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
            payload = {"model": HISTORY_RERANK_MODEL or LOCAL_RERANK_MODEL, "query": query, "documents": [c.get("chunk_text", c.get("content", ""))[:600] for c in candidates], "top_n": top_n}
            timeout = _aiohttp.ClientTimeout(total=8)
            async with _aiohttp.ClientSession(timeout=timeout) as sess:
                async with sess.post(f"{OPENAI_BASE_URL}/rerank", headers=headers, json=payload) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        results = data.get("results", [])
                        results.sort(key=lambda r: r.get("relevance_score", 0), reverse=True)
                        return [candidates[r["index"]] for r in results[:top_n] if 0 <= r["index"] < len(candidates)]
        except Exception:
            logger.exception("Remote rerank failed")
        return candidates[:top_n]

    def _dedupe_adjacent(self, results: list[dict], limit: int) -> list[dict]:
        """Drop overlapping window results: same channel, adjacent in time.

        Each chunk embeds its preceding window, so consecutive messages from one
        conversation produce near-identical results. Keep the best-scoring
        message per conversation window and let get_message_context expand.
        """
        if HISTORY_DEDUPE_WINDOW_MINUTES <= 0 or not results:
            return results[:limit]
        window_s = HISTORY_DEDUPE_WINDOW_MINUTES * 60
        accepted: list[dict] = []
        accepted_ts: list[tuple[str, float]] = []
        for r in results:
            if len(accepted) >= limit:
                break
            cid = str(r.get("channel_id", ""))
            ts = _parse_dt(r.get("ts"))
            if ts is not None:
                dup = any(ac_cid == cid and abs(ts.timestamp() - ac_ts) <= window_s for ac_cid, ac_ts in accepted_ts)
                if dup:
                    continue
                accepted_ts.append((cid, ts.timestamp()))
            accepted.append(r)
        return accepted

    async def _keyword_search(self, query: str, guild_id: int, limit: int, channel_id: str | None, author_ids: set[str] | None, dt_after: datetime.datetime | None, dt_before: datetime.datetime | None, sort_by: str) -> list[dict]:
        """Keyword search served straight from SQLite with filters pushed into SQL."""
        rows = await asyncio.to_thread(
            self._fts_search_rows, query, guild_id, max(limit * 8, 100),
            channel_id, author_ids, dt_after, dt_before,
        )
        if not rows:
            return []
        out = []
        for row, score in rows:
            decay = _time_decay_factor(row.get("ts", ""), HISTORY_TIME_DECAY_LAMBDA) if sort_by == "recent" else 1.0
            out.append(dict(row, _score=float(score * decay), _keyword_score=float(score)))
        if sort_by == "recent":
            out.sort(key=lambda r: r.get("ts", ""), reverse=True)
        return out

    def _keyword_fallback(self, query: str, guild_id: int, limit: int, channel_id: str | None, author_ids: set[str] | None, dt_after: datetime.datetime | None, dt_before: datetime.datetime | None, sort_by: str) -> list[dict]:
        """Substring keyword scan over in-memory chunks (FTS unavailable or empty)."""
        scored = []
        for c in self._chunks.get(guild_id, []):
            if channel_id and c["channel_id"] != str(channel_id):
                continue
            if author_ids and c["author_id"] not in author_ids:
                continue
            if dt_after or dt_before:
                ts = _parse_dt(c.get("ts"))
                if ts is None:
                    continue
                if dt_after and ts < dt_after:
                    continue
                if dt_before and ts > dt_before:
                    continue
            ks = _keyword_score(query, c)
            if ks <= 0:
                continue
            decay = _time_decay_factor(c["ts"], HISTORY_TIME_DECAY_LAMBDA) if sort_by == "recent" and HISTORY_TIME_DECAY_LAMBDA > 0 else 1.0
            scored.append((ks * decay, c))
        if sort_by == "recent":
            # Same rule as the DB keyword path: 'recent' means ts order.
            scored.sort(key=lambda x: x[1]["ts"], reverse=True)
        else:
            scored.sort(reverse=True, key=lambda x: x[0])
        return [dict(c, _score=float(s), _keyword_score=float(s)) for s, c in scored]

    async def search(self, query: str, guild_id: int, limit: int = 5, channel_id: str | None = None, author_id: str | None = None, author_name: str | None = None, after: str | None = None, before: str | None = None, search_mode: str = "hybrid", sort_by: str = "relevance", dedupe: bool = True) -> list[dict]:
        dt_after = _parse_dt(after) if after else None
        dt_before = _parse_dt(before) if before else None
        author_ids = None
        if author_id or author_name:
            author_ids = set(await self._resolve_author(guild_id, author_id, author_name))
            if author_id and str(author_id) not in author_ids:
                author_ids.add(str(author_id))
            if not author_ids and (author_id or author_name):
                return []
        cache_key = f"{guild_id}:{query}:{limit}:{channel_id}:{author_id}:{author_name}:{after}:{before}:{search_mode}:{sort_by}:{dedupe}"
        try:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                return [dict(c) for c in cached]
        except Exception:
            pass
        if search_mode == "keyword":
            res = await self._keyword_search(query, guild_id, limit, channel_id, author_ids, dt_after, dt_before, sort_by)
            if not res:
                res = self._keyword_fallback(query, guild_id, limit, channel_id, author_ids, dt_after, dt_before, sort_by)
            res = self._dedupe_adjacent(res, limit) if dedupe else res[:limit]
            if res:
                # Never negative-cache: guilds still loading/backfilling would
                # serve a confident "nothing found" for the whole TTL otherwise.
                try:
                    self._query_cache[cache_key] = [dict(r) for r in res]
                except Exception:
                    pass
            return res
        if guild_id not in self._chunks or not self._chunks[guild_id]:
            return []
        mat = self._matrices.get(guild_id)
        if mat is None or len(mat) == 0:
            return []
        chunks = self._chunks[guild_id]
        indices = list(range(len(chunks)))
        if channel_id:
            indices = [i for i in indices if chunks[i]["channel_id"] == str(channel_id)]
            if not indices:
                return []
        if author_ids:
            indices = [i for i in indices if chunks[i]["author_id"] in author_ids]
            if not indices:
                return []
        if dt_after or dt_before:
            filtered = []
            for i in indices:
                ts = _parse_dt(chunks[i].get("ts"))
                if ts is None:
                    continue
                if dt_after and ts < dt_after:
                    continue
                if dt_before and ts > dt_before:
                    continue
                filtered.append(i)
            indices = filtered
            if not indices:
                return []
        mat_f = mat[np.array(indices)]
        chunks_f = [chunks[i] for i in indices]
        try:
            q_emb = (await self._embed_batch([query]))[0]
        except Exception:
            logger.exception("History search embed failed")
            return []
        q_arr = np.array(q_emb, dtype=np.float32)
        dots = mat_f @ q_arr
        norms = np.linalg.norm(mat_f, axis=1) * np.linalg.norm(q_arr)
        with np.errstate(invalid='ignore', divide='ignore'):
            vec_scores = np.where(norms > 0, dots / norms, 0.0)
        if search_mode == "hybrid":
            fts_scores_map = await asyncio.to_thread(self._fts_search, query, guild_id, 200, channel_id, author_ids, dt_after, dt_before)
            if fts_scores_map:
                vec_rank = {chunks_f[i]["msg_id"]: float(vec_scores[i]) for i in range(len(chunks_f))}
                allowed_ids = {chunks_f[i]["msg_id"] for i in range(len(chunks_f))}
                fts_filtered = {k: v for k, v in fts_scores_map.items() if k in allowed_ids}
                if fts_filtered:
                    rrf = _rrf_fuse([vec_rank, fts_filtered], k=HISTORY_RRF_K)
                    scores = np.array([rrf.get(chunks_f[i]["msg_id"], 0) for i in range(len(chunks_f))], dtype=np.float32)
                else:
                    kw_scores = np.array([_keyword_score(query, c) for c in chunks_f], dtype=np.float32)
                    w_s = HISTORY_HYBRID_WEIGHT_SEMANTIC
                    w_k = HISTORY_HYBRID_WEIGHT_KEYWORD
                    scores = w_s * vec_scores + w_k * kw_scores
            else:
                kw_scores = np.array([_keyword_score(query, c) for c in chunks_f], dtype=np.float32)
                w_s = HISTORY_HYBRID_WEIGHT_SEMANTIC
                w_k = HISTORY_HYBRID_WEIGHT_KEYWORD
                scores = w_s * vec_scores + w_k * kw_scores
        else:
            scores = vec_scores
        if HISTORY_TIME_DECAY_LAMBDA > 0 and sort_by == "recent":
            decay_factors = np.array([_time_decay_factor(c["ts"], HISTORY_TIME_DECAY_LAMBDA) for c in chunks_f], dtype=np.float32)
            scores = scores * (0.7 + 0.3 * decay_factors)
        if sort_by == "recent" and HISTORY_TIME_DECAY_LAMBDA == 0:
            order = np.argsort([c["ts"] for c in chunks_f])[::-1]
            top_idx = order[: max(limit * 6, 30)]
            candidates = [dict(chunks_f[i], _score=float(scores[i]), _keyword_score=float(_keyword_score(query, chunks_f[i]))) for i in top_idx]
            reranked = await self._rerank_history(query, candidates, top_n=len(candidates))
            reranked = self._dedupe_adjacent(reranked, limit) if dedupe else reranked[:limit]
            if reranked:
                try:
                    self._query_cache[cache_key] = [dict(r) for r in reranked]
                except Exception:
                    pass
            return reranked
        top_k_rerank = max(limit * 6, 30)
        top_idx = np.argsort(scores)[::-1][:top_k_rerank]
        candidates = [dict(chunks_f[i], _score=float(scores[i]), _keyword_score=float(_keyword_score(query, chunks_f[i]))) for i in top_idx]
        if sort_by == "recent" and HISTORY_TIME_DECAY_LAMBDA > 0:
            candidates.sort(key=lambda x: (x["_score"], x["ts"]), reverse=True)
        else:
            # Rerank the full pool so dedupe can pick representatives from beyond
            # the old top-`limit` slice (both rerankers score the whole list anyway).
            candidates = await self._rerank_history(query, candidates, top_n=len(candidates))
        candidates = self._dedupe_adjacent(candidates, limit) if dedupe else candidates[:limit]
        if candidates:
            try:
                self._query_cache[cache_key] = [dict(r) for r in candidates]
            except Exception:
                pass
        return candidates

    async def get_user_stats(self, guild_id: int, author_id: str | None = None, author_name: str | None = None) -> dict:
        if guild_id not in self._chunks:
            return {"error": "Nenhum dado para este servidor."}
        chunks = self._chunks[guild_id]
        if not chunks:
            return {"error": "Nenhum dado."}
        author_ids = await self._resolve_author(guild_id, author_id, author_name)
        if author_id and str(author_id) not in author_ids:
            author_ids.append(str(author_id))
        if not author_ids:
            return {"error": f"Usuário não encontrado: {author_id or author_name}"}
        target_ids = set(author_ids)
        user_chunks = [c for c in chunks if c["author_id"] in target_ids]
        if not user_chunks:
            return {"error": "Nenhuma mensagem deste usuário."}
        total = len(user_chunks)
        by_channel: dict[str, int] = {}
        by_hour: dict[int, int] = {}
        lens = []
        for c in user_chunks:
            by_channel[c.get("channel_name","?")] = by_channel.get(c.get("channel_name","?"), 0) + 1
            try:
                ts = datetime.datetime.fromisoformat(c["ts"].replace("Z","+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=datetime.timezone.utc)
                h = ts.astimezone(BR_TZ).hour if BR_TZ else ts.hour
                by_hour[h] = by_hour.get(h, 0) + 1
            except Exception:
                pass
            lens.append(len(c.get("content","")))
        top_channels = sorted(by_channel.items(), key=lambda x: x[1], reverse=True)[:5]
        top_hours = sorted(by_hour.items(), key=lambda x: x[1], reverse=True)[:5]
        avg_len = sum(lens)/len(lens) if lens else 0
        first_ts = min(c["ts"] for c in user_chunks)
        last_ts = max(c["ts"] for c in user_chunks)
        example = user_chunks[-1]
        return {
            "author_ids": list(target_ids),
            "author_full": user_chunks[0].get("author_full","?"),
            "total_messages": total,
            "avg_length": round(avg_len, 1),
            "top_channels": top_channels,
            "top_hours": top_hours,
            "first_seen": first_ts,
            "last_seen": last_ts,
            "example_jump": example.get("jump_url",""),
            "example_content": example.get("content","")[:300],
        }

    async def count_mentions(self, guild_id: int, query: str, group_by: str = "author", limit: int = 10, after: str | None = None, before: str | None = None) -> list[dict]:
        # dedupe=False: mention counting needs every matching message, not one per conversation window
        results = await self.search(query, guild_id, limit=200, after=after, before=before, search_mode="hybrid", sort_by="relevance", dedupe=False)
        if not results:
            return []
        threshold = 0.15
        relevant = [r for r in results if r.get("_score",0) >= threshold or r.get("_keyword_score",0) > 0]
        if not relevant:
            relevant = results[:30]
        counter: dict[str, dict] = {}
        for r in relevant:
            if group_by == "author":
                key = r.get("author_full", r.get("author_id","?"))
            elif group_by == "channel":
                key = r.get("channel_name","?")
            else:
                try:
                    ts = datetime.datetime.fromisoformat(r["ts"].replace("Z","+00:00"))
                    key = ts.strftime("%Y-%m-%d")
                except Exception:
                    key = r["ts"][:10]
            if key not in counter:
                counter[key] = {"key": key, "count": 0, "example": r}
            counter[key]["count"] += 1
        sorted_groups = sorted(counter.values(), key=lambda x: x["count"], reverse=True)[:limit]
        return sorted_groups

    async def find_users(self, guild_id: int, query: str, limit: int = 5) -> list[dict]:
        """Resolve users by name (current or old), handle, or ID; most active first."""
        return await asyncio.to_thread(self._find_users_db, guild_id, query, limit)

    def _find_users_db(self, guild_id: int, query: str, limit: int) -> list[dict]:
        q = (query or '').strip()
        limit = max(1, min(12, int(limit)))
        con = sqlite3.connect(self._db_path(), timeout=30)
        con.row_factory = sqlite3.Row
        try:
            if q:
                like = f"%{_escape_like(q.lower())}%"
                cur = con.execute(
                    """SELECT * FROM authors
                       WHERE guild_id=? AND (
                           author_id = ?
                           OR lower(display_name) LIKE ? ESCAPE '\\'
                           OR lower(handle) LIKE ? ESCAPE '\\'
                           OR lower(full_label) LIKE ? ESCAPE '\\'
                           OR lower(aliases) LIKE ? ESCAPE '\\')
                       ORDER BY msg_count DESC LIMIT ?""",
                    (guild_id, q, like, like, like, like, limit),
                )
            else:
                cur = con.execute(
                    "SELECT * FROM authors WHERE guild_id=? ORDER BY msg_count DESC LIMIT ?",
                    (guild_id, limit),
                )
            users = []
            for r in cur.fetchall():
                u = dict(r)
                try:
                    u['aliases'] = [a for a in json.loads(u.get('aliases') or '[]') if a]
                except Exception:
                    u['aliases'] = []
                try:
                    chs = con.execute(
                        "SELECT channel_name, COUNT(*) AS n FROM chunks WHERE guild_id=? AND author_id=? GROUP BY channel_name ORDER BY n DESC LIMIT 3",
                        (guild_id, u['author_id']),
                    ).fetchall()
                    u['top_channels'] = [(c['channel_name'], c['n']) for c in chs]
                except Exception:
                    u['top_channels'] = []
                users.append(u)
            return users
        finally:
            con.close()

    async def exec_sql(self, guild_id: int | None, sql: str, *, timeout_s: float | None = None, max_rows: int | None = None) -> str:
        """Run one validated read-only SELECT over the history DB, rendered as text."""
        timeout = HISTORY_SQL_TIMEOUT_SECONDS if timeout_s is None else max(0.1, timeout_s)
        rows_cap = max(1, min(2000, HISTORY_SQL_MAX_ROWS if max_rows is None else max_rows))
        return await asyncio.to_thread(self._exec_sql_sync, guild_id, sql, timeout, rows_cap)

    def _exec_sql_sync(self, guild_id: int | None, sql: str, timeout_s: float, max_rows: int) -> str:
        _validate_history_sql(sql, guild_id)
        started = time.monotonic()
        deadline = started + timeout_s
        path = self._db_path()
        try:
            uri = "file:" + path.replace("?", "%3f").replace("#", "%23") + "?mode=ro"
            con = sqlite3.connect(uri, uri=True, timeout=1)
        except sqlite3.Error:
            # e.g. read-only WAL open impossible in this filesystem: authorizer
            # and query_only below still keep the connection non-mutating.
            con = sqlite3.connect(path, timeout=1)
        try:
            try:
                con.execute("PRAGMA query_only=1")
            except sqlite3.Error:
                pass

            def _auth(action, arg1, arg2, *_):
                if action == sqlite3.SQLITE_READ and (arg1, arg2) == ("chunks", "embedding"):
                    return sqlite3.SQLITE_DENY
                # FTS5 internals: consistency PRAGMA + shadow tables (<fts>_config/_data/...)
                if action == sqlite3.SQLITE_PRAGMA and arg1 == "data_version":
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ and isinstance(arg1, str) and arg1.startswith("chunks_fts"):
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_OK if action in _SQL_AUTHOR_ALLOWED else sqlite3.SQLITE_DENY

            con.set_authorizer(_auth)
            con.set_progress_handler(lambda: time.monotonic() > deadline, 100_000)

            def _regexp(pattern, value):
                if pattern is None or value is None:
                    return None
                if len(str(pattern)) > _SQL_MAX_PATTERN:
                    raise ValueError(f"Padrão REGEXP muito longo (máx. {_SQL_MAX_PATTERN}).")
                try:
                    return re.search(str(pattern), str(value), re.I) is not None
                except re.error as e:
                    raise ValueError(f"REGEXP inválido: {e}")

            con.create_function("regexp", 2, _regexp)
            try:
                cur = con.execute(sql)
                cols = [d[0] for d in cur.description] if cur.description else []
                rows = cur.fetchmany(max_rows + 1)
            except sqlite3.OperationalError as e:
                if "user-defined function raised exception" in str(e):
                    raise ValueError(
                        "Padrão REGEXP inválido (erro dentro da função regexp) — corrija a sintaxe."
                    ) from e
                raise
            truncated = len(rows) > max_rows
            rows = rows[:max_rows]
        finally:
            con.close()
        lines = [" | ".join(cols)] if cols else []
        for r in rows:
            lines.append(" | ".join(_sql_cell(v) for v in r))
            if sum(len(l) for l in lines) > _SQL_OUTPUT_CHARS:
                lines.pop()
                truncated = True
                lines.append(f"…saída truncada em {_SQL_OUTPUT_CHARS} caracteres")
                break
        if truncated and "…saída truncada" not in lines[-1]:
            lines.append(f"…resultado truncado em {max_rows} linhas — adicione LIMIT/WHERE mais restritivo")
        elapsed_ms = (time.monotonic() - started) * 1000
        logger.info("sql_history guild=%s rows=%d %.0fms sql=%s", guild_id, len(rows), elapsed_ms, " ".join(sql.split())[:300])
        return "\n".join(lines) if lines else "(sem linhas)"

    async def get_message_context_from_index(self, guild_id: int, channel_id: str, message_id: str, window: int = 5) -> str | None:
        """Build a ±window context block from the history index (DB-backed).

        Returns None when the message is not indexed — callers fall back to the
        Discord API path.
        """
        try:
            window = max(1, min(10, int(window)))
            rows = await asyncio.to_thread(self._message_context_rows, guild_id, channel_id, message_id, window)
        except Exception:
            logger.exception("Indexed message context failed guild %s msg %s", guild_id, message_id)
            return None
        if rows is None:
            return None
        lines = []
        for r in rows:
            prefix = '▶ ' if str(r.get('msg_id')) == str(message_id) else '  '
            lines.append(prefix + format_chunk_line(r))
        if not lines:
            return None
        chan = rows[0].get('channel_name') or channel_id
        header = (
            f"Contexto ao redor de {message_id} em #{chan} "
            f"(channel_id={channel_id}, ±{window}):\n"
        )
        return header + "\n".join(lines)

    def _message_context_rows(self, guild_id: int, channel_id: str, message_id: str, window: int) -> list[dict] | None:
        con = sqlite3.connect(self._db_path(), timeout=30)
        con.row_factory = sqlite3.Row
        try:
            anchor = con.execute(
                "SELECT ts FROM chunks WHERE guild_id=? AND channel_id=? AND msg_id=?",
                (guild_id, str(channel_id), str(message_id)),
            ).fetchone()
            if anchor is None or not anchor["ts"]:
                return None
            ts = anchor["ts"]
            cols = "msg_id, guild_id, channel_id, channel_name, author_id, author_name, author_full, content, chunk_text, window_line, window_lines, reply_to, ts, jump_url"
            before = con.execute(
                f"SELECT {cols} FROM chunks WHERE guild_id=? AND channel_id=? AND ts<=? ORDER BY ts DESC LIMIT ?",
                (guild_id, str(channel_id), ts, window + 1),
            ).fetchall()
            after = con.execute(
                f"SELECT {cols} FROM chunks WHERE guild_id=? AND channel_id=? AND ts>? ORDER BY ts ASC LIMIT ?",
                (guild_id, str(channel_id), ts, window),
            ).fetchall()
            out = []
            for r in list(reversed(before)) + list(after):
                d = dict(r)
                try:
                    d["window_lines"] = json.loads(d.get("window_lines") or "[]")
                except Exception:
                    d["window_lines"] = []
                out.append(d)
            return out
        finally:
            con.close()

    @app_commands.command(name="history", description="Buscar no histórico do servidor (RAG)")
    @app_commands.describe(query="O que buscar", user="Filtrar por usuário", channel="Filtrar por canal", after="Data inicial YYYY-MM-DD", before="Data final YYYY-MM-DD", limit="Resultados (1-12)", mode="Modo de busca")
    @app_commands.choices(mode=[app_commands.Choice(name="hybrid", value="hybrid"), app_commands.Choice(name="semantic", value="semantic"), app_commands.Choice(name="keyword", value="keyword")])
    async def history_cmd(self, interaction: discord.Interaction, query: str, user: discord.Member | None = None, channel: discord.TextChannel | None = None, after: str | None = None, before: str | None = None, limit: int = 5, mode: str = "hybrid"):
        await interaction.response.defer(thinking=True)
        try:
            await interaction.edit_original_response(content=f'🔎 Buscando no histórico: *{query[:60]}*')
        except discord.HTTPException:
            pass
        guild_id = interaction.guild_id
        if not guild_id:
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            await interaction.followup.send("Este comando só funciona em servidores.", ephemeral=True)
            return
        limit = max(1, min(12, limit))
        try:
            results = await self.search(query, guild_id, limit=limit, channel_id=str(channel.id) if channel else None, author_id=str(user.id) if user else None, after=after, before=before, search_mode=mode)
        except Exception:
            logger.exception("history cmd search failed")
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            await interaction.followup.send("Erro ao buscar no histórico.")
            return
        if not results:
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            await interaction.followup.send(f"Nenhum resultado para `{query}` com os filtros aplicados.")
            return
        embed = discord.Embed(title=f"🔎 Histórico: {query}", color=discord.Color.blurple())
        desc = render_search_results(results, window_chars=900, include_msg_id=False)
        if len(desc) > 4000:
            desc = desc[:3990] + "\n…"
        embed.description = desc
        embed.set_footer(text=f"{len(results)} resultados • modo {mode}")
        try:
            await interaction.edit_original_response(content=None)
        except discord.HTTPException:
            pass
        await interaction.followup.send(embed=embed)

    @app_commands.command(name="userstats", description="Estatísticas de um usuário no histórico")
    @app_commands.describe(user="Usuário (menção ou ID)", nome="Nome parcial se não puder mencionar")
    async def userstats_cmd(self, interaction: discord.Interaction, user: discord.Member | None = None, nome: str | None = None):
        await interaction.response.defer(thinking=True)
        target_label = (user.display_name if user else (nome or "usuário"))[:40]
        try:
            await interaction.edit_original_response(content=f'📊 Coletando estatísticas de *{target_label}*…')
        except discord.HTTPException:
            pass
        guild_id = interaction.guild_id
        if not guild_id:
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            await interaction.followup.send("Só em servidores.", ephemeral=True)
            return
        author_id = str(user.id) if user else None
        author_name = nome
        if not author_id and not author_name and interaction.guild:
            pass
        if not author_id and not author_name:
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            await interaction.followup.send("Informe `user` ou `nome`.", ephemeral=True)
            return
        stats = await self.get_user_stats(guild_id, author_id=author_id, author_name=author_name)
        if "error" in stats:
            try:
                await interaction.edit_original_response(content=None)
            except discord.HTTPException:
                pass
            await interaction.followup.send(stats["error"], ephemeral=True)
            return
        embed = discord.Embed(title=f"📊 {stats['author_full']}", color=discord.Color.gold())
        embed.add_field(name="Mensagens", value=str(stats["total_messages"]), inline=True)
        embed.add_field(name="Média chars", value=str(stats["avg_length"]), inline=True)
        chans = "\n".join(f"#{k}: {v}" for k, v in stats["top_channels"]) or "—"
        embed.add_field(name="Top canais", value=chans[:1024], inline=False)
        hours = "\n".join(f"{h}h: {v}" for h, v in stats["top_hours"]) or "—"
        embed.add_field(name="Horários (BRT)", value=hours, inline=True)
        embed.add_field(name="Período", value=f"{stats['first_seen'][:10]} → {stats['last_seen'][:10]}", inline=True)
        if stats.get("example_jump"):
            embed.add_field(name="Exemplo recente", value=f"[{stats['example_content'][:150]}]({stats['example_jump']})", inline=False)
        try:
            await interaction.edit_original_response(content=None)
        except discord.HTTPException:
            pass
        await interaction.followup.send(embed=embed)

async def setup(bot: commands.Bot):
    if not HISTORY_ENABLED:
        logger.info("HistoryRAG disabled")
        return
    await bot.add_cog(HistoryRAG(bot))
