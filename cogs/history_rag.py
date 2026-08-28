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
    HISTORY_SNAPSHOT_INTERVAL,
    HISTORY_TIME_DECAY_LAMBDA,
    HISTORY_VECTOR_STORE_DIR,
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

def _jump_url(msg: discord.Message) -> str:
    try:
        return f"https://discord.com/channels/{msg.guild.id}/{msg.channel.id}/{msg.id}"
    except Exception:
        return ""

def _keyword_score(query: str, chunk: dict) -> float:
    if not query:
        return 0.0
    q_tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    if not q_tokens:
        return 0.0
    text = (chunk.get("chunk_text", "") + " " + chunk.get("content", "")).lower()
    matched = sum(1 for t in q_tokens if t in text)
    return matched / len(q_tokens)


def _sanitize_fts_query(query: str) -> str | None:
    tokens = [t.lower() for t in re.findall(r"\w+", query) if len(t) > 2]
    if not tokens:
        return None
    escaped = []
    for t in tokens[:10]:
        t = t.replace('"', '""')
        escaped.append(f'"{t}"')
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
        self._snapshot_task: asyncio.Task | None = None
        self._dirty: set[int] = set()

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

    def _store_path(self, guild_id: int) -> tuple[str, str]:
        base = os.path.join(HISTORY_VECTOR_STORE_DIR, str(guild_id))
        return base + ".json", base + ".npy"

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
            con.execute("CREATE INDEX IF NOT EXISTS idx_author ON chunks(author_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_channel ON chunks(channel_id)")
            con.execute("CREATE INDEX IF NOT EXISTS idx_ts ON chunks(ts)")
            try:
                con.execute("""
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    msg_id UNINDEXED, guild_id UNINDEXED, chunk_text, content, tokenize='porter unicode61'
                )""")
                con.execute("CREATE TRIGGER IF NOT EXISTS chunks_fts_delete AFTER DELETE ON chunks BEGIN DELETE FROM chunks_fts WHERE msg_id=old.msg_id; END;")
            except Exception:
                logger.warning("FTS5 not available, keyword search will fallback to substring")
            con.commit()
        finally:
            con.close()

    def _load_guild_from_db(self, guild_id: int) -> bool:
        path = self._db_path()
        if not os.path.exists(path):
            return False
        try:
            con = sqlite3.connect(path)
            con.row_factory = sqlite3.Row
            cur = con.execute("SELECT * FROM chunks WHERE guild_id=? ORDER BY ts ASC", (guild_id,))
            rows = cur.fetchall()
            con.close()
            if not rows:
                return False
            chunks = []
            embs = []
            for r in rows:
                emb = np.frombuffer(r["embedding"], dtype=np.float32)
                if EMBEDDING_PROVIDER == "local" and self._local_dim and emb.shape[0] != self._local_dim:
                    logger.warning("DB dim mismatch guild %s, skipping DB load", guild_id)
                    return False
                chunks.append({
                    "msg_id": r["msg_id"],
                    "channel_id": r["channel_id"],
                    "guild_id": str(r["guild_id"]),
                    "channel_name": r["channel_name"] or r["channel_id"],
                    "author_id": r["author_id"],
                    "author_name": r["author_name"] or "?",
                    "author_full": r["author_full"] or r["author_name"] or "?",
                    "content": r["content"] or "",
                    "chunk_text": r["chunk_text"] or "",
                    "window_line": r["window_line"] or "",
                    "window_lines": json.loads(r["window_lines"]) if r["window_lines"] else [],
                    "reply_to": r["reply_to"],
                    "ts": r["ts"] or "",
                    "jump_url": r["jump_url"] or "",
                    "embedding": emb,
                })
                embs.append(emb)
            if not chunks:
                return False
            self._chunks[guild_id] = chunks
            self._matrices[guild_id] = np.array(embs, dtype=np.float32) if embs else None
            self._msg_index[guild_id] = {int(c["msg_id"]): i for i, c in enumerate(chunks)}
            self._rebuild_recent(guild_id)
            logger.info("Loaded history %d chunks for guild %s from DB", len(chunks), guild_id)
            return True
        except Exception:
            logger.exception("Failed to load history DB for guild %s", guild_id)
            return False

    def _save_guild_to_db(self, guild_id: int):
        chunks = self._chunks.get(guild_id, [])
        if not chunks:
            return
        path = self._db_path()
        self._ensure_db()
        con = sqlite3.connect(path)
        try:
            con.execute("DELETE FROM chunks WHERE guild_id=?", (guild_id,))
            try:
                con.execute("DELETE FROM chunks_fts WHERE guild_id=?", (guild_id,))
            except Exception:
                pass
            for c in chunks:
                emb = c.get("embedding")
                if emb is None:
                    continue
                if isinstance(emb, np.ndarray):
                    blob = emb.astype(np.float32).tobytes()
                else:
                    blob = np.array(emb, dtype=np.float32).tobytes()
                con.execute("""
                INSERT OR REPLACE INTO chunks (msg_id, guild_id, channel_id, channel_name, author_id, author_name, author_full, content, chunk_text, window_line, window_lines, reply_to, ts, jump_url, embedding)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    str(c["msg_id"]), int(guild_id), str(c["channel_id"]), c.get("channel_name",""), str(c["author_id"]), c.get("author_name",""), c.get("author_full",""), c.get("content",""), c.get("chunk_text",""), c.get("window_line",""), json.dumps(c.get("window_lines",[]), ensure_ascii=False), c.get("reply_to"), c.get("ts",""), c.get("jump_url",""), blob
                ))
                try:
                    con.execute("INSERT INTO chunks_fts (msg_id, guild_id, chunk_text, content) VALUES (?,?,?,?)", (str(c["msg_id"]), int(guild_id), c.get("chunk_text",""), c.get("content","")))
                except Exception:
                    pass
            con.commit()
            logger.info("Saved history %d chunks for guild %s to DB", len(chunks), guild_id)
        except Exception:
            logger.exception("Failed to save history DB for guild %s", guild_id)
        finally:
            con.close()

    def _upsert_chunks_to_db(self, guild_id: int, chunks: list[dict]):
        if not chunks:
            return
        path = self._db_path()
        self._ensure_db()
        con = sqlite3.connect(path, timeout=30)
        try:
            rows = []
            for c in chunks:
                emb = c.get("embedding")
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
                con.executemany(
                    "INSERT OR REPLACE INTO chunks_fts (msg_id, guild_id, chunk_text, content) VALUES (?,?,?,?)",
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

    def _load_guild(self, guild_id: int) -> bool:
        if self._load_guild_from_db(guild_id):
            return True
        json_path, npy_path = self._store_path(guild_id)
        if not os.path.exists(json_path):
            return False
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chunks = data.get("chunks", [])
            if not chunks:
                return False
            if os.path.exists(npy_path):
                embs = np.load(npy_path)
                if len(embs) != len(chunks):
                    logger.warning("History guild %s embedding mismatch %d vs %d", guild_id, len(embs), len(chunks))
                    return False
                if EMBEDDING_PROVIDER == "local" and self._local_dim and embs.shape[1] != self._local_dim:
                    logger.warning("History guild %s dim mismatch stored %d vs %d", guild_id, embs.shape[1], self._local_dim)
                    return False
                for c, e in zip(chunks, embs):
                    c["embedding"] = e
            else:
                if chunks and "embedding" in chunks[0]:
                    embs = np.array([c.pop("embedding") for c in chunks], dtype=np.float32)
                    for c, e in zip(chunks, embs):
                        c["embedding"] = e
                else:
                    return False
            self._chunks[guild_id] = chunks
            self._matrices[guild_id] = np.array([c["embedding"] for c in chunks], dtype=np.float32)
            self._msg_index[guild_id] = {int(c["msg_id"]): i for i, c in enumerate(chunks)}
            self._rebuild_recent(guild_id)
            logger.info("Loaded history %d chunks for guild %s from JSON", len(chunks), guild_id)
            try:
                self._save_guild_to_db(guild_id)
            except Exception:
                pass
            return True
        except Exception:
            logger.exception("Failed to load history for guild %s", guild_id)
            return False

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

    def _save_guild_json(self, guild_id: int):
        chunks = list(self._chunks.get(guild_id, []))
        if not chunks:
            return
        json_path, npy_path = self._store_path(guild_id)
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        meta = {
            "guild_id": guild_id,
            "chunks": [
                {k: v for k, v in c.items() if k not in ("embedding",)}
                for c in chunks
            ],
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)
        embs = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        np.save(npy_path, embs)

    def _lock(self, guild_id: int) -> asyncio.Lock:
        if guild_id not in self._locks:
            self._locks[guild_id] = asyncio.Lock()
        return self._locks[guild_id]

    def _load_all_guilds(self):
        if os.path.exists(HISTORY_VECTOR_STORE_DIR):
            for fname in os.listdir(HISTORY_VECTOR_STORE_DIR):
                if fname.endswith(".json"):
                    try:
                        gid = int(fname[:-5])
                        self._load_guild(gid)
                    except ValueError:
                        continue
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
        self._worker_task = asyncio.create_task(self._ingest_worker())
        self._snapshot_task = asyncio.create_task(self._snapshot_loop())
        self.bot.loop.create_task(self._background_backfill_all())

    async def _snapshot_loop(self):
        while True:
            await asyncio.sleep(HISTORY_SNAPSHOT_INTERVAL)
            await self._flush_dirty_snapshots()

    async def _flush_dirty_snapshots(self):
        if not self._dirty:
            return
        gids = list(self._dirty)
        self._dirty.clear()
        for gid in gids:
            try:
                await asyncio.to_thread(self._save_guild_json, gid)
            except Exception:
                logger.exception("Snapshot failed guild %s", gid)

    async def cog_unload(self):
        for task in (self._worker_task, self._snapshot_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        await self._flush_dirty_snapshots()

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
                if self._chunks[guild.id]:
                    self._dirty.add(guild.id)
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
        new_embs = []
        for ch, emb in zip(chunks, embeddings):
            ch["embedding"] = np.asarray(emb, dtype=np.float32)
            self._chunks[gid].append(ch)
            self._msg_index[gid][int(ch["msg_id"])] = len(self._chunks[gid]) - 1
            new_embs.append(ch["embedding"])
        if not new_embs:
            return
        stacked = np.stack(new_embs)
        mat = self._matrices.get(gid)
        if mat is None or mat.shape[0] == 0:
            self._matrices[gid] = stacked
        else:
            self._matrices[gid] = np.vstack([mat, stacked])

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
            await asyncio.to_thread(self._upsert_chunks_to_db, gid, chunks)
        except Exception:
            logger.exception("Failed persisting history batch guild %s", gid)
        self._dirty.add(gid)

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
            emb = (await self._embed_batch([chunk["chunk_text"]]))[0]
            chunk["embedding"] = np.array(emb, dtype=np.float32)
            self._matrices[gid][idx] = chunk["embedding"]
            await asyncio.to_thread(self._upsert_chunks_to_db, gid, [chunk])
            self._dirty.add(gid)
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
                self._matrices[gid] = np.array([c["embedding"] for c in chunks], dtype=np.float32)
                await asyncio.to_thread(self._delete_chunk_from_db, gid, str(message.id))
                self._dirty.add(gid)
            else:
                self._matrices[gid] = None
                j, n = self._store_path(gid)
                try:
                    os.remove(j)
                    os.remove(n)
                except FileNotFoundError:
                    pass
                try:
                    def _purge_guild():
                        con = sqlite3.connect(self._db_path(), timeout=30)
                        try:
                            con.execute("DELETE FROM chunks WHERE guild_id=?", (gid,))
                            try:
                                con.execute("DELETE FROM chunks_fts WHERE guild_id=?", (gid,))
                            except Exception:
                                pass
                            con.commit()
                        finally:
                            con.close()
                    await asyncio.to_thread(_purge_guild)
                    self._dirty.discard(gid)
                except Exception:
                    pass

    def _resolve_author(self, guild_id: int, author_id: str | None, author_name: str | None) -> list[str]:
        if author_id:
            return [str(author_id)]
        if author_name:
            name_low = author_name.lower()
            chunks = self._chunks.get(guild_id, [])
            matched = set()
            for c in chunks:
                if name_low in c.get("author_name","").lower() or name_low in c.get("author_full","").lower() or name_low in c.get("author_id",""):
                    matched.add(c["author_id"])
            return list(matched)
        return []

    def _fts_search(self, query: str, guild_id: int, limit: int = 100) -> dict[str, float]:
        fts_q = _sanitize_fts_query(query)
        if not fts_q:
            return {}
        try:
            con = sqlite3.connect(self._db_path())
            con.row_factory = sqlite3.Row
            cur = con.execute(
                "SELECT msg_id, rank FROM chunks_fts WHERE guild_id=? AND chunks_fts MATCH ? ORDER BY rank LIMIT ?",
                (guild_id, fts_q, limit),
            )
            rows = cur.fetchall()
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

    async def search(self, query: str, guild_id: int, limit: int = 5, channel_id: str | None = None, author_id: str | None = None, author_name: str | None = None, after: str | None = None, before: str | None = None, search_mode: str = "hybrid", sort_by: str = "relevance") -> list[dict]:
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
        author_ids = None
        if author_id or author_name:
            author_ids = set(self._resolve_author(guild_id, author_id, author_name))
            if author_id and str(author_id) not in author_ids:
                author_ids.add(str(author_id))
            if not author_ids and (author_id or author_name):
                return []
            indices = [i for i in indices if chunks[i]["author_id"] in author_ids]
            if not indices:
                return []
        dt_after = _parse_dt(after) if after else None
        dt_before = _parse_dt(before) if before else None
        if dt_after or dt_before:
            filtered = []
            for i in indices:
                try:
                    ts = datetime.datetime.fromisoformat(chunks[i]["ts"].replace("Z","+00:00"))
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=datetime.timezone.utc)
                except Exception:
                    continue
                if dt_after and ts < dt_after:
                    continue
                if dt_before and ts > dt_before:
                    continue
                filtered.append(i)
            indices = filtered
            if not indices:
                return []
        cache_key = f"{guild_id}:{query}:{limit}:{channel_id}:{author_id}:{author_name}:{after}:{before}:{search_mode}:{sort_by}"
        try:
            cached = self._query_cache.get(cache_key)
            if cached is not None:
                return [dict(c) for c in cached]
        except Exception:
            pass
        if search_mode == "keyword":
            fts_scores = await asyncio.to_thread(self._fts_search, query, guild_id, max(limit * 5, 100))
            if fts_scores:
                allowed = {chunks[i]["msg_id"] for i in indices}
                filtered_fts = {k: v for k, v in fts_scores.items() if k in allowed}
                if dt_after or dt_before:
                    pass
                sorted_ids = sorted(filtered_fts, key=lambda k: filtered_fts[k], reverse=True)
                if sort_by == "recent":
                    sorted_ids = sorted(sorted_ids, key=lambda mid: next((chunks[i]["ts"] for i in indices if chunks[i]["msg_id"] == mid), ""), reverse=True)
                result = []
                for mid in sorted_ids[:limit]:
                    idx = next((i for i in indices if chunks[i]["msg_id"] == mid), None)
                    if idx is not None:
                        c = chunks[idx]
                        decay = _time_decay_factor(c["ts"], HISTORY_TIME_DECAY_LAMBDA) if sort_by == "recent" else 1.0
                        result.append(dict(c, _score=float(filtered_fts[mid] * decay), _keyword_score=float(filtered_fts[mid])))
                if result:
                    try:
                        self._query_cache[cache_key] = [dict(r) for r in result]
                    except Exception:
                        pass
                    return result
            scored = []
            for i in indices:
                ks = _keyword_score(query, chunks[i])
                if ks > 0:
                    decay = _time_decay_factor(chunks[i]["ts"], HISTORY_TIME_DECAY_LAMBDA) if sort_by == "recent" and HISTORY_TIME_DECAY_LAMBDA > 0 else 1.0
                    scored.append((ks * decay, i))
            if sort_by == "recent" and HISTORY_TIME_DECAY_LAMBDA == 0:
                scored.sort(key=lambda x: chunks[x[1]]["ts"], reverse=True)
            else:
                scored.sort(reverse=True, key=lambda x: x[0])
            top = scored[:limit]
            res = [dict(chunks[i], _score=float(s), _keyword_score=float(s)) for s, i in top]
            try:
                self._query_cache[cache_key] = [dict(r) for r in res]
            except Exception:
                pass
            return res
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
            fts_scores_map = await asyncio.to_thread(self._fts_search, query, guild_id, 200)
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
            top_idx = order[: max(limit * 4, limit)]
            candidates = [dict(chunks_f[i], _score=float(scores[i]), _keyword_score=float(_keyword_score(query, chunks_f[i]))) for i in top_idx]
            reranked = await self._rerank_history(query, candidates, top_n=limit)
            try:
                self._query_cache[cache_key] = [dict(r) for r in reranked]
            except Exception:
                pass
            return reranked
        top_k_rerank = max(limit * 4, 20)
        top_idx = np.argsort(scores)[::-1][:top_k_rerank]
        candidates = [dict(chunks_f[i], _score=float(scores[i]), _keyword_score=float(_keyword_score(query, chunks_f[i]))) for i in top_idx]
        if sort_by == "recent" and HISTORY_TIME_DECAY_LAMBDA > 0:
            candidates.sort(key=lambda x: (x["_score"], x["ts"]), reverse=True)
            candidates = candidates[:limit]
        else:
            candidates = await self._rerank_history(query, candidates, top_n=limit)
        try:
            self._query_cache[cache_key] = [dict(r) for r in candidates]
        except Exception:
            pass
        return candidates[:limit]

    async def get_user_stats(self, guild_id: int, author_id: str | None = None, author_name: str | None = None) -> dict:
        if guild_id not in self._chunks:
            return {"error": "Nenhum dado para este servidor."}
        chunks = self._chunks[guild_id]
        if not chunks:
            return {"error": "Nenhum dado."}
        author_ids = self._resolve_author(guild_id, author_id, author_name)
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
        results = await self.search(query, guild_id, limit=200, after=after, before=before, search_mode="hybrid", sort_by="relevance")
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
