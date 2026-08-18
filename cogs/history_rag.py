import asyncio
import collections
import datetime
import json
import logging
import os
import re

import discord
import numpy as np
from discord.ext import commands, tasks

from config import (
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    HISTORY_BACKFILL_LIMIT,
    HISTORY_ENABLED,
    HISTORY_EXCLUDE_BOTS,
    HISTORY_MAX_MSG_LENGTH,
    HISTORY_VECTOR_STORE_DIR,
    HISTORY_WINDOW_SIZE,
    LOCAL_EMBEDDING_DEVICE,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
)

logger = logging.getLogger(__name__)

BR_TZ = None
try:
    from zoneinfo import ZoneInfo
    BR_TZ = ZoneInfo("America/Sao_Paulo")
except Exception:
    pass

def _fmt_dt(dt):
    if not dt:
        return "—"
    try:
        if BR_TZ:
            return dt.astimezone(BR_TZ).strftime("%d/%m/%Y %H:%M BRT")
        return dt.strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return dt.isoformat()

def _safe_content(msg: discord.Message) -> str:
    c = msg.content or ""
    if msg.attachments:
        c += " " + " ".join(f"[anexo:{a.filename}]" for a in msg.attachments)
    if msg.embeds and not c:
        # store embed title if no text
        try:
            c = f"[embed: {msg.embeds[0].title or msg.embeds[0].description[:100]}]"
        except Exception:
            c = "[embed]"
    c = c.replace("\n", " ").strip()
    if len(c) > HISTORY_MAX_MSG_LENGTH:
        c = c[:HISTORY_MAX_MSG_LENGTH] + "…"
    return c

def _format_line(msg: discord.Message) -> str:
    ts = _fmt_dt(msg.created_at)
    author = getattr(msg.author, 'display_name', str(msg.author))
    content = _safe_content(msg)
    if not content:
        content = "[sem texto]"
    return f"[{ts}] {author} (@{msg.author.name}): {content}"

def _jump_url(msg: discord.Message) -> str:
    try:
        return f"https://discord.com/channels/{msg.guild.id}/{msg.channel.id}/{msg.id}"
    except Exception:
        return ""

class HistoryRAG(commands.Cog, name="HistoryRAG"):
    """Entire-server RAG: background backfill + live ingestion, 5-msg window."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.client = None
        self._local_model = None
        self._local_dim = None
        if EMBEDDING_PROVIDER != "local":
            try:
                from openai import AsyncOpenAI
                api_key = OPENAI_API_KEY or "not-needed"
                self.client = AsyncOpenAI(base_url=OPENAI_BASE_URL, api_key=api_key)
            except Exception:
                logger.exception("Failed to init OpenAI client for HistoryRAG")
        # per-guild storage
        self._chunks: dict[int, list[dict]] = {}  # guild_id -> list
        self._matrices: dict[int, np.ndarray | None] = {}
        self._locks: dict[int, asyncio.Lock] = {}
        self._recent: dict[int, collections.deque] = {}  # channel_id -> deque of last 5 msgs (for window)
        self._backfilling: set[int] = set()
        # msg_id -> chunk index for dedup
        self._msg_index: dict[int, dict[int, int]] = {}  # guild_id -> {msg_id: idx}

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

    def _load_guild(self, guild_id: int) -> bool:
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
            # rebuild recent deques for window (last 5 per channel)
            self._rebuild_recent(guild_id)
            logger.info("Loaded history %d chunks for guild %s", len(chunks), guild_id)
            return True
        except Exception:
            logger.exception("Failed to load history for guild %s", guild_id)
            return False

    def _rebuild_recent(self, guild_id: int):
        chunks = self._chunks.get(guild_id, [])
        # group by channel, keep last 5 msg_ids chronological
        per_channel: dict[int, list[dict]] = {}
        for c in chunks:
            per_channel.setdefault(int(c["channel_id"]), []).append(c)
        for cid, lst in per_channel.items():
            # lst is chronological (we store oldest first)
            recent = collections.deque(maxlen=HISTORY_WINDOW_SIZE)
            for ch in lst[-HISTORY_WINDOW_SIZE:]:
                # reconstruct minimal msg line for window
                recent.append(ch.get("window_line") or _format_line_from_chunk(ch))
            self._recent[cid] = recent

    def _save_guild(self, guild_id: int):
        chunks = self._chunks.get(guild_id, [])
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
        self._matrices[guild_id] = embs
        logger.info("Saved history %d chunks for guild %s", len(chunks), guild_id)

    def _lock(self, guild_id: int) -> asyncio.Lock:
        if guild_id not in self._locks:
            self._locks[guild_id] = asyncio.Lock()
        return self._locks[guild_id]

    async def cog_load(self):
        if not HISTORY_ENABLED:
            logger.info("HistoryRAG disabled via HISTORY_ENABLED=false")
            return
        if EMBEDDING_PROVIDER == "local":
            try:
                await asyncio.to_thread(self._get_local_model)
            except Exception:
                logger.warning("HistoryRAG local model failed, will fallback to remote if possible")
        # load existing stores
        try:
            if os.path.exists(HISTORY_VECTOR_STORE_DIR):
                for fname in os.listdir(HISTORY_VECTOR_STORE_DIR):
                    if fname.endswith(".json"):
                        try:
                            gid = int(fname[:-5])
                            self._load_guild(gid)
                        except ValueError:
                            continue
        except Exception:
            logger.exception("Failed to scan history store dir")
        # launch background backfill after bot ready
        self.bot.loop.create_task(self._background_backfill_all())

    async def _background_backfill_all(self):
        await self.bot.wait_until_ready()
        # small delay to let other cogs load
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
                # ensure loaded
                if guild.id not in self._chunks:
                    self._chunks[guild.id] = []
                    self._matrices[guild.id] = None
                    self._msg_index[guild.id] = {}
                # collect channels
                channels: list[discord.abc.Messageable] = []
                for ch in guild.channels:
                    if isinstance(ch, (discord.TextChannel, discord.Thread, discord.VoiceChannel)):
                        # VoiceChannel has no history, skip
                        if isinstance(ch, discord.VoiceChannel):
                            continue
                        perms = ch.permissions_for(guild.me)
                        if perms.read_message_history and perms.view_channel:
                            channels.append(ch)
                    elif isinstance(ch, discord.ForumChannel):
                        # include active threads
                        for thread in ch.threads:
                            if thread.permissions_for(guild.me).read_message_history:
                                channels.append(thread)
                    # category etc ignored
                logger.info("History backfill guild %s (%s) %d channels", guild.name, guild.id, len(channels))
                for channel in channels:
                    await self._backfill_channel(guild, channel)
                # final save
                if self._chunks[guild.id]:
                    self._save_guild(guild.id)
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
        # fetch history oldest_first for window correctness
        to_index: list[discord.Message] = []
        try:
            # Use oldest_first to maintain chronological window
            async for msg in channel.history(limit=HISTORY_BACKFILL_LIMIT, oldest_first=True):
                if msg.id in existing:
                    # still update recent deque for window
                    line = _format_line(msg)
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

    async def _index_batch(self, guild: discord.Guild, channel: discord.abc.Messageable, msgs: list[discord.Message]):
        if not msgs:
            return
        gid = guild.id
        cid = channel.id
        recent = self._recent.get(cid)
        if recent is None:
            recent = collections.deque(maxlen=HISTORY_WINDOW_SIZE)
            self._recent[cid] = recent
        chunks: list[dict] = []
        texts: list[str] = []
        for msg in msgs:
            # build window text (5 prev + current)
            window_lines = list(recent)
            cur_line = _format_line(msg)
            # chunk_text includes local context for betting retrieval
            if window_lines:
                chunk_text = "\n".join(window_lines + [cur_line])
            else:
                chunk_text = cur_line
            # also store isolated line for display
            texts.append(chunk_text)
            jump = _jump_url(msg)
            chunk = {
                "msg_id": str(msg.id),
                "channel_id": str(cid),
                "guild_id": str(gid),
                "channel_name": getattr(channel, "name", str(cid)),
                "author_id": str(msg.author.id),
                "author_name": getattr(msg.author, "display_name", str(msg.author)),
                "author_full": f"{getattr(msg.author, 'display_name', str(msg.author))} (@{msg.author.name})",
                "content": _safe_content(msg),
                "chunk_text": chunk_text,
                "window_lines": window_lines.copy(),
                "window_line": cur_line,
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
        # we are already inside lock for backfill, but live path also uses lock
        # if called from backfill we hold lock, so just append without extra lock?
        # Use try to avoid deadlock: if lock locked, just append (caller holds)
        if lock.locked():
            # caller holds lock
            for ch, emb in zip(chunks, embeddings):
                ch["embedding"] = emb
                self._chunks[gid].append(ch)
                self._msg_index[gid][int(ch["msg_id"])] = len(self._chunks[gid]) - 1
            # rebuild matrix incrementally
            self._matrices[gid] = np.array([c["embedding"] for c in self._chunks[gid]], dtype=np.float32)
        else:
            async with lock:
                for ch, emb in zip(chunks, embeddings):
                    ch["embedding"] = emb
                    self._chunks[gid].append(ch)
                    self._msg_index[gid][int(ch["msg_id"])] = len(self._chunks[gid]) - 1
                self._matrices[gid] = np.array([c["embedding"] for c in self._chunks[gid]], dtype=np.float32)
        logger.info("Indexed %d msgs guild %s channel %s (total %d)", len(chunks), gid, cid, len(self._chunks[gid]))

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        if not HISTORY_ENABLED or not message.guild:
            return
        if HISTORY_EXCLUDE_BOTS and message.author.bot:
            return
        if not message.content and not message.attachments and not message.embeds:
            return
        gid = message.guild.id
        # ensure structures exist
        if gid not in self._chunks:
            self._chunks[gid] = []
            self._matrices[gid] = None
            self._msg_index[gid] = {}
            self._load_guild(gid)
        # dedup
        if message.id in self._msg_index.get(gid, {}):
            return
        await self._index_batch(message.guild, message.channel, [message])
        # debounced save
        await asyncio.sleep(2)
        self._save_guild(gid)

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
            # remove
            chunks.pop(idx)
            # rebuild index
            self._msg_index[gid] = {int(c["msg_id"]): i for i, c in enumerate(chunks)}
            if chunks:
                self._matrices[gid] = np.array([c["embedding"] for c in chunks], dtype=np.float32)
                self._save_guild(gid)
            else:
                self._matrices[gid] = None
                # remove files
                j, n = self._store_path(gid)
                try:
                    os.remove(j)
                    os.remove(n)
                except FileNotFoundError:
                    pass

    async def search(self, query: str, guild_id: int, limit: int = 5, channel_id: str | None = None) -> list[dict]:
        if guild_id not in self._chunks or not self._chunks[guild_id]:
            return []
        mat = self._matrices.get(guild_id)
        if mat is None or len(mat) == 0:
            return []
        chunks = self._chunks[guild_id]
        # optional channel filter
        if channel_id:
            indices = [i for i, c in enumerate(chunks) if c["channel_id"] == str(channel_id)]
            if not indices:
                return []
            mat_f = mat[np.array(indices)]
            chunks_f = [chunks[i] for i in indices]
        else:
            mat_f = mat
            chunks_f = chunks
        try:
            q_emb = (await self._embed_batch([query]))[0]
        except Exception:
            logger.exception("History search embed failed")
            return []
        q_arr = np.array(q_emb, dtype=np.float32)
        dots = mat_f @ q_arr
        norms = np.linalg.norm(mat_f, axis=1) * np.linalg.norm(q_arr)
        with np.errstate(invalid='ignore', divide='ignore'):
            scores = np.where(norms > 0, dots / norms, 0.0)
        top_idx = np.argsort(scores)[::-1][:max(limit * 3, limit)]
        candidates = [dict(chunks_f[i], _score=float(scores[i])) for i in top_idx]
        # simple rerank already by cosine; return top limit
        return candidates[:limit]

def _format_line_from_chunk(ch: dict) -> str:
    ts = ch.get("ts", "")[:19]
    author = ch.get("author_full", ch.get("author_name", "?"))
    content = ch.get("content", "")[:300]
    return f"[{ts}] {author}: {content}"

async def setup(bot: commands.Bot):
    if not HISTORY_ENABLED:
        logger.info("HistoryRAG disabled")
        return
    await bot.add_cog(HistoryRAG(bot))
