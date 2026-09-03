"""Standalone smoke test for the history search improvements.

Exercises: phrase-aware FTS, SQL-pushed filters, dedupe, authors table
(rename-proof resolution + find_user + migration), and indexed message
context — all against a throwaway SQLite DB. No Discord or LLM calls.

Run: python3 test_history_search.py
"""

import asyncio
import os
import shutil
import sqlite3
import tempfile

import numpy as np

from cogs.history_rag import (
    HistoryRAG,
    _extract_query_terms,
    _keyword_score,
    _sanitize_fts_query,
)

GID = 1
GENERAL = "100"
DEV = "200"
ALICE_ID = "111"
BOB_ID = "222"

PASS = 0
FAIL = 0


def check(name, cond, extra=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name} {extra}")


def chunk(mid, cid, cname, aid, aname, afull, content, ts, chunk_text=None):
    return {
        "msg_id": str(mid),
        "guild_id": str(GID),
        "channel_id": str(cid),
        "channel_name": cname,
        "author_id": str(aid),
        "author_name": aname,
        "author_full": afull,
        "content": content,
        "chunk_text": chunk_text or content,
        "window_line": content,
        "window_lines": [],
        "reply_to": None,
        "ts": ts,
        "jump_url": f"https://discord.com/channels/{GID}/{cid}/{mid}",
        "embedding": np.zeros(4, dtype=np.float32),
    }


def build_chunks():
    alice = lambda mid, name, content, ts, cid=GENERAL, cname="general": chunk(
        mid, cid, cname, ALICE_ID, name, f"{name} (@alice)", content, ts
    )
    bob = lambda mid, content, ts, cid=DEV, cname="dev": chunk(
        mid, cid, cname, BOB_ID, "Bob", "Bob (@bob)", content, ts
    )
    return [
        alice(1, "Alice", "paper lag fix applied", "2025-06-01T10:00:00+00:00"),
        alice(2, "Alice", "restart after the paper lag fix", "2025-06-01T10:02:00+00:00"),
        alice(3, "Alice", "looks fine now", "2025-06-01T10:03:30+00:00"),
        alice(4, "Alice Smith", "old name was Alice btw", "2025-06-01T10:05:00+00:00"),
        bob(5, "server paper config review", "2025-06-01T11:00:00+00:00"),
        bob(6, "huge lag spike yesterday", "2025-06-03T09:00:00+00:00"),
        alice(7, "Alice Smith", "quarter planning notes", "2025-07-01T10:00:00+00:00"),
    ]


def fresh_rag(tmpdb):
    rag = HistoryRAG(bot=None)
    rag._db_path = lambda: tmpdb  # type: ignore[method-assign]
    rag._ensure_db()
    return rag


async def main():
    tmp = tempfile.mkdtemp(prefix="quillbot_hist_test_")
    tmpdb = os.path.join(tmp, "history.db")
    try:
        rag = fresh_rag(tmpdb)
        chunks = build_chunks()
        rag._upsert_chunks_to_db(GID, chunks)
        rag._upsert_authors(GID, chunks)

        print("== phrase-aware query extraction ==")
        check("phrases kept verbatim", _extract_query_terms('o "paper lag" aconteceu') == ["paper lag", "aconteceu"])
        check("loose words filtered", _extract_query_terms("we fixed the lag") == ["fixed", "the", "lag"])
        fts_q = _sanitize_fts_query('"paper lag" spike')
        check("fts phrase preserved", fts_q == '"paper lag" OR "spike"', fts_q)
        check(
            "keyword score matches phrases",
            _keyword_score('"paper lag"', {"chunk_text": "paper lag fix", "content": ""}) == 1.0
            and _keyword_score('"paper lag"', {"chunk_text": "paper config lag", "content": ""}) == 0.0,
        )

        print("== FTS with SQL-pushed filters ==")
        rows = rag._fts_search_rows('"paper lag"', GID, 100)
        check("phrase matches only adjacent tokens", [r[0]["msg_id"] for r in rows] == ["1", "2"], str(rows))
        rows = rag._fts_search_rows("lag", GID, 100)
        check("loose OR search hits all lag msgs", {r[0]["msg_id"] for r in rows} == {"1", "2", "6"}, str(rows))
        rows = rag._fts_search_rows("lag", GID, 100, channel_id=DEV)
        check("channel filter in SQL", [r[0]["msg_id"] for r in rows] == ["6"], str(rows))
        rows = rag._fts_search_rows("lag", GID, 100, author_ids={ALICE_ID})
        check("author filter in SQL", {r[0]["msg_id"] for r in rows} == {"1", "2"}, str(rows))
        import datetime as _dt
        rows = rag._fts_search_rows("lag", GID, 100, dt_after=_dt.datetime(2025, 6, 2, tzinfo=_dt.timezone.utc))
        check("date filter in SQL", [r[0]["msg_id"] for r in rows] == ["6"], str(rows))
        rows = rag._fts_search_rows("lag", GID, 100, dt_before=_dt.datetime(2025, 6, 2, tzinfo=_dt.timezone.utc))
        check("dt_before filter in SQL", {r[0]["msg_id"] for r in rows} == {"1", "2"}, str(rows))
        rows = rag._fts_search_rows("lag", GID, 100, author_ids={ALICE_ID, BOB_ID})
        check("multi-author IN expansion", {r[0]["msg_id"] for r in rows} == {"1", "2", "6"}, str(rows))

        print("== keyword search end-to-end (guild NOT loaded in memory) ==")
        res = await rag.search("lag", GID, limit=5, search_mode="keyword", dedupe=False)
        check("db-only keyword search works", len(res) == 3 and GID not in rag._chunks, str([(r["msg_id"], r["_score"]) for r in res]))
        check("result rows carry rendering fields", all("chunk_text" in r and "jump_url" in r and "author_full" in r for r in res))

        print("== dedupe of overlapping windows ==")
        res = await rag.search("lag", GID, limit=5, search_mode="keyword", dedupe=True)
        ids = [r["msg_id"] for r in res]
        check("adjacent general msgs collapse to one", ids == ["1", "6"], str(ids))
        raw = await rag.search("lag", GID, limit=5, search_mode="keyword", dedupe=False)
        check("dedupe=False keeps all", {r["msg_id"] for r in raw} == {"1", "2", "6"}, str([r["msg_id"] for r in raw]))

        print("== authors table: rename-proof resolution ==")
        ids = await rag._resolve_author(GID, None, "alice smith")
        check("resolve by current name", ids == [ALICE_ID], str(ids))
        ids = await rag._resolve_author(GID, None, "alice")
        check("resolve by OLD name (alias)", ids == [ALICE_ID], str(ids))
        ids = await rag._resolve_author(GID, BOB_ID, None)
        check("resolve by id passthrough", ids == [BOB_ID], str(ids))
        ids = await rag._resolve_author(GID, None, "nobody")
        check("unknown name → empty", ids == [], str(ids))

        print("== find_user ==")
        users = rag._find_users_db(GID, "alice", 5)
        check("find_user by partial name", len(users) == 1 and users[0]["author_id"] == ALICE_ID, str(users))
        check("find_user has stats + top channels", users[0]["msg_count"] == 5 and users[0]["top_channels"][0][0] == "general", str(users))
        users = rag._find_users_db(GID, "", 5)
        check("find_user no query → most active first", users[0]["author_id"] == ALICE_ID, str(users))
        users = rag._find_users_db(GID, BOB_ID, 5)
        check("find_user by id", len(users) == 1 and users[0]["author_id"] == BOB_ID, str(users))
        users = rag._find_users_db(GID, "zed", 5)
        check("find_user no match → empty", users == [], str(users))

        print("== migration: rebuild authors from legacy chunks ==")
        con = sqlite3.connect(tmpdb)
        con.execute("DELETE FROM authors")
        con.commit()
        con.close()
        n = rag._rebuild_authors_from_chunks()
        check("rebuild populates", n == 2, str(n))
        users = rag._find_users_db(GID, "alice", 5)
        check("aliases survive rebuild", users and "Alice" in users[0]["aliases"] and "Alice Smith" in users[0]["aliases"], str(users))

        print("== indexed message context ==")
        text = await rag.get_message_context_from_index(GID, GENERAL, "2", window=2)
        check("context built from index", text is not None and "▶ " in text and "msg_id=2" in text, text or "")
        lines = text.splitlines()
        check("window: available before + target + 2 after", len(lines) == 5, str(lines))
        check("api-style ordering", "msg_id=1]" in lines[1] and "msg_id=2]" in lines[2] and "msg_id=3]" in lines[3], str(lines))
        text = await rag.get_message_context_from_index(GID, GENERAL, "999", window=2)
        check("unknown message → None (API fallback)", text is None)
        text = await rag.get_message_context_from_index(GID, DEV, "5", window=5)
        check("edge window at channel start", text is not None and "msg_id=5" in text, text or "")

        print("== ingest keeps authors in sync ==")
        new = [chunk(8, GENERAL, "general", ALICE_ID, "Alice Prime", "Alice Prime (@alice)", "new name again", "2025-08-01T10:00:00+00:00")]
        rag._upsert_chunks_to_db(GID, new)
        rag._upsert_authors(GID, new)
        users = rag._find_users_db(GID, "alice", 5)
        check("count grows", users[0]["msg_count"] == 6, str(users[0]["msg_count"]))
        check("new alias registered", "Alice Prime" in users[0]["aliases"], str(users[0]["aliases"]))
        rag._adjust_author_count(GID, ALICE_ID, -1)
        users = rag._find_users_db(GID, "alice", 5)
        check("delete decrements", users[0]["msg_count"] == 5, str(users[0]["msg_count"]))

        print("== re-index is idempotent (no msg_count inflation) ==")
        existing = rag._existing_msg_ids(GID, ["1", "2", "8"])
        check("existing msg probe", existing == {"1", "2", "8"}, str(existing))
        batch = chunks[:2]
        rag._upsert_chunks_to_db(GID, batch)
        rag._upsert_authors(GID, batch, count_msg_ids=set())
        users = rag._find_users_db(GID, "alice", 5)
        check("re-index adds zero count", users[0]["msg_count"] == 5, str(users[0]["msg_count"]))
        con = sqlite3.connect(tmpdb)
        fts_rows = con.execute("SELECT msg_id, COUNT(*) FROM chunks_fts WHERE guild_id=? GROUP BY msg_id ORDER BY msg_id", (GID,)).fetchall()
        con.close()
        check("fts no duplicate rows on re-upsert", all(c == 1 for _, c in fts_rows), str(fts_rows))

        print("== dedupe boundary behavior ==")
        def _res(mid, cid, ts):
            return {"msg_id": str(mid), "channel_id": cid, "ts": ts}
        base = "2025-06-01T10:00:00+00:00"
        edge = "2025-06-01T10:10:00+00:00"  # exactly the 10-minute window
        past = "2025-06-01T10:10:01+00:00"
        out = rag._dedupe_adjacent([_res(1, GENERAL, base), _res(2, GENERAL, edge)], 5)
        check("exact-window edge collapses (inclusive)", [r["msg_id"] for r in out] == ["1"], str(out))
        out = rag._dedupe_adjacent([_res(1, GENERAL, base), _res(2, GENERAL, past)], 5)
        check("past window survives", [r["msg_id"] for r in out] == ["1", "2"], str(out))
        out = rag._dedupe_adjacent([_res(1, GENERAL, base), _res(2, DEV, base)], 5)
        check("same ts different channel kept", [r["msg_id"] for r in out] == ["1", "2"], str(out))
        out = rag._dedupe_adjacent([_res(1, GENERAL, base), _res(2, GENERAL, "not-a-date")], 5)
        check("unparsable ts accepted", [r["msg_id"] for r in out] == ["1", "2"], str(out))
        import cogs.history_rag as hr_mod
        old_win = hr_mod.HISTORY_DEDUPE_WINDOW_MINUTES
        try:
            hr_mod.HISTORY_DEDUPE_WINDOW_MINUTES = 0
            out = rag._dedupe_adjacent([_res(1, GENERAL, base), _res(2, GENERAL, edge)], 5)
            check("window=0 disables dedupe", [r["msg_id"] for r in out] == ["1", "2"], str(out))
        finally:
            hr_mod.HISTORY_DEDUPE_WINDOW_MINUTES = old_win

        print("== keyword fallback from memory (FTS empty) ==")
        rag._chunks[GID] = build_chunks()
        orig_fts_rows = rag._fts_search_rows
        rag._fts_search_rows = lambda *a, **k: []  # type: ignore[method-assign]
        try:
            res = await rag.search("lag", GID, limit=5, search_mode="keyword", author_id=ALICE_ID, dedupe=False)
            check("fallback filters by author", {r["msg_id"] for r in res} == {"1", "2"}, str([r["msg_id"] for r in res]))
            import datetime as _dt2
            res = await rag.search("lag", GID, limit=5, search_mode="keyword", channel_id=DEV, dedupe=False)
            check("fallback filters by channel", [r["msg_id"] for r in res] == ["6"], str([r["msg_id"] for r in res]))
            res = await rag.search("lag", GID, limit=5, search_mode="keyword", before="2025-06-02", dedupe=False)
            check("fallback filters by date", {r["msg_id"] for r in res} == {"1", "2"}, str([r["msg_id"] for r in res]))
        finally:
            rag._fts_search_rows = orig_fts_rows  # type: ignore[method-assign]
        rag._chunks.pop(GID, None)

        print("== resolve fallback when authors table empty for guild ==")
        con = sqlite3.connect(tmpdb)
        con.execute("DELETE FROM authors WHERE guild_id=?", (GID,))
        con.commit()
        con.close()
        rag._chunks[GID] = build_chunks()
        ids = await rag._resolve_author(GID, None, "alice")
        check("memory fallback on empty authors", ids == [ALICE_ID], str(ids))
        rag._chunks.pop(GID, None)

        print("== per-guild migration (one guild must not block another) ==")
        GID2 = 2
        bob2 = chunk(50, "300", "other", BOB_ID, "Bob", "Bob (@bob)", "other guild lag talk", "2025-05-01T10:00:00+00:00")
        bob2["guild_id"] = str(GID2)
        rag._upsert_chunks_to_db(GID2, [bob2])
        n = rag._rebuild_authors_from_chunks()
        users = rag._find_users_db(GID2, "bob", 5)
        check("guild2 rebuilt despite guild1 authors existing", n >= 1 and len(users) == 1, f"n={n} users={users}")
        con = sqlite3.connect(tmpdb)
        con.execute("DELETE FROM authors WHERE guild_id=?", (GID2,))
        con.commit()
        con.close()
        bob_rev = [
            (GID2, BOB_ID, "Bob", "Bob (@bob)", "2025-05-01T10:00:00+00:00"),
            (GID2, BOB_ID, "Bob Renamed", "Bob Renamed (@bob)", "2025-06-01T10:00:00+00:00"),
        ]
        con = sqlite3.connect(tmpdb)
        for gid, aid, name, full, ts in bob_rev:
            con.execute("INSERT OR REPLACE INTO chunks (msg_id, guild_id, channel_id, channel_name, author_id, author_name, author_full, content, chunk_text, window_line, window_lines, reply_to, ts, jump_url, embedding) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (f"rev{ts[:10]}", gid, "300", "other", aid, name, full, "rename test", "rename test", "[]", None, None, ts, "", np.zeros(4, dtype=np.float32).tobytes()))
        con.commit()
        con.close()
        rag._rebuild_authors_from_chunks()
        users = rag._find_users_db(GID2, "bob", 5)
        check("rebuild picks newest name", users and users[0]["display_name"] == "Bob Renamed", str(users))

        print("== LIKE metacharacters escaped ==")
        check("percent literal", rag._find_users_db(GID, "%", 5) == [])
        check("underscore literal", rag._find_users_db(GID, "Ali_e", 5) == [])
        check("backslash literal", rag._find_users_db(GID, "\\", 5) == [])

        print("== purge clears authors when last chunk deleted ==")
        class _FA:
            id = BOB_ID
        class _FM:
            id = 50
            author = _FA()
            guild = type("G", (), {"id": GID2})()
        rag._chunks[GID2] = [{"msg_id": "50", "channel_id": "300", "ts": "2025-05-01T10:00:00+00:00", "embedding": np.zeros(4, dtype=np.float32)}]
        rag._msg_index[GID2] = {50: 0}
        rag._matrices[GID2] = np.zeros((1, 4), dtype=np.float32)
        rag._backfilling.discard(GID2)
        await rag.on_message_delete(_FM())
        users = rag._find_users_db(GID2, "bob", 5)
        con = sqlite3.connect(tmpdb)
        nchunks = con.execute("SELECT COUNT(*) FROM chunks WHERE guild_id=?", (GID2,)).fetchone()[0]
        con.close()
        check("authors purged with guild", users == [] and nchunks == 0, f"users={users} chunks={nchunks}")

        print("== tool wiring (exec_history_tool) ==")
        import cogs.utils as utils_mod
        class _FakeBot:
            def __init__(self, cog): self._cog = cog
            def get_cog(self, name): return self._cog if name == "HistoryRAG" else None
        class _FakeGuild:
            id = GID
        fake_bot = _FakeBot(rag)
        text, _ = await utils_mod.exec_history_tool("find_user", {"query": "alice"}, bot=fake_bot, guild=_FakeGuild(), channel=None)
        check("find_user via tool layer", "author_id=111" in text and "msgs" in text, text[:200])
        text, _ = await utils_mod.exec_history_tool("find_user", {"query": "zed"}, bot=fake_bot, guild=_FakeGuild(), channel=None)
        check("find_user no match hints fallback", "search_history" in text, text[:200])
        text, _ = await utils_mod.exec_history_tool("find_user", {"query": "alice"}, bot=_FakeBot(None), guild=_FakeGuild(), channel=None)
        check("find_user no cog", text == "Histórico não disponível.", text)
        text, _ = await utils_mod.exec_history_tool("find_user", {"query": "alice"}, bot=fake_bot, guild=None, channel=None)
        check("find_user in DM", "servidor" in text, text)
        orig_fetch = utils_mod.fetch_message_context
        async def _fake_fetch(*a, **k):
            return None
        utils_mod.fetch_message_context = _fake_fetch  # type: ignore[assignment]
        try:
            text, _ = await utils_mod.exec_history_tool("get_message_context", {"message_id": "2", "channel_id": GENERAL}, bot=fake_bot, guild=_FakeGuild(), channel=None)
            check("context falls back to index", text is not None and "Contexto ao redor" in text and "msg_id=2" in text, str(text)[:200])
            text, _ = await utils_mod.exec_history_tool("get_message_context", {"message_id": "999", "channel_id": GENERAL, "window": None}, bot=fake_bot, guild=_FakeGuild(), channel=None)
            check("unknown message + null window no crash", "não encontrada" in text, str(text)[:200])
        finally:
            utils_mod.fetch_message_context = orig_fetch  # type: ignore[assignment]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
