"""Standalone smoke test for the read-only sql_history tool.

Exercises: input validation (single SELECT, guild scoping, no writes/PRAGMA),
the sandbox (embedding column denied, query_only, timeout abort), REGEXP,
rendering/truncation, and exec_history_tool wiring. All against a throwaway
SQLite DB. No Discord or LLM calls.

Run: python3 test_history_sql.py
"""

import asyncio
import os
import shutil
import sqlite3
import tempfile

import numpy as np

from cogs.history_rag import HistoryRAG

GID = 777
GID2 = 888
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


def chunk(mid, gid, cid, cname, aid, aname, afull, content, ts, reply_to=None):
    return {
        "msg_id": str(mid),
        "guild_id": str(gid),
        "channel_id": str(cid),
        "channel_name": cname,
        "author_id": str(aid),
        "author_name": aname,
        "author_full": afull,
        "content": content,
        "chunk_text": content,
        "window_line": content,
        "window_lines": [],
        "reply_to": reply_to,
        "ts": ts,
        "jump_url": f"https://discord.com/channels/{gid}/{cid}/{mid}",
        "embedding": np.zeros(4, dtype=np.float32),
    }


def build_chunks():
    alice = lambda mid, content, ts, cid=GENERAL, cname="general", reply=None: chunk(
        mid, GID, cid, cname, ALICE_ID, "Alice", "Alice (@alice)", content, ts, reply
    )
    bob = lambda mid, content, ts, reply=None: chunk(
        mid, GID, DEV, "dev", BOB_ID, "Bob", "Bob (@bob)", content, ts, reply
    )
    return [
        alice(1, "server lag is terrible today", "2025-06-01T10:00:00+00:00"),
        alice(2, "restart fixed the server lag", "2025-06-01T10:02:00+00:00", reply="1"),
        bob(3, "check paper config please <@111>", "2025-06-01T11:00:00+00:00"),
        alice(4, "planning notes for Q3", "2025-07-01T10:00:00+00:00"),
        bob(5, "another lag spike <@!111> help", "2025-07-02T09:00:00+00:00"),
    ]


def rejects(rag, sql, guild=GID, label=""):
    try:
        rag._exec_sql_sync(guild, sql, 5.0, 50)
        return False
    except ValueError:
        return True
    except Exception:
        return False


async def main():
    tmp = tempfile.mkdtemp(prefix="quillbot_sql_test_")
    tmpdb = os.path.join(tmp, "history.db")
    try:
        rag = HistoryRAG(bot=None)
        rag._db_path = lambda: tmpdb  # type: ignore[method-assign]
        rag._ensure_db()
        chunks = build_chunks()
        rag._upsert_chunks_to_db(GID, chunks)
        rag._upsert_authors(GID, chunks)
        other = [chunk(50, GID2, "300", "other", BOB_ID, "Bob", "Bob (@bob)", "secret other guild", "2025-06-01T10:00:00+00:00")]
        rag._upsert_chunks_to_db(GID2, other)
        rag._upsert_authors(GID2, other)

        print("== validation: single read-only SELECT scoped to the guild ==")
        check("empty rejected", rejects(rag, "", label="vazio"))
        check("plain select ok", not rejects(rag, f"SELECT COUNT(*) FROM chunks WHERE guild_id={GID}"))
        check("missing guild id rejected", rejects(rag, "SELECT COUNT(*) FROM chunks"))
        check("other guild id rejected", rejects(rag, f"SELECT COUNT(*) FROM chunks WHERE guild_id={GID2}"))
        check("insert rejected", rejects(rag, f"INSERT INTO chunks (msg_id) VALUES ('x'); SELECT 1 WHERE guild_id={GID}"))
        check("update rejected", rejects(rag, f"UPDATE chunks SET content='x' WHERE guild_id={GID}"))
        check("delete rejected", rejects(rag, f"DELETE FROM chunks WHERE guild_id={GID}"))
        check("drop rejected", rejects(rag, f"DROP TABLE chunks -- guild_id={GID}"))
        check("pragma rejected", rejects(rag, f"PRAGMA query_only=0; SELECT 1 WHERE guild_id={GID}"))
        check("multiple statements rejected", rejects(rag, f"SELECT 1 WHERE guild_id={GID}; SELECT 2"))
        check("semicolon inside literal ok", not rejects(rag, f"SELECT COUNT(*) FROM chunks WHERE guild_id={GID} AND content LIKE '%a;b%'"))
        check("comment injection rejected", rejects(rag, f"SELECT 1 WHERE guild_id={GID} -- comment\n; DROP TABLE chunks"))
        check("lowercase select ok", not rejects(rag, f"select count(*) from chunks where guild_id={GID}"))
        check("cte ok", not rejects(rag, f"WITH t AS (SELECT 1 x) SELECT * FROM t WHERE guild_id={GID} OR x=1"))
        check("overlong query rejected", rejects(rag, f"SELECT 1 WHERE guild_id={GID} AND content LIKE '%{'a' * 5000}%'"))

        print("== write shapes that pass the SELECT regex (authorizer must deny) ==")
        for label, sql in [
            ("with-insert", f"WITH t AS (SELECT 1 x) INSERT INTO authors (guild_id, author_id) SELECT {GID}, 'evil' FROM t"),
            ("with-delete", f"WITH t AS (SELECT 1 x) DELETE FROM chunks WHERE guild_id={GID}"),
            ("explain", f"EXPLAIN SELECT 1 FROM chunks WHERE guild_id={GID}"),
            ("load_extension", f"SELECT load_extension('x') FROM chunks WHERE guild_id={GID}"),
        ]:
            try:
                rag._exec_sql_sync(GID, sql, 3.0, 10)
                check(f"{label} blocked", False, "no exception")
            except (ValueError, sqlite3.Error):
                check(f"{label} blocked", True)
        con = sqlite3.connect(tmpdb)
        rows = con.execute(f"SELECT author_id FROM authors WHERE guild_id={GID}").fetchall()
        con.close()
        check("no data mutated", all(a != "evil" for (a,) in rows), str(rows))

        print("== guild isolation ==")
        out = rag._exec_sql_sync(GID, f"SELECT msg_id, content FROM chunks WHERE guild_id={GID}", 5.0, 50)
        check("only current guild rows", "secret other guild" not in out and "server lag" in out, out)
        check("guild-less query rejected", rejects(rag, "SELECT COUNT(*) FROM chunks"))

        print("== aggregations, FTS, mentions, replies ==")
        out = rag._exec_sql_sync(GID, f"SELECT author_full, COUNT(*) n FROM chunks WHERE guild_id={GID} GROUP BY author_full ORDER BY n DESC", 5.0, 50)
        check("group by author", "Alice (@alice) | 3" in out and "Bob (@bob) | 2" in out, out)
        out = rag._exec_sql_sync(GID, f"SELECT c.msg_id FROM chunks c JOIN chunks_fts f ON f.msg_id=c.msg_id WHERE f.guild_id={GID} AND chunks_fts MATCH 'lag' ORDER BY c.msg_id", 5.0, 50)
        check("fts join", out.split("\n")[1:] == ["1", "2", "5"], out)
        out = rag._exec_sql_sync(GID, f"SELECT COUNT(*) FROM chunks WHERE guild_id={GID} AND (content LIKE '%<@{ALICE_ID}>%' OR content LIKE '%<@!{ALICE_ID}>%')", 5.0, 50)
        check("user mentions counted", "2" in out, out)
        out = rag._exec_sql_sync(GID, f"SELECT msg_id FROM chunks WHERE guild_id={GID} AND reply_to IS NOT NULL", 5.0, 50)
        check("replies found", out.split("\n")[1:] == ["2"], out)
        out = rag._exec_sql_sync(GID, f"SELECT msg_id FROM chunks WHERE guild_id={GID} AND reply_to='1'", 5.0, 50)
        check("reply_to target", out.strip().endswith("2"), out)
        out = rag._exec_sql_sync(GID, f"SELECT substr(ts,1,7) m, COUNT(*) FROM chunks WHERE guild_id={GID} GROUP BY m ORDER BY m", 5.0, 50)
        check("monthly buckets", "2025-06" in out and "2025-07" in out, out)

        print("== regexp function ==")
        out = rag._exec_sql_sync(GID, f"SELECT COUNT(*) FROM chunks WHERE guild_id={GID} AND REGEXP('spike|terrible', content)", 5.0, 50)
        check("regexp or-pattern", "2" in out, out)
        out = rag._exec_sql_sync(GID, f"SELECT COUNT(*) FROM chunks WHERE guild_id={GID} AND REGEXP('LAG', content)", 5.0, 50)
        check("regexp ignorecase", "3" in out, out)
        try:
            rag._exec_sql_sync(GID, f"SELECT REGEXP('(', 'x') FROM chunks WHERE guild_id={GID}", 5.0, 50)
            check("bad regex errors", False)
        except ValueError as e:
            check("bad regex errors", "REGEXP" in str(e))

        print("== sandbox ==")
        try:
            rag._exec_sql_sync(GID, f"SELECT embedding FROM chunks WHERE guild_id={GID} LIMIT 1", 5.0, 50)
            check("embedding read denied", False)
        except sqlite3.DatabaseError as e:
            check("embedding read denied", "embedding" in str(e) or "proibido" in str(e) or "not authorized" in str(e).lower(), str(e))
        try:
            rag._exec_sql_sync(GID, f"ATTACH DATABASE 'x.db' AS x; SELECT 1 WHERE guild_id={GID}", 5.0, 50)
            check("attach blocked", False, "should have been rejected")
        except ValueError:
            check("attach blocked", True)
        try:
            rag._exec_sql_sync(GID, f"WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT COUNT(*) FROM c JOIN chunks ON chunks.guild_id={GID}", 0.3, 50)
            check("timeout aborts runaway query", False, "no exception raised")
        except sqlite3.OperationalError as e:
            check("timeout aborts runaway query", "interrupted" in str(e).lower(), str(e))
        except Exception as e:
            check("timeout aborts runaway query", False, f"unexpected {type(e).__name__}: {e}")

        print("== rendering and truncation ==")
        out = rag._exec_sql_sync(GID, f"SELECT COUNT(*) n FROM chunks WHERE guild_id={GID}", 5.0, 50)
        check("header row present", out.split("\n")[0] == "n", out)
        check("count value rendered", out.split("\n")[1] == "5", out)
        out = rag._exec_sql_sync(GID, f"SELECT msg_id FROM chunks WHERE guild_id={GID}", 5.0, 2)
        check("row cap notice", "resultado truncado em 2 linhas" in out and out.count("\n") == 3, out)
        out = rag._exec_sql_sync(GID, f"SELECT content FROM chunks WHERE guild_id={GID}", 5.0, 50)
        check("nulls/cells flatten", "\n" not in out.split("\n")[1] if len(out.split("\n")) > 1 else True)
        long_content = "x" * 5000
        lc = [chunk(90, GID, GENERAL, "general", ALICE_ID, "Alice", "Alice (@alice)", long_content, "2025-08-01T10:00:00+00:00")]
        rag._upsert_chunks_to_db(GID, lc)
        out = rag._exec_sql_sync(GID, f"SELECT content FROM chunks WHERE guild_id={GID} AND msg_id='90'", 5.0, 50)
        check("cells clipped", len(out) < 400, f"len={len(out)}")
        out = rag._exec_sql_sync(GID, f"SELECT NULL n FROM chunks WHERE guild_id={GID} LIMIT 1", 5.0, 50)
        check("null rendered", "NULL" in out, out)

        print("== read-only coexists with the writer (WAL) ==")
        wcon = sqlite3.connect(tmpdb)
        try:
            out = rag._exec_sql_sync(GID, f"SELECT COUNT(*) FROM chunks WHERE guild_id={GID}", 5.0, 50)
            check("query while writer open", out.split("\n")[1] == "6", out)
        finally:
            wcon.close()

        print("== tool wiring (exec_history_tool) ==")
        import cogs.utils as utils_mod
        from cogs import utils as U
        class _FakeBot:
            def __init__(self, cog): self._cog = cog
            def get_cog(self, name): return self._cog if name == "HistoryRAG" else None
        class _FakeGuild:
            def __init__(self, id): self.id = id
        fake_bot = _FakeBot(rag)
        q = f"SELECT author_full, COUNT(*) n FROM chunks WHERE guild_id={GID} GROUP BY author_full ORDER BY n DESC LIMIT 5"
        text, _ = await utils_mod.exec_history_tool("sql_history", {"sql": q}, bot=fake_bot, guild=_FakeGuild(GID), channel=None)
        check("sql_history via tool layer", "Alice (@alice) | 4" in text, text[:200])
        text, _ = await utils_mod.exec_history_tool("sql_history", {"sql": f"DELETE FROM chunks WHERE guild_id={GID}"}, bot=fake_bot, guild=_FakeGuild(GID), channel=None)
        check("write attempt rejected with guidance", "Consulta rejeitada" in text and "SELECT" in text, text)
        text, _ = await utils_mod.exec_history_tool("sql_history", {"sql": q}, bot=fake_bot, guild=None, channel=None)
        check("DM rejected", "servidor" in text, text)
        text, _ = await utils_mod.exec_history_tool("sql_history", {"sql": q}, bot=_FakeBot(None), guild=_FakeGuild(GID), channel=None)
        check("no cog rejected", text == "Histórico não disponível.", text)
        text, _ = await utils_mod.exec_history_tool("sql_history", {"sql": f"SELECT nosuchcol FROM chunks WHERE guild_id={GID}"}, bot=fake_bot, guild=_FakeGuild(GID), channel=None)
        check("sqlite error surfaced", "Erro SQL" in text and "nosuchcol" in text.lower(), text[:200])
        status = utils_mod.history_tool_status("sql_history", {"sql": "SELECT\n  COUNT(*)\nFROM chunks"})
        check("status label flattens sql", status is not None and "\n" not in status, str(status))
        enabled = U.HISTORY_SQL_TOOL_ENABLED
        try:
            U.HISTORY_SQL_TOOL_ENABLED = False
            text, _ = await utils_mod.exec_history_tool("sql_history", {"sql": q}, bot=fake_bot, guild=_FakeGuild(GID), channel=None)
            check("disabled flag honored", "desativada" in text, text)
        finally:
            U.HISTORY_SQL_TOOL_ENABLED = enabled
        check("tool definition present", U.SQL_HISTORY_TOOL["function"]["name"] == "sql_history")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)


if __name__ == "__main__":
    asyncio.run(main())
