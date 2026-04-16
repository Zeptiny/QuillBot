# QuillBot — Comprehensive Code Review

> **Date:** 2026-04-16  
> **Scope:** Full codebase (`main.py`, `cogs/`, `responses/`)  
> **Application type:** Discord bot for the Miners' Refuge Minecraft server-admin community. Features include slash commands, AI-powered log analysis, and RAG-based documentation Q&A.

---

## 1. User Experience

### 1.1 Feedback & Loading States

| Finding | Severity | Details |
|---------|----------|---------|
| **No feedback during doc indexing at startup** | Medium | When `DocsRAG.index_docs()` runs on first boot (no cached vectors), the bot is available but the `/ask` command will silently return no results until indexing finishes. Users have no way to know indexing is in progress. **Suggestion:** Set a `self._indexing` flag and respond with a "Documentation is being indexed, please try again shortly" message while it's active. |
| **`/reindex` lacks progress reporting** | Low | The command defers with a "thinking" indicator, but for large doc sets the embedding step can take a long time with no intermediate feedback. **Suggestion:** Send a preliminary "Starting reindex…" message, then edit it with progress (e.g., "Embedding chunk 40/120…"). |
| **mclo.gs upload failure message is vague** | Medium | When `upload_mclogs` fails, the user gets "Algo deu errado ao tentar fazer o upload para o mclo.gs." with no actionable guidance. **Suggestion:** Suggest the user upload manually to mclo.gs and include the link. |
| **`/docs` search parameter is decorative** | High | The `busca` parameter on `/docs` does nothing beyond echoing the term. The bot tells the user to press `CTRL+K` on the site. This is misleading — users expect the bot to search for them. **Suggestion:** Either remove the parameter, or integrate it with the existing RAG search in `DocsRAG` so the bot actually searches the docs. |

### 1.2 Error Handling Exposed to Users

| Finding | Severity | Details |
|---------|----------|---------|
| **AI failures return generic messages** | Medium | Both `/ask` and `/analyze` show a generic "Ocorreu um erro" on any exception. Users cannot distinguish between transient errors and misconfiguration. **Suggestion:** Differentiate between rate-limit/quota errors (suggest retry later) vs. other failures. |
| **No validation on `log_link` format before deferring** | Low | `/analyze` defers (shows "thinking") before validating the link format. If the link is invalid, the user waited for the defer animation for nothing. **Suggestion:** Validate the link format _before_ calling `defer()`. |

### 1.3 Accessibility & Discoverability

| Finding | Severity | Details |
|---------|----------|---------|
| **All UI text is in Portuguese only** | Low (intentional) | This is appropriate for the target community but worth noting — there's no i18n support if the community grows internationally. |
| **Follow-up conversation mechanism is non-obvious** | Medium | Users must _reply_ to the bot's `/ask` embed to continue a conversation. Nothing in the embed visually teaches this beyond a small footer text. **Suggestion:** Add an explicit line in the embed body, e.g., "💬 Responda a esta mensagem para fazer perguntas de acompanhamento." |
| **`/help` is manually maintained** | Low | The help embed is hardcoded and can drift out of sync with actual commands. **Suggestion:** Generate help text dynamically from registered commands and their descriptions. |

---

## 2. Consistency

### 2.1 Naming Conventions

| Finding | Details |
|---------|---------|
| **Mixed language in identifiers** | Code comments and docstrings are in English, but slash-command names/descriptions and user-facing strings are in Portuguese. This is fine, but a few identifiers leak Portuguese into code: the command method `plov`, `plano`, parameter `pergunta`, `imagem`, `busca`. The rest of the codebase uses English identifiers (e.g., `check_message`, `read_file_content`). **Suggestion:** Use English for all Python identifiers and reserve Portuguese for user-facing `name=`/`description=` strings only. |
| **Inconsistent variable naming style** | `mclogs_url` vs `upload_link` vs `link` — different names for the same concept (a mclo.gs URL). Standardize to one name. |

### 2.2 Structural Inconsistencies

| Finding | Details |
|---------|---------|
| **`OPENROUTER_API_KEY` read in two places** | Both `log_analyzer.py` and `docs_rag.py` independently call `os.getenv('OPENROUTER_API_KEY')` at module level. **Suggestion:** Centralize configuration in a single `config.py` module. |
| **`CHAT_MODEL` duplicated** | The same env-var default `'google/gemini-2.5-flash-lite'` is repeated in both `log_analyzer.py` and `docs_rag.py`. If someone updates one and not the other, behavior diverges silently. |
| **AI client initialization differs between cogs** | `LogAnalyzer` creates the `AsyncOpenAI` client only if the API key exists (guarded). `DocsRAG` creates it unconditionally — but the entire cog is skipped in `setup()` if the key is missing. The approach works but the patterns are inconsistent. **Suggestion:** Pick one pattern and use it everywhere. |
| **`on_message` listener in two cogs** | Both `LogAnalyzer` and `DocsRAG` register `on_message` listeners. Execution order depends on cog load order which is implicitly defined in `main.py`'s `COGS` list. There is no priority system and no documentation about potential conflicts. **Suggestion:** Document the intended listener order or consolidate message handling. |
| **Error response patterns differ** | `LogAnalyzer.analyze` uses `ephemeral=True` for some errors and not others. Some error responses are plain text, others use embeds. **Suggestion:** Standardize — use ephemeral messages for user-input errors and public messages for system errors. |

### 2.3 Project Structure

| Finding | Details |
|---------|---------|
| **`responses/` package is thin** | The `responses/` package contains only `errors.py` with a pattern dictionary. It is arguably a data file, not a "response" module. **Suggestion:** Consider renaming to `patterns/` or moving into the `cogs/` directory alongside `log_analyzer.py` since it's tightly coupled. |
| **No `__init__.py` exports** | Both `cogs/__init__.py` and `responses/__init__.py` are empty. This is fine but the packages could export their public APIs for clarity. |

---

## 3. Code Quality & Improvements

### 3.1 High Impact

| # | Finding | Details |
|---|---------|---------|
| H1 | **No rate limiting on AI commands** | `/ask`, `/analyze`, and follow-up replies all call the OpenRouter API with no per-user or per-channel cooldown. A single user can spam commands and exhaust the API quota/budget. **Suggestion:** Add a `commands.cooldown` or `app_commands.checks.cooldown` decorator (e.g., 1 use per 30 seconds per user). |
| H2 | **Unbounded conversation cache** | `DocsRAG._conversations` grows until it hits 100 entries, then bulk-deletes the oldest 50. This is a simple but fragile eviction strategy — under load it causes periodic latency spikes. There is also no TTL, so stale conversations linger. **Suggestion:** Use `collections.OrderedDict` with a max size, or an LRU cache with TTL (e.g., `cachetools.TTLCache`). |
| H3 | **`read_file_content` does not enforce size limit on streaming** | The method checks `Content-Length` header but still calls `resp.text()` which reads the entire body. Servers may omit `Content-Length` or lie. For a 100 MB file with no header, the bot will OOM. **Suggestion:** Read in chunks and abort when the accumulated size exceeds `MAX_CONTENT_SIZE`. |
| H4 | **No input sanitization on log content sent to LLM** | Raw log content is inserted into the prompt. A malicious log file could contain prompt-injection text. While the risk is somewhat mitigated by the system prompt, there is no explicit sanitization. **Suggestion:** Strip or escape non-printable characters and add a delimiter/fence to the prompt. |
| H5 | **`BOT_TOKEN` is used without validation** | If `BOT_TOKEN` is `None` (env var not set), `bot.start(None)` will raise a cryptic `LoginFailure`. **Suggestion:** Validate at startup and exit with a clear error message. |

### 3.2 Medium Impact

| # | Finding | Details |
|---|---------|---------|
| M1 | **Sequential document fetching during indexing** | `index_docs()` fetches each doc file one at a time in a `for` loop. For a repo with many docs, this is slow. **Suggestion:** Use `asyncio.gather` or `asyncio.Semaphore` to fetch documents concurrently (e.g., 5 at a time). |
| M2 | **Cosine similarity computed in pure Python** | The `cosine_similarity` function iterates element-by-element. For hundreds of chunks this is measurably slow. **Suggestion:** Use `numpy` for vectorized computation, or compute similarities in batch. |
| M3 | **Embed description truncation is naive** | Both `/ask` and `/analyze` truncate at 3800 characters by slicing the string, which can break markdown formatting mid-tag. **Suggestion:** Find a safe break point (e.g., last `\n` before the limit). |
| M4 | **Vector store file is JSON with embeddings** | Storing float vectors as JSON is space-inefficient. A 768-dim embedding uses ~6 KB in JSON vs ~3 KB in binary. **Suggestion:** Consider using `numpy.save`, `pickle`, or a lightweight vector DB like `lancedb` for better performance at scale. |
| M5 | **No graceful handling of OpenRouter downtime** | If OpenRouter is down, every AI command fails with a generic error. **Suggestion:** Add a circuit-breaker pattern: after N consecutive failures, temporarily disable AI commands and inform users that the service is degraded. |
| M6 | **`on_message` in `LogAnalyzer` processes every message** | The listener runs `check_message` on every non-bot message even if the content is clearly not a log. This is low overhead now but won't scale. **Suggestion:** Add a quick pre-filter (e.g., skip short messages, or only check in designated support channels). |

### 3.3 Low Impact

| # | Finding | Details |
|---|---------|---------|
| L1 | **No type hints on module-level constants** | Constants like `COGS`, `ANALYZE_SYSTEM_PROMPT`, etc. lack type annotations. This is fine for a small project but adding `Final` annotations would prevent accidental mutation. |
| L2 | **`patterns` list in `errors.py` uses `dict` ordering for init** | The `_raw_patterns` dict relies on insertion-order (guaranteed in Python 3.7+). This is fine but a comment noting that order matters (first match wins) would aid maintainability. |
| L3 | **No tests** | There are zero tests in the project. At minimum, `check_message` and `cosine_similarity` are pure functions that are trivially testable. **Suggestion:** Add a `tests/` directory with unit tests for pure logic. |
| L4 | **Missing `py.typed` or type checker config** | No `mypy`, `pyright`, or `ruff` configuration. **Suggestion:** Add a `pyproject.toml` with basic linting/type-checking config. |
| L5 | **`_extract_paths` regex could match non-doc links** | The pattern `r'\(([^)]+\.md)\)'` would also match external `.md` links in SUMMARY.md. **Suggestion:** Restrict to relative paths by excluding `http` prefixes. |
| L6 | **mclo.gs raw URL construction missing for analyze command** | When a user provides an mclo.gs link to `/analyze`, the bot tries to fetch the URL as-is. mclo.gs page URLs return HTML, not raw text. The code should hit the API endpoint (`https://api.mclo.gs/1/raw/<id>`) instead. |

---

## 4. Integrations & Third-Party Opportunities

### 4.1 Spigot / Modrinth / Hangar — Plugin Information

| Integration | Value | Implementation Sketch |
|-------------|-------|-----------------------|
| **Spiget API** (`spiget.org`) | Look up Spigot plugin metadata (version, dependencies, author) when a user mentions a plugin name in logs or asks about one. | Add a `/plugin <name>` command that queries `https://api.spiget.org/v2/search/resources/<name>`. Display version, download count, supported MC versions, and a direct SpigotMC link. |
| **Modrinth API** (`api.modrinth.com`) | Modrinth is increasingly popular for both mods and plugins. Supporting Modrinth extends coverage beyond SpigotMC. | Query `https://api.modrinth.com/v2/search?query=<name>` and present results. Can be combined with the Spiget results in a unified `/plugin` command. |
| **Hangar API** (`hangar.papermc.io`) | PaperMC's official plugin repository. Particularly relevant since many Miners' Refuge users likely use Paper. | Query `https://hangar.papermc.io/api/v1/projects?q=<name>`. |

### 4.2 Documentation Scraping & Indexing

| Integration | Value | Implementation Sketch |
|-------------|-------|-----------------------|
| **PaperMC Docs** (`docs.papermc.io`) | Official Paper documentation — highly relevant for the target audience. | Scrape or use the GitHub source (`PaperMC/docs`) to index Paper's admin docs into the existing RAG pipeline alongside the Miners' Refuge docs. Add a source tag to distinguish origins. |
| **PurpurMC Docs** (`purpurmc.org/docs`) | Popular Paper fork with extra configuration options. | Same scraping approach. |
| **Spark Profiler Docs** (`spark.lucko.me`) | Performance profiling tool — relevant to the "server overloaded" errors the bot already detects. | Index Spark's documentation and link to it when performance-related log patterns are matched. |
| **Minecraft Wiki (Technical)** | General Minecraft server admin knowledge base. | Selectively index admin-relevant pages (server.properties reference, RCON, etc.). Be mindful of content licensing. |

### 4.3 Log & Paste Services

| Integration | Value | Implementation Sketch |
|-------------|-------|-----------------------|
| **Expand paste service support** | Currently only mclo.gs and pastebin.com are recognized. | Add regex patterns for `paste.gg`, `hastebin.com`/`hst.sh`, `gist.github.com`, `bytebin.lucko.me` (used by LuckPerms/Spark). Each has a raw-content API. |
| **Spark profiler link detection** | Spark generates links like `https://spark.lucko.me/<id>` for profiler reports. | Detect these links, fetch the report data via the Spark API, and provide a summary (TPS, tick timings, hotspot analysis). |

### 4.4 Server Status & Monitoring

| Integration | Value | Implementation Sketch |
|-------------|-------|-----------------------|
| **Minecraft Server Status API** | Allow users to check if a server is online and get player count, version, MOTD. | Use the Minecraft protocol query or a service like `api.mcsrvstat.us` — add a `/status <ip>` command. |
| **mcsrvstat.us** | Free API for querying Minecraft server status. | `GET https://api.mcsrvstat.us/2/<address>` returns online status, player count, version, MOTD, and favicon. |

### 4.5 Other Valuable Integrations

| Integration | Value | Implementation Sketch |
|-------------|-------|-----------------------|
| **GitHub Integration (for the docs repo)** | Display recent doc changes, allow users to see what's new. | Use the GitHub API (already partially used for commit SHA) to add a `/changelog` command showing recent doc commits. Could also post to a channel when docs are updated. |
| **Timings / Spark Report Analyzer** | Auto-detect Timings v2 (`timings.aikar.co`) or Spark links and summarize. | Fetch the report JSON and highlight top tick consumers, entity counts, and plugin overhead. Present a simplified summary in an embed. |
| **Flag system (Aikars Flags, GraalVM, etc.)** | Help users generate optimal JVM startup flags. | Add a `/flags <ram> [jvm_type]` command that generates recommended flags based on Aikar's flags guide, customized to their RAM allocation. |

---

## Summary

| Area | Critical/High | Medium | Low |
|------|:---:|:---:|:---:|
| User Experience | 1 | 3 | 2 |
| Consistency | — | 5 | 2 |
| Code Quality | 5 | 6 | 6 |
| Integrations | — | — | — |

**Top 5 recommended actions (by impact):**

1. **Add rate limiting** to AI-powered commands to prevent API quota exhaustion (H1)
2. **Fix `/docs` search** to actually search docs or remove the misleading parameter (UX 1.1)
3. **Enforce size limit on HTTP reads** with streaming to prevent OOM on large files (H3)
4. **Validate `BOT_TOKEN`** at startup with a clear error (H5)
5. **Centralize configuration** (`config.py`) to eliminate duplicated env-var reads and defaults (Consistency 2.2)
