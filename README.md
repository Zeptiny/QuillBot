# QuillBot — Discord Bot for Minecraft Server Administration

AI-powered Discord bot for the **Miners' Refuge** community. Helps Minecraft server admins troubleshoot logs, diagnose performance with Spark, search documentation, and find plugins — all from Discord.

Built with [discord.py](https://discordpy.readthedocs.io/) + RAG (Retrieval-Augmented Generation) over multiple documentation sources.

---

## Table of Contents

- [Features](#features)
- [Slash Commands](#slash-commands)
- [Passive Features](#passive-features)
- [AI & RAG System](#ai--rag-system)
- [Documentation Sources](#documentation-sources)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [Docker](#docker)

---

## Features

| Category | Capability |
|---|---|
| **Q&A (RAG)** | Answers questions grounded in indexed docs (Miners' Refuge, PaperMC, PurpurMC, Spark) |
| **General Chat** | Open-ended assistant with web search, server/history context awareness |
| **Log Analysis** | AI analysis of mclo.gs/pastebin links, file uploads, and screenshots |
| **Error Detection** | Instant passive replies for 20+ known Minecraft error patterns |
| **Spark Profiler** | Full report parsing, bottleneck diagnosis, platform-aware recommendations |
| **Plugin Search** | Concurrent search across Modrinth, Hangar, SpigotMC |
| **Server Tools** | JVM flags, server status checks, docs changelog |
| **Persistent Memory** | The bot remembers: atomic facts about the server and its members, auto-injected into every conversation, with inline LLM writes, pinning, dedupe, full audit history and admin control |

All AI responses are in **Brazilian Portuguese** and formatted for Discord embeds.

---

## Slash Commands

### AI-Powered Commands

#### `/ask <question> [image]`
Ask any Minecraft server administration question. Uses an agentic RAG loop — the LLM automatically calls `search_docs`, `search_plugins`, `memory_search`, `search_history`, and context tools to ground its answer in real documentation.

- **Model:** `CHAT_MODEL` (default: `qwen/qwen3.6-plus`)
- **Tools available:** `search_docs`, `search_plugins`, `memory_search`, `memory_write`, `memory_about`, `get_channel_history`, `get_guild_info`, `search_history`, `get_message_context`, `find_user`, `sql_history`
- **Features:** Image/screenshot analysis, 5-minute ephemeral prompt+report cache, paginated multi-embed output with source links, reply to continue conversation (30 min TTL, 200 conversations), cooldown per user
- **Follow-up:** Reply to the bot's response to continue the conversation with full history

#### `/chat <question> [image]`
General-purpose assistant (same agentic loop as `/ask` but with web search).

- **Tools:** `web_search`, `web_extract` (via Tavily), plus the same context/history tools as `/ask`
- **Web search:** Supports `search_depth` (basic/advanced), `time_range`, domain filtering
- **Reply follow-up** and **@mention mode:** Mention the bot (`@QuillBot <question>`) to chat without a slash command — same rate-limit and conversation handling as `/chat`
- Set `CHAT_MENTION_ENABLED=false` to disable mention mode

#### `/analyze [log_link] [log_file] [image]`
AI-powered log/crash-report analysis.

- **Inputs (at least one required):**
  - `log_link` — mclo.gs or pastebin.com URL (parsed to raw API endpoint)
  - `log_file` — `.log` or `.txt` attachment (auto-uploaded to mclo.gs for sharing, streaming read up to 5 MB)
  - `image` — screenshot of an error/log (vision model)
- **Processing:** Sanitizes non-printable chars, extracts relevant lines for large logs (keeps 60 header lines + error/warning/exception lines, truncates to 12k chars), sends to LLM with a structured diagnostic prompt
- **Output:** Sections — Resumo, Erros Encontrados, Avisos, Recomendações (only non-empty sections shown)
- **Button fallback:** When a pasted log has no regex pattern match, a "🔍 Analisar com IA" button is offered to trigger the same analysis

#### `/spark <url>`
Analyze a [Spark Profiler](https://spark.lucko.me) report with deep AI diagnosis.

- **Input:** Full `https://spark.lucko.me/<code>` URL or bare report code
- **Parser (`spark_parser.py`):** Fetches from `spark-json-service.lucko.me`, extracts server identity, JVM flags, TPS/MSPT, GC, CPU hotspots (chain-collapsed call tree with `self_pct`), world/entity stats, configs, game rules, plugins
- **LLM flow:** Injects a compact summary (~1.7k chars) as synthetic prior exchange, follows an 8-step diagnostic protocol (TPS → MSPT → lag spike detection → waitForNextTick → GC → hotspots → config → grounded recommendation), uses `SPARK_MODEL` (default: `google/gemini-2.5-pro`)
- **Spark tools:** `get_spark_detail` (sections: `hotspots`, `jvm`, `profiler`, `plugins`, `world`, `game_rules`, `configs:<file>`) and `get_config_key` (single-key lookup)
- **Platform awareness:** Detects Paper/Purpur/Spigot/Forge/Fabric/Vanilla and never recommends configs that don't exist for that platform
- **Passive detection:** Any `spark.lucko.me` link in chat triggers a "🔥 Analisar com IA" button

#### `/reindex [source]` *(Admin only)*
Re-index documentation vectors.

- Without `source`: full reindex of all `DOC_SOURCES`
- With `source`: reindex a single source by label (e.g. `PaperMC`)
- Shows `⏳ Indexando...` guard — blocks `/ask`/`/docs` queries during indexing
- Per-source commit SHA tracking for granular updates

#### `/memory <subcommand>` — Persistent Bot Memory
The bot maintains long-term memories: atomic one-sentence facts about the server (guild-wide) or about individual users (scoped, privacy-guarded). Relevant memories are **automatically injected** into every `/ask`, `/chat` and @mention turn — the LLM doesn't need to search first.

| Subcommand | Who | Description |
|---|---|---|
| `/memory list [kind]` | everyone | Paginated list (users see guild memories + their own; admins see all) |
| `/memory show <id>` | everyone | Memory details (own or guild-wide only); admins also see history IDs |
| `/memory edit <id>` | Admin | Edit content / importance |
| `/memory delete <id>` | Admin | Archive (always reversible — no hard deletes) |
| `/memory restore <id>` | Admin | Restore any version by history ID |
| `/memory pin` / `/memory unpin <id>` | Admin | Toggle pinned (always injected; admins bypass the pin limit) |
| `/memory whoami` | everyone | See everything the bot remembers about you |
| `/memory forget-me` | everyone | Hard-delete all memories about you (irreversible) |
| `/memory forget <user>` | Admin | Hard-delete all memories about a user |
| `/memory export` | Admin | Full JSON export |

**How it works:**
- **LLM tools** — `memory_search` (hybrid lexical→semantic recall with importance/recency re-ranking), `memory_write` (`create`/`update`/`forget`, one mutation path, dedupe-guarded), `memory_about` (compact per-user profile)
- **Auto-injection** — every turn embeds the current question and selects the top memories (cosine + importance + participant boost); pinned memories always ride along (importance-ranked, capped). User memories only participate when their subject is in the conversation (speaker, @mention, reply target)
- **Pinning** — bot can pin too (`pinned=true` in `memory_write`, capped at `MEMORY_PIN_LIMIT`/guild, audited via `pinned_by`); admins pin/unpin without the cap
- **Dedupe** — creates are refused when semantically similar (cosine ≥ `MEMORY_DEDUPE_THRESHOLD`) to an existing memory in the same scope; the LLM is told to `update` instead
- **Single mutation path** — every write funnels through `MemoryStore`, which applies the change and records a before/after snapshot into `memory_history` in one SQLite transaction
- **Rate guard** — bot writes capped at `MEMORY_BOT_WRITE_LIMIT`/hour per guild; `MEMORY_MAX_PER_SUBJECT` active memories per scope
- **Transparency channel** — with `MEMORY_LOG_CHANNEL_ID` set, every mutation posts an embed with a **↩️ Reverter** button (admin-only, same guild)
- **Privacy** — `/memory whoami` + `/memory forget-me` are self-service; purges hard-delete memories and their history
- **One-time migration** — existing `lore.db` entries import automatically on first start (curated → pinned, pending skipped; `lore.db` is kept on disk untouched)
- Tool errors degrade gracefully — memory failures return error strings to the LLM instead of aborting `/ask`/`/chat`
- Set `MEMORY_ENABLED=false` to remove the tools and prompt guidance entirely

---

### Utility Commands

| Command | Description | Input |
|---|---|---|
| `/docs [query]` | Search indexed docs (RAG) or show docs link. Supports all sources + source label prefix. | `query` (optional) |
| `/plugin <name>` | Search Modrinth, Hangar, SpigotMC concurrently. Paginated view (one source per page). Autocomplete via Modrinth. | `name` (required) |
| `/status <ip>` | Check Minecraft server status via `mcsrvstat.us`. Shows MOTD, player count, version. | `ip` (hostname or IP, optional port) |
| `/changelog` | Last 5 commits to `MinersRefuge/docs` (SHA, message, author, date) via GitHub API. | — |
| `/flags <ram>` | Generate [Aikar's JVM flags](https://flags.sh) for a given RAM amount in MB. Auto-tunes G1GC thresholds at >= 12 GB. Validates 512–65536 MB range. | `ram` (MB, integer) |
| `/plov` | Embed: **PLOV** info needed to choose hosting — Plano/Players, Localização, Orçamento, Versão. | — |
| `/plano` | Embed: info needed to recommend a plan — Version, Players, Mods/Plugins, Gamemode. | — |
| `/health` | *(Admin)* Bot diagnostics: uptime, concurrent API pings (OpenRouter, Tavily, GitHub, mclo.gs, Modrinth) with latency, vector store stats, web search status, conversation cache usage, WebSocket latency. | — |
| `/sync` | *(Admin)* Re-sync slash commands with Discord. | — |
| `/help` | Auto-generated list of all registered slash commands + passive feature summary. | — |

---

## Passive Features

The bot monitors every non-bot message via `on_message` listeners (load order: `log_analyzer` → `history_rag` → `commands` → `spark` → `docs_rag`).

### 1. Paste Service Link Detection (`log_analyzer`)
- **Triggers on:** `https://mclo.gs/<id>` and `https://pastebin.com/<id>`
- **Behavior:** Reacts with 👀, fetches raw content (streaming, 5 MB cap), checks against known error patterns
  - **If matched:** Replies instantly with the pattern's troubleshooting response
  - **If no match + AI available:** Replies with "🤔 Não reconheci nenhum erro" + **"🔍 Analisar com IA"** button

### 2. File Attachment Handling (`log_analyzer`)
- **Triggers on:** `.log` / `.txt` attachments
- **Behavior:** Reads file (streaming), checks error patterns, uploads to `mclo.gs` (`POST /1/log`), replies with the mclo.gs link (or a fallback message). If large (> 5 MB), notes truncation. Offers AI analysis button when no pattern matches.

### 3. Known Error Pattern Matching (`responses/errors.py`)
25 pre-compiled regex patterns — first match wins (specific before generic):

| Category | Examples |
|---|---|
| Plugin | Ambiguous plugin name, missing dependency, `Error occurred while enabling`, `Could not pass event` |
| Startup / JAR | `Unable to access jarfile`, `Current Java is X but we require Y`, `Unsupported Java`, `JNI error` |
| EULA | `You need to agree to the EULA` |
| Memory | `OutOfMemoryError`, `Can't keep up! Is the server overloaded?` |
| Port / Network | `FAILED TO BIND TO PORT`, `Perhaps a server is already running on that port` |
| World / Data | `Failed to load chunk`, `Region file is truncated`, `Session lock is no longer valid` |
| Permissions | `<player> was denied the command` |
| Version | `Outdated server`, `Outdated client` |
| Crash | `This crash report has been saved to`, `---- Minecraft Crash Report ----` |
| Generic | `The received string length is longer than maximum allowed`, `Connection throttled` |

### 4. Spark URL Detection (`spark`)
Regex `https://spark.lucko.me/<code>` → replies "🔥 Relatório Spark detectado!" with analysis button.

### 5. Reply-Based Follow-Up Conversations (`docs_rag` + `commands`)
Reply to any `/ask`, `/chat`, or `/spark` bot response to continue the thread. The bot replays up to 16 history turns (for `/ask`/`/spark`) with the Spark report carried forward. Per-user cooldown enforced. Supports image attachments in follow-ups.

### 6. @Mention Chat (`commands`)
Mention the bot with a question (`@QuillBot como otimizar meu servidor?`) — equivalent to `/chat` with the same tool loop and conversation storage.

### 7. Server History Indexing (`history_rag`)
Background RAG over the entire Discord server's message history:
- **Backfill** on startup: reads all accessible text channels/threads (oldest-first, configurable limit)
- **Live ingestion:** Every new message is chunked with a 5-message sliding window for local context, embedded, and persisted per-guild (`data/history/<guild_id>.json + .npy`)
- **Dedup + deletion:** Tracks `msg_id → index`, handles `on_message_delete`
- **Search tool:** `search_history` / `search_docs` equivalent available to the LLM in all agentic loops
- **SQL tool:** `sql_history` — LLM-written analytical SELECTs over the history DB (exact counts, GROUP BY, mentions, replies, FTS5, REGEXP), sandboxed four ways: single-SELECT validation + guild scoping, read-only URI + `PRAGMA query_only`, `set_authorizer` (writes/ATTACH/PRAGMA denied, `chunks.embedding` never emitted), and a progress-handler timeout (`HISTORY_SQL_TIMEOUT_SECONDS`). Disable with `HISTORY_SQL_TOOL_ENABLED=false`; rows/output caps via `HISTORY_SQL_MAX_ROWS`.

---

## AI & RAG System

### Embedding & Search
- **Provider:** Switchable via `EMBEDDING_PROVIDER` — `openai` (remote, default) or `local` (sentence-transformers, e.g. `all-MiniLM-L6-v2`)
- **Models:** `EMBEDDING_MODEL` (default `qwen/qwen3-embedding-8b`), `RERANK_MODEL` (default `cohere/rerank-4-fast`)
- **Reranking:** Enabled when `RERANK_ENABLED=true` and `OPENAI_BASE_URL` is OpenRouter — reranks top `3×top_k` candidates via the OpenRouter `/rerank` endpoint, falls back to cosine similarity
- **Storage:** `data/vectors.json` (metadata) + `data/vectors.npy` (float32 binary embeddings) — migrated automatically from old JSON-embedded format
- **Chunking:** Markdown split by headings (`##`/`#`), then by `1500`-char paragraph windows; title extracted from frontmatter or first `#`

### Agentic Tool Loop (`cogs/utils.py:run_tool_loop`)
Shared loop used by `/ask`, `/chat`, and Spark analysis:

```
while tool_calls and rounds < MAX_TOOL_ROUNDS:
  → display status label on deferred interaction
  → execute each tool (deduplicate sources via dedup_key)
  → append tool results (truncated to 6000 chars)
→ final answer (max_tokens=`LLM_MAX_TOKENS`, default 8192)
→ if the answer comes back empty (e.g. a reasoning model spent the whole token budget thinking): one retry without tools + conciseness nudge, then fallback message
→ paginated embeds + source pages
```

- Parallel tool calls per round when the LLM requests multiple

### Trajectory Replay (`CONVERSATIONS_TRAJECTORY_*`)
Each turn captures the **exact rendered user message** plus the serialized internal loop (assistant tool calls, tool results, final answer). On follow-ups, the most recent `CONVERSATIONS_TRAJECTORY_TURNS` turns are replayed **verbatim** — the model literally sees its own prior tool calls and their results instead of a distilled Q/A pair. Older turns (and turns captured before this feature, or whose trajectory was dropped as invalid/oversized) fall back to the compact attributed rendering.

Replay windows advance in `CONVERSATIONS_TRAJECTORY_STEP`-turn steps (hysteresis), so the replayed prefix only changes at deliberate collapse boundaries — not sliding every turn — which is what keeps the provider prefix cache effective. Explicit `cache_control` breakpoints mark the system prompt and the last replayed history message for Anthropic/Gemini via OpenRouter; OpenAI-compatible endpoints ignore them and cache automatically. Captured trajectories are sanitized on persist (no `reasoning`/thinking payloads, tool-call/result pairing enforced).

### Rate Limiting
`COOLDOWN_RATE` / `COOLDOWN_PER` (default `1` per `30s`) per user on `/ask`, `/chat`, `/analyze`. Follow-up replies share a `TTLCache` cooldown. Exceeding returns an ephemeral `⏳ Aguarde Xs` message.

### Context Injection
Every AI call receives a `<contexto>` block with user (display name, account age, join date, roles), guild (name, member count, channels, roles), channel, and temporal (BRT + UTC) context.

When `CHANNEL_CONTEXT_MESSAGES` > 0 (default `10`), `/ask`, `/chat`, @mention and reply follow-ups also receive a `<mensagens_recentes_do_canal>` block — the latest N channel messages in chronological order, same format as the `get_channel_history` tool. The triggering message is excluded; set `0` to disable. Reply follow-ups use the captured gap block (`prior_context`, see `CONVERSATIONS_GAP_MESSAGES`) instead of this recent-channel window whenever a gap was captured.

### Prefix-Cache-Friendly Message Layout
The LLM message list is ordered so conversation follow-ups reuse a cached prefix: `[system (persona + static conversation summary)] → [replayed history turns] → [current message]`. All per-request blocks — `<contexto>` (clock), the semantically-selected memory block, and the recent channel window — ride on the **final user message** instead of the system prompt, so the system prompt + history stay byte-identical across turns and providers can prefix-cache them (typically 80–95% input-token savings on cached prefixes).

---

## Documentation Sources

Configured in `config.py:DOC_SOURCES`. Each entry:

| Key | Required | Description |
|---|---|---|
| `repo` | Yes | GitHub `owner/name` |
| `branch` | Yes | Branch to index |
| `base_url` | Yes | Docs website base URL |
| `label` | Yes | Human-readable label shown in results |
| `summary` | No | Path to mdBook `SUMMARY.md` for discovery |
| `path_prefix` | No | Only index files under this prefix |
| `url_strip_prefix` | No | Strip prefix when building website URLs |
| `max_files` | No | Cap per source (default 200) |

| Source | Label | Discovery | Base URL |
|---|---|---|---|
| `MinersRefuge/docs` | Miners' Refuge | `SUMMARY.md` | https://docs.minersrefuge.com.br |
| `PaperMC/docs` | PaperMC | Tree API (`src/content/docs/paper/admin/`) | https://docs.papermc.io |
| `PurpurMC/PurpurDocs` | PurpurMC | Tree API (`mkdocs/purpur/`) | https://purpurmc.org/docs/purpur/ |
| `lucko/spark-docs` | Spark | Tree API (`docs/`) | https://spark.lucko.me/docs/ |

Periodic reindex every `REINDEX_INTERVAL_HOURS` (default 6h) via composite MD5 of all latest commit SHAs.

---

## Project Structure

```
QuillBot/
├── main.py               # Entrypoint — validates config, loads cogs in order, syncs slash commands
├── config.py             # Centralized env-var config, DOC_SOURCES, feature flags
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── cogs/
│   ├── commands.py       # /chat, /docs, /help, /flags, /plov, /plano, /health, /sync + @mention & reply follow-ups
│   ├── docs_rag.py       # RAG pipeline — indexing, search, reranking, /ask, /reindex, agentic loop + Spark diagnosis
│   ├── history_rag.py    # Server-wide history RAG — backfill, live ingestion, per-guild vector stores
│   ├── log_analyzer.py   # Passive log detection, pattern matching, /analyze, file upload to mclo.gs
│   ├── memory.py         # Persistent memory — auto-injection, LLM write tools, admin commands, audit history, log channel, lore migration
│   ├── plugins.py        # /plugin, /status, /changelog — plugin search & server status
│   ├── spark.py          # /spark command + passive spark.lucko.me detection
│   ├── spark_parser.py   # Spark JSON parsing, summary/detail builders, call-tree rendering
│   ├── plugin_apis.py    # Shared Modrinth/Hangar/SpigotMC API helpers
│   ├── tavily_tools.py   # Tavily web_search / web_extract tool definitions & execution
│   └── utils.py          # Shared: truncate_safe, split_response, PaginatedEmbedView, context builders, run_tool_loop
├── responses/
│   └── errors.py         # 25 compiled Minecraft error regex patterns + pt-BR responses
└── data/
    ├── vectors.json/.npy # Docs RAG vector store
    ├── memory.db          # Persistent memory (memories + history)
    ├── lore.db            # Legacy lore encyclopedia (kept as migration source)
    └── history/          # Per-guild history stores
```

---

## Installation

**Requires Python 3.11+**

```bash
git clone https://github.com/Zeptiny/QuillBot.git
cd QuillBot
pip install -r requirements.txt
```

### Discord Bot Setup

1. Create a bot at https://discord.com/developers/applications
2. Under **Bot → Privileged Gateway Intents** enable **Message Content Intent**
3. Invite with scopes `bot` + `applications.commands` and permissions: Send Messages, Embed Links, Read Message History, Add Reactions, Use Application Commands

---

## Configuration

Copy and fill in your environment file:

```bash
cp .env.example .env   # if available, otherwise create .env manually
```

### Required

| Variable | Description |
|---|---|
| `BOT_TOKEN` | Discord bot token — bot will not start without it |
| `OPENROUTER_API_KEY` | API key for OpenRouter (or any OpenAI-compatible provider). Required for all AI commands. Also aliased as `OPENAI_API_KEY` / `LLM_API_KEY` |

### AI Models

| Variable | Default | Description |
|---|---|---|
| `OPENAI_BASE_URL` | `https://openrouter.ai/api/v1` | OpenAI-compatible endpoint. Aliases: `LLM_BASE_URL`, `OPENROUTER_BASE_URL` |
| `CHAT_MODEL` | `qwen/qwen3.6-plus` | Used for `/ask`, `/chat`, `/analyze` |
| `LLM_MAX_TOKENS` | `8192` | Max completion tokens for the tool loop. Reasoning models count thinking tokens against this |
| `SPARK_MODEL` | `google/gemini-2.5-pro` | Used only during active Spark sessions |
| `EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` | Doc/history embedding model |
| `RERANK_MODEL` | `cohere/rerank-4-fast` | Reranking model |
| `EMBEDDING_PROVIDER` | `openai` | `openai` (remote) or `local` (sentence-transformers) |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model when `EMBEDDING_PROVIDER=local` |
| `LOCAL_EMBEDDING_DEVICE` | `cpu` | Device for local embeddings |
| `RERANK_ENABLED` | `true` | Enable reranking (only when base URL is OpenRouter) |

### Feature Flags

| Variable | Default | Description |
|---|---|---|
| `WEB_SEARCH_ENABLED` | `true` | Enable Tavily web search |
| `TAVILY_API_KEY` | — | Required when web search is enabled |
| `CHAT_MENTION_ENABLED` | `true` | Enable @mention chat mode |
| `CHANNEL_CONTEXT_MESSAGES` | `10` | Latest channel messages auto-injected as context into `/ask`, `/chat`, @mention and reply follow-ups (`0` disables) |
| `HISTORY_ENABLED` | `true` | Enable server history RAG |
| `MEMORY_ENABLED` | `true` | Enable persistent memory (cog not loaded when false) |
| `LOG_LEVEL` | `INFO` | Python logging level |

### History RAG

| Variable | Default | Description |
|---|---|---|
| `HISTORY_VECTOR_STORE_DIR` | `data/history` | Per-guild history storage dir |
| `HISTORY_WINDOW_SIZE` | `5` | Sliding window of prior messages per chunk |
| `HISTORY_BACKFILL_LIMIT` | _(none)_ | Max messages to backfill per channel (unset = all) |
| `HISTORY_MAX_MSG_LENGTH` | `800` | Max chars per message in history chunks |
| `HISTORY_EXCLUDE_BOTS` | `true` | Exclude bot messages from history |

### Memory

| Variable | Default | Description |
|---|---|---|
| `MEMORY_ENABLED` | `true` | Enable the memory cog (cog is not loaded when false) |
| `MEMORY_DB_PATH` | `data/memory.db` | SQLite path for memories + history |
| `MEMORY_LOG_CHANNEL_ID` | _(none)_ | Channel ID for mutation log embeds (Revert button). Unset = console logging only |
| `MEMORY_BOT_WRITE_LIMIT` | `20` | Max bot (`memory_write`) writes per hour per guild |
| `MEMORY_INJECT_LIMIT` | `5` | Semantically selected memories injected per turn (pinned ride along separately) |
| `MEMORY_PIN_LIMIT` | `12` | Max pinned memories per guild (bot is capped; admins bypass) |
| `MEMORY_MAX_PER_SUBJECT` | `200` | Max active memories per scope (guild or user) |
| `MEMORY_SEMANTIC_MIN_SCORE` | `0.35` | Min cosine for semantic search/injection hits |
| `MEMORY_DEDUPE_THRESHOLD` | `0.85` | Cosine above which a create is refused as duplicate |
| `LORE_DB_PATH` | `data/lore.db` | Legacy lore DB — one-time migration source for memory.db |

### Docs / RAG Indexing

| Variable | Default | Description |
|---|---|---|
| `VECTOR_STORE_PATH` | `data/vectors.json` | Docs vector store path |
| `REINDEX_INTERVAL_HOURS` | `6` | Hours between automatic reindex checks |

### Conversations (multi-turn memory)

| Variable | Default | Description |
|---|---|---|
| `CONVERSATIONS_DB_PATH` | `data/conversations.db` | SQLite path for persisted conversations |
| `CONVERSATIONS_TTL_SECONDS` | `1800` | Conversation expiry after **last activity** |
| `CONVERSATIONS_MAX_STORED` | `200` | Max conversations kept per flow (chat/ask) |
| `CONVERSATIONS_MAX_TURNS` | `24` | Max turns stored per conversation |
| `CONVERSATIONS_HISTORY_TURNS` | `16` | Turns replayed to the LLM per request |
| `CONVERSATIONS_GAP_MESSAGES` | `20` | Max channel messages captured between two bot-directed turns as `prior_context` (`0` disables) |
| `CONVERSATIONS_TRAJECTORY_ENABLED` | `true` | Capture and replay the internal LLM trajectory (tool calls/results + exact user message) per turn |
| `CONVERSATIONS_TRAJECTORY_TURNS` | `6` | Most recent N turns replayed **verbatim** with their trajectory (older turns use the compact Q/A rendering) |
| `CONVERSATIONS_TRAJECTORY_STEP` | `3` | Window advances in steps of N turns — prefix-cache hysteresis so the replayed prefix only changes at deliberate boundaries |
| `CONVERSATIONS_TRAJECTORY_MAX_CHARS` | `120000` | Hard budget on the replayed trajectory payload; turns are collapsed to compact rendering when exceeded |

### Other

| Variable | Default | Description |
|---|---|---|
| `COOLDOWN_RATE` | `1` | Allowed uses per cooldown window |
| `COOLDOWN_PER` | `30` | Cooldown window in seconds (per user) |
| `MAX_CONTENT_SIZE` | `5242880` | Max bytes fetched from paste services (5 MB) |
| `MAX_LOG_CONTEXT` | `12000` | Max chars sent to LLM for log analysis |

---

## Running the Bot

```bash
python main.py
```

On startup the bot:

1. Validates `BOT_TOKEN` (exits if missing; warns if `TAVILY_API_KEY` missing while web search is enabled)
2. Loads cogs in order: `log_analyzer` → `history_rag` → `commands` → `plugins` → `spark` → `docs_rag` → `memory` (first run imports legacy `lore.db` automatically)
3. Loads cached vectors from disk or indexes all doc sources
4. Starts periodic reindex loop + background history backfill
5. Syncs slash commands with Discord

> **Cog load order matters** — `log_analyzer`'s `on_message` (pattern matching) runs before `docs_rag`'s (follow-up replies). Do not reorder without reviewing listener interactions.

---

## Docker

```bash
docker compose up --build -d
```

- Uses `python:3.11-slim`, persists `data/` via volume mount, reads config from `.env`.

---

## License

See repository license. Documentation sources retain their original licenses.
