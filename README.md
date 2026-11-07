# QuillBot — Conversational Discord Bot for the Miners' Refuge Community

AI-powered Discord bot that takes part in the **Miners' Refuge** community's conversations: it answers when mentioned, replied to or called by name, remembers people and past conversations, knows when to stay quiet, and brings Minecraft server administration tools into the chat — documentation search, log analysis, Spark diagnosis and plugin search.

Its personality is the server's: a `<personalidade>` block per server — a base the admins write and notes the bot keeps about itself — that rides in every system prompt (see [`/persona`](#persona--personality)).


Built with [discord.py](https://discordpy.readthedocs.io/) + RAG (Retrieval-Augmented Generation) over multiple documentation sources.

---

## Table of Contents

- [Features](#features)
- [Conversations](#conversations)
- [Slash Commands](#slash-commands)
- [Passive Features](#passive-features)
- [Architecture](#architecture)
- [AI & RAG System](#ai--rag-system)
- [Documentation Sources](#documentation-sources)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Radar de Hardware MCP](#radar-de-hardware-via-mcp)
- [Configuration](#configuration)
- [Running the Bot](#running-the-bot)
- [Docker](#docker)
- [Tests](#tests)

---

## Features

| Category | Capability |
|---|---|
| **Conversation** | Answers @mentions, replies to its messages, messages in its threads and being called by name; plain messages like any member (embeds only for summaries, reports and long answers); quick consecutive messages answered as one; may just react or stay quiet |
| **Q&A (RAG)** | Answers questions grounded in indexed docs (Miners' Refuge, PaperMC, PurpurMC, Spark) — in any conversation, not only `/ask` |
| **General Chat** | Web search, server/history context awareness, one voice across every entry point |
| **Episodes** | When a conversation ends, the bot keeps a short summary of it per participant, so it can pick the subject up later ("e aí, deu certo o backup?") |
| **Server Wrapped** | `/wrapped` or asking the bot: who talked most, busiest channels and hours, who replied to whom, favorite emojis, most-reacted messages; optional weekly/monthly automatic recap |
| **Personality** | Per server, in the system prompt: a *base* (who the bot is) that admins edit with `/persona` or ask the bot to change, and *notes* (running jokes, nicknames, catchphrases) the bot rewrites itself, a few times a day at most. Every change is versioned, logged with a Revert button and revertible |
| **Settings** | `/config`: per channel (when to answer, name calls, threads, reply format, log detection, channel instructions) and per server (extra names, automatic Wrapped) |
| **Edits & deletes** | Deleting a message the bot answered deletes the answer; editing it within a minute re-answers it |
| **Chat Threads** | Long or off-topic conversations move into a thread on the asker's message, so the channel keeps one message; inside it the bot answers without a mention and stays quiet when people talk among themselves |
| **Log Analysis** | AI analysis of mclo.gs/pastebin links, file uploads, and screenshots |
| **Error Detection** | Instant passive replies for 20+ known Minecraft error patterns |
| **Spark Profiler** | Full report parsing, bottleneck diagnosis, platform-aware recommendations |
| **Plugin Search** | Concurrent search across Modrinth, Hangar, SpigotMC |
| **Server Tools** | JVM flags, server status checks, docs changelog |
| **Channel Summaries** | `/resumo` and natural-language "o que perdi?" — map-reduce summaries of a channel or thread over any period, with jump links and permission checks |
| **Monitors** | Standing triggers (channels, authors, mentions of a user, a content regex) that call the bot on matching messages; it answers like a mention, knows a monitor sent it, and may stay silent. Managed with `/monitor` or by asking the bot |
| **Persistent Memory** | The bot remembers: atomic facts about the server and its members, auto-injected into every conversation, with inline LLM writes, pinning, dedupe, full audit history and admin control. Each memory records who said it — a fact about someone told by someone else is shown as a report ("segundo Fulano") |

All AI responses are in **Brazilian Portuguese**.

---

## Conversations

Everything below goes through one chat engine (`core/chat_engine.py`), so `/chat`, `/ask`, mentions, replies, threads, name calls, monitors and scheduled tasks share the same behavior.

**When the bot answers** — a reply to one of its messages, an @mention, any message in a thread it opened, or its name in a message ("quill, o que acha?"). Names: its server nickname, its username, those without a trailing "bot", `CHAT_NAME_ALIASES` and the server's `aliases` setting; not inside links, code or other mentions. Name calls and thread messages reach the model marked as not addressed to it, and it may stay silent.

**Bursts instead of a cooldown** — people type in bursts ("ei" / "sabe aquele erro" / "de ontem?"). A message for the bot opens a burst for that person in that channel; everything they send there until they've been quiet `CHAT_BURST_SECONDS` joins it (mention or not) and is answered as one turn, replying to the last message. Messages sent while the bot answers a mention or reply become the next turn of the same conversation, once (what comes while that follow-up is answered isn't taken). While it answers a name call they don't: the person is talking to the channel and names or mentions the bot again to go on. A burst that so far only points at something ("ai oh @Quill", "olha isso": short, no image, link or reply) waits `CHAT_BURST_SHORT_SECONDS` instead, so the screenshot uploaded right after it is part of the turn.

**Budget** — cost is controlled per person (`CHAT_USER_BUDGET` turns per `CHAT_USER_BUDGET_WINDOW`) and per server (`CHAT_GUILD_BUDGET`, monitors included). Over budget, the bot reacts ⏳ instead of answering (a slash command gets an ephemeral notice).

**Doing what was asked** — the personality changes how the bot talks, never what it does: asked to search, it searches (a message that asks for a search — "pesquise", "procura", "busca" — also gets a hint to call `web_search`/`search_docs` first); when someone only points at something ("esse sinal", "isso"), it reads more history with `get_channel_history` instead of guessing; and if they point at an image or link that didn't come with the message, it says so instead of describing a channel image.

**Staying quiet** — on any turn the model may react (`add_reaction`) and answer `[NO_REPLY]`: a "valeu!" gets a ❤️, not a paragraph. Someone who asked directly still hears about a failed answer; unaddressed turns never do.

**How answers look** — plain messages, like any member; `@Name` written by the model becomes a real mention and only those people are pinged (never roles, `@everyone` or bots). Sources go in a small `-# Fontes:` line. Spark reports, channel summaries, Wrapped and answers longer than `CHAT_PLAIN_MAX_CHARS` go in embeds. A channel can force either with `/config canal reply_style`.

**Continuing** — replying to any bot message of a conversation continues it; a mention continues the bot's latest conversation in the last `CHAT_MENTION_CONTINUE_WINDOW` messages. Conversations expire after `CONVERSATIONS_TTL_SECONDS` without activity; then an *episode* — one or two sentences on what was talked about and what was left pending — is saved as a memory for each participant (`EPISODES_*`).

**Edits and deletes** — deleting a message the bot answered deletes the answer and drops that turn (part of a burst keeps the answer); editing it within `CHAT_EDIT_WINDOW_SECONDS` of the answer re-answers it, editing the old answer in place.

**Threads** — see [`/chat`](#chat-question-image) below; per channel, `/config canal threads` is `auto`, `always` or `never`.

---

## Slash Commands

### AI-Powered Commands

#### `/chat <question> [image]`
Talk to the bot from a slash command: the same conversation as a mention (profile `chat`, `core/profiles.py`).

- **Tools:** `web_search`, `web_extract` (via Tavily), `search_docs`, `search_plugins`, the context/history tools, memory, the scheduler and monitor tools (see `/monitor`), `summarize_channel` (see `/resumo`), `server_wrapped` (see `/wrapped`), `add_reaction` and `open_thread` (see threads below)
- **Reactions:** `add_reaction` lets the model react with a Unicode emoji or a `:nome:` custom emoji of the server, only to messages in the current channel (the triggering message by default) and at most 3 per answer. A reaction can be the whole answer (`[NO_REPLY]`). `REACTION_TOOL_ENABLED=false` removes the tool
- **Web search:** Supports `search_depth` (basic/advanced), `time_range`, domain filtering
- **Mentions, replies, name calls:** see [Conversations](#conversations). A mention continues the conversation when one of the bot's answers is among the last `CHAT_MENTION_CONTINUE_WINDOW` channel messages, just like replying to it (unless that conversation moved to a thread); a mention that replies to someone else's message joins it too, with that message as context (`CHAT_MENTION_CONTINUE_ENABLED=false` always starts a new one). `CHAT_MENTION_ENABLED=false` leaves only replies to the bot
- **Threads (`core/chat_threads.py`):** a quick question is answered in the channel, but a conversation moves into a thread when
  - one person has gone back and forth with the bot for `CHAT_THREAD_AFTER_TURNS` turns in a row (default 3, the second follow-up), the last one a reply or mention. Being named in passing, or several people chatting with the bot, never moves the conversation: that is the channel's own conversation
  - the channel's `threads` setting is `always` (or the channel is in `CHAT_THREAD_CHANNELS`): first answer already in a thread, or
  - the model calls `open_thread` for a technical subject that will turn into a long back-and-forth (debugging an error, a step-by-step setup) or an off-topic subject that will go on. Chat, jokes, quick questions and several people talking with the bot stay in the channel (`CHAT_THREAD_TOOL_ENABLED=false` removes the tool)
- `threads: never` keeps every answer in the channel (and `open_thread` tells the model so)
- The thread is opened on the asker's message (for `/chat`, on the command's response) and named by the model: the title it gave `open_thread`, or, when the bot moves the conversation on its own, a title `CHAT_THREAD_TITLE_MODEL` writes from that back-and-forth (the first question if that call fails or takes over 8 s). The channel keeps only that message; the answer goes into the thread and pings the asker, which also adds them to it. Threads archive after `CHAT_THREAD_ARCHIVE_MINUTES` idle
- A conversation that moved keeps answering in its thread, also when someone replies to one of its messages from the channel (an archived thread is reopened). A mention or name call in the channel doesn't follow it there: it starts a new conversation answered in the channel
- Inside a thread the bot opened, every message is for the bot — no mention needed, and an expired conversation just starts a new one there. The model is told the message didn't mention it and may answer `[NO_REPLY]` to stay silent when people are talking among themselves
- No thread is opened where Discord has none (threads, forum posts, voice-channel chats, DMs) or without the Create Public Threads and Send Messages in Threads permissions; the answer stays in place. A missing permission is logged once per channel (`[threads] cannot open threads in #… the bot lacks …`), and `open_thread` tells the model which one, so it can say so instead of guessing. `CHAT_THREADS_ENABLED=false` turns all of this off

#### `/ask <question> [image]`
Ask a Minecraft server administration question with the docs-first profile (`ask`): the model calls `search_docs` right away and grounds every answer in the indexed documentation. Same bot and voice as `/chat` — mentions and `/chat` can search the docs too; `/ask` just insists on it.

- **Model:** `CHAT_MODEL` (default: `qwen/qwen3.6-plus`)
- **Tools available:** `search_docs`, `search_plugins`, `memory_search`, `memory_write`, `memory_about`, `get_channel_history`, `get_guild_info`, `search_history`, `get_message_context`, `get_user_stats`, `count_mentions`, `find_user`, `sql_history`, `add_reaction`
- **Output:** plain message with a `-# Fontes:` line; long answers in an embed titled with the question, with source pages
- **Follow-up:** Reply to the bot's response to continue the conversation with full history (it stays on the `ask` profile)

#### `/resumo [canal] [periodo] [foco]`
Summarize what happened in a channel or thread while you were away.

- **`canal`** — text/voice channel or thread (default: the current one). Archived threads work too
- **`periodo`** — default *since your last message* in that channel (your own messages from the last 10 minutes don't count, so "voltei!" + `/resumo` still covers the gap); also `30m`/`6h`/`2d`, `hoje`, `ontem` or a date like `24/09/2026 08:00` (Brasília time). Autocompletes common presets
- **`foco`** — optional topic to focus on ("backup", "o lag do servidor")
- **Output:** ephemeral embed with **Tópicos**, **Decisões e soluções** and **Pendências**, each item linked (↗) to the message it came from, plus a private **🔔 Mencionaram você** list (messages that mentioned or replied to you). **📢 Publicar no canal** posts it publicly (without your mentions); replying to the published summary continues it as a `/chat` conversation
- **How it works (`cogs/summary.py`):**
  - Messages are read straight from Discord, newest first, until the period start, `SUMMARY_MAX_MESSAGES` or `SUMMARY_MAX_DAYS`. Bot answers are included, since in support channels they're often the fix
  - Ranges that fit `SUMMARY_SEGMENT_CHARS` take one LLM call. Longer ones are split at conversation gaps (≥20 min of silence), each part is summarized in parallel, and the notes are merged
  - The model cites messages as `[msg_id=…]`; code turns them into jump links and drops any ID that wasn't in the fetched range
  - Mentions are rewritten as plain names, so summaries never ping anyone
- **Permissions:** only channels that *you* can view and read the history of (private threads also require membership or Manage Threads), in the current server
- **Natural language:** the same engine is the `summarize_channel` tool in `/chat`, @mention, reply follow-ups and scheduled tasks. For example `@QuillBot o que perdi?`, `resume o que rolou no #suporte desde ontem` or a daily scheduled "resumo do #geral". The chat model turns "hoje"/"desde ontem" into timestamps using its clock context and receives a finished summary, not raw messages, so the 6000-char tool-result cap and conversation replay stay small
- Set `SUMMARY_ENABLED=false` to remove the command and the tool

#### `/monitor <subcommand>` — Monitors
A monitor calls the bot on every message that matches **all** of its filters, as if the bot had been mentioned: same tool loop, the answer is a reply, and replying to it continues the conversation. The model is told which monitor fired and that nobody called it, and it may **not answer**: replying exactly `[NO_REPLY]` posts nothing (it can still react, write memories, schedule, etc. first).

- **Filters** (at least one, combined with AND): `channels` (threads under them count), `authors` (the message was sent by…), `mentions` (the message mentions…), `regex` (case-insensitive search over the content; nested quantifiers like `(a+)+` are refused)
- **Instruction:** what the bot should do on a match, including when to stay silent
- **Never fires on:** bot messages, and messages the bot already answers — the message router offers every message to the chat engine first (mentions, replies to its conversations, its threads, name calls, bursts), and monitors only see what it didn't take. Several monitors matching one message share one run
- **Budget:** a monitor run spends the server budget (`CHAT_GUILD_BUDGET`)
- **Cooldown:** per monitor, `MONITORS_DEFAULT_COOLDOWN` seconds by default (min `MONITORS_MIN_COOLDOWN`), so a busy channel doesn't cost one LLM call per message
- **Commands:** `create`, `list`, `show`, `edit` (a slash option replaces that filter with one value; `clear` removes a filter), `pause`, `resume`, `delete`
- **LLM tools:** `monitor_create`, `monitor_list`, `monitor_update` (lists, clearing, pause/resume) and `monitor_delete` — "sempre que alguém mencionar a staff em #suporte, veja se precisa de ajuda" works from `/chat` or a mention
- Creating, editing and removing needs **Manage Server** (checked against whoever asks the bot); anyone can list
- Set `MONITORS_ENABLED=false` to remove the command, the tools and the listener

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
- **Passive detection:** Any `spark.lucko.me` link in chat triggers a "🔥 Analisar com IA" button (off where the channel's `log_detection` setting is off)
- **Output:** an embed (reports always are); replying continues the analysis with the report loaded, until it expires or the bot restarts

#### `/reindex [source]` *(Admin only)*
Re-index documentation vectors.

- Without `source`: full reindex of all `DOC_SOURCES`
- With `source`: reindex a single source by label (e.g. `PaperMC`)
- Shows `⏳ Indexando...` guard — blocks `/ask`/`/docs` queries during indexing
- Per-source commit SHA tracking for granular updates

#### `/memory <subcommand>` — Persistent Bot Memory
The bot maintains long-term memories: atomic one-sentence facts about the server (guild-wide) or about individual users (scoped, privacy-guarded). Relevant memories are **automatically injected** into every turn — the LLM doesn't need to search first. The bot's personality has its own place now ([`/persona`](#persona--personality)); `/persona importar` moves personality memories there.

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
- **Who said it** — each memory records `told_by`, the person whose message led to it. A fact about someone told by someone else reaches the model as `… (segundo Fulano)` — a report, not something that person said — and `/memory show` lists it as *Contada por*. When the person restates it, `told_by` becomes them
- **Episodes** — kind `episode` (*Conversa*): when a conversation expires, `EPISODES_MODEL` summarizes it in one or two sentences ("Conversa de 03/11/2026 com Ana: …; Ana ia testar o backup") and one memory is saved per participant (importance 2, newest `EPISODES_PER_USER` kept). Conversations with nothing worth remembering are skipped. Only the summarizer writes episodes; `memory_search` can filter them
- **Rate guard** — bot writes capped at `MEMORY_BOT_WRITE_LIMIT`/hour per guild; `MEMORY_MAX_PER_SUBJECT` active memories per scope
- **Transparency channel** — with `MEMORY_LOG_CHANNEL_ID` set, every mutation posts an embed with a **↩️ Reverter** button (admin-only, same guild)
- **Privacy** — `/memory whoami` + `/memory forget-me` are self-service; purges hard-delete memories and their history
- **One-time migration** — existing `lore.db` entries import automatically on first start (curated → pinned, pending skipped; `lore.db` is kept on disk untouched)
- Tool errors degrade gracefully — memory failures return error strings to the LLM instead of aborting the turn
- Set `MEMORY_ENABLED=false` to remove the tools and prompt guidance entirely

#### `/persona` — Personality
The bot's personality on each server, as a `<personalidade>` block in the system prompt (`core/persona.py`, `cogs/persona.py`). Unlike memories, it carries instruction weight, fits a whole document, can't be archived by a stray `memory_write`, and — since it rarely changes — stays inside the provider's prompt cache.

| Section | What | Who changes it | Size |
|---|---|---|---|
| **Base** | Who the bot is: voice, humor, values, hard rules | Admins (`/persona editar`), or the bot when someone with **Manage Server** asks it to | `PERSONA_BASE_MAX_CHARS` (3000) |
| **Notas** | What it picked up on the server: running jokes, nicknames, catchphrases | The bot itself (`persona_update`), at most `PERSONA_BOT_EDITS_PER_DAY` times a day; admins too | `PERSONA_NOTES_MAX_CHARS` (800) |

| Subcommand | Who | Description |
|---|---|---|
| `/persona ver` | everyone | Both sections, their versions and who changed them last |
| `/persona editar [secao]` | Manage Server | Edit a section in a text box pre-filled with the current text |
| `/persona historico [secao]` | Manage Server | Last changes, with ids |
| `/persona reverter <id>` | Manage Server | Put a section back to what it was before a change |
| `/persona travar <sim/não>` | Manage Server | Stop (or allow again) the bot changing its own notes |
| `/persona importar <ids>` | Manage Server | Append server memories (e.g. a pinned personality memory) to the base and unpin them |

- **The tool** — `persona_update(section, content, reason)` replaces a whole section (the model rewrites it short instead of appending forever). A member asking to change the base is refused, and the model is told to suggest talking to an admin; "become someone else" requests are not a reason to touch the notes
- **Audit** — every change is a new version with before/after (`persona_history`); with `PERSONA_LOG_CHANNEL_ID` (defaults to `MEMORY_LOG_CHANNEL_ID`) each one is posted with a **↩️ Reverter** button (admins of that server; the button lasts a day, `/persona reverter` always works)
- Server memories that say how the bot should act still count, but the block is the place for it
- Set `PERSONA_ENABLED=false` to remove the command, the tool and the block

#### `/wrapped [periodo]` — Server Wrapped
A recap of the server over a period (`semana` = 7 days, default; `mes`; `ano`; or `14d`): messages and active members, who talked most, busiest channels, peak hour and weekday, who replied to whom most, favorite emojis (from messages and reactions) and the most-reacted messages with links, introduced in two or three lines in the bot's own voice. People are named, never pinged.

- **Ask the bot:** the `server_wrapped` tool — "faz um wrapped do mês", "quem mais falou essa semana?"
- **Automatic:** `/config servidor wrapped weekly|monthly` + `/config servidor wrapped_channel #canal` posts the last full week every Monday (or the last month on the 1st) at `WRAPPED_HOUR` (Brasília)
- **Data:** messages from the history index; reactions from a `reactions` table recorded live since this version (no backfill), also queryable with `sql_history`
- Set `WRAPPED_ENABLED=false` to remove the command, the tool and the automatic posts

#### `/config` — Server and channel settings *(Manage Server)*
| Subcommand | Description |
|---|---|
| `/config ver [canal]` | Effective settings of a channel (and where each comes from: channel, parent channel, server or default) plus the server settings |
| `/config canal <opção> <valor> [canal]` | Set a channel's value (threads follow their parent channel) |
| `/config servidor <opção> <valor>` | Set a server setting, or the default for every channel |
| `/config limpar <opção> [canal]` | Back to the default |

| Option | Scope | Values |
|---|---|---|
| `respond` | channel | `on` (mentions, replies, name calls, its threads), `mentions` (only mentions and replies), `off` (doesn't converse; slash commands still work) |
| `name_trigger` | channel | answer when called by name without @ (default `CHAT_NAME_TRIGGER_ENABLED`) |
| `threads` | channel | `auto`, `always`, `never` |
| `reply_style` | channel | `auto`, `plain`, `embed` (default `CHAT_REPLY_STYLE`) |
| `log_detection` | channel | react to log links, `.log`/`.txt` files and Spark links |
| `instructions` | channel | text the model follows in that channel ("seja direto e técnico", "aqui é papo livre") |
| `aliases` | server | extra names the bot answers to, comma-separated |
| `wrapped`, `wrapped_channel` | server | automatic Wrapped: `off`/`weekly`/`monthly` and where |

Stored in `SETTINGS_DB_PATH`.

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

Every message goes through one router (`core/router.py`): passive *observers* see each message concurrently (history indexing, log and Spark detection), and *handlers* get human messages in priority order until one takes it — the chat engine first, monitors after. Load order doesn't matter.

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

Detection 1–2 and 4 are off in channels whose `log_detection` setting is off.

### 4. Spark URL Detection (`spark`)
Regex `https://spark.lucko.me/<code>` → replies "🔥 Relatório Spark detectado!" with analysis button.

### 5. Conversations (`core/chat_engine.py`)
Replies to the bot, @mentions, name calls and messages in its threads — see [Conversations](#conversations). Replies continue any conversation (`/chat`, `/ask`, Spark, monitors, scheduled posts, published summaries) on its own profile, replaying up to 16 history turns, with the Spark report carried forward. Supports image attachments.

### 6. Edits and Deletes (`cogs/chat.py`)
Raw message delete/edit events: a deleted question takes its answer with it; an edit shortly after the answer re-answers it.

### 7. Monitors (`monitors`)
Messages matching a monitor's filters run the bot on them like a mention, and it may choose not to answer — see [`/monitor`](#monitor-subcommand--monitors).

### 8. Server History Indexing (`history_rag`)
Background RAG over the entire Discord server's message history:
- **Backfill** on startup: reads all accessible text channels/threads (oldest-first, configurable limit)
- **Live ingestion:** Every new message is chunked with a 5-message sliding window for local context, embedded, and persisted per-guild (`data/history/<guild_id>.json + .npy`)
- **Dedup + deletion:** Tracks `msg_id → index`, handles `on_message_delete`
- **Reactions:** every reaction added/removed is recorded in a `reactions` table (for Wrapped and `sql_history`)
- **Search tool:** `search_history` / `search_docs` equivalent available to the LLM in all agentic loops
- **SQL tool:** `sql_history` — LLM-written analytical SELECTs over the history DB (exact counts, GROUP BY, mentions, replies, FTS5, REGEXP), sandboxed four ways: single-SELECT validation + guild scoping, read-only URI + `PRAGMA query_only`, `set_authorizer` (writes/ATTACH/PRAGMA denied, `chunks.embedding` never emitted), and a progress-handler timeout (`HISTORY_SQL_TIMEOUT_SECONDS`). Disable with `HISTORY_SQL_TOOL_ENABLED=false`; rows/output caps via `HISTORY_SQL_MAX_ROWS`.

---

## Architecture

Every feature talks to the others through shared services on `bot.services` (`core/services.py`), never through another cog's private methods:

| Service | What it is |
|---|---|
| `llm` (`core/llm.py`) | The one OpenAI-compatible client: streamed completions retried on 429/timeouts/connection errors/5xx (`LLM_RETRY_*`), and embeddings (remote or the one shared local model) |
| `tools` (`core/tools.py`) | Tool registry — each feature registers its tools (schema, coroutine, status line) when it loads |
| `router` (`core/router.py`) | The single `on_message`: observers + handlers in priority order |
| `settings` (`core/settings.py`) | Server/channel settings (`/config`) |
| `conversations` (`core/conversations.py`) | The conversation store (one for every kind of conversation) |
| `chat` (`core/chat_engine.py`) | The chat engine: answers messages, monitor hits, scheduled tasks, slash commands and Spark analyses; anchors conversations on posts |
| `memory`, `history`, `docs`, `persona` | Set by those cogs when they load |

A turn is one `TurnContext` (`core/turn.py`: who, where, the stored conversation, images, per-turn tool state) run by `core/runner.py:run_turn` with a **profile** (`core/profiles.py`): `chat` (mentions, replies, threads, name calls, `/chat`, monitors, scheduled tasks), `ask` (docs-first `/ask`) or `spark`. The profile picks the prompt (`core/prompts.py` — one role and one set of rules for all of them, plus the server's `<personalidade>` block), the tools by name in a fixed order, the model and how answers look; `core/delivery.py` then decides plain message vs embed vs thread.

## AI & RAG System

### Embedding & Search
- **Provider:** Switchable via `EMBEDDING_PROVIDER` — `openai` (remote, default) or `local` (sentence-transformers, e.g. `all-MiniLM-L6-v2`)
- **Models:** `EMBEDDING_MODEL` (default `qwen/qwen3-embedding-8b`), `RERANK_MODEL` (default `cohere/rerank-4-fast`)
- **Reranking:** Enabled when `RERANK_ENABLED=true` and `OPENAI_BASE_URL` is OpenRouter — reranks top `3×top_k` candidates via the OpenRouter `/rerank` endpoint, falls back to cosine similarity
- **Storage:** `data/vectors.json` (metadata) + `data/vectors.npy` (float32 binary embeddings) — migrated automatically from old JSON-embedded format
- **Chunking:** Markdown split by headings (`##`/`#`), then by `1500`-char paragraph windows; title extracted from frontmatter or first `#`

### Agentic Tool Loop (`core/tool_loop.py:run_tool_loop`)
Shared loop used by every turn (`core/runner.py`):

```
while tool_calls and rounds < profile.max_rounds:
  → show the tool's status line on a deferred slash command
  → execute each tool through the registry (deduplicate sources by URL)
  → append tool results (truncated to 6000 chars)
→ final answer (max_tokens=`LLM_MAX_TOKENS`, default 8192)
→ if the answer comes back empty (e.g. a reasoning model spent the whole token budget thinking): one retry without tools + conciseness nudge, then fallback message
→ core/delivery.py: plain message(s), or embeds + source pages
```

- Parallel tool calls per round when the LLM requests multiple

### Trajectory Replay (`CONVERSATIONS_TRAJECTORY_*`)
Each turn captures the **exact rendered user message** plus the serialized internal loop (assistant tool calls, tool results, final answer). On follow-ups, the most recent `CONVERSATIONS_TRAJECTORY_TURNS` turns are replayed **verbatim** — the model literally sees its own prior tool calls and their results instead of a distilled Q/A pair. Older turns (and turns captured before this feature, or whose trajectory was dropped as invalid/oversized) fall back to the compact attributed rendering.

Replay windows advance in `CONVERSATIONS_TRAJECTORY_STEP`-turn steps (hysteresis), so the replayed prefix only changes at deliberate collapse boundaries — not sliding every turn — which is what keeps the provider prefix cache effective. Explicit `cache_control` breakpoints mark the system prompt and the last replayed history message for Anthropic/Gemini via OpenRouter; OpenAI-compatible endpoints ignore them and cache automatically. Captured trajectories are sanitized on persist (no `reasoning`/thinking payloads, tool-call/result pairing enforced).

### Rate Limiting
Conversations (`/chat`, `/ask`, mentions, replies, name calls, monitors) use the sliding-window budgets `CHAT_USER_BUDGET` / `CHAT_GUILD_BUDGET` — over budget the bot reacts ⏳ (slash commands get an ephemeral notice); quick consecutive messages are merged into one turn, never dropped. `COOLDOWN_RATE` / `COOLDOWN_PER` (default `1` per `30s`) still apply per user to `/analyze`, `/spark`, `/resumo` and `/wrapped`.

### Context Injection
Every AI call receives a `<contexto>` block with user (display name, account age, join date, roles), guild (name, member count, channels, roles), channel, and temporal (BRT + UTC) context, plus a `<canal>` block with the channel's `instructions` setting when it has one.

When `CHANNEL_CONTEXT_MESSAGES` > 0 (default `30`), every conversational turn also receives a `<mensagens_recentes_do_canal>` block — the latest N channel messages in chronological order, trimmed from the oldest side to `CHANNEL_CONTEXT_MAX_CHARS` (in a fast chat ten messages is a minute of talk, and what "esse sinal" points at can be further back), same format as the `get_channel_history` tool. Image attachments in that context are persisted and sent as vision parts, up to the four-image per-message budget (current-message images take priority, then the newest channel images); filenames remain visible in the text context. Only images posted within `CHANNEL_CONTEXT_IMAGE_MAX_AGE_MINUTES` (default `60`) are sent, and `CHANNEL_CONTEXT_IMAGES_ENABLED=false` turns channel images off entirely (use it with text-only `CHAT_MODEL`s). Attachments are downloaded once and reused by attachment id. Stickers (`[figurinha:Nome]`) and GIF/image link embeds (Tenor, Giphy, direct image links; `[gif: …]`) count as images too; animated GIFs and stickers are sent as one numbered sheet of four evenly spaced frames, and Tenor/Giphy links use a small animated rendition with the still preview as fallback. Lottie stickers are shown by name only. Custom emojis appear as `:nome:` in the text, and each message's custom emojis are drawn on one labelled sheet that only uses image slots left over after real media. The same media from the triggering message (and the message it replies to) is sent as the user's own images. Reactions are listed on each line before `msg_id` as `[reações: 👍 3 (Ana, Bruno +1)]`, with up to `REACTION_USERS_LIMIT` reactor names per reaction. The triggering message is excluded; set `0` to disable. Reply follow-ups use the captured gap block (`prior_context`, see `CONVERSATIONS_GAP_MESSAGES`) instead of this recent-channel window whenever a gap was captured, including images from those gap messages.

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
├── main.py               # Entrypoint — validates config, builds bot.services, routes on_message, loads cogs, syncs slash commands
├── config.py             # Centralized env-var config, DOC_SOURCES, feature flags
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── core/                 # Shared machinery (no Discord cogs)
│   ├── services.py       # bot.services: llm, tools, router, settings, conversations, chat (+ memory/history/docs)
│   ├── llm.py            # The one LLM client: streamed completions with retry, embeddings
│   ├── tool_loop.py      # Agentic tool loop, text tool-call recovery, trajectory capture
│   ├── tools.py          # Tool + ToolRegistry
│   ├── turn.py           # TurnContext / TurnResult, [NO_REPLY] and the unaddressed-turn headers
│   ├── profiles.py       # chat / ask / spark profiles (prompt, tools, model, look)
│   ├── prompts.py        # System prompts (one voice) and the Spark diagnostic protocol
│   ├── runner.py         # run_turn: builds the request (cache-friendly layout) and runs the loop
│   ├── chat_engine.py    # When the bot talks, bursts, budget, conversations, delivery, edits/deletes
│   ├── delivery.py       # Plain messages vs embeds vs threads; pings
│   ├── router.py         # The single on_message: observers + handlers
│   ├── burst.py          # Merging a person's quick consecutive messages
│   ├── budget.py         # Per-person / per-server sliding-window budgets
│   ├── settings.py       # Server/channel settings store
│   ├── episodes.py       # Summaries of ended conversations → episode memories
│   ├── persona.py        # Per-server personality sections, versions, the <personalidade> block
│   ├── conversations.py  # Conversation store + history/current-message builders
│   ├── context.py        # <contexto> block, channel history/window/gap fetchers
│   ├── context_tools.py  # Core tools: channel history, server info, message context, reactions, threads, web
│   ├── discord_text.py   # Mentions/pings, identity block, canonical message lines, splitting
│   ├── chat_threads.py   # When conversations move into threads, the open_thread tool, bot-thread detection
│   ├── message_media.py  # Stickers, GIFs, custom emojis, reactions, the add_reaction tool
│   ├── image_store.py    # Persisted, re-encoded images for vision
│   ├── tavily_tools.py   # Tavily web_search / web_extract
│   ├── local_inference.py# Single-thread executor for local models
│   └── ui.py             # PaginatedEmbedView, source pages
├── cogs/
│   ├── chat.py           # /chat, /ask, the chat router handler, edit/delete events, the episode worker
│   ├── commands.py       # /docs, /help, /flags, /plov, /plano, /health, /sync
│   ├── server_config.py  # /config — server and channel settings
│   ├── persona.py        # /persona, the persona_update tool, change log with Revert
│   ├── wrapped.py        # /wrapped, server_wrapped tool, automatic weekly/monthly recaps
│   ├── docs_rag.py       # Docs RAG — indexing, search, reranking, /reindex, search_docs/search_plugins tools
│   ├── history_rag.py    # Server-wide history RAG — backfill, live ingestion, reactions, history tools, Wrapped stats
│   ├── log_analyzer.py   # Passive log detection, pattern matching, /analyze, file upload to mclo.gs
│   ├── monitors.py       # /monitor + monitor_* tools — standing message triggers that run the chat engine and may stay silent
│   ├── memory.py         # Persistent memory — auto-injection, tools, provenance, episodes, admin commands, audit history
│   ├── plugins.py        # /plugin, /status, /changelog — plugin search & server status
│   ├── summary.py        # /resumo + summarize_channel tool — period parsing, fetch, map-reduce summary, permission checks
│   ├── scheduler.py      # /schedule + schedule_* tools — cron and one-shot tasks run by the chat engine
│   ├── spark.py          # /spark, Spark tools, passive spark.lucko.me detection
│   ├── spark_parser.py   # Spark JSON parsing, summary/detail builders, call-tree rendering
│   └── plugin_apis.py    # Shared Modrinth/Hangar/SpigotMC API helpers
├── responses/
│   └── errors.py         # 25 compiled Minecraft error regex patterns + pt-BR responses
├── tests/                # pytest suite, one file per module/feature (see Tests)
└── data/
    ├── vectors.json/.npy # Docs RAG vector store
    ├── memory.db          # Persistent memory (memories + history)
    ├── settings.db        # /config settings
    ├── persona.db         # Personality sections + history
    ├── conversations.db   # Conversations (handles, triggers, episode state)
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
| `LLM_REASONING_EFFORT` | *(unset)* | Reasoning effort sent on every chat completion: `minimal`, `low`, `medium` or `high`. Unset omits the parameter entirely (providers/models that don't accept it reject the request) |
| `LLM_CACHE_CONTROL` | `auto` | Explicit `cache_control` prompt-cache breakpoints on the system prompt and conversation tail. `auto` sends them only when `OPENAI_BASE_URL` is OpenRouter or the Merge Gateway (they forward them to Anthropic/Gemini); strict providers such as Mistral reject the field with a 422. `true`/`false` forces it on or off |
| `LLM_RETRY_ATTEMPTS` | `3` | Total attempts for a chat completion on 429, timeouts, connection errors and 5xx (`1` disables retries) |
| `LLM_RETRY_BASE_DELAY` | `1.0` | Seconds before the first retry, doubling after (a provider's `Retry-After` wins, capped at 10s) |
| `MERGE_PRIORITY_ORDER` | `particle,wafer,fireworks,modal,zai` | When `OPENAI_BASE_URL` points at the Merge Gateway (merge.dev), every chat completion carries an inline `priority_order` so the gateway tries these vendors in order (failing over on throttles, outages and timeouts). Comma-separated vendor slugs, highest priority first; empty disables |
| `SPARK_MODEL` | `google/gemini-2.5-pro` | Used only during active Spark sessions |
| `EMBEDDING_MODEL` | `qwen/qwen3-embedding-8b` | Doc/history embedding model |
| `RERANK_MODEL` | `cohere/rerank-4-fast` | Reranking model |
| `EMBEDDING_PROVIDER` | `openai` | `openai` (remote) or `local` (sentence-transformers) |
| `LOCAL_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Model when `EMBEDDING_PROVIDER=local` |
| `LOCAL_EMBEDDING_DEVICE` | `cpu` | Device for local embeddings |
| `RERANK_ENABLED` | `true` | Enable reranking (only when base URL is OpenRouter) |
| `RERANK_PROVIDER` | `auto` | `auto` (remote on OpenRouter, local elsewhere), `local` (cross-encoder) or `remote` (OpenRouter `/rerank`) |
| `LOCAL_RERANK_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder used when the rerank provider is local |
| `LOCAL_RERANK_DEVICE` | `cpu` | Device for the local reranker (defaults to `LOCAL_EMBEDDING_DEVICE`) |

### Feature Flags

| Variable | Default | Description |
|---|---|---|
| `WEB_SEARCH_ENABLED` | `true` | Enable Tavily web search |
| `TAVILY_API_KEY` | — | Required when web search is enabled |
| `CHAT_MENTION_ENABLED` | `true` | Answer @mentions, name calls and messages in the bot's threads (`false` leaves only replies to the bot) |
| `CHAT_NAME_TRIGGER_ENABLED` | `true` | Answer when called by name without @ (per channel: `/config canal name_trigger`) |
| `CHAT_NAME_ALIASES` | — | Extra names the bot answers to, comma-separated (servers add theirs with `/config servidor aliases`) |
| `CHAT_REPLY_STYLE` | `auto` | `auto` (plain messages; embeds for summaries, Spark, Wrapped and long answers), `plain` or `embed` (per channel: `reply_style`) |
| `CHAT_PLAIN_MAX_CHARS` | `2000` | In `auto`, answers longer than this go in an embed |
| `CHAT_BURST_SECONDS` | `2.5` | Quiet time after a person's last message before their burst is answered as one turn |
| `CHAT_BURST_SHORT_SECONDS` | `6` | Wait instead while the burst only points at something (short, no image/link/reply), so the image sent right after joins it |
| `CHAT_USER_BUDGET` / `CHAT_USER_BUDGET_WINDOW` | `8` / `300` | Answered turns per person per window in seconds (`0` = unlimited); over it the bot reacts ⏳ |
| `CHAT_GUILD_BUDGET` / `CHAT_GUILD_BUDGET_WINDOW` | `150` / `3600` | Answered turns per server per window, monitor runs included (`0` = unlimited) |
| `CHAT_DELETE_WITH_TRIGGER` | `true` | Deleting a message the bot answered deletes the answer |
| `CHAT_EDIT_WINDOW_SECONDS` | `60` | Editing it within this many seconds of the answer re-answers it (`0` never) |
| `CHAT_MENTION_CONTINUE_ENABLED` | `true` | A mention continues the newest bot conversation within the last `CHAT_MENTION_CONTINUE_WINDOW` (`10`) messages instead of starting a new one (a quoted message is added as context) |
| `CHAT_THREADS_ENABLED` | `true` | Move long or off-topic chat conversations into threads; the bot answers un-mentioned messages in its threads |
| `CHAT_THREAD_AFTER_TURNS` | `3` | One person's turn in a row with the bot, a reply or mention, that moves their conversation into a thread (`3` = the second follow-up; `0` never moves by length) |
| `CHAT_THREAD_CHANNELS` | — | Comma-separated channel IDs where every conversation opens a thread on its first answer |
| `CHAT_THREAD_TOOL_ENABLED` | `true` | Give the chat model the `open_thread` tool for long technical back-and-forths and off-topic subjects that will go on (not chat or quick questions) |
| `CHAT_THREAD_TITLE_MODEL` | `SUMMARY_MODEL` → `CHAT_MODEL` | Model that names the threads the bot opens on its own; a cheap model is enough |
| `CHAT_THREAD_ARCHIVE_MINUTES` | `60` | Idle minutes before a bot thread archives (Discord accepts `60`, `1440`, `4320`, `10080`; other values round up) |
| `CHANNEL_CONTEXT_MESSAGES` | `30` | Latest channel messages auto-injected as text and vision context into `/ask`, `/chat`, @mention and reply follow-ups (`0` disables) |
| `CHANNEL_CONTEXT_MAX_CHARS` | `6000` | Character budget of that window; the oldest lines are dropped first (`0` = no limit) |
| `CHAT_MENTION_CONTINUE_WINDOW` | `10` | How many recent channel messages a mention looks back for the bot's conversation |
| `CHANNEL_CONTEXT_IMAGES_ENABLED` | `true` | Send images attached to channel-context and follow-up gap messages as vision parts (`false` for text-only models) |
| `CHANNEL_CONTEXT_IMAGE_MAX_AGE_MINUTES` | `60` | Only channel-context images newer than this are sent (`0` = no age limit) |
| `CHANNEL_CONTEXT_REACTIONS_ENABLED` | `true` | Show reactions on channel-context lines (history tool, recent window, follow-up gaps, message context) |
| `REACTION_TOOL_ENABLED` | `true` | Give the chat model the `add_reaction` tool (current channel only, 3 per answer) |
| `REACTION_USERS_LIMIT` | `5` | Reactor names shown per reaction (one API call per reaction, cached while the count is unchanged; `0` = counts only) |
| `HISTORY_ENABLED` | `true` | Enable server history RAG |
| `MEMORY_ENABLED` | `true` | Enable persistent memory (cog not loaded when false) |
| `LOG_LEVEL` | `INFO` | Python logging level |

### History RAG

| Variable | Default | Description |
|---|---|---|
| `HISTORY_VECTOR_STORE_DIR` | `data/history` | Per-guild history storage dir |
| `HISTORY_DB_PATH` | `data/history/history.db` | SQLite history DB (defaults to `<HISTORY_VECTOR_STORE_DIR>/history.db`) |
| `HISTORY_WINDOW_SIZE` | `5` | Sliding window of prior messages per chunk |
| `HISTORY_WINDOW_OVERLAP` | `1` | Overlap between consecutive chunks |
| `HISTORY_BACKFILL_LIMIT` | _(none)_ | Max messages to backfill per channel per run (unset = all); restarts resume after the newest indexed message |
| `HISTORY_MAX_MSG_LENGTH` | `800` | Max chars per message in history chunks |
| `HISTORY_EXCLUDE_BOTS` | `true` | Exclude bot messages from history |
| `HISTORY_INGEST_BATCH_SIZE` | `10` | Messages per embedding batch during live ingestion |
| `HISTORY_INGEST_FLUSH_SECONDS` | `2.0` | Max delay before a partial batch is flushed |
| `HISTORY_SNAPSHOT_INTERVAL` | `300` | Seconds between vector-store snapshots to disk |
| `HISTORY_QUERY_CACHE_SIZE` | `200` | LRU query result cache size |
| `HISTORY_DEDUPE_WINDOW_MINUTES` | `10` | Adjacent results in the same channel treated as one window and deduplicated (`0` disables) |
| `HISTORY_RERANK_ENABLED` | `true` | Rerank history search results |
| `HISTORY_RERANK_PROVIDER` | `auto` | Rerank provider for history search (defaults to `RERANK_PROVIDER`) |
| `HISTORY_RERANK_MODEL` | `LOCAL_RERANK_MODEL` | Remote rerank model for history search |
| `HISTORY_TIME_DECAY_LAMBDA` | `0.0` | Exponential decay on recency for `recent` sorting (`0` disables) |
| `HISTORY_HYBRID_WEIGHT_SEMANTIC` | `0.65` | Semantic weight of the semantic/keyword hybrid blend |
| `HISTORY_HYBRID_WEIGHT_KEYWORD` | `0.35` | Keyword (FTS5) weight of the hybrid blend |
| `HISTORY_RRF_K` | `60` | Reciprocal-rank-fusion constant |
| `HISTORY_SQL_TOOL_ENABLED` | `true` | Enable the read-only `sql_history` LLM tool |
| `HISTORY_SQL_TIMEOUT_SECONDS` | `30` | Query timeout enforced by the SQLite progress handler |
| `HISTORY_SQL_MAX_ROWS` | `200` | Max rows returned / shown to the LLM |

### Vision Images

| Variable | Default | Description |
|---|---|---|
| `IMAGES_DIR` | `data/images` | Where downloaded images are persisted |
| `IMAGE_MAX_SIDE` | `640` | Max dimension (px) of re-encoded JPEGs (channel-context images) |
| `IMAGE_MAX_SIDE_SHOWN` | `1280` | Max dimension of images someone shows the bot — attached to the message it answers or the one that replies to, or to `/chat`, `/ask`, `/analyze` — so screenshots of text stay readable. They're labelled in the message as what the person is showing, apart from channel images |
| `IMAGE_JPEG_QUALITY` | `80` | JPEG quality of re-encoded images |
| `IMAGE_RETENTION_SECONDS` | `86400` | Stored images swept after this many seconds |
| `CONVERSATIONS_IMAGE_TURNS` | `3` | Inline stored images only for the most recent N history turns (older become text markers) |

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

### Channel Summaries

| Variable | Default | Description |
|---|---|---|
| `SUMMARY_ENABLED` | `true` | Enable `/resumo` and the `summarize_channel` tool (cog not loaded when false) |
| `SUMMARY_MODEL` | `CHAT_MODEL` | Model for the summary calls; a cheaper model is usually fine |
| `SUMMARY_MAX_MESSAGES` | `1000` | Max messages read per summary (the newest are kept) |
| `SUMMARY_MAX_DAYS` | `7` | Max lookback in days |
| `SUMMARY_SEGMENT_CHARS` | `24000` | Rendered message chars per LLM call; longer ranges are summarized map-reduce style |

### Monitors

| Variable | Default | Description |
|---|---|---|
| `MONITORS_ENABLED` | `true` | Enable `/monitor`, the `monitor_*` tools and the listener (cog not loaded when false) |
| `MONITORS_DB_PATH` | `data/monitors.db` | SQLite store for monitors |
| `MONITORS_MAX_PER_GUILD` | `25` | Max monitors (active or paused) per server |
| `MONITORS_MAX_PROMPT` | `500` | Max instruction length |
| `MONITORS_DEFAULT_COOLDOWN` | `60` | Seconds a monitor waits after firing before it can fire again |
| `MONITORS_MIN_COOLDOWN` | `10` | Lowest cooldown a monitor can be given |

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
| `CONVERSATIONS_MAX_STORED` | `200` | Max conversations kept |
| `CONVERSATIONS_MAX_TURNS` | `24` | Max turns stored per conversation |
| `CONVERSATIONS_HISTORY_TURNS` | `16` | Turns replayed to the LLM per request |
| `CONVERSATIONS_GAP_MESSAGES` | `20` | Max channel messages captured between two bot-directed turns as text and vision `prior_context` (`0` disables) |
| `CONVERSATIONS_TRAJECTORY_ENABLED` | `true` | Capture and replay the internal LLM trajectory (tool calls/results + exact user message) per turn |
| `CONVERSATIONS_TRAJECTORY_TURNS` | `6` | Most recent N turns replayed **verbatim** with their trajectory (older turns use the compact Q/A rendering) |
| `CONVERSATIONS_TRAJECTORY_STEP` | `3` | Window advances in steps of N turns — prefix-cache hysteresis so the replayed prefix only changes at deliberate boundaries |
| `CONVERSATIONS_TRAJECTORY_MAX_CHARS` | `120000` | Hard budget on the replayed trajectory payload; turns are collapsed to compact rendering when exceeded |

### Personality, Episodes, Settings and Wrapped

| Variable | Default | Description |
|---|---|---|
| `PERSONA_ENABLED` | `true` | The `<personalidade>` block, `/persona` and the `persona_update` tool |
| `PERSONA_DB_PATH` | `data/persona.db` | Personality sections and their history |
| `PERSONA_BASE_MAX_CHARS` / `PERSONA_NOTES_MAX_CHARS` | `3000` / `800` | Section sizes (at most 4000, Discord's text box limit) |
| `PERSONA_BOT_EDITS_PER_DAY` | `5` | Times a day the bot may rewrite a server's notes on its own (`0` never) |
| `PERSONA_LOG_CHANNEL_ID` | `MEMORY_LOG_CHANNEL_ID` | Where personality changes are posted with a Revert button |
| `EPISODES_ENABLED` | `true` | Summarize expired conversations into one *conversa* memory per participant (needs `MEMORY_ENABLED`) |
| `EPISODES_MODEL` | `SUMMARY_MODEL` → `CHAT_MODEL` | Model for those summaries |
| `EPISODES_PER_USER` | `20` | Newest episodes kept per person (older ones archived) |
| `EPISODES_INTERVAL_SECONDS` | `300` | How often expired conversations are summarized |
| `SETTINGS_DB_PATH` | `data/settings.db` | `/config` settings |
| `WRAPPED_ENABLED` | `true` | `/wrapped`, the `server_wrapped` tool and automatic recaps |
| `WRAPPED_HOUR` | `10` | Brasília hour automatic recaps are posted (Mondays / the 1st) |

### Other

| Variable | Default | Description |
|---|---|---|
| `COOLDOWN_RATE` | `1` | Allowed uses per cooldown window of `/analyze`, `/spark`, `/resumo`, `/wrapped` |
| `COOLDOWN_PER` | `30` | Cooldown window in seconds (per user) |
| `MAX_CONTENT_SIZE` | `5242880` | Max bytes fetched from paste services (5 MB) |
| `MAX_LOG_CONTEXT` | `12000` | Max chars sent to LLM for log analysis |

### API Request Logging

Implemented in [`api_logger.py`](api_logger.py) — patches the `aiohttp`/`httpx`/`httpx2` transports so every outbound HTTP request (Discord REST, LLM API, Tavily, GitHub, plugin APIs) and every inbound Discord interaction is appended as one JSON line to a rotating file. See [SETUP.md](SETUP.md) for sample lines and privacy notes.

| Variable | Default | Description |
|---|---|---|
| `API_REQUEST_LOG_ENABLED` | `true` | Master switch |
| `API_REQUEST_LOG_PATH` | `data/api_requests.log` | Log file path (parent dirs created) |
| `API_REQUEST_LOG_MAX_BYTES` | `10485760` | Rotate after this many bytes |
| `API_REQUEST_LOG_BACKUPS` | `5` | Rotated files to keep |
| `API_REQUEST_LOG_DISCORD` | `true` | `false` omits outbound Discord REST traffic (inbound interactions are logged regardless) |
| `API_REQUEST_LOG_CONSOLE` | `false` | Also mirror log lines to stdout / docker logs |
| `API_REQUEST_LOG_BODY` | `openai,tavily` | Services whose request/response bodies are logged (`all`/`none` accepted) |
| `API_REQUEST_LOG_BODY_MAX_CHARS` | `20000` | Per-body truncation limit |

---

## Running the Bot

```bash
python main.py
```

On startup the bot:

1. Validates `BOT_TOKEN` (exits if missing; warns if `TAVILY_API_KEY` missing while web search is enabled)
2. Builds `bot.services` (LLM client, tool registry, router, settings, conversation store) and loads the settings
3. Loads the cogs; each registers its tools, router observers/handlers and services (first run imports legacy `lore.db` automatically)
4. Loads cached vectors from disk or indexes all doc sources
5. Starts periodic reindex, background history backfill, the scheduler, the episode worker and automatic Wrapped
6. Syncs slash commands with Discord

Cog load order doesn't matter: the router decides who sees a message, not the order listeners were registered in.

---

## Docker

```bash
docker compose up --build -d
```

- Uses `python:3.11-slim`, persists `data/` via volume mount, reads config from `.env`.

---

## Tests

The suite lives in `tests/` and runs with pytest. It needs no Discord token, network or models: Discord objects, the LLM client and embeddings are faked, and SQLite databases go in temporary directories.

```bash
pip install -r requirements-dev.txt
pytest                         # whole suite
pytest tests/test_summary.py   # one file
pytest -k reindex              # tests whose name matches
ruff check .                   # lint (rules configured in pyproject.toml)
```

`.github/workflows/ci.yml` runs `ruff check .` and `pytest` on every push to `main` and on pull requests, on Python 3.11 — the version the Docker image ships, so a newer local interpreter can't hide an incompatibility from production.

| File | Covers |
|---|---|
| `test_api_logger.py` | Request logging for aiohttp/httpx/httpx2, redaction, body capture, inbound interactions (runs in its own process) |
| `test_attribution.py` | Author tags on every question, byte-identical replay of per-turn blocks, memory block labels and person-memory rules |
| `test_channel_context_images.py` | Images from automatically fetched channel context |
| `test_chat_threads.py` | When conversations move into threads (incl. the `threads` setting; one person's back-and-forth, not name calls or a group), delivery and pings, reusing a moved conversation's thread, mentions and name calls in the channel not following it there, un-mentioned messages and silence in bot threads, the `open_thread` tool (only for long or off-topic subjects), the model naming threads it opens on its own (falling back to the first question) |
| `test_context_and_requests.py` | The wider channel window and its budget, waiting for the image after "ai oh @bot", sharper shown images labelled apart from channel images, the search hint |
| `test_conversation_flow.py` | Bursts (follow-ups while answering only after a mention or reply), budget, name calls, aliases and `respond` settings, channel instructions, reply style, staying quiet, edits and deletes, `/chat` and `/ask` |
| `test_delivery.py` | Plain vs embed, splitting long answers (code blocks stay valid), real pings for humans only, sources line, threads |
| `test_conversation_participants.py` | Participant scope for shared conversations and memory writes |
| `test_docs_rag.py` | Docs `/reindex`: partial and full reindex, failed sources kept |
| `test_history_backfill.py` | History backfill watermarks and the duplicate-append guard |
| `test_history_perf.py` | History search/indexing performance fixes (threading, FTS deletes, matrix buffer) |
| `test_history_search.py` | FTS search, filters, dedupe, authors table, message context, `find_user` |
| `test_history_sql.py` | The read-only `sql_history` tool and its sandbox |
| `test_memory_provenance_episodes.py` | Who told a fact (`told_by`, "segundo Fulano"), episode memories with their cap, expired conversations waiting for their summary, the episode writer |
| `test_mention_continuation.py` | A mention continues the bot's recent conversation (or not), with a quoted message as context |
| `test_mentions.py` | Which mentions in a reply become real pings |
| `test_message_media.py` | Stickers, GIFs, custom emojis, reactions and the `add_reaction` tool |
| `test_monitors.py` | Monitor validation, filter matching, store, cooldowns, skipping messages the bot already answers, tools and permissions, silent vs. reply runs |
| `test_persona.py` | Personality store and versions, the block in the system prompt, who may change base vs notes, daily limit and lock, revert and log, importing memories, the edit modal |
| `test_runner.py` | The runner: docs tools in normal chat, profile tool order, server personality and channel instructions in the prompt, structured tools, LLM retries, status lines |
| `test_scheduler.py` | Cron job firing and rescheduling |
| `test_settings_router.py` | `/config` values and lookup order (channel → parent → server → default), the message router |
| `test_summary.py` | `/resumo`: period parsing, fetching, map-reduce, permissions |
| `test_tool_loop.py` | Tool errors reported to the model instead of aborting the turn |
| `test_trajectory_replay.py` | Trajectory capture, verbatim replay, text tool-call recovery |
| `test_wrapped.py` | Wrapped numbers, weekly/monthly schedule, automatic posts once per period, the tool |

`tests/conftest.py` puts the repo root on the import path, runs `async def` tests without a plugin, and provides a `tmpdb` fixture. `tests/helpers.py` has `check(label, condition, detail)`, a labelled assert. `tests/chat_fakes.py` has fake Discord objects and `make_engine`, a real chat engine on temp stores whose turns answer from a script. New tests go in the file for the module they cover, or a new `test_<module>.py`.

---

## Radar de Hardware via MCP

Optional integration with [Radar de Hardware](https://radardehardware.com.br)
for current Brazilian prices, price history and specifications of GPUs, CPUs
and RAM through its Streamable HTTP MCP endpoint.

Install `requirements.txt`, set `RADAR_MCP_ENABLED=true` in `.env`, and restart
the bot. No API key or Tavily configuration is required. Disabled by default.
`RADAR_MCP_URL` defaults to `https://radardehardware.com.br/mcp`;
`RADAR_MCP_TIMEOUT_SECONDS` limits each request (default: 30 seconds).

In `/chat`, mentions and conversations, ask for example: “Quais placas de vídeo
até R$ 2.000 estão disponíveis?” or “Como evoluiu o preço da RTX 5070 Ti?”.
The model uses `radar_list_tools` to discover current schemas, then
`radar_call_tool` to query the server. MCP links, structured content and error
flags are preserved. Network failures are reported to the model so it can
continue the conversation. `/ask` and Spark keep their documentation tools.

---

## License

See repository license. Documentation sources retain their original licenses.
