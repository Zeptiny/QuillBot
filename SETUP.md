# QuillBot — Setup & Configuration Guide

Complete guide to configure and run the QuillBot Discord bot for Minecraft server administration.

---

## Prerequisites

- **Python 3.11+**
- A **Discord bot** created via the [Discord Developer Portal](https://discord.com/developers/applications)
- An **OpenRouter API key** from [openrouter.ai](https://openrouter.ai) (required for AI-powered commands)

---

## 1. Installation

```bash
git clone https://github.com/Zeptiny/QuillBot.git
cd QuillBot
pip install -r requirements.txt
```

### Dependencies

| Package       | Purpose                               |
|---------------|---------------------------------------|
| `discord.py`  | Discord API wrapper                   |
| `discord.py[voice]`, `discord-ext-voice-recv` | Voice transport, DAVE encryption and receiving audio |
| `websockets`, `certifi` | TLS connection to Qwen's real-time audio API |
| `aiohttp`     | Async HTTP requests                   |
| `openai`      | OpenRouter API client (OpenAI compat) |
| `cachetools`  | TTL cache for conversation history    |
| `python-dotenv` | Load `.env` file at startup         |

---

## 2. Environment Variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

### Required

| Variable           | Description                                                  |
|--------------------|--------------------------------------------------------------|
| `BOT_TOKEN`        | Your Discord bot token. The bot **will not start** without it. |
| `OPENROUTER_API_KEY` | API key from OpenRouter. Required for every AI feature: conversations (`/chat`, `/ask`, mentions, replies), `/analyze`, Spark, summaries and RAG indexing. |

### Optional — AI Models

| Variable         | Default                        | Description                                  |
|------------------|--------------------------------|----------------------------------------------|
| `CHAT_MODEL`     | `qwen/qwen3.6-plus` | LLM model used for conversations and `/analyze`. |
| `EMBEDDING_MODEL`| `qwen/qwen3-embedding-8b`     | Model used to embed documentation chunks.    |
| `RERANK_MODEL`   | `cohere/rerank-4-fast`         | Model used to rerank search results.         |

> Text and embedding models use the configured OpenAI-compatible provider. Voice connects directly to Token Plan.

### Optional — Real-time Voice

1. Install the updated `requirements.txt`. Local Linux installations also need Opus (`sudo apt-get install libopus0` on Debian/Ubuntu or `sudo dnf install opus` on Fedora); macOS uses `brew install opus`. The Docker image includes it, and discord.py bundles it on Windows.
2. Set `TOKEN_PLAN_API_KEY` to your Token Plan subscription key in the bot environment. `ANTHROPIC_AUTH_TOKEN` is the fallback. The default endpoint is Singapore; set `QWEN_REALTIME_URL` for the subscription's region if different. Do not use the OpenRouter key for voice.
3. Grant **View Channel**, **Connect**, **Speak**, and **Send Messages** in the voice channel, then restart the bot.
4. Join a normal voice channel and run `/voice entrar`. The bot announces that channel audio goes to Qwen, then listens and speaks. Use `/voice sair` to stop or `/voice status` to check the session.

The default model is `qwen-audio-3.0-realtime-plus`, with voice `longanqian`. All human participants can talk to it; use headphones and take turns. The bot leaves empty channels and disconnects on failures, after `VOICE_IDLE_TIMEOUT_SECONDS=300` of inactivity, or after `VOICE_MAX_SESSION_SECONDS=1800` total. `VOICE_ENABLED=false` disables the commands. It uses the server's personality, but does not access text tools or memory, or save audio/transcripts. See [voice configuration](README.md#real-time-voice) for all variables and behavior.

### Optional — Rate Limiting

| Variable       | Default | Description                                         |
|----------------|---------|-----------------------------------------------------|
| `CHAT_USER_BUDGET` / `CHAT_USER_BUDGET_WINDOW` | `8` / `300` | Answered conversation turns per person per window (seconds). Over it the bot reacts ⏳ instead of answering. |
| `CHAT_GUILD_BUDGET` / `CHAT_GUILD_BUDGET_WINDOW` | `150` / `3600` | Answered turns per server per window, monitor runs included. |
| `CHAT_BURST_SECONDS` | `2.5` | Quick consecutive messages from one person are answered together once they've been quiet this long. |
| `COOLDOWN_RATE`| `1`     | Uses allowed per cooldown window of `/analyze`, `/spark`, `/resumo`, `/wrapped`. |
| `COOLDOWN_PER` | `30`    | Cooldown window in seconds (per user).               |

Users who exceed a slash-command cooldown receive an ephemeral "wait X seconds" message.

### Optional — API Request Logging

Every outbound HTTP request (discord.py REST, the OpenAI-compatible LLM API, Tavily, GitHub, plugin APIs) and every inbound Discord interaction is appended as one JSON line to a rotating file. Implemented in [`api_logger.py`](api_logger.py) by patching the `aiohttp`, `httpx` and `httpx2` transports (recent `openai` SDKs send through `httpx2`) — no call-site changes needed.

| Variable | Default | Description |
|---|---|---|
| `API_REQUEST_LOG_ENABLED` | `true` | Master switch. |
| `API_REQUEST_LOG_PATH` | `data/api_requests.log` | Log file path (parent dirs are created). |
| `API_REQUEST_LOG_MAX_BYTES` | `10485760` | Rotate the file after this many bytes. |
| `API_REQUEST_LOG_BACKUPS` | `5` | Rotated files to keep. |
| `API_REQUEST_LOG_DISCORD` | `true` | Set `false` to omit outbound Discord REST traffic (message sends, history fetches) — the noisiest service. Inbound interactions are logged regardless. |
| `API_REQUEST_LOG_CONSOLE` | `false` | Also mirror API log lines to stdout (visible in `docker compose logs`). |
| `API_REQUEST_LOG_BODY` | `openai,tavily` | Comma-separated services whose request/response JSON bodies are logged. `openai` = whatever host `OPENAI_BASE_URL` points at (OpenRouter by default). Accepts `all` or `none`. |
| `API_REQUEST_LOG_BODY_MAX_CHARS` | `20000` | Per-body truncation limit. Longer bodies keep their first and last halves (a chat request's current message is at the end); inline base64 images are replaced by their size. Raise it (e.g. `200000`) to see whole conversations while debugging. |

Example lines:

```json
{"ts":"2026-08-28T17:18:47.123+00:00","dir":"outbound","service":"openai","method":"POST","url":"https://openrouter.ai/api/v1/chat/completions","status":200,"duration_ms":1834.2,"model":"qwen/qwen3.6-plus","request_body":{"model":"...","messages":[...]},"response_body":{"choices":[...],"usage":{"prompt_tokens":9123}}}
{"ts":"2026-08-28T17:18:51.780+00:00","dir":"inbound","service":"discord","interaction_type":"application_command","command":"ask","user":"nyuu","user_id":123,"guild_id":456,"channel_id":789}
```

Notes:

- **Privacy:** captured bodies include users' message content sent to the LLM and search queries — they persist to disk.
- Bodies are captured for **httpx/httpx2-based** services (`openai`, `tavily`); **aiohttp-based** services (Discord REST, GitHub, Modrinth/Hangar/SpigotMC, mclo.gs, spark) log metadata only.
- Headers are never logged; API keys travel in headers. Sensitive query-string values (`token`, `key`, …) are redacted to `REDACTED`.

### Optional — RAG / Docs

| Variable               | Default              | Description                                              |
|------------------------|----------------------|----------------------------------------------------------|
| `VECTOR_STORE_PATH`    | `data/vectors.json`  | Where indexed doc embeddings are persisted on disk.      |
| `REINDEX_INTERVAL_HOURS`| `6`                 | How often (in hours) the bot checks for doc repo updates.|

---

## 3. Documentation Sources (`DOC_SOURCES`)

The bot indexes documentation from multiple GitHub repositories for RAG-powered search. This is configured in [`config.py`](config.py) via the `DOC_SOURCES` list.

Each entry is a dictionary with the following keys:

| Key                | Required | Description                                                        |
|--------------------|----------|--------------------------------------------------------------------|
| `repo`             | Yes      | GitHub repo in `owner/name` format (e.g. `MinersRefuge/docs`).    |
| `branch`           | Yes      | Branch to index (e.g. `main`).                                    |
| `base_url`         | Yes      | Base URL of the docs website (used to build clickable links).      |
| `label`            | Yes      | Human-readable source label shown in search results.               |
| `summary`          | No       | Path to an mdBook `SUMMARY.md` file for page discovery.           |
| `path_prefix`      | No       | Only index files whose repo path starts with this prefix.          |
| `url_strip_prefix` | No       | Strip this prefix from the repo path when building website URLs.   |
| `max_files`        | No       | Maximum number of files to index per source (default: 200).        |

### Default sources

```python
DOC_SOURCES = [
    {
        'repo': 'MinersRefuge/docs',
        'branch': 'main',
        'base_url': 'https://docs.minersrefuge.com.br',
        'label': "Miners' Refuge",
        'summary': 'SUMMARY.md',          # Discovers pages via mdBook SUMMARY
    },
    {
        'repo': 'PaperMC/docs',
        'branch': 'main',
        'base_url': 'https://docs.papermc.io',
        'label': 'PaperMC',
        'path_prefix': 'docs/paper/admin', # Only admin docs
        'url_strip_prefix': 'docs/',       # Strip "docs/" from path for URL
        'max_files': 30,
    },
]
```

### Adding a new source

Add a new dictionary to the `DOC_SOURCES` list in `config.py`. For repositories that use an mdBook-style `SUMMARY.md`, set the `summary` key. Otherwise, the bot uses the GitHub tree API and filters by `path_prefix` and file extension (`.md`, `.mdx`).

After adding a source, restart the bot or run `/reindex` (admin-only) to index it.

---

## 4. Discord Bot Setup

### Required Permissions

- **Send Messages** — respond to commands and auto-detect logs
- **Embed Links** — all responses use Discord embeds
- **Read Message History** — follow-up conversations via replies
- **Add Reactions** — the 👀 reaction on detected paste links
- **Use Application Commands** — slash commands
- **View Channel**, **Connect**, **Speak** — real-time voice; **Send Messages** must also be allowed in the voice channel for its start notice
- **Create Public Threads** and **Send Messages in Threads** — move long or off-topic chats into threads (without them the bot answers in the channel)

### Required Intents

- **Message Content** — needed for passive log/error detection in chat
- **Voice States** — enabled in code for voice sessions; not a privileged portal toggle

Enable this in the Developer Portal under **Bot → Privileged Gateway Intents → Message Content Intent**.

### Inviting the Bot

Use the OAuth2 URL generator in the Developer Portal:

1. Go to **OAuth2 → URL Generator**
2. Select scopes: `bot`, `applications.commands`
3. Select the permissions listed above
4. Use the generated URL to invite the bot to your server

---

## 5. Running the Bot

```bash
python main.py
```

On first start the bot will:
1. Validate that `BOT_TOKEN` is set (exits immediately if not)
2. Build the shared services (`bot.services`) and load the `/config` settings
3. Load all cogs; each registers its tools and its part of the message router
4. Index documentation from all `DOC_SOURCES` (if no cached vectors exist)
5. Sync slash commands with Discord
6. Start the periodic reindex loop, history backfill, scheduler, episode worker and automatic Wrapped

### Message routing

Every message goes through one router (`core/router.py`): passive observers (history indexing, log and Spark detection) run concurrently, then the chat engine and monitors are offered the message in that order. Cog load order doesn't matter.

---

## 6. Commands Reference

| Command     | Description                                         | Requires AI |
|-------------|-----------------------------------------------------|:-----------:|
| `/chat`     | Talk to the bot (same as mentioning it)              | Yes         |
| `/ask`      | Ask a question answered from indexed documentation  | Yes         |
| `/resumo`   | Summarize what happened in a channel                 | Yes         |
| `/wrapped`  | Server recap: who talked most, channels, emojis…     | Optional**  |
| `/config`   | Server and channel settings (Manage Server)          | No          |
| `/persona`  | The bot's personality on this server (editing needs Manage Server) | No |
| `/memory`, `/monitor`, `/schedule` | Memories, standing triggers, scheduled tasks | Yes |
| `/analyze`  | Analyze a server log with AI (mclo.gs, pastebin, or file) | Yes   |
| `/docs`     | Search docs or get the documentation link           | No*         |
| `/plugin`   | Search for a plugin on Modrinth, Hangar, SpigotMC   | No          |
| `/status`   | Check if a Minecraft server is online               | No          |
| `/changelog`| Show recent commits to the docs repo                | No          |
| `/flags`    | Generate Aikar's JVM flags for a given RAM amount   | No          |
| `/plov`     | Info needed to choose a hosting provider             | No          |
| `/plano`    | Info needed to recommend a hosting plan              | No          |
| `/reindex`  | Re-index documentation (admin only)                  | Yes         |
| `/help`     | List all available commands (auto-generated)         | No          |

*`/docs` with a query delegates to RAG search if available, otherwise shows a link.
**The numbers come from the history index; the AI only writes the intro.

---

## 7. Passive Features

The bot automatically monitors messages for:

- **mclo.gs / pastebin.com links** — reacts with 👀, fetches the log, and matches known error patterns
- **.log / .txt file attachments** — reads the file, matches patterns, and uploads to mclo.gs for easy sharing
- **Known Minecraft error messages** — replies with targeted troubleshooting advice
- **Conversations** — @mentions, replies to the bot, messages in its threads and its name ("quill, …"); quick consecutive messages are answered together (see README → Conversations)
- **Edits and deletes** — deleting a message the bot answered deletes the answer; editing it shortly after re-answers it

---

## 8. Project Structure

```
QuillBot/
├── .env.example        # Template for environment variables
├── config.py           # Centralized configuration (env vars, doc sources)
├── main.py             # Bot entrypoint, cog loading, command sync
├── requirements.txt    # Python dependencies
├── core/               # Shared machinery: LLM client, tool registry, router, chat engine, delivery…
├── cogs/
│   ├── chat.py         # /chat, /ask, conversations, edits/deletes, episodes
│   ├── commands.py     # Utility slash commands (/docs, /help, /flags, etc.)
│   ├── server_config.py# /config
│   ├── persona.py      # /persona and the bot's personality per server
│   ├── wrapped.py      # /wrapped and automatic recaps
│   ├── docs_rag.py     # RAG pipeline: indexing, search, /reindex, search_docs tool
│   ├── log_analyzer.py # Log analysis: pattern matching, /analyze, AI analysis
│   ├── plugins.py      # External API commands: /plugin, /status, /changelog
│   └── …               # memory, history_rag, monitors, scheduler, summary, spark (see README)
└── responses/
    └── errors.py       # Known Minecraft error patterns and responses
```
