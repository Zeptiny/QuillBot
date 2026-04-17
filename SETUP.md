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
| `OPENROUTER_API_KEY` | API key from OpenRouter. Required for `/ask`, `/analyze`, RAG indexing, and follow-up conversations. If not set, the `DocsRAG` and `LogAnalyzer` AI features are disabled. |

### Optional — AI Models

| Variable         | Default                        | Description                                  |
|------------------|--------------------------------|----------------------------------------------|
| `CHAT_MODEL`     | `google/gemini-2.5-flash-lite` | LLM model used for `/ask` and `/analyze`.    |
| `EMBEDDING_MODEL`| `qwen/qwen3-embedding-8b`     | Model used to embed documentation chunks.    |
| `RERANK_MODEL`   | `cohere/rerank-4-fast`         | Model used to rerank search results.         |

> All models are routed through OpenRouter. You can swap to any model available there by changing the identifier.

### Optional — Rate Limiting

| Variable       | Default | Description                                         |
|----------------|---------|-----------------------------------------------------|
| `COOLDOWN_RATE`| `1`     | Number of command uses allowed per cooldown window.  |
| `COOLDOWN_PER` | `30`    | Cooldown window in seconds (per user).               |

These apply to `/ask` and `/analyze`. Users who exceed the limit receive an ephemeral "wait X seconds" message.

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

### Required Intents

- **Message Content** — needed for passive log/error detection in chat

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
2. Load all cogs in order: `log_analyzer` → `commands` → `plugins` → `docs_rag`
3. Index documentation from all `DOC_SOURCES` (if no cached vectors exist)
4. Sync slash commands with Discord
5. Start the periodic reindex loop

### Cog load order

The load order in `main.py` matters because both `LogAnalyzer` and `DocsRAG` register `on_message` listeners. The `LogAnalyzer` listener (pattern matching) runs first, followed by `DocsRAG` (follow-up replies). Do not reorder without reviewing listener interactions.

---

## 6. Commands Reference

| Command     | Description                                         | Requires AI |
|-------------|-----------------------------------------------------|:-----------:|
| `/ask`      | Ask a question answered from indexed documentation  | Yes         |
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

---

## 7. Passive Features

The bot automatically monitors messages for:

- **mclo.gs / pastebin.com links** — reacts with 👀, fetches the log, and matches known error patterns
- **.log / .txt file attachments** — reads the file, matches patterns, and uploads to mclo.gs for easy sharing
- **Known Minecraft error messages** — replies with targeted troubleshooting advice
- **Reply-based follow-ups** — reply to a `/ask` response to continue the conversation

---

## 8. Project Structure

```
QuillBot/
├── .env.example        # Template for environment variables
├── config.py           # Centralized configuration (env vars, doc sources)
├── main.py             # Bot entrypoint, cog loading, command sync
├── requirements.txt    # Python dependencies
├── cogs/
│   ├── commands.py     # Utility slash commands (/docs, /help, /flags, etc.)
│   ├── docs_rag.py     # RAG pipeline: indexing, search, /ask, /reindex
│   ├── log_analyzer.py # Log analysis: pattern matching, /analyze, AI analysis
│   └── plugins.py      # External API commands: /plugin, /status, /changelog
└── responses/
    └── errors.py       # Known Minecraft error patterns and responses
```
