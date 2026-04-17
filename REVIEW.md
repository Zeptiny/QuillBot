# QuillBot — Audit Review: LLM, Prompting & Feature Roadmap

Items identified during the architectural audit that are **not yet implemented**.
Organized by priority.

---

## 1. LLM / Prompting Improvements

### 1.3 New Tools to Expose

| Tool | Source | Purpose |
|---|---|---|
| `check_server_status` | mcsrvstat.us API | Let the LLM check if a user's server is online when diagnosing |
| `get_paper_config_docs` | RAG + PaperMC docs | Dedicated config key lookup |
| `check_plugin_compatibility` | Modrinth version API | Check specific MC version support |
| `get_java_version_info` | Static mapping | MC version → recommended Java version |

### 1.4 Context Window Management

**Current problems:**
- `_truncate_safe(result_text, limit=6000)` per tool × `MAX_TOOL_ROUNDS=4` = up to 24K chars of tool results
- History replays 3 full Q&A pairs verbatim
- `max_tokens=1024` truncates complex answers

**Recommendations:**
```python
# Track cumulative context size
MAX_CONTEXT_CHARS = 20000

# Summarize older history
if history and len(history) > 1:
    summary = f"Resumo: {history[-2]['question']} → {history[-2]['answer'][:200]}"
    messages.append({'role': 'user', 'content': summary})
    # Only keep last exchange verbatim
    messages.append({'role': 'user', 'content': history[-1]['question']})
    messages.append({'role': 'assistant', 'content': history[-1]['answer']})

# Increase max_tokens
max_tokens=2048
```

### 1.5 Anthropic Best Practices Checklist

| Principle | Status | Action Needed |
|---|---|---|
| Clear and direct instructions | ✅ Good | — |
| Give Claude a role | ✅ Good | — |
| XML tags for structure | ❌ Missing | Wrap in `<role>`, `<instructions>`, `<response_format>` |
| Few-shot examples | ❌ Missing | Add 1-2 examples of ideal answers |
| Positive instructions over "don't" | ⚠️ Partial | Rephrase "NÃO inclua" → "Omita" |
| Tool descriptions: when NOT to use | ⚠️ Partial | Add negative guidance to tool descriptions |

---

## 2. Feature Roadmap

### 2.1 Interactive Timings/Spark Analysis (High Impact)

Add `/timings` command:
- Accept Spark report URL (`spark.lucko.me`)
- Parse heap summary, tick stats, top plugins from the Spark JSON API
- Feed structured data to the LLM for diagnosis
- Most commonly shared artifact in MC admin channels

### 2.2 Server Configuration Wizard (Medium Impact)

Multi-step `discord.ui.Modal`:
1. Collect: RAM, player count, MC version, server type (Paper/Purpur/Fabric)
2. LLM generates optimized config snippets from RAG context
3. Outputs `server.properties`, `paper-global.yml`, `purpur.yml` changes
4. Converts passive documentation into active, personalized guidance

### 2.3 Plugin Compatibility Checker (Medium Impact)

`/compat <plugin> <version>`:
- Query Modrinth/Hangar version-specific API endpoints
- Cross-reference with known incompatibilities
- Expose as an LLM tool (`check_compatibility`) for natural-language queries
- Addresses the most repetitive question type in support channels

---

## 3. UX Enhancements (Minor)

- **Autocomplete on `/plugin name`** — Use `app_commands.autocomplete` with Modrinth search suggestions
- **Pagination for long results** — Use `discord.ui.View` with prev/next buttons for `/plugin` and `/docs` results
- **Inform users when RAG isn't ready** — `/docs` with query should say "indexação em andamento" instead of silently falling back
- **Add `/sync` admin command** — Replace the removed `on_ready` auto-sync with a manual admin command:

```python
@app_commands.command(name='sync', description='Sincronizar comandos (Admin)')
@app_commands.checks.has_permissions(administrator=True)
async def sync_commands(self, interaction: discord.Interaction):
    synced = await interaction.client.tree.sync()
    await interaction.response.send_message(
        f'✅ {len(synced)} comandos sincronizados.', ephemeral=True
    )
```
